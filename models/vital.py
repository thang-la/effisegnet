import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBNFeatures
from monai.networks.nets.efficientnet import get_efficientnet_image_size


class MAGMA(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projection layers (no change if visual_dim == text_dim)
        self.proj_v = nn.Linear(visual_dim, visual_dim)
        if text_dim != visual_dim:
            self.proj_t = nn.Linear(text_dim, visual_dim)
        else:
            self.proj_t = nn.Identity()

        # Query, Key, Value projections for cross-attention
        self.W_q_v = nn.Linear(visual_dim, visual_dim)
        self.W_k_t = nn.Linear(visual_dim, visual_dim)
        self.W_v_t = nn.Linear(visual_dim, visual_dim)

        self.W_q_t = nn.Linear(visual_dim, visual_dim)
        self.W_k_v = nn.Linear(visual_dim, visual_dim)
        self.W_v_v = nn.Linear(visual_dim, visual_dim)

        # Channel-wise Learnable Gating
        self.gate_mlp = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.ReLU(inplace=True),
            nn.Linear(visual_dim, visual_dim * 2),
            nn.Sigmoid(),
        )

        # Final fusion
        self.W_f = nn.Linear(visual_dim * 2, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, F_v, F_t):
        B, C, H, W = F_v.shape
        N = H * W

        # Handle both 2D and 3D text features
        if F_t.dim() == 2:  # [B, C_t]
            F_t = F_t.unsqueeze(1)  # [B, 1, C_t]
        elif F_t.dim() == 3:  # [B, C_t, M]
            F_t = F_t.transpose(1, 2)  # [B, M, C_t]

        # Reshape visual features
        F_v_flat = F_v.flatten(2).transpose(1, 2)  # [B, N, C]

        # Project features
        F_v_proj = self.proj_v(F_v_flat)  # [B, N, C]
        F_t_proj = self.proj_t(F_t)  # [B, M, C] where M is 1 for our case

        # Global Average Pooling for gating
        alpha_v_input = F_v_proj.mean(dim=1)  # [B, C]
        alpha_t_input = F_t_proj.mean(dim=1)  # [B, C]

        # Cross-Attention: Text -> Visual
        Q_v = (
            self.W_q_v(F_v_proj)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K_t = (
            self.W_k_t(F_t_proj)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V_t = (
            self.W_v_t(F_t_proj)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_t2v = torch.softmax(
            torch.matmul(Q_v, K_t.transpose(-2, -1)) * self.scale, dim=-1
        )
        A_t2v = torch.matmul(attn_t2v, V_t).transpose(1, 2).reshape(B, N, C)

        # Cross-Attention: Visual -> Text
        Q_t = (
            self.W_q_t(F_t_proj)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K_v = (
            self.W_k_v(F_v_proj)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V_v = (
            self.W_v_v(F_v_proj)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_v2t = torch.softmax(
            torch.matmul(Q_t, K_v.transpose(-2, -1)) * self.scale, dim=-1
        )
        A_v2t = torch.matmul(attn_v2t, V_v).transpose(1, 2).reshape(B, -1, C)

        # Channel-wise Learnable Gating
        z = torch.cat([alpha_v_input, alpha_t_input], dim=1)
        gates = self.gate_mlp(z)
        alpha_v, alpha_t = gates.chunk(2, dim=1)

        # Apply gating
        F_v_tilde = alpha_v.unsqueeze(1) * (F_v_proj + A_t2v)
        A_v2t_pooled = A_v2t.mean(dim=1, keepdim=True).expand(-1, N, -1)
        F_t_tilde = alpha_t.unsqueeze(1) * (F_v_proj + A_v2t_pooled)

        # Concat -> Linear -> Norm
        F_c = torch.cat([F_v_tilde, F_t_tilde], dim=-1)
        F_c = self.W_f(F_c)
        F_c = self.norm(F_c)

        # Reshape back to spatial
        F_c = F_c.transpose(1, 2).reshape(B, C, H, W)

        return F_c


class LearnableUpsampling(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
        )

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        modulation = self.mlp(z).unsqueeze(-1).unsqueeze(-1)

        offset = 0.5 * torch.sigmoid(self.conv1(x) + modulation) * self.conv2(x)

        x_up = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        offset_up = F.interpolate(
            offset, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        return x_up + offset_up


class SpatialChannelAttentionGate(nn.Module):
    def __init__(self, visual_channels, text_channels):
        super().__init__()

        self.conv_v_spatial = nn.Conv2d(visual_channels, visual_channels, 1)
        self.conv_t_spatial = nn.Conv2d(text_channels, visual_channels, 1)
        self.conv_spatial = nn.Conv2d(visual_channels, 1, 1)

        self.fc_v_channel = nn.Linear(visual_channels, visual_channels)
        self.fc_t_channel = nn.Linear(text_channels, visual_channels)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))

    def forward(self, visual_feat, text_feat):
        v_spatial = self.conv_v_spatial(visual_feat)
        t_spatial = self.conv_t_spatial(text_feat)
        spatial_attn = torch.sigmoid(self.conv_spatial(F.relu(v_spatial + t_spatial)))

        v_pool = F.adaptive_avg_pool2d(visual_feat, 1).squeeze(-1).squeeze(-1)
        t_pool = F.adaptive_avg_pool2d(text_feat, 1).squeeze(-1).squeeze(-1)
        channel_attn = torch.sigmoid(
            self.fc_v_channel(v_pool) + self.fc_t_channel(t_pool)
        )
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1)

        W_sc = self.gamma_s * spatial_attn * self.gamma_c * channel_attn

        return visual_feat * W_sc


class DeformableSqueezeAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, text_channels):
        super().__init__()

        self.upsample = LearnableUpsampling(in_channels, scale_factor=2)
        self.attention_gate = SpatialChannelAttentionGate(skip_channels, text_channels)
        self.deformable_attn = DeformableSqueezeAttention(in_channels + skip_channels)

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip, text_feat):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        skip = self.attention_gate(skip, text_feat)
        x = torch.cat([x, skip], dim=1)
        x = self.deformable_attn(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        return x


class VITAL(nn.Module):
    def __init__(
        self,
        ch=64,
        pretrained=True,
        freeze_encoder=False,
        deep_supervision=False,
        model_name="efficientnet-b0",
    ):
        super(VITAL, self).__init__()
        self.model_name = model_name
        self.deep_supervision = deep_supervision

        # EfficientNet encoder (same as baseline)
        self.encoder = EfficientNetBNFeatures(
            model_name=model_name,
            pretrained=pretrained,
        )

        # Remove unused layers
        del self.encoder._avg_pooling
        del self.encoder._dropout
        del self.encoder._fc

        # Extract model number
        b = int(model_name[-1])

        # Channel configuration
        num_channels_per_output = [
            (16, 24, 40, 112, 320),
            (16, 24, 40, 112, 320),
            (16, 24, 48, 120, 352),
            (24, 32, 48, 136, 384),
            (24, 32, 56, 160, 448),
            (24, 40, 64, 176, 512),
            (32, 40, 72, 200, 576),
            (32, 48, 80, 224, 640),
            (32, 56, 88, 248, 704),
            (72, 104, 176, 480, 1376),
        ]

        channels_per_output = num_channels_per_output[b]

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Upsampling layers (same as baseline)
        upsampled_size = get_efficientnet_image_size(model_name)
        self.up1 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up2 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up3 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up4 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up5 = nn.Upsample(size=upsampled_size, mode="nearest")

        # Feature projection layers (same as baseline)
        self.conv1 = nn.Conv2d(
            channels_per_output[0], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(
            channels_per_output[1], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(ch)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(ch)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(ch)

        self.relu = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(ch)

        # MAGMA modules for each encoder stage
        self.text_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(768, ch), nn.ReLU(inplace=True)) for _ in range(5)]
        )

        self.magma_modules = nn.ModuleList(
            [MAGMA(ch, ch, num_heads=8) for _ in range(5)]
        )

        self.decoder4 = DecoderBlock(ch, ch, ch, ch)
        self.decoder3 = DecoderBlock(ch, ch, ch, ch)
        self.decoder2 = DecoderBlock(ch, ch, ch, ch)
        self.decoder1 = DecoderBlock(ch, ch, ch, ch)

        # Final conv for enhanced decoder
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, 1),
        )

        # Deep supervision heads (same as baseline)
        if self.deep_supervision:
            self.conv7 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn7 = nn.BatchNorm2d(ch)
            self.conv8 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn8 = nn.BatchNorm2d(ch)
            self.conv9 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn9 = nn.BatchNorm2d(ch)
            self.conv10 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn10 = nn.BatchNorm2d(ch)
            self.conv11 = nn.Conv2d(
                ch, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.bn11 = nn.BatchNorm2d(ch)

    def forward(self, x, text_embed=None):
        # Encode image
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Project features
        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.bn1(x0)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)

        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        features = [x0, x1, x2, x3, x4]

        if text_embed is not None:
            fused_features = []
            for i, feat in enumerate(features):
                text_proj = self.text_projections[i](text_embed)
                fused_feat = self.magma_modules[i](feat, text_proj)
                fused_features.append(fused_feat)
            features = fused_features

        x0, x1, x2, x3, x4 = features

        # Upsample to common size
        x0 = self.up1(x0)
        x1 = self.up2(x1)
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)

        if text_embed is not None:
            text_feats = []
            for i in range(4):
                text_proj = self.text_projections[i](text_embed)
                B, C = text_proj.shape
                text_spatial = text_proj.view(B, C, 1, 1)
                text_feats.append(text_spatial)
        else:
            B = x.shape[0]
            text_feats = [
                torch.zeros(B, x4.shape[1], 1, 1, device=x.device) for _ in range(4)
            ]

        # Expand text features to match spatial dimensions
        text_feat_3 = text_feats[3].expand(-1, -1, x3.shape[2], x3.shape[3])
        text_feat_2 = text_feats[2].expand(-1, -1, x2.shape[2], x2.shape[3])
        text_feat_1 = text_feats[1].expand(-1, -1, x1.shape[2], x1.shape[3])
        text_feat_0 = text_feats[0].expand(-1, -1, x0.shape[2], x0.shape[3])

        # Hierarchical decoding
        d4 = self.decoder4(x4, x3, text_feat_3)
        d3 = self.decoder3(d4, x2, text_feat_2)
        d2 = self.decoder2(d3, x1, text_feat_1)
        d1 = self.decoder1(d2, x0, text_feat_0)

        out = self.final_conv(d1)

        # Deep supervision
        if self.deep_supervision:
            x0_ds = self.bn7(x0)
            x0_ds = self.conv7(x0_ds)

            x1_ds = self.bn8(x1)
            x1_ds = self.conv8(x1_ds)

            x2_ds = self.bn9(x2)
            x2_ds = self.conv9(x2_ds)

            x3_ds = self.bn10(x3)
            x3_ds = self.conv10(x3_ds)

            x4_ds = self.bn11(x4)
            x4_ds = self.conv11(x4_ds)

            return out, [x0_ds, x1_ds, x2_ds, x3_ds, x4_ds]

        return out
