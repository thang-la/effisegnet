import lightning as L
import torch
from hydra.utils import instantiate
from monai import metrics as mm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import os
from skimage import measure


class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        self.model = model

        self.get_dice = mm.DiceMetric(include_background=False)
        self.get_iou = mm.MeanIoU(include_background=False)
        self.get_recall = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="sensitivity"
        )
        self.get_precision = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="precision"
        )
        self.get_hausdorff = mm.HausdorffDistanceMetric(
            include_background=False,
            percentile=95
        )

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

        self.val_hd_values = []
        self.test_hd_values = []

    def forward(self, image, text_embedding):
        return self.model(image, text_embedding)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": instantiate(self.scheduler, optimizer=optimizer),
                "monitor": "val_loss",
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        image, mask, text, image_filename = batch

        if self.model.deep_supervision:
            logits, logits_aux = self(image, text)

            aux_loss = sum(self.criterion(z, mask) for z in logits_aux)
            loss = (self.criterion(logits, mask) + aux_loss) / (1 + len(logits_aux))
        else:
            logits = self(image, text)
            loss = self.criterion(logits, mask)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        image, mask, text, image_filename = batch

        if self.model.deep_supervision:
            logits, _ = self(image, text)
        else:
            logits = self(image, text)

        loss = self.criterion(logits, mask)
        self.log("val_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, mask)
        self.get_iou(preds, mask)
        self.get_recall(preds, mask)
        self.get_precision(preds, mask)
        self.get_hausdorff(preds, mask)

        return loss
    
    def test_step(self, batch, batch_idx):
        image, mask, text, image_filename = batch

        if self.model.deep_supervision:
            logits, _ = self(image, text)
        else:
            logits = self(image, text)

        loss = self.criterion(logits, mask)
        self.log("test_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, mask)
        self.get_iou(preds, mask)
        self.get_recall(preds, mask)
        self.get_precision(preds, mask)
        self.get_hausdorff(preds, mask)

        return loss

    def on_validation_epoch_end(self):
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        hausdorff = self.get_hausdorff.aggregate().item()

        self.log("val_dice", dice)
        self.log("val_iou", iou)
        self.log("val_recall", recall)
        self.log("val_precision", precision)
        self.log("val_f1", 2 * (precision * recall) / (precision + recall + 1e-8))
        self.log("val_hausdorff", hausdorff, sync_dist=True)

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()
        self.get_hausdorff.reset()
        self.val_hd_values = []
    
    def on_test_epoch_end(self):
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()
        hausdorff = self.get_hausdorff.aggregate().item()

        self.log("test_dice", dice)
        self.log("test_iou", iou)
        self.log("test_recall", recall)
        self.log("test_precision", precision)
        self.log("test_f1", 2 * (precision * recall) / (precision + recall + 1e-8))
        self.log("test_hausdorff", hausdorff, sync_dist=True)

        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()
        self.get_hausdorff.reset()
        self.test_hd_values = []
