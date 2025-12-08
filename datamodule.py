import os

import albumentations as A
import cv2
import lightning as L
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd


class KvasirSEGDatagen(Dataset):
    def __init__(self, pairs, csv_path=None, transform=None, text_encoder=None):
        self.transform = transform
        self.pairs = pairs
        self.text_encoder = text_encoder

        self.image2desc = {}
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            self.image2desc = dict(zip(df["Image"], df["Description"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image_filename = os.path.basename(image_path)
        description = self.image2desc.get(image_filename, "")

        if self.text_encoder is not None and description.strip() != "":
            tokenizer, model = self.text_encoder
            inputs = tokenizer(
                description, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
                text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        else:
            text_embedding = torch.zeros(768)

        return image, mask.long().unsqueeze(0), text_embedding, image_filename


class KvasirSEGDataset(L.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        root_dir="./Kvasir-SEG",
        num_workers=2,
        train_val_ratio=0.8,
        img_size=(224, 224),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    *(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
                ),
                A.Affine(
                    p=0.5,
                    scale=(0.5, 1.5),
                    translate_percent=0.125,
                    rotate=90,
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_val_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    *(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def get_test_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    *(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None):
        train_images = os.listdir(os.path.join(self.root_dir, "train/images"))
        train_masks = os.listdir(os.path.join(self.root_dir, "train/masks"))
        train_images = [
            os.path.join(self.root_dir, "train/images", img) for img in train_images
        ]
        train_masks = [
            os.path.join(self.root_dir, "train/masks", mask) for mask in train_masks
        ]

        val_images = os.listdir(os.path.join(self.root_dir, "validation/images"))
        val_masks = os.listdir(os.path.join(self.root_dir, "validation/masks"))
        val_images = [
            os.path.join(self.root_dir, "validation/images", img) for img in val_images
        ]
        val_masks = [
            os.path.join(self.root_dir, "validation/masks", mask) for mask in val_masks
        ]

        test_images = os.listdir(os.path.join(self.root_dir, "test/images"))
        test_masks = os.listdir(os.path.join(self.root_dir, "test/masks"))
        test_images = [
            os.path.join(self.root_dir, "test/images", img) for img in test_images
        ]
        test_masks = [
            os.path.join(self.root_dir, "test/masks", mask) for mask in test_masks
        ]

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()
        text_encoder = (tokenizer, model)

        train_csv = os.path.join(self.root_dir, "validation/val.csv")
        val_csv = os.path.join(self.root_dir, "validation/val.csv")
        test_csv = os.path.join(self.root_dir, "test/test.csv")

        train_pairs = list(zip(train_images, train_masks))
        val_pairs = list(zip(val_images, val_masks))
        test_pairs = list(zip(test_images, test_masks))

        self.train_set = KvasirSEGDatagen(
            train_pairs,
            transform=self.get_train_transforms(),
            csv_path=train_csv,
            text_encoder=text_encoder,
        )
        self.val_set = KvasirSEGDatagen(
            val_pairs,
            transform=self.get_val_transforms(),
            csv_path=val_csv,
            text_encoder=text_encoder,
        )
        self.test_set = KvasirSEGDatagen(
            test_pairs,
            transform=self.get_test_transforms(),
            csv_path=test_csv,
            text_encoder=text_encoder,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
