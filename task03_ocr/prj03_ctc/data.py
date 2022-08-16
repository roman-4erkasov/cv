import os
import re
import json
import torch
from typing import Optional, List
import numpy as np
import pytorch_lightning as pl
import cv2
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize


class RecDataset(Dataset):
    def __init__(self, data_path, config, abc, split=None, transforms=None):
        super(RecDataset, self).__init__()
        self.data_path = data_path
        self.abc = abc
        self.transforms = transforms
        # self.split = split
        self.config = config

    def __len__(self):
        return len(self.config)

    def text_to_seq(self, text):
        seq = [self.abc.find(c) + 1 for c in text]
        return seq
    
    def __getitem__(self, idx):
        fname, text = self.config[idx]
        fpath = os.path.join(self.data_path, fname)
        try:
            image = cv2.imread(fpath).astype(np.float32) / 255.
        except Exception as exc:
            raise Exception(f"fpath={fpath}") from exc
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        if self.transforms is not None:
            image_rec = self.transforms(image=image)
            image = image_rec["image"]
        item = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        return item

    @staticmethod
    def collate_fn(batch):
        images = list()
        seqs = list()
        seq_lens = list()
        for sample in batch:
            images.append(torch.from_numpy(sample["image"].transpose((2, 0, 1))).float())
            seqs.extend(sample["seq"])
            seq_lens.append(sample["seq_len"])
        images = torch.stack(images)
        seqs = torch.Tensor(seqs).int()
        seq_lens = torch.Tensor(seq_lens).int()
        batch = {"images": images, "seqs": seqs, "seq_lens": seq_lens,}
        return batch


class RecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        config: List,
        split_proportions = (0.6, 0.8),
        batch_size = 32,
        image_size=256,
        seed = None,
    ):
        self.data_path = data_path
        self.split_proportions = split_proportions
        self.config = config
        self.n_img = len(self.config)
        self.batch_size = batch_size
        self.image_size = image_size
        self.transforms = self.get_transforms(image_size=self.image_size)
        self.abc = "0123456789ABCEHKMOPTXY"
        
    @staticmethod
    def get_transforms(image_size: int = 256):
        return Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.),
            Resize(height=image_size, width=image_size, ),
            # ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "3" or stage is None:
            # print(self.images[:5])
            # print(tuple(int(self.n_img*x) for x in self.split_proportions))
            self.train, self.val, self.test = np.split(
                self.config, 
                tuple(int(self.n_img*x) for x in self.split_proportions)
            )
        elif stage == "2":
            self.train, self.val = np.split(
                self.config, tuple(int(n_img*self.split_proportions[0]),)
            )
        elif stage == "1":
            self.train = self.images        
    
    def train_dataloader(self):
        # train_transforms = get_train_transforms(256)
        ds = RecDataset(
            data_path=self.data_path, config=self.train, transforms=self.transforms, abc=self.abc
        )
        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            # num_workers=2,
            collate_fn=ds.collate_fn
        )

    def val_dataloader(self):
        ds = RecDataset(data_path=self.data_path, config=self.val, abc=self.abc)
        return DataLoader(
            ds, 
            batch_size=self.batch_size,
            shuffle=True, 
            # num_workers=2,
            collate_fn=ds.collate_fn
        )

    def test_dataloader(self):
        ds = RecDataset(data_path=self.data_path, config=self.test, abc=self.abc)
        return DataLoader(
            ds, 
            batch_size=self.batch_size,
            shuffle=True, 
            # num_workers=2,
            collate_fn=ds.collate_fn
        )
