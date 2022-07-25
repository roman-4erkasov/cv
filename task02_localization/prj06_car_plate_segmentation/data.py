import os
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import cv2
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, data_path, config, transforms=None,):
        super(DetectionDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.images = [
            os.path.join(self.data_path,f"{name}.{ext}") for name, ext in config
        ]
        self.masks = [
            os.path.join(self.data_path,f"{name}.mask.{ext}") for name, ext in config
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx]).astype(np.float32) / 255.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        item = dict(image=image, mask=mask)

        if self.transforms is not None:
            item = self.transforms(**item)
        item["mask"] = item["mask"][None, :, :]
        return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        split_proportions = (0.6, 0.8),
        batch_size = 32,
        image_size=256,
        seed = None,
    ):
        self.data_dir = data_dir
        self.split_proportions = split_proportions
        self.images = list(
            {
                tuple(x for x in fn.split(".") if x != "mask") 
                for fn in os.listdir(data_dir)
            }
        )
        self.n_img = len(self.images)
        self.batch_size = batch_size
        self.image_size = image_size
        self.transforms = self.get_transforms(image_size=self.image_size)
        
    @staticmethod
    def get_transforms(image_size: int = 256):
        return Compose([
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.),
            Resize(height=image_size, width=image_size, ),
            ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "3" or stage is None:
            print(self.images[:5])
            print(tuple(int(n_img*x) for x in self.split_proportions))
            self.train, self.val, self.test = np.split(
                self.images, 
                tuple(int(n_img*x) for x in self.split_proportions)
            )
        elif stage == "2":
            self.train, self.val = np.split(
                self.images, tuple(int(n_img*self.split_proportions[0]),)
            )
        elif stage == "1":
            self.train = images        
    
    def train_dataloader(self):
        # train_transforms = get_train_transforms(256)
        ds = DetectionDataset(data_path=self.data_dir, config=self.train, transforms=self.transforms)
        return DataLoader(ds, batch_size=self.batch_size)

    def val_dataloader(self):
        ds = DetectionDataset(data_path=self.data_dir, config=self.val)
        return DataLoader(ds, batch_size=self.batch_size)

    def test_dataloader(self):
        ds = DetectionDataset(data_path=self.data_dir, config=self.test)
        return DataLoader(ds, batch_size=self.batch_size)

