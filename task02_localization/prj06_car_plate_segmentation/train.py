import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from data import DataModule
from model import SegmModel


DEBUG = False
PATH_DATA = "/home/ubuntu/datasets/segment_car_plate/data/train"


dmodule = DataModule(data_dir=PATH_DATA)
dmodule.setup()
dtrain = dmodule.train_dataloader()
dval = dmodule.val_dataloader()
dtest = dmodule.test_dataloader()

model = SegmModel(
    architecture="FPN",
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    out_classes=1
)
if DEBUG:
    trainer = pl.Trainer(
        gpus=0, 
        max_epochs=2,
        # fast_dev_run=3,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
    )
else:
    trainer = pl.Trainer(
        gpus=0, 
        max_epochs=8,
    )
trainer.fit(
    model, 
    train_dataloaders=dtrain, 
    val_dataloaders=dval,
)
trainer.test(model=model, dataloaders=dtest)
