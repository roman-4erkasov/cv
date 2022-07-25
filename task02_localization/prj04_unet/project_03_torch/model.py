import torch
import os
import cv2
import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import config

def load_model(path):
    unet = UNet().to(config.DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=config.INIT_LR)
    checkpoint = torch.load(path)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    return unet


class Block(torch.nn.Module):
    """Conv->ReLU->Conv->ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    

class Encoder(torch.nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.enc_blocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        outputs = [] # intermediate outputs
        for block in self.enc_blocks:
            x = block(x)
            outputs.append(x)
            x = self.pool(x)
        return outputs

    
class Decoder(torch.nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upconvs = torch.nn.ModuleList(
            [
                torch.nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)
            ]
        )
        self.dec_blocks = torch.nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        
    def forward(self, enc_output):
        x = enc_output[-1]
        for enc, dec, up in zip(enc_output[::-1][1:], self.dec_blocks, self.upconvs):
            x_up = up(x)
            x = dec(torch.cat([x_up, self.crop(enc, x_up)], dim=1))
        return x
    
    @staticmethod
    def crop(enc_features, x):
        (_, _, h, w) = x.shape
        return torchvision.transforms.CenterCrop([h, w])(enc_features)


class UNet(torch.nn.Module):
    def __init__(
        self, 
        enc_channels=(3, 16, 32, 64),
        dec_channels=(64, 32, 16),
        nb_classes=1, retain_dim=True,
        out_size=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)
    ):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.head = torch.nn.Conv2d(
            in_channels=dec_channels[-1],
            out_channels=nb_classes,
            kernel_size=1,
        )
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output)
        segm_map = self.head(dec_output)
        if self.retain_dim:
            segm_map = torch.nn.functional.interpolate(segm_map, self.out_size)
        return segm_map
