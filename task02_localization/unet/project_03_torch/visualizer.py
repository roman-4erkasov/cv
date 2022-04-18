import cv2
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_tensor
from matplotlib import pyplot as plt
from copy import deepcopy


class Visualizer:
    @staticmethod
    def add_boxes(img, boxes):        
        img_bbox = draw_bounding_boxes(
            image=img,
            boxes=boxes
        )
        return img_bbox

    @staticmethod
    def add_masks(img, masks):
        img_mask = draw_segmentation_masks(
            image=img,
            masks=masks,
            alpha=0.9,
            colors="yellow"
        )
        return img_mask

    @classmethod
    def save(cls, img, filepath, boxes=None, masks=None):
        img = (deepcopy(img) * 255).to(torch.uint8)
        if boxes is not None: img = cls.add_boxes(img, boxes)
        if masks is not None: img = cls.add_masks(img, masks)
        img_save = cv2.cvtColor(
            img.permute(1,2,0).numpy(),
            cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(filename=filepath, img=img_save)

    @classmethod
    def show(cls, img, boxes=None, masks=None, ax=None):
        img = (deepcopy(img) * 255).to(torch.uint8)
        if boxes is not None: img = cls.add_boxes(img, boxes)
        if masks is not None: img = cls.add_masks(img, masks)
        if ax is None:
            plt.figure(figsize = (10,10))
            plt.imshow(img.permute(1,2,0))
        else:
            # ax.figure(figsize = (10,10))
            ax.imshow(img.permute(1,2,0))
