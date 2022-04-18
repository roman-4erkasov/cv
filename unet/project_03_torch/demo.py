import config
import torch
import cv2
import numpy as np
import streamlit as st
from visualizer import Visualizer
from model import load_model, UNet
import torchvision
from copy import deepcopy
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt

def prepare_image(image):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
            ),
            torchvision.transforms.ToTensor()
        ]
    )
    if transforms is not None:
        image = transforms(image)
    return image

model = load_model(config.PATH_MODEL)

st.title("Image segmentation")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    st.header(body="Original image")
    st.image(
        uploaded_file, caption='Uploaded Image.', use_column_width=True
    )
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.cvtColor(
        cv2.imdecode(file_bytes, cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    image = image.astype("float32") / 255.0
    image = cv2.resize(image, (128, 128))
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).to(config.DEVICE)
    image = prepare_image(image)
    pred_mask = torch.sigmoid(
        model(image.unsqueeze(0)).squeeze()
    ).cpu()
    pred_mask = (pred_mask > config.THRESHOLD).to(torch.bool)
    img = (deepcopy(image) * 255).to(torch.uint8)
    img_mask = draw_segmentation_masks(
        image=img,
        masks=pred_mask,
        alpha=0.2,
        colors="yellow"
    )
    fig, ax = plt.subplots(figsize = (10,10))
    st.header(body="Prediction")
    ax.imshow(img_mask.permute(1,2,0))
    st.pyplot(fig)
