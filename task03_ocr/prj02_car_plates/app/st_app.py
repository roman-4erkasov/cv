import streamlit as st
import pandas as pd
import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
from detector_v01_baseline import Detector
from ocr_v01_baseline import CRNN, decode, ResizeImage

PATH_MODEL = "./crnn.pth.tar"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.title('Car license plate OCR')
uploaded_file = st.file_uploader("Upload photo with a car")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.cvtColor(
        cv2.imdecode(file_bytes, cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    fig, ax = plt.subplots(figsize = (10,10))
    st.header(body="Original")
    ax.imshow(image)
    st.pyplot(fig)
    det = Detector()
    plate, plate_count = det(image)
    if plate is not None:
        fig, ax = plt.subplots(figsize = (10,10))
        st.header(body="Detection")
        ax.imshow(plate)
        st.pyplot(fig)
        
        crnn = CRNN()
        with open(PATH_MODEL, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        crnn.load_state_dict(state_dict)
        transforms=ResizeImage(size=(320, 64))
        plate2 = transforms(plate/255.)
        plate_tensor = torch.from_numpy(plate2)
        plate_img = torch.stack(
            [plate_tensor, plate_tensor, plate_tensor]
        ).float().unsqueeze(0).to(device)
        st.write(plate.shape, plate_img.shape)
        preds = crnn(plate_img).cpu().detach()
        texts_pred = decode(preds, crnn.alphabet)
        st.write(texts_pred)

