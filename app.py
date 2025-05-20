import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import yaml

with open("data.yaml") as f:
    data = yaml.safe_load(f)
classes = data['names']

st.title("🚆 Railway Object Detection")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Загрузите изображение (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Оригинальное изображение", use_container_width=True)

    img_array = np.array(image)

    results = model(img_array)

    annotated_img = results[0].plot()


    st.image(annotated_img, caption="Распознанные объекты", use_container_width=True)