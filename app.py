import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import yaml

# Загружаем кастомные имена классов
with open("data.yaml") as f:
    data = yaml.safe_load(f)
classes = data['names']  # используем это!

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
    result = results[0]

    # Копируем изображение для кастомной отрисовки
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        label = f"{classes[cls_id]} {conf:.2f}"

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="red", font=font)

    st.image(annotated_image, caption="Распознанные объекты", use_container_width=True)
