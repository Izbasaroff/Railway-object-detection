import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

st.title("🚆 Railway Object Detection")

# Загружаем модель один раз при старте
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Загрузите изображение (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Оригинальное изображение", use_container_width=True)

    # Конвертируем в numpy-массив для модели
    img_array = np.array(image)

    # Запускаем предсказание
    results = model(img_array)

    # Получаем изображение с нанесёнными бокcами и метками
    annotated_img = results[0].plot()

    # Показываем результат
    st.image(annotated_img, caption="Распознанные объекты", use_container_width=True)