import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Загружаем модель
model = YOLO("best.pt") 

st.title("🚆 Railway Object Detection")
st.write("Загрузите изображения")

# Загрузка изображений
uploaded_files = st.file_uploader("Выберите одно или несколько изображений", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"Оригинал: {uploaded_file.name}", use_column_width=True)

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        results = model.predict(image_array, conf=0.25)

        result_image = Image.fromarray(results[0].plot())
        st.image(result_image, caption="Результат YOLOv8", use_column_width=True)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        data = []
        for box, score, cls in zip(boxes, scores, classes):
            class_name = model.names[cls]
            data.append({
                "Класс": class_name,
                "Уверенность": round(float(score), 2),
                "X1": int(box[0]),
                "Y1": int(box[1]),
                "X2": int(box[2]),
                "Y2": int(box[3]),
            })

        st.write("📋 Детали предсказаний:")
        st.dataframe(data)
