import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

# Load model YOLO (có thể thay bằng yolov8n.pt, yolov8s.pt tùy nhu cầu)
model = YOLO("yolov11-experiment/exp/weights/best.pt")

st.title("🎯 Object Detection App")
st.write("Upload ảnh hoặc video để phát hiện đối tượng với YOLOv8")

# Chọn loại input
option = st.radio("Chọn kiểu dữ liệu:", ("Ảnh", "Video"))

if option == "Ảnh":
    uploaded_file = st.file_uploader("Upload một ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh gốc", use_container_width=True)

        # Dự đoán
        results = model(image)
        res_img = results[0].plot()  # render kết quả
        st.image(res_img, caption="Kết quả phát hiện", use_container_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload một video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        os.remove(tfile.name)
