import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_DIR = "."  # Verzeichnis mit den .h5-Modellen

@st.cache_resource
def load_yolo_model():
    return YOLO("face_detection_project/yolov8_face_detector/weights/best.pt")

@st.cache_resource
def get_available_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

@st.cache_resource
def load_cnn_model(path):
    return load_model(path)

def extract_face_yolo(image, model, conf_threshold=0.5):
    results = model.predict(source=image, imgsz=640, conf=conf_threshold, save=False)
    result = results[0]

    if not result.boxes or len(result.boxes) == 0:
        return image  # Kein Gesicht erkannt

    box = result.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    return image[y1:y2, x1:x2]

def preprocess_image(img, target_size=(256, 256)):
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = preprocess_input(img_rgb.astype("float32"))
    return np.expand_dims(img_preprocessed, axis=0)

def main():
    st.title("KI-Bildprüfung mit Gesichtsextraktion")

    # Modell-Auswahl
    available_models = get_available_models()
    selected_model_name = st.selectbox("Wähle ein Modell:", available_models)

    uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        yolo_model = load_yolo_model()
        face_img = extract_face_yolo(img, yolo_model)

        if face_img.shape[0] < 10 or face_img.shape[1] < 10:
            st.warning("❌ Gesicht konnte nicht zuverlässig erkannt werden.")
            return

        st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Erkanntes Gesicht", use_container_width=True)

        model_input = preprocess_image(face_img)
        cnn_model = load_cnn_model(os.path.join(MODEL_DIR, selected_model_name))
        prediction = cnn_model.predict(model_input)

        label = "Echt" if prediction[0][0] < 0.5 else "KI-generiert"
        st.write(f"**Vorhersage:** {label}")
        st.write(f"Vorhersagewert: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
