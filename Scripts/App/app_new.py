#!/usr/bin/env python3
"""Streamlit app for detecting AI-generated faces with YOLO cropping."""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT_DIR / "Models"
CLASSIFIER_DIRS = [
    MODEL_ROOT,
    MODEL_ROOT / "ResNet50_Deepfake_detection",
]
MODEL_EXTENSION = ".h5"
YOLO_WEIGHTS = MODEL_ROOT / "YOLOv8_Face_Detection" / "weights" / "best.pt"


def _list_classifier_paths() -> List[Path]:
    """Return all available Keras classifier checkpoints."""
    paths: List[Path] = []
    for directory in CLASSIFIER_DIRS:
        if not directory.exists():
            continue
        paths.extend(sorted(directory.glob(f"*{MODEL_EXTENSION}")))
    return paths


@st.cache_resource(show_spinner="Loading YOLO face detector ...")
def load_yolo_model() -> YOLO:
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {YOLO_WEIGHTS}. "
            "Please download the face detector by running Scripts/PrepData/download_all_models.py."
        )
    return YOLO(str(YOLO_WEIGHTS))


@st.cache_resource(show_spinner="Loading classifier ...")
def load_classifier(model_path: Path):
    return load_model(model_path)


def _extract_face(image: np.ndarray, detector: YOLO, conf_threshold: float = 0.5) -> Optional[np.ndarray]:
    """Run YOLO inference and return the first detected face crop."""
    results = detector.predict(image, imgsz=640, conf=conf_threshold, verbose=False)
    first = results[0]
    if not first.boxes or len(first.boxes) == 0:
        return None
    x1, y1, x2, y2 = map(int, first.boxes[0].xyxy[0].cpu().numpy())
    x1, y1 = max(0, x1), max(0, y1)
    return image[y1:y2, x1:x2]


def _preprocess_face(face_image: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize and preprocess the cropped face for ResNet50."""
    resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    processed = preprocess_input(rgb.astype("float32"))
    return np.expand_dims(processed, axis=0)


def main() -> None:
    st.set_page_config(page_title="AI Image Detector", page_icon="AI")
    st.title("AI Image Detector with Face Extraction")
    st.caption(
        "Upload a portrait, let YOLO isolate the face, and evaluate if it is real or AI-generated."
    )

    classifier_paths = _list_classifier_paths()
    if not classifier_paths:
        st.error(
            "No classifier checkpoints were found under the Models/ directory. "
            "Please add at least one `.h5` model."
        )
        return

    selected_label = st.selectbox(
        "Select classifier", options=[path.name for path in classifier_paths], index=0
    )
    selected_path = next(path for path in classifier_paths if path.name == selected_label)

    uploaded_file = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
    if not uploaded_file:
        st.info("Supported input: JPG or PNG portrait images.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original is None:
        st.error("The file could not be decoded as an image.")
        return

    st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original image", use_container_width=True)

    detector = load_yolo_model()
    face = _extract_face(original, detector)

    if face is None or face.size == 0 or min(face.shape[:2]) < 20:
        st.warning("No reliable face crop detected. Please try another image.")
        return

    st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Detected face", use_container_width=True)

    classifier = load_classifier(selected_path)
    model_input = _preprocess_face(face)
    prediction = classifier.predict(model_input)

    score = float(prediction.ravel()[0])
    label = "real" if score < 0.5 else "AI-generated"
    st.markdown(f"### Prediction: **{label.title()}**")
    st.write(f"Confidence score: {score:.3f} (values closer to 0 indicate real images).")


if __name__ == "__main__":
    main()
