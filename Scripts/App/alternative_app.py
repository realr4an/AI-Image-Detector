#!/usr/bin/env python3
"""Minimal Streamlit app for quick binary classification of uploaded images."""

from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT_DIR / "Models"
CLASSIFIER_DIRS = [
    MODEL_ROOT,
    MODEL_ROOT / "ResNet50_Deepfake_detection",
]
MODEL_EXTENSION = ".h5"


def _list_classifier_paths() -> List[Path]:
    paths: List[Path] = []
    for directory in CLASSIFIER_DIRS:
        if not directory.exists():
            continue
        paths.extend(sorted(directory.glob(f"*{MODEL_EXTENSION}")))
    return paths


@st.cache_resource(show_spinner="Loading classifier ...")
def load_classifier(model_path: Path):
    return tf.keras.models.load_model(model_path)


def main() -> None:
    st.set_page_config(page_title="Quick Deepfake Check", page_icon="AI")
    st.title("Quick Deepfake Check")
    st.caption("Upload an image to classify it with a selected Keras model.")

    classifier_paths = _list_classifier_paths()
    if not classifier_paths:
        st.error("No `.h5` checkpoints found under Models/. Please add a trained model first.")
        return

    selected_label = st.selectbox(
        "Select classifier",
        options=[path.name for path in classifier_paths],
    )
    selected_path = next(path for path in classifier_paths if path.name == selected_label)
    model = load_classifier(selected_path)

    if len(model.input_shape) != 4:
        st.error(f"Unexpected input shape: {model.input_shape}")
        return
    _, height, width, _ = model.input_shape

    uploaded = st.file_uploader("Upload an image", type=("png", "jpg", "jpeg"))
    if not uploaded:
        st.info("Supported input formats: PNG, JPG, JPEG.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized = image.resize((width, height))
    img_array = np.array(resized)
    img_array = preprocess_input(img_array.astype("float32"))
    batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(batch)
    score = float(prediction.ravel()[0])
    label = "real" if score > 0.5 else "deepfake"

    st.markdown(f"### Prediction: **{label.title()}**")
    st.write(f"Confidence score: {score:.3f} (higher values indicate real images).")


if __name__ == "__main__":
    main()
