#!/usr/bin/env python3
"""Streamlit app for classifying AI-generated images with Grad-CAM explanation."""

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model

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
    return load_model(model_path)


def preprocess_image(img: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    resized = cv2.resize(img, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_input(rgb.astype("float32"))
    return np.expand_dims(preprocessed, axis=0)


def _find_last_conv_layer(model: Model) -> Optional[str]:
    """Locate the last 2D convolution layer within nested sub-models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, Model):
            nested = _find_last_conv_layer(layer)
            if nested:
                return nested
    return None


def make_gradcam_heatmap(
    img_array: np.ndarray, model: Model, last_conv_layer_name: str, pred_index: Optional[int] = None
) -> np.ndarray:
    """Generate a Grad-CAM heatmap highlighting influential regions."""
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay the Grad-CAM heatmap on top of the original image."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return superimposed


def main() -> None:
    st.set_page_config(page_title="AI Image Detector", page_icon="AI")
    st.title("AI Image Detector with Grad-CAM Explanation")
    st.caption(
        "Upload an image to predict whether it is real or AI-generated while visualising the most relevant regions."
    )

    classifier_paths = _list_classifier_paths()
    if not classifier_paths:
        st.error(
            "No classifier checkpoints were found under the Models/ directory. "
            "Please add at least one `.h5` model."
        )
        return

    selected_label = st.selectbox(
        "Select classifier",
        options=[path.name for path in classifier_paths],
    )
    selected_path = next(path for path in classifier_paths if path.name == selected_label)

    uploaded = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
    if not uploaded:
        st.info("Supported input formats: JPG and PNG.")
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("The file could not be decoded as an image.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    model = load_classifier(selected_path)
    processed = preprocess_image(image_bgr)
    prediction = model.predict(processed)

    score = float(prediction.ravel()[0])
    label = "real" if score > 0.5 else "AI-generated"
    st.markdown(f"### Prediction: **{label.title()}**")
    st.write(f"Confidence score: {score:.3f} (higher values indicate real images).")

    last_conv = _find_last_conv_layer(model)
    if not last_conv:
        st.warning("No convolution layer found. Grad-CAM explanation is unavailable for this model.")
        return

    st.write("### Explanation")
    st.caption("Highlighted regions contribute the most to the model decision.")
    heatmap = make_gradcam_heatmap(processed, model, last_conv)
    overlay = overlay_heatmap(heatmap, image_rgb)
    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)


if __name__ == "__main__":
    main()
