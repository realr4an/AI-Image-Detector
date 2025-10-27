#!/usr/bin/env python3
"""Evaluate all trained classifiers on the held-out test set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT_DIR / "Data" / "test"
DEFAULT_MODEL_DIR = ROOT_DIR / "Models"
BATCH_SIZE = 32


class ImageDataGenerator(Sequence):
    """Lazy generator that loads and preprocesses batches of images from disk."""

    def __init__(self, image_paths, labels, batch_size, target_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            image = load_img(path, target_size=self.target_size)
            image_array = preprocess_input(img_to_array(image))
            batch_images.append(image_array)

        return np.array(batch_images), np.array(batch_labels)


def _collect_test_paths():
    fake_dir = TEST_DIR / "fake"
    real_dir = TEST_DIR / "real"

    fake_paths = sorted(path for path in fake_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    real_paths = sorted(path for path in real_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    labels = np.array([0] * len(fake_paths) + [1] * len(real_paths))
    paths = fake_paths + real_paths

    return paths, labels


def _list_models():
    model_paths = sorted(DEFAULT_MODEL_DIR.rglob("*.h5"))
    if not model_paths:
        raise FileNotFoundError("No .h5 checkpoints found under the Models/ directory.")
    return model_paths


def evaluate_model(model_path, image_paths, labels):
    model = load_model(model_path, compile=False)
    width, height = model.input_shape[1], model.input_shape[2]
    generator = ImageDataGenerator(image_paths, labels, BATCH_SIZE, (width, height))

    print(f"\nEvaluating model: {model_path.name}")
    predictions = model.predict(generator, verbose=1)
    predicted_labels = (predictions > 0.5).astype("int32").flatten()

    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


def main():
    image_paths, labels = _collect_test_paths()
    if not image_paths:
        raise FileNotFoundError("Test set is empty. Prepare Data/test/{real,fake} before evaluation.")

    for model_path in _list_models():
        evaluate_model(model_path, image_paths, labels)


if __name__ == "__main__":
    main()
