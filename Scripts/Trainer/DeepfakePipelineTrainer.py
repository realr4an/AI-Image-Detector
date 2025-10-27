#!/usr/bin/env python3
"""Train a ResNet50 classifier on face crops produced by a YOLO detector."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence as SeqType, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "Models"
LOG_DIR = ROOT_DIR / "logs"
YOLO_WEIGHTS = MODEL_DIR / "YOLOv8_Face_Detection" / "weights" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _list_image_paths(directory: Path) -> List[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


class FaceDataGenerator(Sequence):
    """Keras sequence that crops faces via YOLO on the fly."""

    def __init__(
        self,
        file_paths: SeqType[Path],
        labels: SeqType[int],
        img_size: tuple[int, int],
        batch_size: int,
        face_detector: YOLO,
    ):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.img_size = img_size
        self.batch_size = batch_size
        self.face_detector = face_detector

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.file_paths))
        batch_files = self.file_paths[start:end]
        batch_labels = self.labels[start:end]

        images: List[np.ndarray] = []
        labels: List[int] = []

        for path, label in zip(batch_files, batch_labels):
            image = cv2.imread(str(path))
            if image is None:
                continue

            face = self._detect_face(image)
            if face is None:
                # Fallback to the original image if detection fails
                face = image

            face_resized = cv2.resize(face, self.img_size, interpolation=cv2.INTER_AREA)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            processed = preprocess_input(face_rgb.astype("float32"))

            images.append(processed)
            labels.append(label)

        if not images:
            raise ValueError("FaceDataGenerator could not load any images for this batch.")

        return np.stack(images, axis=0), np.array(labels, dtype=np.float32)

    def _detect_face(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[np.ndarray]:
        results = self.face_detector.predict(image, imgsz=640, conf=conf_threshold, verbose=False)
        first = results[0]
        if not first.boxes or len(first.boxes) == 0:
            return None
        x1, y1, x2, y2 = map(int, first.boxes[0].xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        return image[y1:y2, x1:x2]


class DeepfakePipelineTrainer:
    """Training pipeline that combines YOLO face extraction with a ResNet50 classifier."""

    def __init__(
        self,
        train_dir: Optional[Path] = None,
        val_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        img_size: tuple[int, int] = (256, 256),
        batch_size: int = 64,
        epochs: int = 25,
    ) -> None:
        self.train_dir = train_dir or (DATA_DIR / "train")
        self.val_dir = val_dir or (DATA_DIR / "validation")
        self.test_dir = test_dir or (DATA_DIR / "test")
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name_prefix = "ResNet50_YOLO_Pipeline"

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {YOLO_WEIGHTS}. "
                "Run Scripts/PrepData/download_all_models.py first."
            )
        self.face_detector = YOLO(str(YOLO_WEIGHTS))
        print("> Loaded YOLO face detector.")

    @staticmethod
    def _gather_files(directory: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
        classes = sorted(
            [path for path in directory.iterdir() if path.is_dir()],
            key=lambda p: p.name.lower(),
        )
        class_to_index = {cls.name: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls.name for cls, idx in class_to_index.items()}

        file_paths: List[Path] = []
        labels: List[int] = []

        for cls in classes:
            images = _list_image_paths(cls)
            file_paths.extend(images)
            labels.extend([class_to_index[cls.name]] * len(images))

        return file_paths, labels, idx_to_class

    def build_model(self) -> Sequential:
        """Create the classifier head on top of ResNet50."""
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3),
        )
        base_model.trainable = False

        model = Sequential(
            [
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.4),
                Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
                Dropout(0.3),
                Dense(1, activation="sigmoid", dtype="float32"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self) -> Path:
        """Train the model and return the path to the best checkpoint."""
        _enable_memory_growth()

        train_files, train_labels, _ = self._gather_files(self.train_dir)
        val_files, val_labels, _ = self._gather_files(self.val_dir)

        train_gen = FaceDataGenerator(train_files, train_labels, self.img_size, self.batch_size, self.face_detector)
        val_gen = FaceDataGenerator(val_files, val_labels, self.img_size, self.batch_size, self.face_detector)

        model = self.build_model()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = MODEL_DIR / f"{self.model_name_prefix}_{timestamp}.h5"
        log_dir = LOG_DIR / "fit" / f"{self.model_name_prefix}-{timestamp}"

        callbacks = [
            ModelCheckpoint(filepath=str(model_path), save_best_only=True, monitor="val_accuracy", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7),
            TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        ]

        print("> Starting training ...")
        model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=False,
        )

        print(f"> Training complete. Best model saved to {model_path}")

        if self.test_dir.exists():
            self.evaluate(model_path)

        return model_path

    def evaluate(self, model_path: Path) -> None:
        """Evaluate the best checkpoint on the test split."""
        test_files, test_labels, idx_to_class = self._gather_files(self.test_dir)
        if not test_files:
            print("! No test samples found. Skipping evaluation.")
            return

        test_gen = FaceDataGenerator(test_files, test_labels, self.img_size, self.batch_size, self.face_detector)
        model = tf.keras.models.load_model(model_path)

        print("> Evaluating the model on the test set ...")
        loss, accuracy = model.evaluate(test_gen, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        predictions = model.predict(test_gen)
        predicted_labels = (predictions > 0.5).astype(int).flatten()

        y_true = np.array(test_labels[: len(predicted_labels)])
        target_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

        print("\nClassification Report:")
        print(classification_report(y_true, predicted_labels, target_names=target_names))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, predicted_labels))


def _enable_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("! No GPU detected. Falling back to CPU execution.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"! Could not set memory growth for {gpu}: {exc}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    trainer = DeepfakePipelineTrainer()
    trainer.train()
