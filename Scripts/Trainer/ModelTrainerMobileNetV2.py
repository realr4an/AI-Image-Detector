#!/usr/bin/env python3
"""Train a MobileNetV2 classifier on the curated dataset."""

from __future__ import annotations

import datetime
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Data"
MODEL_DIR = ROOT_DIR / "Models"
LOG_DIR = ROOT_DIR / "logs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
MODEL_NAME = "MobileNetV2"


def _enable_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("! No GPU detected. Training will run on CPU.")
        return

    print(f"> Detected GPUs: {gpus}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"! Could not enable memory growth on {gpu}: {exc}")


def _build_model() -> Sequential:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = True

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu", kernel_regularizer=l2(2e-3)),
            Dropout(0.6),
            Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _create_generators():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_dir = DATA_DIR / "train"
    validation_dir = DATA_DIR / "validation"

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    val_gen = datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )

    return train_gen, val_gen, datagen


def _train_model(model: Sequential, train_gen, val_gen) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = MODEL_DIR / f"{MODEL_NAME}_{timestamp}.h5"
    log_dir = LOG_DIR / "fit" / f"{MODEL_NAME}-{timestamp}"

    callbacks = [
        ModelCheckpoint(filepath=str(model_path), save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
    ]

    print("> Starting MobileNetV2 training ...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False,
    )

    print(f"> Training complete. Best model stored at {model_path}")
    return model_path


def _evaluate(model_path: Path, datagen: ImageDataGenerator) -> None:
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        print("! Test directory not found. Skipping evaluation.")
        return

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(test_gen)
    predicted_labels = (predictions > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(test_gen.classes, predicted_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(test_gen.classes, predicted_labels))


def train_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _enable_memory_growth()
    mixed_precision.set_global_policy("mixed_float16")

    model = _build_model()
    train_gen, val_gen, datagen = _create_generators()
    model_path = _train_model(model, train_gen, val_gen)
    _evaluate(model_path, datagen)
    return model_path


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    train_model()
