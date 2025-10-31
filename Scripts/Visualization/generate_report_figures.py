#!/usr/bin/env python3
"""
Generate refreshed report figures for the AI-Image-Detector project.

This script assembles dataset collages, training diagnostics, ROC/PR curves,
and supporting diagrams required for the written report. All outputs are written
to the figures/ directory.
"""

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


RNG = random.Random(1234)
np.random.seed(1234)

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SELECTION_LOG = FIGURES_DIR / "selection_log.csv"
DOC_FIGURES_DIR = ROOT / "docs" / "figures"

DATA_DIR = ROOT / "data"
VALIDATION_REAL = DATA_DIR / "validation" / "real"
VALIDATION_FAKE = DATA_DIR / "validation" / "fake"

TRAINING_BASELINE_LOG = ROOT / "docs" / "training_logs" / "Training_ResNet50.txt"
TRAINING_PIPELINE_LOG = ROOT / "docs" / "training_logs" / "Training_Pipeline.txt"

BASELINE_MODEL_PATH = ROOT / "Models" / "ResNet50_20250710-061102.h5"
PIPELINE_RESULTS_CSV = FIGURES_DIR / "pipeline_validation_scores.csv"
BASELINE_SCORES_CSV = FIGURES_DIR / "resnet50_baseline_scores.csv"
PIPELINE_MODEL_PATH = ROOT / "Models" / "ResNet50_YOLO_Pipeline_20250713-230159.h5"
YOLO_WEIGHTS_PATH = ROOT / "Models" / "YOLOv8_Face_Detection" / "weights" / "best.pt"
PIPELINE_SKIPPED_LOG = FIGURES_DIR / "pipeline_skipped.log"
PIPELINE_TARGET_SIZE = (256, 256)

COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_ACCENT = "#2ca02c"
COLOR_BACKGROUND = "#f6f7fb"
LABEL_COLORS = {"Real": "#3cb44b", "Fake": "#ffa500"}

COLLAGE_CONFIG = {
    "datasets_deepfake_vs_real60k.png": {
        "title": "DeepFake vs. Real (Validation)",
        "panels": [
            {"label": "Real", "path": "data/validation/real/real_6906.jpg", "type": "image"},
            {"label": "Real", "path": "data/validation/real/0001 (6466).png", "type": "image"},
            {"label": "Fake", "path": "data/validation/fake/0001 (14814).jpg", "type": "image"},
            {"label": "Fake", "path": "data/validation/fake/fake_14112.jpg", "type": "image"},
        ],
    },
    "datasets_detect_ai_generated.png": {
        "title": "Detecting AI Generated Faces",
        "panels": [
            {"label": "Real", "path": "data/validation/real/non-child-885.png", "type": "image"},
            {"label": "Real", "path": "data/validation/real/real_12717.jpg", "type": "image"},
            {"label": "Fake", "path": "data/validation/fake/fake_19420.jpg", "type": "image"},
            {"label": "Fake", "path": "data/validation/fake/fake_2342.jpg", "type": "image"},
        ],
    },
    "datasets_deepfake_and_real_images.png": {
        "title": "DeepFake Artefacts vs. Real Details",
        "panels": [
            {"label": "Real", "path": "data/validation/real/real_2723.jpg", "type": "face"},
            {"label": "Real", "path": "data/validation/real/real_1745.jpg", "type": "face"},
            {"label": "Fake", "path": "data/validation/fake/fake_13622.jpg", "type": "face"},
            {"label": "Fake", "path": "data/validation/fake/fake_8291.jpg", "type": "face"},
        ],
    },
}

selection_records: list[dict[str, str]] = []
detection_cache: dict[Path, "DetectionResult | None"] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_image_paths(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]


def load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def record_selection(figure: str, label: str, path: Path | str, note: str = "", score: float | None = None) -> None:
    rel = str(path.relative_to(ROOT)) if isinstance(path, Path) else str(path)
    entry = {
        "figure": figure,
        "label": label,
        "path": rel,
        "score": "" if score is None else f"{score:.4f}",
        "note": note,
    }
    selection_records.append(entry)


def save_figure(fig: plt.Figure, filename: str, directory: Path | None = None) -> Path:
    target_dir = directory if directory is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / filename
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def ensure_scores_available(fake_paths: list[Path], real_paths: list[Path]) -> pd.DataFrame:
    if PIPELINE_RESULTS_CSV.exists():
        return pd.read_csv(PIPELINE_RESULTS_CSV)
    return compute_pipeline_scores(fake_paths, real_paths)


# ---------------------------------------------------------------------------
# Face detection helpers for pipeline evaluation
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    confidence: float
    face_for_model: np.ndarray
    display_crop: np.ndarray


def expand_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int], margin: float = 0.15) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    dx = int(box_w * margin)
    dy = int(box_h * margin)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(width - 1, x2 + dx)
    ny2 = min(height - 1, y2 + dy)
    return nx1, ny1, nx2, ny2


def detect_face(face_detector: YOLO, image_path: Path) -> DetectionResult | None:
    if image_path in detection_cache:
        return detection_cache[image_path]

    image = cv2.imread(str(image_path))
    if image is None:
        detection_cache[image_path] = None
        return None

    results = face_detector(image, verbose=False)
    if not results:
        detection_cache[image_path] = None
        return None

    boxes = results[0].boxes
    if boxes is None or boxes.data.shape[0] == 0:
        detection_cache[image_path] = None
        return None

    data = boxes.data.cpu().numpy()
    best_idx = int(np.argmax(data[:, 4]))
    x1, y1, x2, y2, conf, _ = data[best_idx]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        detection_cache[image_path] = None
        return None

    ex1, ey1, ex2, ey2 = expand_bbox((x1, y1, x2, y2), (h, w))
    face_crop = image[ey1:ey2, ex1:ex2]
    if face_crop.size == 0:
        detection_cache[image_path] = None
        return None

    resized = cv2.resize(face_crop, PIPELINE_TARGET_SIZE, interpolation=cv2.INTER_AREA)
    face_for_model = preprocess_input(resized.astype(np.float32))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    result = DetectionResult(
        bbox=(ex1, ey1, ex2, ey2),
        confidence=float(conf),
        face_for_model=face_for_model,
        display_crop=face_rgb,
    )
    detection_cache[image_path] = result
    return result


# ---------------------------------------------------------------------------
# Dataset collage helpers
# ---------------------------------------------------------------------------


def render_collage_panels(
    panels: Sequence[tuple[str, Path, np.ndarray, str]],
    filename: str,
    title: str,
    directory: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.patch.set_facecolor("white")

    for ax, (label, path, image, color) in zip(axes.flatten(), panels):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(label, fontsize=18, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(5)
            spine.set_edgecolor(color)
        record_selection(filename, label, path, note="docs collage")

    fig.suptitle(title, fontsize=24, weight="bold")
    fig.subplots_adjust(wspace=0.02, hspace=0.18)
    return save_figure(fig, filename, directory=directory)


def generate_dataset_collages(real_paths: list[Path], fake_paths: list[Path]) -> list[Path]:
    DOC_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    face_detector: YOLO | None = None

    for filename, config in COLLAGE_CONFIG.items():
        panels: list[tuple[str, Path, np.ndarray, str]] = []
        requires_face = any(panel["type"] == "face" for panel in config["panels"])
        if requires_face and face_detector is None:
            face_detector = YOLO(str(YOLO_WEIGHTS_PATH))

        for panel in config["panels"]:
            label = panel["label"]
            path = ROOT / panel["path"].replace("\\", "/")
            if panel["type"] == "face":
                assert face_detector is not None
                detection = detect_face(face_detector, path)
                if detection is None:
                    raise RuntimeError(f"Failed to detect face in {path}")
                image = cv2.resize(detection.display_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
            else:
                image = load_image_rgb(path)
            color = LABEL_COLORS.get(label, COLOR_PRIMARY)
            panels.append((label, path, image, color))

        outputs.append(
            render_collage_panels(
                panels,
                filename,
                config["title"],
                DOC_FIGURES_DIR,
            )
        )

    return outputs


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

def pick_intro_samples() -> tuple[list[Path], list[Path]]:
    real_paths = list_image_paths(VALIDATION_REAL)
    fake_paths = list_image_paths(VALIDATION_FAKE)
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise RuntimeError("Need at least two real and two fake validation images for intro collage.")
    RNG.shuffle(real_paths)
    RNG.shuffle(fake_paths)
    return real_paths[:2], fake_paths[:2]


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def generate_intro_example() -> Path:
    real_samples, fake_samples = pick_intro_samples()
    samples = [("Real", path, "#3cb44b") for path in real_samples] + [("Fake", path, "#ffa500") for path in fake_samples]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.patch.set_facecolor(COLOR_BACKGROUND)

    for ax, (label, path, color) in zip(axes.flatten(), samples):
        image = load_image_rgb(path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(label, fontsize=18, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(6)
            spine.set_edgecolor(color)
        record_selection("intro_example.png", label, path, note="validation split sample")

    fig.suptitle("Real vs. Synthetic (Validation Samples)", fontsize=24, weight="bold")
    return save_figure(fig, "intro_example.png")


def generate_pipeline_process_diagram() -> Path:
    stages = [
        ("Data\nPreparation", "Cleaning · Augmentation"),
        ("YOLO Face\nDetection", "Localized face crops"),
        ("ResNet50\nClassification", "Deep feature extraction"),
        ("Evaluation", "Accuracy · F1"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    x_positions = np.linspace(0.14, 0.86, len(stages))
    box_width = 0.15
    box_height = 0.30

    for idx, ((title, subtitle), x) in enumerate(zip(stages, x_positions)):
        box = FancyBboxPatch(
            (x - box_width / 2, 0.35),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=2,
            facecolor=COLOR_BACKGROUND,
            edgecolor=COLOR_PRIMARY,
        )
        ax.add_patch(box)
        ax.text(x, 0.53, title, ha="center", va="center", fontsize=11, weight="bold", color="#222222", wrap=True, linespacing=1.2)
        ax.text(x, 0.41, subtitle, ha="center", va="center", fontsize=9, color="#444444", wrap=True)

        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x + box_width / 2 + 0.035, 0.5),
                (x_positions[idx + 1] - box_width / 2 - 0.035, 0.5),
                arrowstyle="->",
                mutation_scale=20,
                color=COLOR_PRIMARY,
                linewidth=2,
            )
            ax.add_patch(arrow)

    fig.suptitle("Pipeline Overview", fontsize=20, weight="bold")
    return save_figure(fig, "pipeline_process_diagram.png")


def generate_frequency_visualization(sample_path: Path) -> Path:
    image = load_image_rgb(sample_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft) + 1e-8)

    h, w = gray.shape
    crow, ccol = h // 2, w // 2
    low_pass_mask = np.zeros_like(gray, dtype=np.float32)
    high_pass_mask = np.ones_like(gray, dtype=np.float32)
    radius = int(min(h, w) * 0.08)
    cv2.circle(low_pass_mask, (ccol, crow), radius, 1, -1)
    cv2.circle(high_pass_mask, (ccol, crow), radius * 2, 0, -1)

    low_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(fft * low_pass_mask)))
    high_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(fft * high_pass_mask)))

    conv_filter = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )
    feature_map = cv2.filter2D(gray.astype(np.float32), -1, conv_filter)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Eingabebild", fontsize=14, weight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(magnitude, cmap="magma")
    axes[0, 1].set_title("Frequenzanalyse (Log-Magnitude)", fontsize=14, weight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(conv_filter, cmap="coolwarm")
    axes[1, 0].set_title("Konvolutionsfilter (Kanten)", fontsize=14, weight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(feature_map, cmap="viridis")
    axes[1, 1].set_title("Resultierende Feature Map", fontsize=14, weight="bold")
    axes[1, 1].axis("off")

    fig.suptitle("Frequenzsensitivität in CNNs", fontsize=20, weight="bold")

    record_selection("frequency_cnn_visualization.png", "input_image", sample_path, note="frequency illustration sample")
    return save_figure(fig, "frequency_cnn_visualization.png")


# ---------------------------------------------------------------------------
# Training history parsing
# ---------------------------------------------------------------------------

loss_pattern = re.compile(r"- loss:\s*([0-9.]+)")
val_loss_pattern = re.compile(r"- val_loss:\s*([0-9.]+)")


def parse_training_history(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")
    train_losses: list[float] = []
    val_losses: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if " - loss:" in line and " - val_loss:" in line:
                train_match = loss_pattern.search(line)
                val_match = val_loss_pattern.search(line)
                if train_match and val_match:
                    train_losses.append(float(train_match.group(1)))
                    val_losses.append(float(val_match.group(1)))
    if not train_losses:
        raise RuntimeError(f"No loss values parsed from {path}")
    return train_losses, val_losses


def plot_loss_curves(train_losses: Sequence[float], val_losses: Sequence[float], title: str, filename: str) -> Path:
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.plot(epochs, train_losses, label="Training", color=COLOR_PRIMARY, linewidth=2.5)
    ax.plot(epochs, val_losses, label="Validation", color=COLOR_SECONDARY, linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=18, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    max_loss = max(max(train_losses), max(val_losses))
    upper = math.ceil(max_loss * 20) / 20
    ax.set_ylim(0, upper)
    ax.set_yticks(np.linspace(0, upper, 6))

    fig.tight_layout()

    return save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def chunk(iterable: Sequence[tuple[Path, int]], size: int) -> Iterable[Sequence[tuple[Path, int]]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def compute_baseline_scores(model_path: Path, fake_paths: list[Path], real_paths: list[Path], batch_size: int = 64) -> pd.DataFrame:
    if BASELINE_SCORES_CSV.exists():
        return pd.read_csv(BASELINE_SCORES_CSV)

    model = load_model(model_path, compile=False)
    dataset = [(path, 0) for path in fake_paths] + [(path, 1) for path in real_paths]
    dataset.sort(key=lambda item: str(item[0]))

    records: list[dict[str, object]] = []
    for batch in chunk(dataset, batch_size):
        images = []
        for path, _ in batch:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Could not load baseline image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            image = preprocess_input(image.astype(np.float32))
            images.append(image)

        batch_array = np.stack(images, axis=0)
        preds = model.predict(batch_array, batch_size=len(batch), verbose=0).flatten()
        for (path, label), score in zip(batch, preds):
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "label": label,
                    "score": float(score),
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(BASELINE_SCORES_CSV, index=False)
    return df


def compute_pipeline_scores(fake_paths: list[Path], real_paths: list[Path], batch_size: int = 32) -> pd.DataFrame:
    if not PIPELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pipeline model not found at {PIPELINE_MODEL_PATH}")
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS_PATH}")

    face_detector = YOLO(str(YOLO_WEIGHTS_PATH))
    pipeline_model = load_model(PIPELINE_MODEL_PATH, compile=False)

    dataset = [(path, 0) for path in fake_paths] + [(path, 1) for path in real_paths]
    dataset.sort(key=lambda item: str(item[0]))

    records: list[dict[str, object]] = []
    pipeline_batch: list[np.ndarray] = []
    meta: list[tuple[Path, int, DetectionResult]] = []
    skipped: list[tuple[Path, int, str]] = []

    for path, label in dataset:
        detection = detect_face(face_detector, path)
        if detection is None:
            skipped.append((path, label, "no_face"))
            continue

        pipeline_batch.append(detection.face_for_model)
        meta.append((path, label, detection))

        if len(pipeline_batch) >= batch_size:
            preds = pipeline_model.predict(np.stack(pipeline_batch), batch_size=len(pipeline_batch), verbose=0).flatten()
            for (meta_path, meta_label, det), score in zip(meta, preds):
                records.append(
                    {
                        "path": str(meta_path.relative_to(ROOT)),
                        "label": meta_label,
                        "score": float(score),
                        "bbox_x1": det.bbox[0],
                        "bbox_y1": det.bbox[1],
                        "bbox_x2": det.bbox[2],
                        "bbox_y2": det.bbox[3],
                        "det_conf": det.confidence,
                    }
                )
            pipeline_batch.clear()
            meta.clear()

    if pipeline_batch:
        preds = pipeline_model.predict(np.stack(pipeline_batch), batch_size=len(pipeline_batch), verbose=0).flatten()
        for (meta_path, meta_label, det), score in zip(meta, preds):
            records.append(
                {
                    "path": str(meta_path.relative_to(ROOT)),
                    "label": meta_label,
                    "score": float(score),
                    "bbox_x1": det.bbox[0],
                    "bbox_y1": det.bbox[1],
                    "bbox_x2": det.bbox[2],
                    "bbox_y2": det.bbox[3],
                    "det_conf": det.confidence,
                }
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(PIPELINE_RESULTS_CSV, index=False)

    if skipped:
        with PIPELINE_SKIPPED_LOG.open("w", encoding="utf-8") as handle:
            for path, label, reason in skipped:
                handle.write(f"{path.relative_to(ROOT)}\t{label}\t{reason}\n")

    return df


def compute_metrics_from_scores(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    labels = df["label"].astype(int).values
    scores = df["score"].astype(float).values
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_real": precision_score(labels, preds, zero_division=0),
        "recall_real": recall_score(labels, preds, zero_division=0),
        "f1_real": f1_score(labels, preds, zero_division=0),
        "precision_fake": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_fake": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_fake": f1_score(labels, preds, pos_label=0, zero_division=0),
    }


def generate_pr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    filename: str,
    *,
    mark_best: bool = False,
) -> Path:
    positive = (labels == 0).astype(int)
    fake_scores = 1.0 - scores
    fpr, tpr, thresholds = roc_curve(positive, fake_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.5, label="Chance")
    ax.set_xlabel("False Positive Rate (Fake misclassified)")
    ax.set_ylabel("True Positive Rate (Fake detected)")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.01)

    # Mark threshold corresponding to real-score 0.50
    fake_threshold = 0.5
    # thresholds returned correspond to decision boundary for fake_scores >= threshold
    idx = np.argmin(np.abs(thresholds - fake_threshold))
    threshold_fpr = fpr[idx]
    threshold_tpr = tpr[idx]
    ax.scatter([threshold_fpr], [threshold_tpr], color=COLOR_SECONDARY, s=80, zorder=5)
    ax.annotate(
        f"Threshold 0.50\nTPR={threshold_tpr:.2f}\nFPR={threshold_fpr:.2f}",
        xy=(threshold_fpr, threshold_tpr),
        xytext=(min(threshold_fpr + 0.15, 0.9), max(threshold_tpr - 0.15, 0.1)),
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=11,
    )

    ax.legend(loc="lower right")

    return save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Additional diagrams
# ---------------------------------------------------------------------------

def generate_implementation_cycle() -> Path:
    labels = [
        ("Data Preparation", "Cleaning · Curation"),
        ("Training", "YOLO + ResNet50"),
        ("Evaluation", "Metrics · Review"),
        ("Fine-tuning", "Hyperparameter sweep"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    radius = 0.35
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)[:-1]

    for angle, (title, subtitle) in zip(angles, labels):
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        box = FancyBboxPatch(
            (x - 0.14, y - 0.08),
            0.28,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=COLOR_BACKGROUND,
            edgecolor=COLOR_PRIMARY,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.03, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x, y - 0.03, subtitle, ha="center", va="center", fontsize=10, color="#555555")

    for angle, next_angle in zip(angles, np.roll(angles, -1)):
        start_center = np.array([0.5 + radius * math.cos(angle), 0.5 + radius * math.sin(angle)])
        end_center = np.array([0.5 + radius * math.cos(next_angle), 0.5 + radius * math.sin(next_angle)])
        direction = end_center - start_center
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        unit = direction / norm
        offset = 0.18
        start = start_center + unit * offset
        end = end_center - unit * offset
        arrow = FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="->",
            color=COLOR_PRIMARY,
            linewidth=2,
            connectionstyle="arc3,rad=0.2",
            mutation_scale=18,
        )
        ax.add_patch(arrow)

    ax.add_patch(Circle((0.5, 0.5), 0.02, color=COLOR_PRIMARY))
    fig.suptitle("Implementation Cycle", fontsize=18, weight="bold")
    return save_figure(fig, "implementation_cycle.png")


def generate_pipeline_results_chart(baseline_metrics: dict[str, float], pipeline_metrics: dict[str, float]) -> Path:
    metrics = ["accuracy", "f1_fake", "precision_fake", "recall_fake"]
    labels = ["Accuracy", "F1 (Fake)", "Precision (Fake)", "Recall (Fake)"]

    baseline_values = [baseline_metrics[m] for m in metrics]
    pipeline_values = [pipeline_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.bar(x - width / 2, baseline_values, width, label="Baseline ResNet50", color=COLOR_PRIMARY, alpha=0.8)
    ax.bar(x + width / 2, pipeline_values, width, label="YOLO + ResNet50 Pipeline", color=COLOR_SECONDARY, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", axis="y", alpha=0.3)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline vs. Baseline Results", fontsize=18, weight="bold", pad=20)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_yticks(np.linspace(0, 1.0, 6))

    for idx, (baseline_val, pipeline_val) in enumerate(zip(baseline_values, pipeline_values)):
        ax.text(idx - width / 2, baseline_val + 0.02, f"{baseline_val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.text(idx + width / 2, pipeline_val + 0.02, f"{pipeline_val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout(pad=2)

    return save_figure(fig, "implementation_pipeline_results.png")


# ---------------------------------------------------------------------------
# Evaluation figure updates (no text overlays)
# ---------------------------------------------------------------------------

def render_detection_image(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    filename: str,
    color: str,
) -> Path:
    image = load_image_rgb(image_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(image)
    ax.axis("off")
    x1, y1, x2, y2 = bbox
    ax.add_patch(
        FancyBboxPatch(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            boxstyle="round,pad=0.01",
            linewidth=4,
            edgecolor=color,
            facecolor="none",
        )
    )
    return save_figure(fig, filename)


def regenerate_evaluation_images(pipeline_df: pd.DataFrame, baseline_df: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []

    # False positive: real image predicted fake by pipeline
    fp_candidates = pipeline_df[(pipeline_df["label"] == 1) & (pipeline_df["score"] < 0.5)].copy()
    if not fp_candidates.empty:
        fp_candidates["margin"] = 0.5 - fp_candidates["score"]
        fp_row = fp_candidates.sort_values("margin", ascending=False).iloc[0]
        path = ROOT / fp_row["path"]
        bbox = (int(fp_row["bbox_x1"]), int(fp_row["bbox_y1"]), int(fp_row["bbox_x2"]), int(fp_row["bbox_y2"]))
        outputs.append(render_detection_image(path, bbox, "eval_false_positive.png", "#ff4136"))
        record_selection("eval_false_positive.png", "Real (FP)", path, score=fp_row["score"], note="bbox only")

    # False negative recovered: fake image pipeline correct but baseline wrong
    merged = pipeline_df.merge(baseline_df, on="path", suffixes=("_pipeline", "_baseline"))
    candidates = merged[
        (merged["label_pipeline"] == 0)
        & (merged["score_baseline"] >= 0.5)
        & (merged["score_pipeline"] < 0.5)
    ].copy()
    if not candidates.empty:
        candidates["baseline_margin"] = candidates["score_baseline"] - 0.5
        row = candidates.sort_values("baseline_margin", ascending=False).iloc[0]
        path = ROOT / row["path"]
        bbox = (
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
            int(row["bbox_x2"]),
            int(row["bbox_y2"]),
        )
        outputs.append(render_detection_image(path, bbox, "eval_false_negative.png", "#2ca02c"))
        record_selection("eval_false_negative.png", "Fake (Recovered)", path, score=row["score_pipeline"], note="bbox only")

    return outputs


# ---------------------------------------------------------------------------
# Selection log
# ---------------------------------------------------------------------------

def write_selection_log() -> None:
    if not selection_records:
        return
    fieldnames = ["figure", "label", "path", "score", "note"]
    with SELECTION_LOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_records)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    real_paths = list_image_paths(VALIDATION_REAL)
    fake_paths = list_image_paths(VALIDATION_FAKE)
    if not real_paths or not fake_paths:
        raise RuntimeError("Validation dataset images not found. Please populate data/validation/(real|fake).")

    outputs: list[Path] = []
    doc_outputs = generate_dataset_collages(real_paths, fake_paths)
    outputs.extend(doc_outputs)

    outputs.append(generate_intro_example())
    outputs.append(generate_pipeline_process_diagram())

    frequency_sample = real_paths[0]
    outputs.append(generate_frequency_visualization(frequency_sample))

    baseline_train, baseline_val = parse_training_history(TRAINING_BASELINE_LOG)
    outputs.append(plot_loss_curves(baseline_train, baseline_val, "ResNet50 Baseline Loss", "resnet50_baseline_loss.png"))

    pipeline_train, pipeline_val = parse_training_history(TRAINING_PIPELINE_LOG)
    outputs.append(plot_loss_curves(pipeline_train, pipeline_val, "Pipeline Loss (YOLO + ResNet50)", "pipeline_loss.png"))

    baseline_df = compute_baseline_scores(BASELINE_MODEL_PATH, fake_paths, real_paths)
    pipeline_df = ensure_scores_available(fake_paths, real_paths)

    outputs.append(
        generate_pr_curve(
            baseline_df["score"].values,
            baseline_df["label"].values,
            "Baseline Precision-Recall (Fake Detection)",
            "resnet50_baseline_prroc.png",
            mark_best=False,
        )
    )

    outputs.append(
        generate_pr_curve(
            pipeline_df["score"].values,
            pipeline_df["label"].values,
            "Pipeline Precision-Recall (Fake Detection)",
            "pipeline_prroc.png",
            mark_best=True,
        )
    )

    outputs.append(generate_implementation_cycle())

    baseline_metrics = compute_metrics_from_scores(baseline_df)
    pipeline_metrics = compute_metrics_from_scores(pipeline_df)
    outputs.append(generate_pipeline_results_chart(baseline_metrics, pipeline_metrics))

    outputs.extend(regenerate_evaluation_images(pipeline_df, baseline_df))

    write_selection_log()

    print("Generated figures:")
    for path in outputs:
        print(f" - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
