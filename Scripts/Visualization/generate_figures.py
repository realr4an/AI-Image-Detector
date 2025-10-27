#!/usr/bin/env python3
"""Generate reporting figures for the AI-Image-Detector pipeline."""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, roc_curve, auc


RNG = random.Random(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(*parts: str) -> Path:
    for base_name in ("data", "Data"):
        base = ROOT / base_name
        candidate = base.joinpath(*parts)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve path for parts={parts}")


FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
SELECTION_LOG = FIGURES_DIR / "selection_log.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_image_paths(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.glob("**/*")) if p.suffix.lower() in IMAGE_EXTENSIONS]


def record_selection(figure: str, label: str, path: Path, score: float | None = None, note: str | None = None) -> None:
    entry = {
        "figure": figure,
        "label": label,
        "path": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "score": "" if score is None else f"{score:.4f}",
        "note": note or "",
    }
    selection_records.append(entry)


selection_records: list[dict[str, str]] = []

def average_saturation(image_path: Path) -> float | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean() / 255.0)

from dataclasses import dataclass
@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    confidence: float
    face_for_model: np.ndarray
    face_rgb: np.ndarray

detection_cache: dict[Path, tuple[tuple[int, int, int, int], float] | None] = {}

def expand_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int], margin: float = 0.1) -> tuple[int, int, int, int]:
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


def detect_face(face_detector: YOLO, image_path: Path, target_size: tuple[int, int]) -> DetectionResult | None:
    image = cv2.imread(str(image_path))
    if image is None:
        detection_cache[image_path] = None
        return None

    if image_path in detection_cache:
        cache_entry = detection_cache[image_path]
    else:
        cache_entry = None

    if cache_entry is None and image_path in detection_cache:
        return None

    if cache_entry is None:
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
        ex1, ey1, ex2, ey2 = expand_bbox((x1, y1, x2, y2), (h, w), margin=0.15)
        cache_entry = ((ex1, ey1, ex2, ey2), float(conf))
        detection_cache[image_path] = cache_entry

    (bx1, by1, bx2, by2), conf_val = cache_entry
    bx1 = max(0, bx1)
    by1 = max(0, by1)
    bx2 = min(image.shape[1] - 1, bx2)
    by2 = min(image.shape[0] - 1, by2)
    if bx2 <= bx1 or by2 <= by1:
        detection_cache[image_path] = None
        return None

    face_crop = image[by1:by2, bx1:bx2]
    if face_crop.size == 0:
        detection_cache[image_path] = None
        return None

    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    face_for_model = preprocess_input(resized.astype(np.float32))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    return DetectionResult(
        bbox=(bx1, by1, bx2, by2),
        confidence=conf_val,
        face_for_model=face_for_model,
        face_rgb=face_rgb,
    )
def load_image_rgb(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_for_base(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    return preprocess_input(resized.astype(np.float32))

def save_figure(fig: plt.Figure, filename: str) -> Path:
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path

def list_image_paths(directory: Path) -> list[Path]:
    return [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

def generate_collage_basic(real_paths: list[Path], fake_paths: list[Path]) -> Path:
    if len(real_paths) < 2 or len(fake_paths) < 2:
        raise ValueError("Need at least 2 real and 2 fake images for the collage")

    selected_real = RNG.sample(real_paths, 2)
    selected_fake = RNG.sample(fake_paths, 2)
    entries = [("Real", selected_real[0]), ("Real", selected_real[1]), ("Fake", selected_fake[0]), ("Fake", selected_fake[1])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (label, path) in zip(axes.flatten(), entries):
        image = load_image_rgb(path)
        if image is None:
            ax.axis("off")
            continue
        ax.imshow(image)
        ax.axis("off")
        ax.text(0.02, 0.08, label, transform=ax.transAxes, fontsize=24, weight="bold", color="white", backgroundcolor=(0, 0, 0, 0.5))
        record_selection("datasets_deepfake_vs_real60k.png", label, path, note="validation random pick")

    fig.suptitle("DeepFake vs. Real (Validation Samples)", fontsize=26, weight="bold")
    return save_figure(fig, "datasets_deepfake_vs_real60k.png")

def select_by_saturation(paths: list[Path], k: int, pick_high: bool) -> list[Path]:
    sample_count = min(len(paths), 200)
    sampled = RNG.sample(paths, sample_count)
    scored: list[tuple[float, Path]] = []
    for path in sampled:
        sat = average_saturation(path)
        if sat is not None:
            scored.append((sat, path))
    if len(scored) < k:
        return RNG.sample(paths, k)
    scored.sort(key=lambda item: item[0], reverse=pick_high)
    return [path for _, path in scored[:k]]


def generate_collage_contrast(real_paths: list[Path], fake_paths: list[Path]) -> Path:
    selected_real = select_by_saturation(real_paths, 2, pick_high=False)
    selected_fake = select_by_saturation(fake_paths, 2, pick_high=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [("Real", selected_real[0]), ("Real", selected_real[1]), ("Fake", selected_fake[0]), ("Fake", selected_fake[1])]
    subtitle_map = {"Real": "Low saturation portrait", "Fake": "High saturation synthetic"}

    for ax, (label, path) in zip(axes.flatten(), panels):
        image = load_image_rgb(path)
        if image is None:
            ax.axis("off")
            continue
        ax.imshow(image)
        ax.axis("off")
        ax.text(0.02, 0.08, label, transform=ax.transAxes, fontsize=24, weight="bold", color="white", backgroundcolor=(0, 0, 0, 0.55))
        ax.text(0.02, 0.92, subtitle_map[label], transform=ax.transAxes, fontsize=14, color="white", backgroundcolor=(0, 0, 0, 0.4))
        record_selection("datasets_detect_ai_generated.png", label, path, note="saturation guided pick")

    fig.suptitle("Detecting AI Generated Faces: Stylized vs. Natural", fontsize=26, weight="bold")
    return save_figure(fig, "datasets_detect_ai_generated.png")

def pick_detected_samples(face_detector: YOLO, paths: list[Path], count: int, target_size: tuple[int, int]) -> list[tuple[Path, DetectionResult]]:
    shuffled = paths.copy()
    RNG.shuffle(shuffled)
    picks: list[tuple[Path, DetectionResult]] = []
    for path in shuffled:
        detection = detect_face(face_detector, path, target_size)
        if detection is None:
            continue
        picks.append((path, detection))
        if len(picks) == count:
            break
    if len(picks) < count:
        raise RuntimeError(f"Unable to find {count} detected faces in selection")
    return picks


def generate_face_detail_collage(face_detector: YOLO, real_paths: list[Path], fake_paths: list[Path], target_size: tuple[int, int]) -> Path:
    real_detected = pick_detected_samples(face_detector, real_paths, 2, target_size)
    fake_detected = pick_detected_samples(face_detector, fake_paths, 2, target_size)
    panels = real_detected + fake_detected

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (path, detection) in zip(axes.flatten(), panels):
        label = "Real" if path in [item[0] for item in real_detected] else "Fake"
        face_rgb = cv2.resize(detection.face_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
        ax.imshow(face_rgb)
        ax.axis("off")
        ax.text(0.02, 0.08, label, transform=ax.transAxes, fontsize=24, weight="bold", color="white", backgroundcolor=(0, 0, 0, 0.5))
        ax.text(0.02, 0.92, "Zoomed-in facial features", transform=ax.transAxes, fontsize=14, color="white", backgroundcolor=(0, 0, 0, 0.35))
        record_selection("datasets_deepfake_and_real_images.png", label, path, note="face crop via YOLO")

    fig.suptitle("DeepFake Artefacts vs. Real Details", fontsize=26, weight="bold")
    return save_figure(fig, "datasets_deepfake_and_real_images.png")

def generate_face_detail_collage(face_detector: YOLO, real_paths: list[Path], fake_paths: list[Path], target_size: tuple[int, int]) -> Path:
    real_detected = pick_detected_samples(face_detector, real_paths, 2, target_size)
    fake_detected = pick_detected_samples(face_detector, fake_paths, 2, target_size)
    panels = real_detected + fake_detected
    real_path_set = {path for path, _ in real_detected}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (path, detection) in zip(axes.flatten(), panels):
        label = "Real" if path in real_path_set else "Fake"
        face_rgb = cv2.resize(detection.face_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
        ax.imshow(face_rgb)
        ax.axis("off")
        ax.text(0.02, 0.08, label, transform=ax.transAxes, fontsize=24, weight="bold", color="white", backgroundcolor=(0, 0, 0, 0.5))
        ax.text(0.02, 0.92, "Highlighted facial micro-details", transform=ax.transAxes, fontsize=14, color="white", backgroundcolor=(0, 0, 0, 0.35))
        record_selection("datasets_deepfake_and_real_images.png", label, path, note="face crop via YOLO")

    fig.suptitle("DeepFake Artefacts vs. Real Details", fontsize=26, weight="bold")
    return save_figure(fig, "datasets_deepfake_and_real_images.png")

def widerface_label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = image_path.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def load_widerface_boxes(image_path: Path, images_root: Path, labels_root: Path) -> list[tuple[int, int, int, int]]:
    label_path = widerface_label_path(image_path, images_root, labels_root)
    if not label_path.exists():
        return []
    with label_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if not lines:
        return []
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    h, w = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        box_w = bw * w
        box_h = bh * h
        x_center = cx * w
        y_center = cy * h
        x1 = int(max(0, x_center - box_w / 2))
        y1 = int(max(0, y_center - box_h / 2))
        x2 = int(min(w - 1, x_center + box_w / 2))
        y2 = int(min(h - 1, y_center + box_h / 2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def pick_widerface_samples(images_root: Path, labels_root: Path, count: int = 3) -> list[tuple[Path, list[tuple[int, int, int, int]]]]:
    all_images = [p for p in images_root.glob("**/*.jpg")]
    RNG.shuffle(all_images)
    selections: list[tuple[Path, list[tuple[int, int, int, int]]]] = []
    for image_path in all_images:
        boxes = load_widerface_boxes(image_path, images_root, labels_root)
        if not boxes:
            continue
        selections.append((image_path, boxes))
        if len(selections) == count:
            break
    if len(selections) < count:
        raise RuntimeError("Not enough WIDER FACE samples with boxes found")
    return selections


def generate_widerface_collage(images_root: Path, labels_root: Path) -> Path:
    samples = pick_widerface_samples(images_root, labels_root, count=3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (image_path, boxes) in zip(axes, samples):
        image = load_image_rgb(image_path)
        if image is None:
            ax.axis("off")
            continue
        ax.imshow(image)
        ax.axis("off")
        for x1, y1, x2, y2 in boxes:
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#00FF88", facecolor="none")
            ax.add_patch(rect)
        ax.text(0.02, 0.08, "Faces", transform=ax.transAxes, fontsize=18, color="white", backgroundcolor=(0, 0, 0, 0.45))
        record_selection("datasets_widerface.png", "WIDER FACE", image_path, note=f"{len(boxes)} boxes")

    fig.suptitle("WIDER FACE Validation Samples with YOLO Boxes", fontsize=22, weight="bold")
    return save_figure(fig, "datasets_widerface.png")

def generate_yolo_loss_curve(results_csv: Path) -> Path:
    df = pd.read_csv(results_csv)
    epochs = df["epoch"].values
    metrics = [
        ("Box Loss", "train/box_loss", "val/box_loss"),
        ("Classification Loss", "train/cls_loss", "val/cls_loss"),
        ("DFL Loss", "train/dfl_loss", "val/dfl_loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    for ax, (title, train_key, val_key) in zip(axes, metrics):
        ax.plot(epochs, df[train_key], label="Train", color="#1f77b4", linewidth=2)
        if val_key in df.columns:
            ax.plot(epochs, df[val_key], label="Validation", color="#ff7f0e", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.suptitle("YOLOv8 Training Losses", fontsize=20, weight="bold")
    return save_figure(fig, "yolo_loss_curve.png")

def generate_yolo_precision_recall(results_csv: Path) -> Path:
    df = pd.read_csv(results_csv)
    precision = df["metrics/precision(B)"].values
    recall = df["metrics/recall(B)"].values
    epochs = df["epoch"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, marker="o", color="#2ca02c", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("YOLOv8 Precision-Recall Across Epochs", fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    if len(epochs) > 0:
        ax.annotate(f"Epoch {epochs[-1]}", xy=(recall[-1], precision[-1]), xytext=(recall[-1] + 0.01, precision[-1] - 0.05),
                    arrowprops=dict(arrowstyle="->", color="#444"), fontsize=10)
    return save_figure(fig, "yolo_precision_recall.png")

def evaluate_pipeline(face_detector: YOLO,
                      pipeline_model_path: Path,
                      base_model_path: Path,
                      fake_paths: list[Path],
                      real_paths: list[Path],
                      target_size: tuple[int, int] = (256, 256),
                      pipeline_batch_size: int = 32,
                      base_batch_size: int = 32) -> tuple[list[dict[str, object]], dict[Path, float], list[tuple[Path, int, str]]]:
    pipeline_model = load_model(pipeline_model_path, compile=False)
    base_model = load_model(base_model_path, compile=False)

    dataset = [(path, 0) for path in fake_paths] + [(path, 1) for path in real_paths]
    dataset.sort(key=lambda item: str(item[0]))

    pipeline_batch: list[np.ndarray] = []
    pipeline_meta: list[tuple[Path, int, DetectionResult]] = []
    base_batch: list[np.ndarray] = []
    base_meta: list[Path] = []

    results: list[dict[str, object]] = []
    base_scores: dict[Path, float] = {}
    skipped: list[tuple[Path, int, str]] = []

    for path, label in dataset:
        image = cv2.imread(str(path))
        if image is None:
            skipped.append((path, label, "read_error"))
            continue

        base_batch.append(preprocess_for_base(image, target_size))
        base_meta.append(path)
        if len(base_batch) >= base_batch_size:
            base_preds = base_model.predict(np.stack(base_batch), batch_size=base_batch_size, verbose=0).flatten()
            for meta_path, score in zip(base_meta, base_preds):
                base_scores[meta_path] = float(score)
            base_batch.clear()
            base_meta.clear()

        detection = detect_face(face_detector, path, target_size)
        if detection is None:
            skipped.append((path, label, "no_face"))
            continue

        pipeline_batch.append(detection.face_for_model)
        pipeline_meta.append((path, label, detection))
        if len(pipeline_batch) >= pipeline_batch_size:
            preds = pipeline_model.predict(np.stack(pipeline_batch), batch_size=pipeline_batch_size, verbose=0).flatten()
            for (meta_path, meta_label, det), score in zip(pipeline_meta, preds):
                results.append({
                    "path": meta_path,
                    "label": meta_label,
                    "score": float(score),
                    "bbox": det.bbox,
                    "det_conf": det.confidence,
                })
            pipeline_batch.clear()
            pipeline_meta.clear()

    if base_batch:
        base_preds = base_model.predict(np.stack(base_batch), batch_size=len(base_batch), verbose=0).flatten()
        for meta_path, score in zip(base_meta, base_preds):
            base_scores[meta_path] = float(score)
    if pipeline_batch:
        preds = pipeline_model.predict(np.stack(pipeline_batch), batch_size=len(pipeline_batch), verbose=0).flatten()
        for (meta_path, meta_label, det), score in zip(pipeline_meta, preds):
            results.append({
                "path": meta_path,
                "label": meta_label,
                "score": float(score),
                "bbox": det.bbox,
                "det_conf": det.confidence,
            })

    return results, base_scores, skipped

def pipeline_results_to_df(results: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for item in results:
        bbox = item["bbox"]
        records.append({
            "path": item["path"],
            "label": item["label"],
            "score": item["score"],
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "det_conf": item["det_conf"],
        })
    df = pd.DataFrame(records)
    df["path"] = df["path"].apply(lambda p: str(p.relative_to(ROOT)) if isinstance(p, Path) else str(p))
    return df

def pipeline_results_to_df(results: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for item in results:
        bbox = item["bbox"]
        path_obj = item["path"] if isinstance(item["path"], Path) else Path(str(item["path"]))
        records.append({
            "path": str(path_obj.relative_to(ROOT)),
            "path_obj": path_obj,
            "label": int(item["label"]),
            "score": float(item["score"]),
            "bbox_x1": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x2": int(bbox[2]),
            "bbox_y2": int(bbox[3]),
            "det_conf": float(item["det_conf"]),
        })
    df = pd.DataFrame(records)
    return df

def generate_pipeline_roc(df: pd.DataFrame) -> tuple[Path, dict[str, float]]:
    y_true_fake = (df["label"].values == 0).astype(int)
    fake_scores = 1.0 - df["score"].values
    fpr, tpr, thresholds = roc_curve(y_true_fake, fake_scores)
    roc_auc = auc(fpr, tpr)

    positive_mask = y_true_fake == 1
    negative_mask = ~positive_mask
    tp = ((fake_scores >= 0.5) & positive_mask).sum()
    fn = ((fake_scores < 0.5) & positive_mask).sum()
    fp = ((fake_scores >= 0.5) & negative_mask).sum()
    tn = ((fake_scores < 0.5) & negative_mask).sum()
    tpr_point = tp / (tp + fn) if (tp + fn) else 0.0
    fpr_point = fp / (fp + tn) if (fp + tn) else 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.scatter([fpr_point], [tpr_point], color="red", s=80, label="Threshold 0.50")
    ax.set_xlabel("False Positive Rate (Fake as Real)")
    ax.set_ylabel("True Positive Rate (Fake detected)")
    ax.set_title("Pipeline ROC Curve", fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    output_path = save_figure(fig, "pipeline_roc.png")
    stats = {"roc_auc": roc_auc, "threshold_0_5_fpr": fpr_point, "threshold_0_5_tpr": tpr_point}
    return output_path, stats

def generate_pipeline_pr(df: pd.DataFrame) -> tuple[Path, dict[str, float]]:
    y_true_fake = (df["label"].values == 0).astype(int)
    fake_scores = 1.0 - df["score"].values
    precision, recall, thresholds = precision_recall_curve(y_true_fake, fake_scores)

    f1_scores = np.zeros_like(precision)
    for idx, (p, r) in enumerate(zip(precision, recall)):
        denom = p + r
        if denom == 0:
            f1_scores[idx] = 0.0
        else:
            f1_scores[idx] = 2 * p * r / denom
    best_idx = int(np.argmax(f1_scores))
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    if best_idx < len(thresholds):
        best_fake_threshold = thresholds[best_idx]
    else:
        best_fake_threshold = thresholds[-1]
    best_real_threshold = 1.0 - best_fake_threshold

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="#d62728", linewidth=2)
    ax.scatter([best_recall], [best_precision], color="navy", s=80, label=f"Max F1={best_f1:.3f}\nScore~{best_real_threshold:.2f}")
    ax.set_xlabel("Recall (Fake detected)")
    ax.set_ylabel("Precision (Fake detected)")
    ax.set_title("Pipeline Precision-Recall", fontsize=16, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left")

    output_path = save_figure(fig, "pipeline_pr.png")
    stats = {
        "best_f1": float(best_f1),
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
        "best_real_threshold": float(best_real_threshold),
    }
    return output_path, stats

def generate_pipeline_process_diagram() -> Path:
    stages = [
        ("Data Preparation", "Augmentation\nBalancing"),
        ("YOLO Face Detection", "WIDER pretraining"),
        ("ResNet50 Classification", "Fine-tuned on faces"),
        ("Evaluation", "Accuracy / F1"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    total = len(stages)
    for idx, (title, subtitle) in enumerate(stages):
        x = 0.05 + idx * 0.23
        box = FancyBboxPatch(
            (x, 0.4), 0.2, 0.35,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=2,
            edgecolor="#1f77b4",
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 0.1, 0.57, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 0.1, 0.47, subtitle, ha="center", va="center", fontsize=10)
        if idx < total - 1:
            ax.annotate(
                "",
                xy=(x + 0.2, 0.575),
                xytext=(x + 0.23, 0.575),
                arrowprops=dict(arrowstyle="->", linewidth=2, color="#444"),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("Pipeline Overview", fontsize=18, weight="bold")
    return save_figure(fig, "pipeline_process_diagram.png")

def render_detection_figure(image_path: Path,
                             bbox: tuple[int, int, int, int],
                             title: str,
                             caption: str,
                             filename: str) -> Path:
    image = load_image_rgb(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    x1, y1, x2, y2 = bbox
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="#ff4136", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(title, fontsize=18, weight="bold")
    fig.text(0.5, 0.03, caption, ha="center", fontsize=12)
    return save_figure(fig, filename)

def generate_false_positive_figure(df: pd.DataFrame, base_scores: dict[Path, float]) -> Path:
    candidates = df[(df["label"] == 1) & (df["score"] < 0.5)].copy()
    if candidates.empty:
        raise RuntimeError("No false positives found in pipeline predictions")
    candidates["margin"] = 0.5 - candidates["score"]
    row = candidates.sort_values("margin", ascending=False).iloc[0]
    path_obj: Path = row["path_obj"]
    pipeline_score = float(row["score"])
    base_score = base_scores.get(path_obj)
    caption = f"Pipeline real-score={pipeline_score:.2f} (pred Fake)."
    if base_score is not None:
        caption += f" Base real-score={base_score:.2f}."
    output = render_detection_figure(path_obj, (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])),
                                     "False Positive: Real ? Pred Fake", caption, "eval_false_positive.png")
    record_selection("eval_false_positive.png", "Real image", path_obj, score=pipeline_score, note="Pipeline false positive")
    return output

def generate_false_negative_figure(df: pd.DataFrame, base_scores: dict[Path, float]) -> Path:
    rows = []
    for _, row in df.iterrows():
        if row["label"] != 0:
            continue
        base_score = base_scores.get(row["path_obj"])
        if base_score is None:
            continue
        if base_score >= 0.5 and row["score"] < 0.5:
            rows.append((row, base_score))
    if not rows:
        raise RuntimeError("No sample where base misclassifies but pipeline correct")
    rows.sort(key=lambda item: item[1], reverse=True)
    row, base_score = rows[0]
    path_obj: Path = row["path_obj"]
    pipeline_score = float(row["score"])
    caption = f"Base real-score={base_score:.2f} (wrong). Pipeline real-score={pipeline_score:.2f} (correct Fake)."
    output = render_detection_figure(path_obj, (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])),
                                     "Recovered False Negative", caption, "eval_false_negative.png")
    record_selection("eval_false_negative.png", "Fake image", path_obj, score=pipeline_score, note="Pipeline recovered base FN")
    return output

def write_selection_log() -> None:
    if not selection_records:
        return
    fieldnames = ["figure", "label", "path", "score", "note"]
    with SELECTION_LOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_records)

def main() -> None:
    print("Preparing figure generation...")
    val_real_dir = resolve_path("validation", "real")
    val_fake_dir = resolve_path("validation", "fake")
    real_paths = list_image_paths(val_real_dir)
    fake_paths = list_image_paths(val_fake_dir)
    if not real_paths or not fake_paths:
        raise RuntimeError("Validation directories missing images")

    face_weights = ROOT / "Models" / "YOLOv8_Face_Detection" / "weights" / "best.pt"
    if not face_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found at {face_weights}")
    face_detector = YOLO(str(face_weights))
    target_size = (256, 256)

    outputs: list[Path] = []
    outputs.append(generate_collage_basic(real_paths, fake_paths))
    outputs.append(generate_collage_contrast(real_paths, fake_paths))
    outputs.append(generate_face_detail_collage(face_detector, real_paths, fake_paths, target_size))

    wider_images_root = resolve_path("widerface", "images", "val")
    wider_labels_root = resolve_path("widerface", "labels", "val")
    outputs.append(generate_widerface_collage(wider_images_root, wider_labels_root))

    yolo_results_csv = ROOT / "Models" / "YOLOv8_Face_Detection" / "results.csv"
    if not yolo_results_csv.exists():
        fallback = ROOT / "docs" / "raw_training_artifacts" / "results.csv"
        if fallback.exists():
            yolo_results_csv = fallback
        else:
            raise FileNotFoundError("YOLO results.csv not found")
    outputs.append(generate_yolo_loss_curve(yolo_results_csv))
    outputs.append(generate_yolo_precision_recall(yolo_results_csv))

    pipeline_model_path = ROOT / "Models" / "ResNet50_YOLO_Pipeline_20250713-230159.h5"
    base_model_path = ROOT / "Models" / "ResNet50_20250710-061102.h5"
    print("Running pipeline evaluation on validation set...")
    pipeline_results, base_scores, skipped = evaluate_pipeline(face_detector, pipeline_model_path, base_model_path, fake_paths, real_paths, target_size=target_size)
    if skipped:
        skipped_log = FIGURES_DIR / "pipeline_skipped.log"
        with skipped_log.open("w", encoding="utf-8") as handle:
            for path, label, reason in skipped:
                handle.write(f"{path.relative_to(ROOT)}\t{label}\t{reason}\n")
        print(f"Skipped {len(skipped)} images (see {skipped_log.relative_to(ROOT)})")

    df = pipeline_results_to_df(pipeline_results)
    if df.empty:
        raise RuntimeError("No pipeline predictions available for plotting")
    scores_csv = FIGURES_DIR / "pipeline_validation_scores.csv"
    df.drop(columns=["path_obj"], inplace=False).to_csv(scores_csv, index=False)

    roc_path, roc_stats = generate_pipeline_roc(df)
    pr_path, pr_stats = generate_pipeline_pr(df)
    outputs.append(roc_path)
    outputs.append(pr_path)
    outputs.append(generate_pipeline_process_diagram())
    outputs.append(generate_false_positive_figure(df, base_scores))
    outputs.append(generate_false_negative_figure(df, base_scores))

    write_selection_log()

    print("Created figures:")
    for path in outputs:
        print(f" - {path.relative_to(ROOT)}")
    print(f"ROC stats: {roc_stats}")
    print(f"PR stats: {pr_stats}")


if __name__ == "__main__":
    main()






