#!/usr/bin/env python3
"""Train a YOLOv8 detector on WIDER FACE to extract facial regions."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import requests
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "Models" / "YOLOv8_Face_Detection"
DATA_ROOT = ROOT_DIR / "Data"
WIDERFACE_DIR = DATA_ROOT / "widerface"
DATASET_CONFIG_PATH = WIDERFACE_DIR / "data.yaml"

WIDERFACE_URLS: Dict[str, str] = {
    "train_images": "https://huggingface.co/datasets/wider-face/resolve/main/WIDER_train.zip",
    "val_images": "https://huggingface.co/datasets/wider-face/resolve/main/WIDER_val.zip",
    "annotations": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
}


class YOLOv8FaceTrainer:
    """Download WIDER FACE, convert annotations, and fine-tune a YOLOv8 detector."""

    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
    ) -> None:
        self.base_model = base_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.project_name = "YOLOv8_Face_Detection"

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        WIDERFACE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset preparation helpers
    # ------------------------------------------------------------------

    def _download_file(self, url: str, destination: Path) -> None:
        """Download a file with a progress bar if it does not exist yet."""
        if destination.exists():
            print(f"> {destination.name} already exists. Skipping download.")
            return

        print(f"> Downloading {destination.name} ...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with destination.open("wb") as file, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Download",
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress.update(len(chunk))

        print(f"> Downloaded {destination.name}")

    def _convert_annotations(self, image_dir: Path, annotation_file: Path, target_label_dir: Path) -> None:
        """Convert the WIDER FACE annotation format into YOLO text files."""
        target_label_dir.mkdir(parents=True, exist_ok=True)
        print(f"> Converting annotations from {annotation_file.name} ...")

        with annotation_file.open("r") as handle:
            lines = handle.readlines()

        index = 0
        created_files = 0
        total_lines = len(lines)

        with tqdm(total=total_lines, desc="Annotations", unit="line") as bar:
            while index < total_lines:
                image_rel_path = lines[index].strip()
                index += 1
                bar.update(1)

                if not image_rel_path.endswith(".jpg"):
                    continue

                box_count = int(lines[index].strip())
                index += 1
                bar.update(1)

                image_path = image_dir / image_rel_path
                image = cv2.imread(str(image_path))
                if image is None:
                    index += box_count
                    bar.update(box_count)
                    continue

                height, width = image.shape[:2]
                label_path = target_label_dir / image_rel_path.replace(".jpg", ".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)

                yolo_annotations: List[str] = []
                for _ in range(box_count):
                    fields = lines[index].strip().split()
                    index += 1
                    bar.update(1)

                    if len(fields) < 4:
                        continue

                    x, y, w, h = map(float, fields[:4])
                    if w <= 0 or h <= 0:
                        continue

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    norm_w = w / width
                    norm_h = h / height

                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                if yolo_annotations:
                    with label_path.open("w") as label_file:
                        label_file.write("\n".join(yolo_annotations))
                    created_files += 1
                else:
                    label_path.unlink(missing_ok=True)

        print(f"> Finished conversion. Created {created_files} label files.")

    def prepare_widerface_dataset(self) -> None:
        """Download and prepare the WIDER FACE dataset for YOLO training."""
        if DATASET_CONFIG_PATH.exists():
            print("> WIDER FACE dataset already prepared.")
            return

        zip_paths = {name: WIDERFACE_DIR / f"{name}.zip" for name in WIDERFACE_URLS}
        for name, url in WIDERFACE_URLS.items():
            self._download_file(url, zip_paths[name])

        print("> Extracting archives ...")
        for zip_path in zip_paths.values():
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(WIDERFACE_DIR)

        train_images = WIDERFACE_DIR / "WIDER_train" / "images"
        val_images = WIDERFACE_DIR / "WIDER_val" / "images"
        annotations_dir = WIDERFACE_DIR / "wider_face_split"

        if not train_images.exists() or not val_images.exists() or not annotations_dir.exists():
            raise FileNotFoundError("The WIDER FACE archive structure has changed. Please update the script.")

        yolo_images_train = WIDERFACE_DIR / "images" / "train"
        yolo_images_val = WIDERFACE_DIR / "images" / "val"
        yolo_labels_train = WIDERFACE_DIR / "labels" / "train"
        yolo_labels_val = WIDERFACE_DIR / "labels" / "val"

        yolo_images_train.parent.mkdir(parents=True, exist_ok=True)
        yolo_images_val.parent.mkdir(parents=True, exist_ok=True)

        if not yolo_images_train.exists():
            shutil.move(str(train_images), str(yolo_images_train))
        if not yolo_images_val.exists():
            shutil.move(str(val_images), str(yolo_images_val))

        train_annotation_file = annotations_dir / "wider_face_train_bbx_gt.txt"
        val_annotation_file = annotations_dir / "wider_face_val_bbx_gt.txt"

        self._convert_annotations(yolo_images_train, train_annotation_file, yolo_labels_train)
        self._convert_annotations(yolo_images_val, val_annotation_file, yolo_labels_val)

        print("> Creating data.yaml configuration ...")
        data_config = {
            "train": str(yolo_images_train.resolve()),
            "val": str(yolo_images_val.resolve()),
            "nc": 1,
            "names": ["face"],
        }
        with DATASET_CONFIG_PATH.open("w") as config_file:
            yaml.safe_dump(data_config, config_file)
        print(f"> Saved dataset configuration to {DATASET_CONFIG_PATH}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Prepare data and fine-tune YOLO. Returns the path to the best weights."""
        self.prepare_widerface_dataset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"> Using device: {device}")

        model = YOLO(self.base_model)
        model.train(
            data=str(DATASET_CONFIG_PATH),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=str(MODEL_DIR),
            name=self.project_name,
            exist_ok=True,
            patience=20,
            device=device,
        )

        best_model = MODEL_DIR / self.project_name / "weights" / "best.pt"
        if not best_model.exists():
            raise FileNotFoundError("Training finished but best.pt was not created.")

        print(f"> Training complete. Best weights stored at {best_model}")
        return best_model


if __name__ == "__main__":
    trainer = YOLOv8FaceTrainer(epochs=50)
    trainer.train()
