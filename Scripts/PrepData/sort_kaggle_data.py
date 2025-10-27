#!/usr/bin/env python3
"""Interactively map Kaggle dataset folders into the project structure."""

import gc
import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import tkinter as tk
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Data"

SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def ensure_structure() -> None:
    """Ensure the project-wide train/validation/test folders exist."""
    for split in SPLITS:
        for label in ("real", "fake"):
            (DATA_DIR / split / label).mkdir(parents=True, exist_ok=True)


def collect_images_from_dir(directory: Path) -> List[Path]:
    """Recursively collect all image files from a directory."""
    return [file for file in directory.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS]


def split_list(items: List[Path]) -> Dict[str, List[Path]]:
    """Split a list of items into train/validation/test subsets."""
    if not items:
        return {"train": [], "validation": [], "test": []}

    train, temp = train_test_split(items, train_size=SPLITS["train"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    validation, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"train": train, "validation": validation, "test": test}


def copy_files(files: Iterable[Path], split: str, label: str) -> None:
    """Copy files into the project data structure, avoiding name collisions."""
    target_dir = DATA_DIR / split / label
    target_dir.mkdir(parents=True, exist_ok=True)

    for src in tqdm(files, desc=f"{split}/{label}", unit="file"):
        destination = target_dir / src.name
        if destination.exists():
            destination = target_dir / f"{uuid.uuid4().hex}_{src.name}"
        shutil.copy2(src, destination)
    gc.collect()


def process_split_structure(folder: Path) -> None:
    """Handle datasets that follow a Train/Validation/Test structure."""
    print("Detected structured dataset with Train/Validation/Test folders.")
    for split in ("Train", "Validation", "Test"):
        split_dir = folder / split
        if not split_dir.exists():
            continue
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name.lower()
            if label not in ("real", "fake"):
                continue
            images = collect_images_from_dir(label_dir)
            copy_files(images, split.lower(), label)


def process_flat_folder(folder: Path, label_map: Dict[str, Iterable[str]]) -> None:
    """Handle datasets that only separate images by label without splits."""
    print("No explicit splits found; creating train/validation/test splits locally.")
    for label, directories in label_map.items():
        images: List[Path] = []
        for subdir in directories:
            candidate = folder / subdir
            if candidate.is_dir():
                images.extend(collect_images_from_dir(candidate))
        splits = split_list(images)
        for split_name, files in splits.items():
            copy_files(files, split_name, label)


def select_folder_menu() -> Optional[Path]:
    """Open a folder selection dialog to pick the Kaggle export folder."""
    root = tk.Tk()
    root.withdraw()
    print("\nPlease select the extracted Kaggle dataset folder.")
    selected = filedialog.askdirectory(title="Select Kaggle dataset directory")

    if not selected:
        print("No folder selected. Aborting.")
        return None

    path = Path(selected)
    print(f"Selected folder: {path}")
    return path


def main() -> None:
    ensure_structure()
    selected_folder = select_folder_menu()
    if not selected_folder:
        return

    if all((selected_folder / split).exists() for split in ("Train", "Validation", "Test")):
        process_split_structure(selected_folder)
    else:
        print("\nNo split folders detected. Please map subdirectories to labels.")
        label_map: Dict[str, Iterable[str]] = {}
        for label in ("real", "fake"):
            user_input = input(
                f"Which subfolders correspond to '{label}'? (comma separated): "
            ).strip()
            label_map[label] = [entry.strip() for entry in user_input.split(",") if entry.strip()]
        process_flat_folder(selected_folder, label_map)

    print("\nDataset sorting complete.")


if __name__ == "__main__":
    main()
