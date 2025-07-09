#!/usr/bin/env python3

import os
import shutil
import uuid
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# ─────────────────────────────────────────────────────
# 📁 Pfade relativ zur Datei in Scripts/PrepData/
# ─────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent        # → PrepData/
SCRIPTS_DIR = BASE_DIR.parent                     # → Scripts/
ROOT_DIR = SCRIPTS_DIR.parent                     # → AI-Image-Detector/

DOWNLOADS_DIR = ROOT_DIR / "downloads"
DATA_DIR = ROOT_DIR / "Data"

# ─────────────────────────────────────────────────────

SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def ensure_structure():
    for split in SPLITS:
        for label in ["real", "fake"]:
            os.makedirs(DATA_DIR / split / label, exist_ok=True)


def collect_images_from_dir(directory: Path):
    return [f for f in directory.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]


def split_list(lst):
    train, temp = train_test_split(lst, train_size=SPLITS["train"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    val, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"train": train, "validation": val, "test": test}


def copy_files(files, split: str, label: str):
    target_dir = DATA_DIR / split / label
    for src in tqdm(files, desc=f"{split}/{label}", unit="file"):
        filename = src.name
        dest = target_dir / filename
        if dest.exists():
            filename = f"{uuid.uuid4().hex}_{filename}"
            dest = target_dir / filename
        shutil.copy2(src, dest)
    gc.collect()


def process_split_structure(folder: Path):
    print("📦 Split-Ordnerstruktur erkannt (Train/Validation/Test)")
    for split in ["Train", "Validation", "Test"]:
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


def process_flat_folder(folder: Path, label_map: dict):
    print("🔍 Automatische Aufteilung (kein Splitordner gefunden)")
    for label, dirs in label_map.items():
        all_imgs = []
        for sub in dirs:
            path = folder / sub
            if path.exists() and path.is_dir():
                all_imgs.extend(collect_images_from_dir(path))
        splits = split_list(all_imgs)
        for split, files in splits.items():
            copy_files(files, split, label)


def select_folder_menu():

    root = tk.Tk()
    root.withdraw()  # Verstecke das Hauptfenster

    print("\n📂 Bitte wähle den entpackten Kaggle-Ordner aus …")
    selected_path = filedialog.askdirectory(title="Ordner auswählen")

    if not selected_path:
        print("❌ Kein Ordner ausgewählt. Abbruch.")
        return None

    folder_path = Path(selected_path)
    print(f"📁 Ausgewählt: {folder_path}")
    return folder_path



def main():
    ensure_structure()
    selected_folder = select_folder_menu()
    if not selected_folder:
        return

    # Fall A: Struktur mit Split-Ordnern vorhanden
    if all((selected_folder / s).exists() for s in ["Train", "Validation", "Test"]):
        process_split_structure(selected_folder)
    else:
        # Fall B: Nur Label-Ordner vorhanden → manuelles Mapping
        print("\n🧠 Keine Splitstruktur erkannt. Manuelles Mapping erforderlich.")
        print("📌 Gib echte Labelnamen an (z.B. 'real', 'fake') und ordne vorhandene Unterordner zu.")
        label_map = {}
        for label in ["real", "fake"]:
            dirs = input(f"🔧 Welche Unterordner gehören zu '{label}'? (Komma-getrennt): ").strip().split(",")
            label_map[label] = [d.strip() for d in dirs]
        process_flat_folder(selected_folder, label_map)

    print("\n✅ Einsortieren abgeschlossen.")


if __name__ == "__main__":
    main()
