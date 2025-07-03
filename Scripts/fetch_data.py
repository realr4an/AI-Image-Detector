#!/usr/bin/env python3
# download_and_prepare.py

import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Basisverzeichnisse: einen Ordner über dem Skript
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DOWNLOAD_ROOT = os.path.join(PARENT_DIR, "downloads")
OUTPUT_ROOT = os.path.join(PARENT_DIR, "data")

# Konfiguration für Datensätze mit unterschiedlichen Strukturen
DATASETS_INFO = {
    "manjilkarki/deepfake-and-real-images": {"mode": "pre_split"},
    "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset": {
        "mode": "pre_label", "labels": {"real": ["real"], "fake": ["AI"]}
    },
    "prithivsakthiur/deepfake-vs-real-60k": {
        "mode": "pre_label", "labels": {"real": ["Real"], "fake": ["Fake"]}
    }
}

SPLITS = {"training": 0.7, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def download_and_unzip(dataset_id: str, download_dir: str):
    """Lädt den Dataset-Zip herunter und entpackt ihn manuell, um multiprocessing-Semaphore-Leaks zu vermeiden."""
    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_dir, exist_ok=True)
    print(f"> Downloading {dataset_id} into {download_dir} …")
    # Herunterladen ohne automatisches Entpacken
    api.dataset_download_files(
        dataset_id,
        path=download_dir,
        unzip=False,
        quiet=False
    )
    # Manuelles Entpacken
    ds_name = dataset_id.split("/")[-1]
    zip_path = os.path.join(download_dir, f"{ds_name}.zip")
    if os.path.exists(zip_path):
        print(f"> Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in tqdm(zf.infolist(), desc="Extracting files", unit="file"):
                zf.extract(member, download_dir)
        os.remove(zip_path)
    else:
        print(f"Warnung: {zip_path} nicht gefunden, kein manuelles Entpacken möglich.")


def collect_images_from_dirs(dir_list):
    """Sammelt alle Bilddateien aus einer Liste von Verzeichnissen."""
    imgs = []
    for d in dir_list:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                imgs.append(os.path.join(d, f))
    return imgs


def split_list(lst):
    """Teilt eine Liste von Pfaden in train/val/test nach SPLITS."""
    train, temp = train_test_split(lst, train_size=SPLITS["training"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    val, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"training": train, "validation": val, "test": test}


def copy_files(file_list, dest_dir):
    """Kopiert Dateien mit Fortschrittsanzeige nach dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in tqdm(file_list, desc=f"Copy to {dest_dir}", unit="file"):
        shutil.copy2(src, dest_dir)


def process_pre_split(root_dir: str, output_dir: str):
    """Für Datensätze mit bereits vorhandenen Split-Ordnern."""
    for split in ["Train", "Validation", "Test"]:
        split_lower = split.lower()
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            print(f"Warnung: Ordner {split_path} nicht gefunden, überspringe.")
            continue
        for label in os.listdir(split_path):
            label_lower = label.lower()
            if label_lower not in ("real", "fake"):
                continue
            src_dir = os.path.join(split_path, label)
            dest_dir = os.path.join(output_dir, split_lower, label_lower)
            files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
            copy_files(files, dest_dir)


def process_pre_label(root_dir: str, output_dir: str, label_map: dict):
    """Für Datensätze, die erst nach 'real'/'fake' aufgeteilt und dann gesplittet werden."""
    images = {l: collect_images_from_dirs([os.path.join(root_dir, d) for d in label_map[l]]) for l in label_map}
    splits = {l: split_list(images[l]) for l in images}
    for label, split_dict in splits.items():
        for split_name, files in split_dict.items():
            dest_dir = os.path.join(output_dir, split_name, label)
            copy_files(files, dest_dir)


def process_dataset(dataset_id: str):
    info = DATASETS_INFO[dataset_id]
    ds_name = dataset_id.replace("/", "-")
    download_dir = os.path.join(DOWNLOAD_ROOT, ds_name)
    download_and_unzip(dataset_id, download_dir)
    subdirs = [d for d in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, d))]
    root_dir = os.path.join(download_dir, subdirs[0]) if len(subdirs) == 1 else download_dir
    output_dir = os.path.join(OUTPUT_ROOT, ds_name)
    if info["mode"] == "pre_split":
        process_pre_split(root_dir, output_dir)
    else:
        process_pre_label(root_dir, output_dir, info["labels"])


if __name__ == "__main__":
    for ds in tqdm(DATASETS_INFO.keys(), desc="Verarbeite Datasets", unit="dataset"):
        process_dataset(ds)
    print("✅ Alle Datensätze heruntergeladen und einsortiert.")
