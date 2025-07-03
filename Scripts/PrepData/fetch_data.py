#!/usr/bin/env python3
# download_and_prepare.py

import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import uuid

# Pfade (Root: AI-IMAGE-DETECTOR/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # → PrepData
SCRIPTS_DIR = os.path.dirname(BASE_DIR)                      # → Scripts
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)                      # → AI-IMAGE-DETECTOR

DATA_DIR = os.path.join(ROOT_DIR, "Data")                    # → AI-IMAGE-DETECTOR/Data
DOWNLOAD_ROOT = os.path.join(ROOT_DIR, "downloads")          # → AI-IMAGE-DETECTOR/downloads

# Dataset-Konfiguration
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


def ensure_structure():
    """Erstellt die Zielstruktur: Data/train|val|test/real|fake"""
    for split in SPLITS.keys():
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(DATA_DIR, split, label), exist_ok=True)


def download_and_unzip(dataset_id: str, download_dir: str):
    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_dir, exist_ok=True)
    print(f"> Downloading {dataset_id} …")
    api.dataset_download_files(dataset_id, path=download_dir, unzip=False, quiet=False)

    ds_name = dataset_id.split("/")[-1]
    zip_path = os.path.join(download_dir, f"{ds_name}.zip")
    if os.path.exists(zip_path):
        print(f"> Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in tqdm(zf.infolist(), desc="Extracting", unit="file"):
                zf.extract(member, download_dir)
        os.remove(zip_path)
    else:
        print(f"⚠️ Warnung: {zip_path} nicht gefunden!")


def collect_images_from_dirs(dir_list):
    images = []
    for d in dir_list:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(d, f))
    return images


def split_list(lst):
    train, temp = train_test_split(lst, train_size=SPLITS["training"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    val, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"training": train, "validation": val, "test": test}


def copy_files(file_list, split, label):
    dest_dir = os.path.join(DATA_DIR, split, label)
    for src in tqdm(file_list, desc=f"{split}/{label}", unit="file"):
        filename = os.path.basename(src)
        dest_path = os.path.join(dest_dir, filename)

        # Falls Datei schon existiert → neuen Namen vergeben
        if os.path.exists(dest_path):
            filename = f"{uuid.uuid4().hex}_{filename}"
            dest_path = os.path.join(dest_dir, filename)

        shutil.copy2(src, dest_path)


def process_pre_split(root_dir: str):
    for split in ["Train", "Validation", "Test"]:
        split_lower = split.lower()
        split_path = os.path.join(root_dir, split)
        if not os.path.isdir(split_path):
            print(f"⚠️ {split_path} nicht gefunden, überspringe.")
            continue
        for label in os.listdir(split_path):
            label_lower = label.lower()
            if label_lower not in ("real", "fake"):
                continue
            src_dir = os.path.join(split_path, label)
            files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
            copy_files(files, split_lower, label_lower)


def process_pre_label(root_dir: str, label_map: dict):
    images = {l: collect_images_from_dirs([os.path.join(root_dir, d) for d in label_map[l]]) for l in label_map}
    for label, image_list in images.items():
        splits = split_list(image_list)
        for split_name, files in splits.items():
            copy_files(files, split_name, label)


def process_dataset(dataset_id: str):
    info = DATASETS_INFO[dataset_id]
    ds_name = dataset_id.replace("/", "-")
    download_dir = os.path.join(DOWNLOAD_ROOT, ds_name)
    download_and_unzip(dataset_id, download_dir)

    subdirs = [d for d in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, d))]
    root_dir = os.path.join(download_dir, subdirs[0]) if len(subdirs) == 1 else download_dir

    if info["mode"] == "pre_split":
        process_pre_split(root_dir)
    else:
        process_pre_label(root_dir, info["labels"])


def clean_downloads():
    """Löscht den gesamten downloads/-Ordner nach Abschluss."""
    if os.path.exists(DOWNLOAD_ROOT):
        print(f"🧹 Lösche temporären Ordner: {DOWNLOAD_ROOT}")
        shutil.rmtree(DOWNLOAD_ROOT)


if __name__ == "__main__":
    ensure_structure()
    for ds in tqdm(DATASETS_INFO.keys(), desc="Verarbeite Datasets", unit="dataset"):
        process_dataset(ds)
    clean_downloads()
    print("✅ Alle Datensätze wurden in Data/ einsortiert – downloads/ ist gelöscht.")
