import os
import shutil
import zipfile
import uuid
import sys
import gc
import psutil
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Pfade (Root: AI-IMAGE-DETECTOR/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # → PrepData
SCRIPTS_DIR = os.path.dirname(BASE_DIR)                      # → Scripts
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)                      # → AI-IMAGE-DETECTOR

DATA_DIR = os.path.join(ROOT_DIR, "Data")
DOWNLOAD_ROOT = os.path.join(ROOT_DIR, "downloads")

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

SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def ensure_structure():
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
    train, temp = train_test_split(lst, train_size=SPLITS["train"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    val, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"train": train, "validation": val, "test": test}


def copy_files_in_chunks(file_list, split, label, chunk_size=2000, min_free_ram_gb=2):
    dest_dir = os.path.join(DATA_DIR, split, label)
    os.makedirs(dest_dir, exist_ok=True)
    total = len(file_list)

    for i in range(0, total, chunk_size):
        chunk = file_list[i:i + chunk_size]
        tqdm_desc = f"{split}/{label} [{i + 1}-{min(i + chunk_size, total)} von {total}]"
        for src in tqdm(chunk, desc=tqdm_desc, unit="file"):
            filename = os.path.basename(src)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(dest_path):
                filename = f"{uuid.uuid4().hex}_{filename}"
                dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src, dest_path)

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"🧠 Freier RAM: {available_gb:.2f} GB")
        gc.collect()
        if available_gb < min_free_ram_gb:
            print("⚠️ RAM niedrig. Warte kurz oder reduziere Chunk-Größe.")


def process_pre_split(root_dir: str):
    split_map = {"Train": "train", "Validation": "validation", "Test": "test"}
    for split in ["Train", "Validation", "Test"]:
        split_lower = split_map[split]
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
            copy_files_in_chunks(files, split_lower, label_lower)


def process_pre_label(root_dir: str, label_map: dict):
    images = {l: collect_images_from_dirs([os.path.join(root_dir, d) for d in label_map[l]]) for l in label_map}
    for label, image_list in images.items():
        splits = split_list(image_list)
        for split_name, files in splits.items():
            copy_files_in_chunks(files, split_name, label)


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
    if os.path.exists(DOWNLOAD_ROOT):
        print(f"🧹 Lösche temporären Ordner: {DOWNLOAD_ROOT}")
        shutil.rmtree(DOWNLOAD_ROOT)


if __name__ == "__main__":
    ensure_structure()

    dataset_keys = list(DATASETS_INFO.keys())

    print("\n📦 Verfügbare Datensätze:")
    for i, key in enumerate(dataset_keys, 1):
        print(f"[{i}] {key}")
    print("[0] Alle Datensätze verarbeiten")

    try:
        choice = int(input("\n🔍 Auswahl (Zahl eingeben): ").strip())
    except ValueError:
        print("❌ Ungültige Eingabe. Abbruch.")
        sys.exit(1)

    if choice == 0:
        for ds in tqdm(DATASETS_INFO.keys(), desc="Verarbeite Datasets", unit="dataset"):
            process_dataset(ds)
    elif 1 <= choice <= len(dataset_keys):
        selected_ds = dataset_keys[choice - 1]
        print(f"\n🎯 Verarbeite: {selected_ds}\n")
        process_dataset(selected_ds)
    else:
        print("❌ Ungültige Auswahl. Abbruch.")
        sys.exit(1)

    clean_downloads()
    print("✅ Fertig. Datensatz(e) wurden in Data/ einsortiert.")
