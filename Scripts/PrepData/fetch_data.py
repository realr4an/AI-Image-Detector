import gc
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import psutil
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Directory layout relative to the project root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Data"
DOWNLOAD_ROOT = ROOT_DIR / "downloads"

# Dataset configuration
DATASETS_INFO: Dict[str, Dict[str, object]] = {
    "manjilkarki/deepfake-and-real-images": {"mode": "pre_split"},
    "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset": {
        "mode": "pre_label",
        "labels": {"real": ["real"], "fake": ["AI"]},
    },
    "prithivsakthiur/deepfake-vs-real-60k": {
        "mode": "pre_label",
        "labels": {"real": ["Real"], "fake": ["Fake"]},
    },
}

SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def ensure_structure() -> None:
    """Create the output directory structure (train/validation/test per class)."""
    for split in SPLITS:
        for label in ("real", "fake"):
            (DATA_DIR / split / label).mkdir(parents=True, exist_ok=True)


def download_and_unzip(dataset_id: str, download_dir: Path) -> Path:
    """Download a dataset via the Kaggle API and extract the archive."""
    api = KaggleApi()
    api.authenticate()
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"> Downloading {dataset_id} ...")
    api.dataset_download_files(dataset_id, path=str(download_dir), unzip=False, quiet=False)

    zip_name = dataset_id.split("/")[-1]
    zip_path = download_dir / f"{zip_name}.zip"

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive {zip_path} was not created.")

    print(f"> Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in tqdm(archive.infolist(), desc="Extracting", unit="file"):
            archive.extract(member, str(download_dir))
    zip_path.unlink(missing_ok=True)

    return download_dir


def collect_images_from_dirs(dir_list: Iterable[Path]) -> List[Path]:
    """Collect all image paths from directories that actually exist."""
    images: List[Path] = []
    for directory in dir_list:
        if not directory.is_dir():
            continue
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(directory / file_name)
    return images


def split_list(items: List[Path]) -> Dict[str, List[Path]]:
    """Create train/validation/test splits based on the configured ratios."""
    if not items:
        return {"train": [], "validation": [], "test": []}

    train, temp = train_test_split(items, train_size=SPLITS["train"], random_state=RANDOM_SEED)
    val_ratio = SPLITS["validation"] / (SPLITS["validation"] + SPLITS["test"])
    validation, test = train_test_split(temp, train_size=val_ratio, random_state=RANDOM_SEED)
    return {"train": train, "validation": validation, "test": test}


def copy_files_in_chunks(
    file_list: List[Path],
    split: str,
    label: str,
    chunk_size: int = 2000,
    min_free_ram_gb: int = 2,
) -> None:
    """Copy files in manageable chunks while monitoring available RAM."""
    dest_dir = DATA_DIR / split / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = len(file_list)
    for start_index in range(0, total, chunk_size):
        chunk = file_list[start_index : start_index + chunk_size]
        desc = f"{split}/{label} [{start_index + 1}-{min(start_index + chunk_size, total)} of {total}]"

        for src in tqdm(chunk, desc=desc, unit="file"):
            filename = src.name
            dest_path = dest_dir / filename

            if dest_path.exists():
                filename = f"{uuid.uuid4().hex}_{filename}"
                dest_path = dest_dir / filename

            shutil.copy2(src, dest_path)

        print(f"> Copied {len(chunk)} files to {dest_dir}")
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"> Available RAM: {available_gb:.2f} GB")
        gc.collect()

        if available_gb < min_free_ram_gb:
            print("! Low RAM detected. Consider pausing the copy process.")


def process_pre_split(root_dir: Path) -> None:
    """Handle datasets that already contain train/validation/test subfolders."""
    split_map = {"Train": "train", "Validation": "validation", "Test": "test"}

    for original_split, target_split in split_map.items():
        split_path = root_dir / original_split
        if not split_path.is_dir():
            print(f"! Skipping missing directory: {split_path}")
            continue

        for label in os.listdir(split_path):
            label_lower = label.lower()
            if label_lower not in ("real", "fake"):
                continue

            src_dir = split_path / label
            files = [
                src_dir / fname
                for fname in os.listdir(src_dir)
                if fname.lower().endswith(IMAGE_EXTENSIONS)
            ]
            copy_files_in_chunks(files, target_split, label_lower)


def process_pre_label(root_dir: Path, label_map: Dict[str, Iterable[str]]) -> None:
    """Handle datasets that only provide label folders without data splits."""
    images = {
        label: collect_images_from_dirs(root_dir / subdir for subdir in subdirs)
        for label, subdirs in label_map.items()
    }

    for label, image_list in images.items():
        splits = split_list(image_list)
        for split_name, files in splits.items():
            copy_files_in_chunks(files, split_name, label)


def process_dataset(dataset_id: str) -> None:
    """Download, unpack, and sort a dataset into the project structure."""
    info = DATASETS_INFO[dataset_id]
    download_dir = DOWNLOAD_ROOT / dataset_id.replace("/", "-")
    extracted_root = download_and_unzip(dataset_id, download_dir)

    subdirs = [d for d in extracted_root.iterdir() if d.is_dir()]
    root_dir = subdirs[0] if len(subdirs) == 1 else extracted_root

    if info["mode"] == "pre_split":
        process_pre_split(root_dir)
    else:
        process_pre_label(root_dir, info["labels"])  # type: ignore[arg-type]


def clean_downloads() -> None:
    """Remove the temporary download directory to reclaim disk space."""
    if DOWNLOAD_ROOT.exists():
        print(f"> Removing temporary directory: {DOWNLOAD_ROOT}")
        shutil.rmtree(DOWNLOAD_ROOT)


def _print_dataset_menu(dataset_keys: List[str]) -> None:
    print("\nAvailable datasets:")
    for index, key in enumerate(dataset_keys, start=1):
        print(f"[{index}] {key}")
    print("[0] Process all datasets")


if __name__ == "__main__":
    ensure_structure()

    dataset_keys = list(DATASETS_INFO.keys())
    _print_dataset_menu(dataset_keys)

    try:
        choice = int(input("\nSelect dataset (number): ").strip())
    except ValueError:
        print("Invalid input. Aborting.")
        sys.exit(1)

    if choice == 0:
        for ds in tqdm(DATASETS_INFO.keys(), desc="Datasets", unit="dataset"):
            process_dataset(ds)
    elif 1 <= choice <= len(dataset_keys):
        dataset_id = dataset_keys[choice - 1]
        print(f"\nProcessing {dataset_id} ...\n")
        process_dataset(dataset_id)
    else:
        print("Invalid selection. Aborting.")
        sys.exit(1)

    clean_downloads()
    print("Done. Datasets have been copied into the Data/ folder.")
