#!/usr/bin/env python3
"""Download all published Hugging Face models for the configured account."""

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: The HF_TOKEN environment variable is required.")
    sys.exit(1)

HF_USERNAME = os.getenv("HF_USERNAME", "realr4an")
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "Models"
REPO_ALIASES = {
    "yolov8_face_detector": "YOLOv8_Face_Detection",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("hf-downloader")


def ensure_models_dir() -> None:
    """Create the local models directory if it does not yet exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Target directory: %s", MODELS_DIR)


def list_model_repos(api: HfApi, username: str) -> Iterable[str]:
    """Return all repository ids owned by the given user."""
    LOGGER.info("Fetching model list for account: %s", username)
    try:
        models = api.list_models(author=username, full=True)
        return [model.modelId for model in models]
    except Exception as exc:
        LOGGER.error("Failed to fetch repository list: %s", exc)
        return []


def download_repo_snapshot(repo_id: str, token: str) -> None:
    """Download a full snapshot of the given repository into Models/<repo>."""
    repo_name = repo_id.split("/")[-1]
    model_name = REPO_ALIASES.get(repo_name, repo_name)
    target_dir = MODELS_DIR / model_name

    if target_dir.exists():
        LOGGER.info("Skipping %s. Directory already exists.", model_name)
        return

    LOGGER.info("Downloading snapshot for %s ...", repo_id)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        LOGGER.info("Stored %s in %s", model_name, target_dir)
    except RepositoryNotFoundError:
        LOGGER.error("Repository not found: %s", repo_id)
    except Exception as exc:
        LOGGER.error("Download failed for %s: %s", repo_id, exc)


def main() -> None:
    ensure_models_dir()
    api = HfApi(token=HF_TOKEN)

    for repo in list_model_repos(api, HF_USERNAME):
        download_repo_snapshot(repo, HF_TOKEN)

    LOGGER.info("Finished downloading Hugging Face models.")


if __name__ == "__main__":
    main()
