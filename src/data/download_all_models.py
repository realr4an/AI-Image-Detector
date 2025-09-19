#!/usr/bin/env python3
# fetch_all_models.py

import os
import sys
import logging
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

# ─────────────────────────────────────────────────────────────────────────────
# Setup & Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("✖ FEHLER: Die Umgebungsvariable HF_TOKEN ist nicht gesetzt.")
    sys.exit(1)

HF_USERNAME = os.getenv("HF_USERNAME", "realr4an")

BASE_DIR = Path(__file__).resolve().parent              # → src/data
ROOT_DIR = BASE_DIR.parent.parent                       # → AI-IMAGE-DETECTOR
MODELS_DIR = ROOT_DIR / "models" / "pretrained"        # → AI-IMAGE-DETECTOR/models/pretrained


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def ensure_models_dir():
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
        logger.info(f"📁 Erstelle Zielordner: {MODELS_DIR}")


def list_repos(api: HfApi, username: str):
    logger.info(f"🔍 Hole alle Model-Repos von: {username}")
    try:
        models = api.list_models(author=username, full=True)
        return [m.modelId for m in models]
    except Exception as e:
        logger.error(f"✖ Fehler beim Abrufen der Repos: {e}")
        return []


def download_repo_snapshot(repo_id: str, destination: Path, token: str):
    model_name = repo_id.split("/")[-1]
    target_dir = destination / model_name
    if target_dir.exists():
        logger.info(f"✔ Repo '{model_name}' existiert bereits. Überspringe.")
        return

    logger.info(f"⬇ Lade Snapshot von '{repo_id}' …")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=token
        )
        logger.info(f"✅ Fertig: {model_name} gespeichert in {target_dir}")
    except RepositoryNotFoundError:
        logger.error(f"✖ Repo '{repo_id}' nicht gefunden.")
    except Exception as e:
        logger.error(f"✖ Fehler beim Download von '{repo_id}': {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_models_dir()
    api = HfApi(token=HF_TOKEN)
    repos = list_repos(api, HF_USERNAME)

    if not repos:
        logger.warning("⚠️ Keine Repos gefunden.")
        return

    for repo_id in repos:
        download_repo_snapshot(repo_id, MODELS_DIR, HF_TOKEN)

    logger.info("🏁 Alle Modelle wurden erfolgreich heruntergeladen.")


if __name__ == "__main__":
    main()
