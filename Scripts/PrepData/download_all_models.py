#!/usr/bin/env python3
# download_only_models.py

import os
import sys
import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

# 1) Lies HF_TOKEN aus der Umgebung (muss vorher gesetzt sein)
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is None:
    print("✖ FEHLER: Die Umgebungsvariable HF_TOKEN ist nicht gesetzt.")
    print("  Lege sie z.B. so an: export HF_TOKEN=\"hf_...\"")
    sys.exit(1)

# 2) Optional: Lies HF_USERNAME aus der Umgebung oder harteingestellt
HF_USERNAME = os.getenv("HF_USERNAME", "realr4an")
# Wenn du einen anderen User/Namespace herunterladen willst, ändere hier entsprechend.

# 3) Pfad, in dem alle Modelle gespeichert werden sollen
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

# 4) Welche Dateiendungen sollen tatsächlich heruntergeladen werden?
#    Du kannst hier beliebig anpassen oder erweitern:
MODEL_EXTENSIONS = {".h5"}  # Beispiel: nur .h5-Dateien ziehen


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfs‐Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def ensure_models_dir():
    """Erstelle ./models, falls noch nicht existiert."""
    if not MODELS_DIR.exists():
        logger.info(f"Erstelle den Ordner: {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)


def list_all_model_repos(api: HfApi, username: str) -> list[str]:
    """
    Liefert alle Model-Repos, die unter einem bestimmten Nutzer/Namespace existieren.
    Gibt zurück: Liste von Repo-IDs im Format "<username>/<reponame>".
    """
    logger.info(f"Hole Liste aller Modelle von HuggingFace-User '{username}' …")
    try:
        all_models = api.list_models(author=username, full=True)
    except Exception as e:
        logger.error(f"✖ Fehler beim Abrufen der Model-Liste: {e}")
        return []
    repo_ids: list[str] = []
    for m in all_models:
        # m.modelId ist z.B. "realr4an/ResNet50_Deepfake_detection"
        repo_ids.append(m.modelId)
    logger.info(f"Insgesamt {len(repo_ids)} Modell-Repos gefunden.")
    return repo_ids


def download_only_model_files(repo_id: str, dst_base: Path, token: str) -> None:
    """
    Lädt aus dem Repo 'repo_id' nur Dateien mit Endungen aus MODEL_EXTENSIONS
    ins lokale Verzeichnis ./models/<modellname>/ herunter.
    """
    model_name = repo_id.split("/")[-1]
    target_dir = dst_base / model_name

    # Erstelle das Zielverzeichnis, falls es nicht existiert
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Erstelle Zielordner: {target_dir}")
    else:
        logger.info(f"Zielordner existiert bereits: {target_dir}")

    logger.info(f"  → Prüfe Repo '{repo_id}' auf Dateien mit Endung {MODEL_EXTENSIONS} …")
    api = HfApi(token=token)

    try:
        # 1) Liste alle Dateien (Pfadstrings) im Repo
        all_files = api.list_repo_files(repo_id=repo_id)
    except RepositoryNotFoundError:
        logger.error(f"  ✖ Repo '{repo_id}' nicht gefunden.")
        return
    except Exception as e:
        logger.error(f"  ✖ Fehler beim Auflisten der Dateien von '{repo_id}': {e}")
        return

    # 2) Filter: nur Dateien, die mit einer der erlaubten Endungen enden
    model_files = [
        f for f in all_files
        if Path(f).suffix.lower() in MODEL_EXTENSIONS
    ]

    if not model_files:
        logger.info(f"  → Keine Modelldateien mit Endung {MODEL_EXTENSIONS} in '{repo_id}' gefunden.")
        return

    logger.info(f"  → Gefundene Modelldateien: {model_files}")

    # 3) Lade jede gefilterte Datei einzeln herunter
    for file_in_repo in model_files:
        local_path = target_dir / Path(file_in_repo).name
        # Falls die Datei bereits existiert, überspringen
        if local_path.exists():
            logger.info(f"    • '{local_path.name}' existiert bereits. Überspringe.")
            continue

        logger.info(f"    • Lade herunter: '{file_in_repo}' …")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_in_repo,
                local_dir=str(target_dir),
                token=token,
                local_dir_use_symlinks=False  # zwingt auf echte Kopie statt Symlink
            )
            logger.info(f"      ✓ Fertig: '{local_path.name}'")
        except RevisionNotFoundError:
            logger.error(f"      ✖ Revision nicht gefunden für '{repo_id}'")
        except Exception as e:
            logger.error(f"      ✖ Fehler beim Herunterladen von '{file_in_repo}': {e}")
        
    # 4) Entferne nachträglich das Unterverzeichnis ".cache" im Zielordner
    cache_folder = target_dir / ".cache"
    if cache_folder.exists() and cache_folder.is_dir():
        try:
            shutil.rmtree(cache_folder)
            logger.info(f"    • Entferne überschüssigen Ordner: {cache_folder.name}")
        except Exception as e:
            logger.warning(f"    • Konnte {cache_folder} nicht löschen: {e}")

    logger.info(f"  → Alle gewünschten Dateien für '{model_name}' wurden bearbeitet.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Hauptteil
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Ordner models/ erstellen (falls nicht bereits vorhanden)
    ensure_models_dir()

    # 2) HfApi‐Instanz anlegen (wir nutzen sie u.a. in list_all_model_repos)
    api = HfApi(token=HF_TOKEN)

    # 3) Liste aller Modell-Repos des Users holen
    repo_ids = list_all_model_repos(api, HF_USERNAME)
    if not repo_ids:
        logger.warning("Keine Model-Repos gefunden. Prüfe, ob HF_USERNAME korrekt ist.")
        sys.exit(0)

    # 4) Jeden Repo einzeln durchgehen und nur .h5-Dateien (bzw. MODEL_EXTENSIONS) ziehen
    for repo_id in repo_ids:
        download_only_model_files(repo_id, MODELS_DIR, HF_TOKEN)

    logger.info("Alle Downloads abgeschlossen.")

if __name__ == "__main__":
    main()
