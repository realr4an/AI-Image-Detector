# 🤖 AI Image Detector

Dieses Projekt untersucht, ob sich KI-generierte und echte Bilder mithilfe von Convolutional Neural Networks (CNNs) automatisch unterscheiden lassen. Im Repository finden sich Streamlit-Demos, Skripte für den Datenimport, Trainingspipelines für ResNet50/MobileNetV2 sowie Hilfswerkzeuge zur Modellbewertung.

## Projektstruktur

```
AI-Image-Detector/
├── data/
│   ├── downloads/            # temporäre Kaggle-Downloads (.gitkeep)
│   ├── processed/            # train/validation/test-Struktur für Trainingsläufe
│   └── raw/                  # unbearbeitete Datensätze (z.B. WIDER FACE)
├── docs/
│   ├── experiments/
│   │   ├── figures/          # Trainingskurven, Confusion-Matrizen usw.
│   │   └── notes/            # Versuchsdokumentationen und Auswertungen
│   └── thesis/               # LaTeX-Projektarbeit
├── logs/
│   └── training_runs/        # TensorBoard-Logs vergangener Läufe
├── models/
│   ├── checkpoints/          # erzeugte Modell-Checkpoints
│   └── pretrained/           # heruntergeladene Gewichte (z.B. YOLO)
├── src/
│   ├── app/                  # Streamlit-Anwendungen
│   ├── data/                 # Skripte zum Daten-/Modell-Download
│   ├── training/             # Trainingspipelines (ResNet50, MobileNetV2, YOLO)
│   └── evaluation.py         # Bewertung gespeicherter Modelle
├── requirements.txt          # Python-Abhängigkeiten
└── README.md
```

> ℹ️ Die leeren Ordner enthalten `.gitkeep`-Dateien, damit die Struktur im Git-Repository sichtbar bleibt.

## Installation

```bash
pip install -r requirements.txt
```

Es wird empfohlen, ein eigenes virtuelles Environment (z.B. `python -m venv .venv`) zu verwenden.

## Datensätze herunterladen (Kaggle)

1. Speichere deine Kaggle-Credentials in `~/.kaggle/kaggle.json` **oder** exportiere `KAGGLE_USERNAME` und `KAGGLE_KEY`.
2. Starte anschließend das Download- und Aufbereitungsskript:

```bash
python src/data/fetch_data.py
```

Die Rohdaten landen unter `data/downloads/`, aufbereitete Train-/Validation-/Test-Splits unter `data/processed/`.

### Manuell sortierte Kaggle-Daten

Falls ein Datensatz manuell einsortiert werden muss, unterstützt `src/data/sort_kaggle_data.py` beim Verteilen in die train/val/test-Struktur.

## Vortrainierte Modelle (Hugging Face)

Setze vor dem Download die folgenden Variablen:

```bash
export HF_TOKEN=hf_xxx
# optional: export HF_USERNAME=dein_name
python src/data/download_all_models.py
```

Die Repositories werden in `models/pretrained/` gespiegelt.

## Streamlit-App starten

```bash
streamlit run src/app/app.py
```

Alternative Varianten befinden sich im selben Verzeichnis (`app_new.py`, `alternative_app.py`).

## Training

Typische Trainingsläufe lassen sich über die Skripte im Ordner `src/training/` starten, z.B.:

```bash
python src/training/ModelTrainerResNet50.py
python src/training/ModelTrainerMobileNetV2.py
python src/training/DeepfakePipelineTrainer.py
python src/training/FaceExtractionTrainer.py
```

Checkpoints werden unter `models/checkpoints/` abgelegt, TensorBoard-Logs unter `logs/training_runs/`.

## Evaluation

Gespeicherte Modelle können mit dem Evaluationsskript getestet werden (erwartet Daten in `data/processed/test/` und Checkpoints in `models/checkpoints/`):

```bash
python src/evaluation.py
```

## Dokumentation & Ergebnisse

- **`docs/experiments/`** enthält Versuchsdokumentationen, Kennzahlen und Visualisierungen.
- **`docs/thesis/`** bündelt die zugehörige wissenschaftliche Ausarbeitung im LaTeX-Format.

Viel Erfolg beim Experimentieren! 🎯
