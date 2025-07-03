# 🤖 AI Image Detector

> ⚠️ **Dieses Repository befindet sich in aktiver Entwicklung. Inhalte und Strukturen können sich noch ändern.**

Dieses Projekt zielt darauf ab, mithilfe von Convolutional Neural Networks (CNNs) automatisch zu erkennen, ob Bilder von einer KI erzeugt wurden oder ob es sich um echte Fotografien handelt. Neben Trainingsskripten stehen Werkzeuge zum Download von Datensätzen und vortrainierten Modellen sowie mehrere Streamlit-Apps für die Demonstration bereit.

## Projektstruktur

```
AI-Image-Detector/
├── Scripts/
│   ├── App/                # Streamlit-Anwendungen
│   │   ├── app.py
│   │   ├── app_new.py
│   │   └── alternative_app.py
│   ├── PrepData/           # Daten- und Modell-Downloads
│   │   ├── fetch_data.py
│   │   └── download_all_models.py
│   ├── Trainer/            # Trainingsskripte
│   │   ├── train_model.py
│   │   ├── train_model_new.py
│   │   ├── MobileNetV2Trainer.py
│   │   └── DeepfakePipelineTrainer.py
│   ├── evaluate_models.py  # Modelle bewerten
│   └── requirements.txt    # benötigte Python-Pakete
├── Projektarbeit/          # wissenschaftliche Ausarbeitung (LaTeX)
└── README.md
```

## Installation

```bash
pip install -r Scripts/requirements.txt
```

## Datensätze herunterladen

Die Datensätze werden über die Kaggle API geladen. Lege dazu deine Kaggle-Credentials in `~/.kaggle/kaggle.json` ab oder exportiere `KAGGLE_USERNAME` und `KAGGLE_KEY`. Anschließend ruft

```bash
python Scripts/PrepData/fetch_data.py
```

die in `fetch_data.py` definierten Datensätze ab. Die Rohdaten landen unter `Scripts/downloads/`, das aufbereitete Trainings-, Validierungs- und Testmaterial unter `Scripts/data/`.

## Vortrainierte Modelle beziehen

Um Modelle von Hugging Face herunterzuladen, müssen folgende Variablen gesetzt sein:

- `HF_TOKEN` – dein Zugriffstoken (Pflicht)
- `HF_USERNAME` – optionaler Benutzername, standardmäßig `realr4an`

Dann genügt

```bash
export HF_TOKEN=hf_xxx
# optional: export HF_USERNAME=dein_name
python Scripts/PrepData/download_all_models.py
```

Die Modelle werden im Ordner `Scripts/PrepData/models/` abgelegt.

## Streamlit-App starten

Wähle eine der Apps im Verzeichnis `Scripts/App/` aus, z.B.

```bash
streamlit run Scripts/App/app.py
```

Weitere Varianten sind `app_new.py` und `alternative_app.py`.

## Training

Ein einfaches Training lässt sich mit folgendem Befehl starten:

```bash
python Scripts/Trainer/train_model.py
```

Weitere Trainer und Optionen befinden sich im Unterordner `Scripts/Trainer/`.
