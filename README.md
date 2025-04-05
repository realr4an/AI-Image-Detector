> ⚠️ **ACHTUNG: Dieses Repository befindet sich aktuell in aktiver Entwicklung. Inhalte, Funktionen und Strukturen können sich kurzfristig ändern!**  
> Bitte beachte, dass noch nicht alles vollständig implementiert oder final getestet ist. Feedback ist dennoch willkommen!

# 🤖 AI Image Detector

**CNN-basiertes Tool zur automatischen Erkennung von KI-generierten Bildern (Deepfakes & mehr).**

---

## 📌 Projektbeschreibung
Dieses Projekt nutzt ein Convolutional Neural Network (CNN), um automatisch zu erkennen, ob ein Bild von einer Künstlichen Intelligenz (wie DALL·E, MidJourney oder Stable Diffusion) generiert wurde oder ob es sich um eine echte Fotografie handelt.

Die Anwendung bietet zusätzlich Erklärungen (Grad-CAM), um nachvollziehen zu können, welche Bildbereiche für die Entscheidung des CNN entscheidend sind.

---

## 🚀 Features

- ✅ **Benutzerfreundliche Oberfläche:** Einfacher Upload und Klassifikation von Bildern.
- ✅ **CNN-Modell:** Hohe Genauigkeit bei der Unterscheidung von echten und KI-generierten Bildern.
- ✅ **Erklärbarkeit:** Grad-CAM Visualisierung zur Erklärung der Modellentscheidung.

---

## 📦 Installation

Um das Projekt lokal zu starten, folge diesen Schritten:

1. **Repository klonen:**
```bash
git clone https://github.com/realr4an/ai-image-detector.git
cd ai-image-detector
```

2. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

3. **Streamlit App starten:**
```bash
streamlit run app.py
```

---

## 🛠 Verwendete Technologien

- **Python** (TensorFlow/Keras, OpenCV, NumPy)
- **Streamlit** (für die Benutzeroberfläche)
- **CNN-Modell** (Convolutional Neural Network)
- **Grad-CAM** (zur Visualisierung der Entscheidungsgrundlage)

---

## 📁 Projektstruktur

```
ai-image-detector/
├── app.py                 # Streamlit App
├── model.h5               # Vortrainiertes CNN-Modell (nicht enthalten, selbst trainieren!)
├── requirements.txt       # Abhängigkeiten
├── README.md              # Diese Datei
└── images/                # Beispielbilder (optional)
```

---

## 🧑‍💻 Autor

- **realr4an** - [realr4an](https://github.com/realr4an)
