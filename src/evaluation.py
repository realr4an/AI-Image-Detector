import os
import math
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# Ordnerpfade
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # → src/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))   # → AI-Image-Detector/

DATA_DIR = os.path.join(ROOT_DIR, "data", "processed", "test")
MODELS_ROOT = os.path.join(ROOT_DIR, "models", "checkpoints")
MODEL_DIR = os.path.join(MODELS_ROOT, "resnet50_deepfake_detection")

# ─────────────────────────────────────────────────────────────────────────────
# Memory-Efficient Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataGenerator(Sequence):
    """Lädt und prozessiert Bilddaten stapelweise, um den RAM zu schonen."""

    def __init__(self, image_paths, labels, batch_size, target_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        for img_path in batch_paths:
            img = load_img(img_path, target_size=self.target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            batch_images.append(img_array)

        return np.array(batch_images), np.array(batch_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Bewertung pro Modell
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 32  # Du kannst die Batch-Größe je nach verfügbarem RAM anpassen


def evaluate_models():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Testdaten nicht gefunden unter: {DATA_DIR}")

    test_dir_fake = os.path.join(DATA_DIR, "fake")
    test_dir_real = os.path.join(DATA_DIR, "real")

    for directory in (test_dir_fake, test_dir_real):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Erwarte Testdaten unter: {directory}")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Kein Modellverzeichnis unter: {MODEL_DIR}")

    model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError(f"Keine .h5-Dateien im Ordner {MODEL_DIR} gefunden.")

    for model_file in model_files:
        print(f"\n🔍 Evaluating model: {os.path.basename(model_file)}")

        model = load_model(model_file, compile=False)
        width, height = model.input_shape[1], model.input_shape[2]
        target_size = (width, height)

        fake_paths = [os.path.join(test_dir_fake, f) for f in os.listdir(test_dir_fake) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        real_paths = [os.path.join(test_dir_real, f) for f in os.listdir(test_dir_real) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        all_paths = fake_paths + real_paths
        all_labels = np.array([0] * len(fake_paths) + [1] * len(real_paths))

        test_generator = ImageDataGenerator(
            image_paths=all_paths,
            labels=all_labels,
            batch_size=BATCH_SIZE,
            target_size=target_size
        )

        print(f"🔄 Vorhersagen werden für {len(all_paths)} Bilder berechnet (Batch-Größe: {BATCH_SIZE})...")
        predictions = model.predict(test_generator, verbose=1)

        predicted_labels = (predictions > 0.5).astype("int32").flatten()

        acc = accuracy_score(all_labels, predicted_labels)
        prec = precision_score(all_labels, predicted_labels)
        rec = recall_score(all_labels, predicted_labels)
        f1 = f1_score(all_labels, predicted_labels)

        print(f"\n📊 Accuracy : {acc:.4f}")
        print(f"📊 Precision: {prec:.4f}")
        print(f"📊 Recall   : {rec:.4f}")
        print(f"📊 F1-Score : {f1:.4f}")


if __name__ == "__main__":
    evaluate_models()
