import os
import numpy as np
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Ordnerpfade
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Scripts/Trainer/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))   # AI-IMAGE-DETECTOR/

DATA_DIR = os.path.join(ROOT_DIR, "Data", "test")          # Data/test/
MODEL_DIR = os.path.join(ROOT_DIR, "Models", "ResNet50_Deepfake_detection")

TEST_DIR_FAKE = os.path.join(DATA_DIR, "fake")
TEST_DIR_REAL = os.path.join(DATA_DIR, "real")

model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

# ─────────────────────────────────────────────────────────────────────────────
# Memory-Efficient Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataGenerator(Sequence):
    """
    Lädt und prozessiert Bilddaten stapelweise, um den RAM zu schonen.
    """
    def __init__(self, image_paths, labels, batch_size, target_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        # Gibt die Gesamtzahl der Batches zurück
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        # Lädt und liefert einen Batch von Bildern
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

for model_file in model_files:
    print(f"\n🔍 Evaluating model: {os.path.basename(model_file)}")
    
    # Modell laden und Input-Größe bestimmen
    model = load_model(model_file, compile=False)
    width, height = model.input_shape[1], model.input_shape[2]
    target_size = (width, height)

    # 1. Dateipfade und Labels sammeln (ohne die Bilder zu laden)
    fake_paths = [os.path.join(TEST_DIR_FAKE, f) for f in os.listdir(TEST_DIR_FAKE) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    real_paths = [os.path.join(TEST_DIR_REAL, f) for f in os.listdir(TEST_DIR_REAL) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    all_paths = fake_paths + real_paths
    all_labels = np.array([0] * len(fake_paths) + [1] * len(real_paths))

    # 2. Generator erstellen
    # Wir übergeben hier nur die Pfade, nicht die Bilder selbst!
    # Die `true_labels` benötigen wir am Ende für den Vergleich.
    test_generator = ImageDataGenerator(
        image_paths=all_paths, 
        labels=all_labels,  # Die Labels werden hier nur zur Vollständigkeit mitgegeben, für predict() aber nicht benötigt
        batch_size=BATCH_SIZE, 
        target_size=target_size
    )

    # 3. Vorhersagen effizient mit dem Generator berechnen
    print(f"🔄 Vorhersagen werden für {len(all_paths)} Bilder berechnet (Batch-Größe: {BATCH_SIZE})...")
    predictions = model.predict(test_generator, verbose=1)
    
    # 4. Metriken berechnen
    predicted_labels = (predictions > 0.5).astype("int32").flatten()

    acc = accuracy_score(all_labels, predicted_labels)
    prec = precision_score(all_labels, predicted_labels)
    rec = recall_score(all_labels, predicted_labels)
    f1 = f1_score(all_labels, predicted_labels)

    print(f"\n📊 Accuracy : {acc:.4f}")
    print(f"📊 Precision: {prec:.4f}")
    print(f"📊 Recall   : {rec:.4f}")
    print(f"📊 F1-Score : {f1:.4f}")