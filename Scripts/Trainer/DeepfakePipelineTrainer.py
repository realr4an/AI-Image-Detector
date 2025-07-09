#!/usr/bin/env python3

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Pfad-Konfiguration basierend auf der Projektstruktur
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

class DeepfakePipelineTrainer:
    """
    Trainiert ein Deepfake-Erkennungsmodell mit Gesichtserkennung in der Datenpipeline.
    """

    # Der FaceDataGenerator ist jetzt eine "innere Klasse", was die Organisation verbessert.
    class FaceDataGenerator(Sequence):
        def __init__(self, file_list, labels, img_size, batch_size, face_cascade):
            self.file_list = file_list
            self.labels = labels
            self.img_size = img_size
            self.batch_size = batch_size
            self.face_cascade = face_cascade

        def __len__(self):
            return int(np.ceil(len(self.file_list) / self.batch_size))

        def __getitem__(self, idx):
            batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_labels = [self.labels[f] for f in batch_files]
            
            X, y = [], []
            for fpath, label in zip(batch_files, batch_labels):
                try:
                    img = cv2.imread(fpath)
                    if img is None: continue # Überspringt beschädigte Dateien
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) == 0: continue # Überspringt Bilder ohne gefundenes Gesicht
                    
                    x_face, y_face, w, h = faces[0]
                    face = img[y_face:y_face+h, x_face:x_face+w]
                    face = cv2.resize(face, self.img_size)
                    face = preprocess_input(face.astype('float32'))
                    
                    X.append(face)
                    y.append(label)
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {fpath}: {e}")
                    continue

            return np.array(X), np.array(y)

    def __init__(self,
                 train_dir=os.path.join(DATA_DIR, 'train'),
                 val_dir=os.path.join(DATA_DIR, 'validation'),
                 test_dir=os.path.join(DATA_DIR, 'test'),
                 img_size=(224, 224),
                 batch_size=64,
                 initial_epochs=50,
                 output_dir=os.path.join(ROOT_DIR, 'misclassified_faces')):
        
        # Initialisierung der Pfade und Parameter
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = initial_epochs
        self.output_dir = output_dir
        self.base_model_name = "MobileNetV2_Pipeline"

        # Erstellen der notwendigen Verzeichnisse
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Mixed Precision für Performance
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        # Laden des Haar-Cascade-Klassifikators für die Gesichtserkennung
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _gather_files(self, directory):
        """Sammelt Dateipfade und weist Labels basierend auf Unterordnern zu."""
        files, labels = [], {}
        class_indices = {}
        for idx, cls_name in enumerate(sorted(os.listdir(directory))):
            cls_dir = os.path.join(directory, cls_name)
            if not os.path.isdir(cls_dir): continue
            
            class_indices[cls_name] = idx
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(cls_dir, fname)
                    files.append(path)
                    labels[path] = idx
        return files, labels, class_indices

    def build_model(self):
        """Erstellt das CNN-Modell mit MobileNetV2 als Basis."""
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        base.trainable = True # Fine-Tuning des gesamten Modells
        
        model = Sequential([
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
            Dropout(0.6),
            Dense(1, activation='sigmoid', dtype='float32') # float32 für die letzte Schicht
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        """Führt den Trainings- und Evaluationsprozess aus."""
        train_files, train_labels, _ = self._gather_files(self.train_dir)
        val_files, val_labels, _ = self._gather_files(self.val_dir)

        train_gen = self.FaceDataGenerator(train_files, train_labels, self.img_size, self.batch_size, self.face_cascade)
        val_gen = self.FaceDataGenerator(val_files, val_labels, self.img_size, self.batch_size, self.face_cascade)

        model = self.build_model()

        # Dynamische Dateinamen und Callbacks
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f"{self.base_model_name}_{timestamp}.h5"
        model_filepath = os.path.join(MODEL_DIR, model_filename)
        
        log_dir = os.path.join(LOG_DIR, "fit", f"{self.base_model_name}-{timestamp}")

        callbacks = [
            ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_accuracy', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        print("\n🚀 Starte das Training...")
        model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=False
        )
        print(f"\n✅ Training abgeschlossen. Das beste Modell wurde hier gespeichert:\n{model_filepath}")

        # Evaluation mit dem besten Modell
        self.evaluate(model_filepath)

    def evaluate(self, model_path):
        """Evaluiert das trainierte Modell auf dem Test-Set."""
        if not os.path.exists(self.test_dir):
            print("⚠️ Testverzeichnis nicht gefunden. Evaluation wird übersprungen.")
            return

        print("\n🧪 Evaluiere das beste Modell auf dem Test-Set...")
        best_model = tf.keras.models.load_model(model_path)
        
        test_files, test_labels, class_indices = self._gather_files(self.test_dir)
        # Batch Size von 1 für die Evaluation, um jede Datei einzeln zu verarbeiten
        test_gen = self.FaceDataGenerator(test_files, test_labels, self.img_size, 1, self.face_cascade)

        y_true, y_pred = [], []
        
        for i in range(len(test_gen)):
            Xb, yb = test_gen[i]
            if len(Xb) == 0: continue
            
            pred_prob = best_model.predict(Xb, verbose=0)
            pred_class = (pred_prob > 0.5).astype(int)[0][0]
            true_class = yb[0]

            y_true.append(true_class)
            y_pred.append(pred_class)
            
            # Speichere falsch klassifizierte Bilder zur Analyse
            if pred_class != true_class:
                img_array = ((Xb[0] * 127.5) + 127.5).astype('uint8')
                label_true_name = [k for k, v in class_indices.items() if v == true_class][0]
                label_pred_name = [k for k, v in class_indices.items() if v == pred_class][0]
                
                img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                output_path = os.path.join(self.output_dir, f"test_{i}_true-{label_true_name}_pred-{label_pred_name}.png")
                img.save(output_path)

        print("\n📊 Klassifikationsreport:")
        print(classification_report(y_true, y_pred, target_names=class_indices.keys()))
        print("\n📊 Konfusionsmatrix:")
        print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    trainer = DeepfakePipelineTrainer()
    trainer.train()