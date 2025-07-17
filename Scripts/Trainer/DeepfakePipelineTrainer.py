#!/usr/bin/env python3

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import cv2
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Pfad-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

class DeepfakePipelineTrainer:
    """
    Trainiert ein Deepfake-Erkennungsmodell mit ResNet50 als Basis.
    Nutzt ein YOLOv8-Modell zur Gesichtsextraktion, ansonsten identisch zur klassischen Pipeline.
    """

    class FaceDataGenerator(Sequence):
        def __init__(self, file_list, labels, img_size, batch_size, face_detector):
            self.file_list = file_list
            self.labels = labels
            self.img_size = img_size
            self.batch_size = batch_size
            self.face_detector = face_detector

        def __len__(self):
            return int(np.ceil(len(self.file_list) / self.batch_size))

        def __getitem__(self, idx):
            batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_labels = [self.labels[f] for f in batch_files]
            
            X, y = [], []
            for fpath, label in zip(batch_files, batch_labels):
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    results = self.face_detector(img, verbose=False)

                    if len(results) > 0 and len(results[0].boxes) > 0:
                        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        face = img[y1:y2, x1:x2]

                        if face.size == 0:
                            continue
                        
                        face_resized = cv2.resize(face, self.img_size)
                        face_preprocessed = preprocess_input(face_resized.astype('float32'))

                        X.append(face_preprocessed)
                        y.append(label)

                except Exception as e:
                    print(f"Fehler bei {fpath}: {e}")
                    continue

            return np.array(X), np.array(y)

    def __init__(self,
                 train_dir=os.path.join(DATA_DIR, 'train'),
                 val_dir=os.path.join(DATA_DIR, 'validation'),
                 test_dir=os.path.join(DATA_DIR, 'test'),
                 img_size=(256, 256),
                 batch_size=64,
                 initial_epochs=25):
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = initial_epochs
        self.base_model_name = "ResNet50_YOLO_Pipeline"

        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        yolo_model_path = os.path.join(ROOT_DIR, 'Models', 'YOLOv8_Face_Detection', 'weights', 'best.pt')
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO-Modell nicht gefunden unter: {yolo_model_path}")
        self.face_detector = YOLO(yolo_model_path)
        print("✅ YOLOv8-Modell erfolgreich geladen.")

    def _gather_files(self, directory):
        files, labels = [], {}
        class_indices = {}
        for idx, cls_name in enumerate(sorted(os.listdir(directory))):
            cls_dir = os.path.join(directory, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            class_indices[cls_name] = idx
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(cls_dir, fname)
                    files.append(fpath)
                    labels[fpath] = idx
        return files, labels, class_indices

    def build_model(self):
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        base.trainable = True

        model = Sequential([
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
            Dropout(0.6),
            Dense(1, activation='sigmoid', dtype='float32')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        """Führt den Trainingsprozess aus und speichert das beste Modell."""
        
        # ── GPU-Verfügbarkeits-Check ──────────────────────────────────────────────
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Gefundene GPU(s): {gpus}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        else:
            print("⚠️ Keine GPU gefunden. Das Training läuft auf der CPU.")

        # ── Daten vorbereiten ─────────────────────────────────────────────────────
        train_files, train_labels, _ = self._gather_files(self.train_dir)
        val_files, val_labels, _ = self._gather_files(self.val_dir)

        train_gen = self.FaceDataGenerator(train_files, train_labels, self.img_size, self.batch_size, self.face_detector)
        val_gen = self.FaceDataGenerator(val_files, val_labels, self.img_size, self.batch_size, self.face_detector)

        # ── Modell aufbauen ───────────────────────────────────────────────────────
        model = self.build_model()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = f"{self.base_model_name}_{timestamp}.h5"
        model_filepath = os.path.join(MODEL_DIR, model_filename)

        log_dir = os.path.join(LOG_DIR, "fit", f"{self.base_model_name}-{timestamp}")

        callbacks = [
            ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_accuracy', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        # ── Training starten ──────────────────────────────────────────────────────
        print(f"\n🚀 Starte Training mit {self.base_model_name}...")
        model.fit(
            train_gen,
            steps_per_epoch=len(train_files) // self.batch_size,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=len(val_files) // self.batch_size,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=False
        )

        print(f"\n✅ Training abgeschlossen. Modell gespeichert unter:\n{model_filepath}")

        # ── Evaluation auf Testdaten ──────────────────────────────────────────────
        if os.path.exists(self.test_dir):
            test_files, test_labels, _ = self._gather_files(self.test_dir)
            test_gen = self.FaceDataGenerator(test_files, test_labels, self.img_size, self.batch_size, self.face_detector)

            print("\n🧪 Evaluiere das Modell auf dem Test-Set...")
            best_model = tf.keras.models.load_model(model_filepath)

            test_loss, test_acc = best_model.evaluate(test_gen)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

            y_pred = best_model.predict(test_gen)
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true = [test_labels[f] for f in test_gen.file_list]

            print("\n📊 Klassifikationsreport:")
            print(classification_report(y_true, y_pred_classes))

            print("📊 Konfusionsmatrix:")
            print(confusion_matrix(y_true, y_pred_classes))



if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    tf.get_logger().setLevel('ERROR')
    trainer = DeepfakePipelineTrainer()
    trainer.train()
