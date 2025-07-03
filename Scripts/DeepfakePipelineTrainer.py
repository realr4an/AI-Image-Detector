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

class DeepfakePipelineTrainer:
    def __init__(self,
                 train_dir='Dataset/train',
                 val_dir='Dataset/validation',
                 test_dir='Dataset/test',
                 img_size=(256,256),
                 batch_size=64,
                 initial_epochs=50,
                 output_dir='misclassified_faces'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = initial_epochs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

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
            X, y = [], []
            for fpath in batch_files:
                img = cv2.imread(fpath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    continue
                x, y0, w, h = faces[0]
                face = img[y0:y0+h, x:x+w]
                face = cv2.resize(face, self.img_size)
                face = preprocess_input(face.astype('float32'))
                X.append(face)
                y.append(self.labels[fpath])
            return np.array(X), np.array(y)

    def _gather_files(self, directory):
        files, labels = [], {}
        class_indices = {}
        for idx, cls in enumerate(sorted(os.listdir(directory))):
            cls_dir = os.path.join(directory, cls)
            if not os.path.isdir(cls_dir): continue
            class_indices[cls] = idx
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    path = os.path.join(cls_dir, fname)
                    files.append(path)
                    labels[path] = idx
        return files, labels, class_indices

    def build_model(self):
        base = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=(*self.img_size,3))
        base.trainable = True
        model = Sequential([
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
            Dropout(0.6),
            Dense(1, activation='sigmoid', dtype='float32')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        train_files, train_labels, _ = self._gather_files(self.train_dir)
        val_files, val_labels, _   = self._gather_files(self.val_dir)

        train_gen = self.FaceDataGenerator(train_files, train_labels,
                                           self.img_size, self.batch_size,
                                           self.face_cascade)
        val_gen = self.FaceDataGenerator(val_files, val_labels,
                                         self.img_size, self.batch_size,
                                         self.face_cascade)

        model = self.build_model()

        log_dir = os.path.join("logs", "pipeline", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            ModelCheckpoint('best_pipeline.h5', save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        model.fit(train_gen,
                  epochs=self.epochs,
                  validation_data=val_gen,
                  callbacks=callbacks,
                  workers=4,
                  use_multiprocessing=False)

        test_files, test_labels, class_indices = self._gather_files(self.test_dir)
        test_gen = self.FaceDataGenerator(test_files, test_labels,
                                          self.img_size, 1,
                                          self.face_cascade)

        y_true, y_pred = [], []
        for i in range(len(test_gen)):
            Xb, yb = test_gen[i]
            if len(Xb)==0: continue
            pred = (model.predict(Xb) > 0.5).astype(int)[0][0]
            y_true.append(yb[0])
            y_pred.append(pred)
            if pred != yb[0]:
                img = ( (Xb[0] + 1)*127.5 ).astype('uint8')
                label_true = [k for k,v in class_indices.items() if v==yb[0]][0]
                label_pred = [k for k,v in class_indices.items() if v==pred][0]
                out = Image.fromarray(img)
                out.save(os.path.join(self.output_dir, f"{i}_{label_true}_as_{label_pred}.png"))

        print("Test Classification Report:")
        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        model.save("pipeline_final.h5")
        print("Modell gespeichert als 'pipeline_final.h5'")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    trainer = DeepfakePipelineTrainer()
    trainer.train()
