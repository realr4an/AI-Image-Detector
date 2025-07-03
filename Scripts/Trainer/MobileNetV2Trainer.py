#!/usr/bin/env python3
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

class MobileNetV2Trainer:
    def __init__(self,
                 train_dir='Dataset/train',
                 val_dir='Dataset/validation',
                 test_dir='Dataset/test',
                 img_size=(256,256),
                 batch_size=64,
                 initial_epochs=50,
                 output_dir='misclassified'):
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

    def build_model(self):
        base = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=(*self.img_size, 3))
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
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_gen = datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )
        val_gen = datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        model = self.build_model()

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks = [
            ModelCheckpoint('best_mobilenetv2.h5', save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // self.batch_size,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=False
        )

        if os.path.exists(self.test_dir):
            test_gen = datagen.flow_from_directory(
                self.test_dir,
                target_size=self.img_size,
                batch_size=1,
                class_mode='binary',
                shuffle=False
            )
            preds = model.predict(test_gen, workers=4)
            y_true = test_gen.classes
            y_pred = (preds > 0.5).astype(int).flatten()

            loss, acc = model.evaluate(test_gen, workers=4)
            print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))

            for idx, fname in enumerate(test_gen.filepaths):
                if y_true[idx] != y_pred[idx]:
                    img = Image.open(fname).resize(self.img_size)
                    label_true = test_gen.class_indices
                    label_name = list(test_gen.class_indices.keys())[y_true[idx]]
                    label_pred = list(test_gen.class_indices.keys())[y_pred[idx]]
                    out_name = f"{idx}_{label_name}_as_{label_pred}.png"
                    img.save(os.path.join(self.output_dir, out_name))

        model.save("mobilenetv2_final.h5")
        print("Modell gespeichert als 'mobilenetv2_final.h5'")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    trainer = MobileNetV2Trainer()
    trainer.train()
