#!/usr/bin/env python3
"""
train_model.py

Dieses Skript trainiert ein CNN zur binären Klassifikation (z.B. Echt vs. KI-generiert)
unter Nutzung von Transfer Learning (ResNet50) und verschiedenen Trainings-Callbacks.
Stelle sicher, dass die Verzeichnisstruktur wie folgt aufgebaut ist:

Dataset/
├── train/
│   ├── Klasse1/
│   └── Klasse2/
├── validation/
│   ├── Klasse1/
│   └── Klasse2/
└── test/  (optional, nur zur abschließenden Evaluation)

Die Bilder werden mit Data Augmentation vorverarbeitet.
"""

#import os
# Setze den XLA-Pfad, damit libdevice gefunden wird (angepasst an deine CUDA-Installation)
#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/nvvm/libdevice"

import tensorflow as tf
import os

def train_model():
    # Optional: XLA-Optimierung aktivieren (experimentell, kann die Leistung verbessern)
    #tf.config.optimizer.set_jit(True)
    
    # Überprüfen, ob eine GPU verfügbar ist
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Gefundene GPU(s):", gpus)
    else:
        print("Keine GPU gefunden. Das Training läuft auf der CPU.")

    # Mixed Precision aktivieren (sinnvoll bei GPU-Nutzung)
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision aktiviert mit Policy:", policy.name)

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    # Hyperparameter
    img_width, img_height = 224, 224
    batch_size = 64  # Angepasst für 8GB VRAM
    epochs = 25

    # Verzeichnispfade
    train_data_dir = 'Dataset/train'
    validation_data_dir = 'Dataset/validation'
    test_data_dir = 'Dataset/test'  # Optional: Nur für finale Evaluation

    # Data Augmentation und Preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Trainingsdaten laden
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Validierungsdaten laden
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Aufbau des Modells mit Transfer Learning (ResNet50 als Basis)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    for layer in base_model.layers:
        layer.trainable = False

    # Das finale Dense-Layer wird explizit als float32 definiert, um numerische Probleme bei Mixed Precision zu vermeiden
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks definieren
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    # Training des Modells (Multiprocessing hier deaktiviert, um Pickling-Probleme unter Windows zu vermeiden)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        workers=4,
        use_multiprocessing=False
    )

    # Optional: Evaluation auf dem Testdatensatz, falls vorhanden
    if os.path.exists(test_data_dir):
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Speichern des finalen Modells
    model.save("model.h5")
    print("Das Modell wurde erfolgreich als 'model.h5' gespeichert.")

if __name__ == '__main__':
    # Wichtiger Einstiegspunkt für Multiprocessing unter Windows
    tf.get_logger().setLevel('ERROR')  # Optional: Weniger Logausgaben
    train_model()
