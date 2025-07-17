#!/usr/bin/env python3

import tensorflow as tf
import os
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Pfad-Konfiguration basierend auf der Projektstruktur
# ─────────────────────────────────────────────────────────────────────────────
# Aktueller Pfad des Skripts (z.B. AI-IMAGE-DETECTOR/Scripts/Trainer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wurzelverzeichnis des Projekts (z.B. AI-IMAGE-DETECTOR)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Pfade für Daten, Modelle und Logs definieren
DATA_DIR = os.path.join(ROOT_DIR, "Data")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")


def train_model():
    """
    Trainiert ein Bildklassifikationsmodell und speichert das Ergebnis.
    """
    # 1. GPU- und Mixed-Precision-Setup
    # ─────────────────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Gefundene GPU(s): {gpus}")
        try:
            # Speichereservierung für GPUs dynamisch gestalten
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ Keine GPU gefunden. Das Training läuft auf der CPU.")

    # Mixed Precision für schnelleres Training auf modernen GPUs
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Weitere Imports
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from tensorflow.keras.regularizers import l2
    from sklearn.metrics import classification_report, confusion_matrix

    # 2. Trainingsparameter
    # ─────────────────────────────────────────────────────────────────────────
    img_width, img_height = 256, 256
    batch_size = 64
    initial_epochs = 25
    base_model_name = "ResNet50" # Wichtig für den Dateinamen des Modells

    # Korrekte Pfade zu den Daten-Unterordnern
    train_data_dir = os.path.join(DATA_DIR, 'train')
    validation_data_dir = os.path.join(DATA_DIR, 'validation')
    test_data_dir = os.path.join(DATA_DIR, 'test')
    
    # Sicherstellen, dass die Zielordner existieren
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 3. Daten-Generatoren
    # ─────────────────────────────────────────────────────────────────────────
    # Dieser Generator wendet die notwendige Vorverarbeitung für ResNet50 an
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # 4. Modellerstellung
    # ─────────────────────────────────────────────────────────────────────────
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = True # Wir trainieren das gesamte Modell (Fine-Tuning)

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
        Dropout(0.6),
        # Die letzte Schicht benötigt float32 für Mixed Precision
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 5. Callbacks und dynamische Dateinamen
    # ─────────────────────────────────────────────────────────────────────────
    # Zeitstempel für eindeutige Namen
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Dynamischer Dateipfad für das zu speichernde Modell
    model_filename = f"{base_model_name}_{timestamp}.h5"
    model_filepath = os.path.join(MODEL_DIR, model_filename)

    # Speichert nur das beste Modell basierend auf der Validierungsgenauigkeit
    model_checkpoint = ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_accuracy', verbose=1)
    
    # Reduziert die Lernrate, wenn der Fortschritt stagniert
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7)

    # Log-Verzeichnis für TensorBoard
    tensorboard_log_dir = os.path.join(LOG_DIR, "fit", f"{base_model_name}-{timestamp}")
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    callbacks = [model_checkpoint, reduce_lr, tensorboard_callback]

    # 6. Training des Modells
    # ─────────────────────────────────────────────────────────────────────────
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=initial_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False
    )

    print(f"\n✅ Training abgeschlossen. Das beste Modell wurde hier gespeichert:\n{model_filepath}")

    # 7. Evaluation auf dem Test-Set (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if os.path.exists(test_data_dir):
        print("\n🧪 Evaluiere das Modell auf dem Test-Set...")
        # Lade das BESTE Modell für die finale Evaluation
        best_model = tf.keras.models.load_model(model_filepath)
        
        test_generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        test_loss, test_accuracy = best_model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        y_pred = best_model.predict(test_generator)
        y_pred_classes = (y_pred > 0.5).astype(int)

        print("\n📊 Klassifikationsreport:")
        print(classification_report(test_generator.classes, y_pred_classes))

        print("📊 Konfusionsmatrix:")
        print(confusion_matrix(test_generator.classes, y_pred_classes))


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    train_model()