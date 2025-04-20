#!/usr/bin/env python3
"""
train_model.py

Dieses Skript trainiert ein CNN zur binären Klassifikation (z.B. Echt vs. KI-generiert)
unter Nutzung von Transfer Learning (ResNet50), Data Augmentation, Callbacks und TensorBoard.
"""

import tensorflow as tf
import os

def train_model():
    # Überprüfen, ob eine GPU verfügbar ist
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Gefundene GPU(s):", gpus)
    else:
        print("Keine GPU gefunden. Das Training läuft auf der CPU.")

    # Mixed Precision aktivieren
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision aktiviert mit Policy:", policy.name)

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from tensorflow.keras.regularizers import l2
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import datetime

    # Hyperparameter
    img_width, img_height = 224, 224
    batch_size = 64
    initial_epochs = 10
    fine_tune_epochs = 10

    # Verzeichnispfade
    train_data_dir = 'Dataset/train'
    validation_data_dir = 'Dataset/validation'
    test_data_dir = 'Dataset/test'

    # Data Augmentation
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

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Modell mit Transfer Learning
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # TensorBoard-Callback
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Callbacks für erstes Training
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model_initial.h5', save_best_only=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    callbacks_initial = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]

    # Erstes Training
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=initial_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks_initial,
        workers=4,
        use_multiprocessing=False
    )

    # Fine-Tuning: letzte ResNet-Schichten freigeben
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    model_checkpoint_fine = ModelCheckpoint('best_model_finetuned.h5', save_best_only=True, monitor='val_accuracy')
    callbacks_finetune = [early_stopping, model_checkpoint_fine, reduce_lr, tensorboard_callback]

    # Fine-Tuning
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks_finetune,
        workers=4,
        use_multiprocessing=False
    )

    # Optional: Evaluation
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

        y_pred = model.predict(test_generator)
        y_pred_classes = (y_pred > 0.5).astype(int)

        print("\nKlassifikationsreport:")
        print(classification_report(test_generator.classes, y_pred_classes))

        print("Konfusionsmatrix:")
        print(confusion_matrix(test_generator.classes, y_pred_classes))

    # Speichern des finalen Modells
    model.save("model_final.h5")
    print("Das Modell wurde erfolgreich als 'model_final.h5' gespeichert.")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    train_model()
