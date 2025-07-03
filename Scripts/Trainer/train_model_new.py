#!/usr/bin/env python3

import tensorflow as tf
import os

def train_model():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Gefundene GPU(s):", gpus)
    else:
        print("Keine GPU gefunden. Das Training läuft auf der CPU.")

    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from tensorflow.keras.regularizers import l2
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import datetime

    img_width, img_height = 256, 256
    batch_size = 64
    initial_epochs = 50

    train_data_dir = 'Dataset/train'
    validation_data_dir = 'Dataset/validation'
    test_data_dir = 'Dataset/test'

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

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
        Dropout(0.6),
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_checkpoint = ModelCheckpoint('best_model_initial.h5', save_best_only=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7)
    callbacks = [model_checkpoint, reduce_lr, tensorboard_callback]

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

    if os.path.exists(test_data_dir):
        test_generator = datagen.flow_from_directory(
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

    model.save("model_final.h5")
    print("Das Modell wurde erfolgreich als 'model_final.h5' gespeichert.")

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    train_model()
