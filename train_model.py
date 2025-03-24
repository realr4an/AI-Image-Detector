import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Bildgrößen und Trainingsparameter definieren
img_width, img_height = 224, 224
batch_size = 32
epochs = 25

# Pfade zum Trainings- und Validierungsdatensatz
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Data Augmentation für den Trainingsdatensatz
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Für den Validierungsdatensatz wird nur reskaliert
test_datagen = ImageDataGenerator(rescale=1.0/255)

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

# Aufbau des CNN-Modells
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binäre Klassifikation: 0 = echt, 1 = KI-generiert
])

# Kompilieren des Modells
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Modell trainieren
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Speichern des Modells als .h5-Datei
model.save("model.h5")
print("Das Modell wurde erfolgreich als 'model.h5' gespeichert.")
