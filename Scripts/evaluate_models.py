import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Testdatenpfade
test_dir_fake = "Dataset/Test/Fake"
test_dir_real = "Dataset/Test/Real"
print(os.listdir("models/ResNet50_Deepfake_detection"))
model_dir = "models/ResNet50_Deepfake_detection"
model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".h5")]

# Bildgröße (wird aus erstem Modell gelesen)
def get_model_input_size(model_path):
    model = load_model(model_path, compile=False)
    input_shape = model.input_shape
    return input_shape[1], input_shape[2]

# Bilder laden
def load_images_from_folder(folder, label, target_size):
    images = []
    labels = []
    filenames = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for filename in tqdm(filenames, desc=f"Lade Bilder aus {folder}", unit="Bild"):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
        labels.append(label)
    return images, labels

# Bewertung pro Modell
for model_file in model_files:
    print(f"\n🔍 Evaluating model: {model_file}")
    model = load_model(model_file, compile=False)

    # Bildgröße anpassen
    width, height = get_model_input_size(model_file)
    target_size = (width, height)

    # Bilder laden mit Fortschrittsanzeige
    fake_images, fake_labels = load_images_from_folder(test_dir_fake, 0, target_size)
    real_images, real_labels = load_images_from_folder(test_dir_real, 1, target_size)

    # Gesamtdaten
    all_images = np.array(fake_images + real_images)
    all_labels = np.array(fake_labels + real_labels)

    # Vorhersagen mit Fortschrittsanzeige
    print("🔄 Vorhersagen werden berechnet...")
    predictions = []
    for img in tqdm(all_images, desc="Model Predictions", unit="Bild"):
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        predictions.append(pred[0][0])
    predicted_labels = (np.array(predictions) > 0.5).astype("int32")

    # Metriken berechnen
    acc = accuracy_score(all_labels, predicted_labels)
    prec = precision_score(all_labels, predicted_labels)
    rec = recall_score(all_labels, predicted_labels)
    f1 = f1_score(all_labels, predicted_labels)

    # Ausgabe
    print(f"📊 Accuracy : {acc:.4f}")
    print(f"📊 Precision: {prec:.4f}")
    print(f"📊 Recall   : {rec:.4f}")
    print(f"📊 F1-Score : {f1:.4f}")
