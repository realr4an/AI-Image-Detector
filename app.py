import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input  # Preprocessing importieren

# ----------------------------
# Hilfsfunktionen
# ----------------------------

@st.cache_resource
def load_cnn_model():
    """
    Lädt das vortrainierte CNN-Modell (Dateiname: best_model.h5).
    Achte darauf, dass das Modell im selben Verzeichnis liegt oder passe den Pfad an.
    """
    model = load_model("best_model.h5")
    return model

def preprocess_image(img, target_size=(224, 224)):
    """
    Passt das hochgeladene Bild an die benötigte Eingabegröße an und preprocessiert es mit preprocess_input.
    """
    # Resize und Konvertierung in RGB (falls im BGR-Format)
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Statt Division durch 255: Preprocessing-Funktion anwenden
    img_preprocessed = preprocess_input(img_rgb.astype("float32"))
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    return img_expanded

def get_layer_from_model(model, layer_name):
    """
    Sucht nach einem Layer mit dem gegebenen Namen im Modell oder in enthaltenen Submodellen.
    """
    try:
        return model.get_layer(layer_name)
    except ValueError:
        # Suche in Submodellen
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(layer_name)
                except ValueError:
                    continue
    raise ValueError(f"No such layer: {layer_name}. Existing layers are: {[layer.name for layer in model.layers]}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Erstellt eine Grad-CAM Heatmap, die zeigt, welche Bildregionen zum Modell-Entscheid beigetragen haben.
    """
    conv_layer = get_layer_from_model(model, last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    # Berechnung der Gradienten
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Gewichtung der Feature Maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # ReLU und Normierung der Heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Überlagert die Grad-CAM Heatmap auf das Originalbild.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# ----------------------------
# Hauptanwendung (Streamlit)
# ----------------------------

def main():
    st.title("Erkennung von KI-generierten Bildern mittels CNN")
    st.write("Lade ein Bild hoch, um zu prüfen, ob es KI-generiert wurde.")

    # Bild-Upload
    uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Lese das Bild als Numpy-Array ein
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Anzeige des hochgeladenen Bildes (konvertiert zu RGB)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Hochgeladenes Bild", use_container_width=True)

        # Bildvorverarbeitung
        processed_img = preprocess_image(img)

        # Laden des vortrainierten Modells
        model = load_cnn_model()

        # Vorhersage: Annahme – Modell gibt einen Wert zwischen 0 und 1 zurück
        prediction = model.predict(processed_img)
        # Angepasste Logik: Hoher Wert -> Echt, niedriger Wert -> KI-generiert
        if prediction[0][0] > 0.5:
            label = "Echt"
        else:
            label = "KI-generiert"
        st.write(f"**Vorhersage:** Das Bild wird als **{label}** eingestuft.")
        st.write(f"Vorhersagewert: {prediction[0][0]:.2f}")

        # Grad-CAM: Ermittlung der letzten Convolution-Schicht
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):  # z.B. ResNet50 als Submodell
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.write("Keine Convolution-Schicht gefunden – Grad-CAM kann nicht angewendet werden.")
        else:
            st.write("**Erklärung (Grad-CAM):** Die hervorgehobenen Bereiche zeigen, welche Bildregionen für die Entscheidung wichtig waren.")
            heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name)
            # Overlay der Heatmap auf das Originalbild (RGB)
            original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_heatmap(heatmap, original_img)
            st.image(superimposed_img, caption="Grad-CAM Erklärung", use_container_width=True)

if __name__ == '__main__':
    main()
