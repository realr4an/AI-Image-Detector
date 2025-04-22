import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input  # Import hinzufügen

# Verwende st.cache_resource, um das Modell zu laden und zwischenzuspeichern.
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_finetuned.h5")
    return model

model = load_model()

# Bestimme die erwartete Eingabeform des Modells (angenommen: (None, Höhe, Breite, Kanäle))
input_shape = model.input_shape  # z.B. (None, height, width, channels)
if len(input_shape) == 4:
    height, width = input_shape[1], input_shape[2]
else:
    st.error("Das Modell hat eine unerwartete Eingabeform!")
    # Fallback-Werte, falls nötig
    height, width = 224, 224

st.title("Deep Fake Erkennung")
st.write("Laden Sie ein Bild hoch, um zu prüfen, ob es sich um einen Deep Fake handelt.")

# Bild-Upload
uploaded_file = st.file_uploader("Wählen Sie ein Bild (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Bild laden und in RGB konvertieren (falls nicht bereits)
    image = Image.open(uploaded_file).convert("RGB")
    
    # Bild auf die vom Modell erwartete Größe anpassen
    image_resized = image.resize((width, height))
    
    # Bild anzeigen
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
    
    # In ein numpy-Array konvertieren
    img_array = np.array(image_resized)
    # Wende das ResNet50 Preprocessing an (statt /255.0)
    img_array = preprocess_input(img_array)
    # Batch-Dimension hinzufügen
    img_array = np.expand_dims(img_array, axis=0)
    
    # Vorhersage mit dem Modell
    prediction = model.predict(img_array)
    
    # Anzeige der rohen Vorhersage-Werte
    st.write("Model Prediction (Rohwerte):", prediction)
    
    # Annahme: Binäre Klassifikation (Deep Fake vs. Authentisch)
    # Hier wird angenommen, dass ein Wert > 0.5 als Deep Fake klassifiziert wird.
    if prediction[0][0] > 0.5:
        st.info("Das Bild wird als authentisch klassifiziert.")
    else:
        st.success("Das Bild wird als Deep Fake klassifiziert.")

