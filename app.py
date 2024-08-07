import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Charger le modèle
@st.cache_data()
def load():
    model = tf.keras.models.load_model('model/cnn/pneumonie_model1.h5')
    return model

model = load()

# Fonction pour préparer l'image
def preprocess_image(image):
    image = Image.open(image)
    image = image.convert('L')  # Convertir en niveaux de gris
    image = image.resize((224, 224))  # Redimensionner
    image = np.array(image)
    image = image / 255.0  # Normalisation
    image = np.expand_dims(image, axis=-1)  # Ajouter une dimension pour les canaux
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    return image

# Interface utilisateur Streamlit
st.title('Prédiction d\'Image de Pneumonie')

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

c1, c2 = st.columns(2)
if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    prediction = model.predict(image)
    class_names = ["PNEUMONIA", "NORMAL"]
    predicted_class = class_names[np.argmax(prediction)]
    c1.image(uploaded_file, caption='Image', use_column_width=True)
    c2.write(f"{predicted_class} avec une probabilité de {np.max(prediction*100):.2f}%")
