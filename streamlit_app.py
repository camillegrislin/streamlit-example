import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

# Charger le modèle de machine learning
model = tf.keras.models.load_model('chemin_vers_votre_modele.h5')

# Fonction pour extraire les caractéristiques audio
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0)
    return features

# Interface utilisateur avec Streamlit
st.write("""
# Prédiction audio avec Streamlit

Cette application utilise un modèle de machine learning pour prédire une sortie en fonction d'un fichier audio chargé.
""")

# Chargement du fichier audio
audio_file = st.file_uploader("Veuillez sélectionner un fichier audio", type=["wav"])

if audio_file is not None:
    # Extraction des caractéristiques du fichier audio
    features = extract_features(audio_file)
    
    # Redimensionner les caractéristiques pour correspondre aux attentes du modèle
    features = np.expand_dims(features, axis=0)
    
    # Prédiction avec le modèle
    prediction = model.predict(features)
    
    # Affichage des résultats
    st.subheader('Caractéristiques audio extraites:')
    st.write(features)
    
    st.subheader('Prédiction:')
    st.write(prediction)
