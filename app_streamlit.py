"""
Interface Streamlit pour la détection de véhicules
Fichier : app_streamlit.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os

# Configuration de la page
st.set_page_config(
    page_title="Détection de Véhicules",
    page_icon="🚗",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("🚗 Système de Détection de Véhicules")
st.markdown("**Détection automatique de véhicules utilisant l'intelligence artificielle**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Informations")
    st.info("""
    **Modèle** : YOLO/RT-DETR/YOLOv8l
    
    **Classes détectées** :
    - 🚌 Bus
    - 🚗 Car
    - 🏍️ Motorcycle
    - 🚙 Pickup-truck
    - 🚛 Semi-trailer
    - 🚐 Van
    """)
    
    st.header("⚙️ Paramètres")
    conf_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Seuil minimum de confiance pour afficher les détections"
    )
    
    st.markdown("---")
    st.markdown("**Projet IATP - 2026**")

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle YOLO"""
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle non trouvé : {model_path}")
        st.info("Placez le fichier best.pt dans le même dossier que cette application")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle
model = load_model()

# Layout en deux colonnes
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload d'image")
    uploaded_file = st.file_uploader(
        "Téléchargez une image à analyser",
        type=['jpg', 'jpeg', 'png'],
        help="Formats acceptés : JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image originale
        image = Image.open(uploaded_file)
        st.image(image, caption="Image originale", use_container_width=True)

with col2:
    st.header("📊 Résultats")
    
    if uploaded_file is not None and model is not None:
        if st.button("🔍 Détecter les objets", type="primary"):
            with st.spinner("Analyse en cours..."):
                # Convertir l'image en numpy array
                image_np = np.array(image)
                
                # Prédiction
                results = model.predict(image_np, conf=conf_threshold)
                
                # Afficher l'image avec détections
                for result in results:
                    # Sauvegarder temporairement
                    result.save("temp_result.jpg")
                    result_img = Image.open("temp_result.jpg")
                    st.image(result_img, caption="Image avec détections", use_container_width=True)
                    
                    # Nombre de détections
                    num_detections = len(result.boxes)
                    
                    if num_detections > 0:
                        st.success(f"✅ {num_detections} objet(s) détecté(s)")
                        
                        # Créer un DataFrame avec les détections
                        st.subheader("📋 Détails des détections")
                        
                        detections = []
                        for box in result.boxes:
                            cls = int(box.cls)
                            conf = float(box.conf)
                            class_name = result.names.get(cls, f"Classe {cls}")
                            bbox = box.xyxy[0].tolist()
                            
                            detections.append({
                                "Classe": class_name,
                                "Confiance": f"{conf:.2%}",
                                "X_min": f"{bbox[0]:.0f}",
                                "Y_min": f"{bbox[1]:.0f}",
                                "X_max": f"{bbox[2]:.0f}",
                                "Y_max": f"{bbox[3]:.0f}"
                            })
                        
                        # Afficher le tableau
                        df = pd.DataFrame(detections)
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistiques par classe
                        st.subheader("📈 Statistiques")
                        class_counts = df['Classe'].value_counts()
                        st.bar_chart(class_counts)
                        
                        # Métriques
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Total détections", num_detections)
                        with col_m2:
                            avg_conf = sum([float(d['Confiance'].strip('%'))/100 for d in detections]) / len(detections)
                            st.metric("Confiance moyenne", f"{avg_conf:.2%}")
                        with col_m3:
                            st.metric("Classes différentes", len(class_counts))
                    else:
                        st.warning("⚠️ Aucun objet détecté. Essayez de réduire le seuil de confiance.")
    
    elif uploaded_file is None:
        st.info("👆 Téléchargez une image pour commencer l'analyse")
    
    elif model is None:
        st.error("❌ Le modèle n'a pas pu être chargé")

# Instructions
with st.expander("📖 Instructions d'utilisation"):
    st.markdown("""
    ### Comment utiliser cette application ?
    
    1. **Téléchargez une image** contenant des véhicules (JPG, PNG)
    2. **Ajustez le seuil de confiance** dans la barre latérale si nécessaire
    3. **Cliquez sur "Détecter les objets"** pour lancer l'analyse
    4. **Consultez les résultats** : image annotée, tableau des détections, statistiques
    
    ### Classes détectées :
    - Bus
    - Car (voiture)
    - Motorcycle (moto)
    - Pickup-truck (pickup)
    - Semi-trailer (semi-remorque)
    - Van
    
    ### Conseils :
    - Utilisez des images de bonne qualité pour de meilleurs résultats
    - Si aucun objet n'est détecté, essayez de réduire le seuil de confiance
    - Les détections avec une confiance > 80% sont généralement très fiables
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Projet IATP - Détection de Véhicules</strong></p>
    <p>Développé avec ❤️ en utilisant Streamlit et YOLO</p>
</div>
""", unsafe_allow_html=True)

"""
INSTALLATION :
pip install streamlit ultralytics pillow numpy pandas

UTILISATION :
streamlit run app_streamlit.py

DÉPLOIEMENT :
1. Créer un repository GitHub
2. Uploader : app_streamlit.py, best.pt, requirements.txt
3. Déployer sur Streamlit Cloud (https://share.streamlit.io)
"""