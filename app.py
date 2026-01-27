"""
API Flask pour le déploiement du modèle de détection de véhicules
Fichier : app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Charger le meilleur modèle
# Remplacez par le chemin de votre modèle
MODEL_PATH = "best.pt"

@app.before_first_request
def load_model():
    """Charge le modèle au démarrage"""
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Modèle chargé : {MODEL_PATH}")
    else:
        print(f"❌ Modèle non trouvé : {MODEL_PATH}")
        model = None

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Request:
        - file: image (JPG, PNG)
        - conf: seuil de confiance (optionnel, défaut: 0.5)
    
    Response:
        {
            'detections': [...],
            'count': int,
            'model': str
        }
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Vérifier qu'un fichier est fourni
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        # Récupérer le seuil de confiance
        conf = float(request.form.get('conf', 0.5))
        
        # Lire l'image
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Prédiction
        results = model.predict(image_np, conf=conf)
        
        # Extraire les détections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)
                conf_score = float(box.conf)
                class_name = result.names.get(cls, f"Class {cls}")
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf_score,
                    'bbox': {
                        'x_min': bbox[0],
                        'y_min': bbox[1],
                        'x_max': bbox[2],
                        'y_max': bbox[3]
                    }
                })
        
        return jsonify({
            'detections': detections,
            'count': len(detections),
            'model': 'Vehicle Detection Model'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil"""
    return """
    <html>
        <head>
            <title>Vehicle Detection API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                }
                h1 { color: #1f77b4; }
                .endpoint {
                    background: #f0f2f6;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                code {
                    background: #e0e0e0;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <h1>🚗 Vehicle Detection API</h1>
            <p>API pour la détection de véhicules</p>
            
            <h2>Endpoints disponibles :</h2>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Vérifier l'état de l'API</p>
                <code>curl http://localhost:5000/health</code>
            </div>
            
            <div class="endpoint">
                <h3>POST /predict</h3>
                <p>Détecter les véhicules dans une image</p>
                <code>curl -X POST -F "file=@image.jpg" http://localhost:5000/predict</code>
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Charger le modèle au démarrage
    load_model()
    
    # Lancer l'application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

"""
INSTALLATION :
pip install flask flask-cors ultralytics pillow numpy

UTILISATION :
python app.py

TEST :
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
"""