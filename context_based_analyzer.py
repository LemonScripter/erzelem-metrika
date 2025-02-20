# ====================================================================
# ALKALMAZÁS INTEGRÁCIÓ - KONTEXTUÁLIS ELEMZÉS
# ====================================================================
# Fájlnév: context_based_analyzer.py 
# Verzió: 1.0
# Leírás: Kontextuális érzelmi elemzés integrálása a webalkalmazásba
# ====================================================================

from flask import Blueprint, request, jsonify
from train_context_emotion_model import ContextualEmotionModel
from emotion_cube_modeler import EmotionCubeModeler
import os
import traceback

# Blueprint létrehozása
context_routes = Blueprint('context', __name__, url_prefix='/context')

# Globális modell változók
context_model = None
cube_modeler = None

def load_models():
    """Modellek betöltése"""
    global context_model, cube_modeler
    
    model_path = "models/context_emotion_model"
    if os.path.exists(model_path):
        try:
            context_model = ContextualEmotionModel.load(model_path)
            cube_modeler = EmotionCubeModeler()
            print("Kontextuális érzelmi modell sikeresen betöltve!")
            return True
        except Exception as e:
            print(f"Hiba a kontextuális modell betöltésekor: {e}")
            traceback.print_exc()
    else:
        print(f"A kontextuális modell nem található: {model_path}")
    
    return False

@context_routes.route('/analyze', methods=['POST'])
def analyze_with_context():
    """Szöveg elemzése adott kontextusban"""
    try:
        # Modellek betöltése, ha még nem történt meg
        if context_model is None:
            if not load_models():
                return jsonify({
                    'success': False,
                    'error': 'Kontextuális modell nem elérhető.'
                }), 500
                
        # Adatok kinyerése
        data = request.get_json()
        text = data.get('text', '')
        context = data.get('context', 'general')
        
        if not text:
            return jsonify({'error': 'Nem adott meg szöveget!'}), 400
            
        # Elemzés a modellel
        results = context_model.predict([text], [context])[0]
        
        # Kategorizálás
        category_info = cube_modeler.classify_emotion(
            results['valence'], 
            results['arousal'], 
            results['dominance']
        )
        
        # Eredmények összeállítása
        response = {
            'success': True,
            'text': text,
            'context': context,
            'valence': results['valence'],
            'arousal': results['arousal'],
            'dominance': results['dominance'],
            'emotion_category': category_info['label'],
            'confidence': category_info['confidence'],
            'description': category_info.get('description', '')
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_message = str(e)
        print(f"Hiba a kontextuális elemzés során: {error_message}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_message,
        }), 500

# Kontextusok lekérdezése
@context_routes.route('/available', methods=['GET'])
def get_available_contexts():
    """Elérhető kontextusok lekérdezése"""
    try:
        if context_model is None:
            if not load_models():
                return jsonify({
                    'success': False,
                    'error': 'Kontextuális modell nem elérhető.'
                }), 500
        
        # Kontextusok kinyerése az encoderből
        contexts = context_model.context_encoder.categories_[0].tolist()
        
        return jsonify({
            'success': True,
            'contexts': contexts
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500