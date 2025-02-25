# app.py

# TensorFlow warnings kikapcsolása
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3: csak ERROR logok
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU kikapcsolása

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import json
import traceback
import tensorflow as tf
from datetime import datetime

# TensorFlow warning kikapcsolása
tf.get_logger().setLevel('ERROR')

# Flask app inicializálása
app = Flask(__name__)

# Környezeti változók beállítása
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')

# Globális változók
OUTPUT_DIR = 'static/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Globális változók a modulok lusta betöltéséhez
demo = None
context_detector = None

# Importálásokat késleltetjük, hogy csak akkor történjenek, 
# amikor tényleg szükség van rájuk
text_preprocessor = None
emotion_analyzer = None
emotion_cube_modeler = None
emotion_visualizer = None

def init_detector():
    """Kontextus-felismerő inicializálása - csak amikor szükséges"""
    global context_detector
    if context_detector is None:
        # Csak ekkor importáljuk a modult
        from context_detector import ContextDetector
        context_detector = ContextDetector()
        print("Kontextus-felismerő inicializálva")

def init_demo():
    """Demó osztály inicializálása - csak amikor szükséges"""
    global demo
    if demo is None:
        # Csak ekkor importáljuk a szükséges modulokat
        from text_preprocessor import TextPreprocessor
        from emotion_analyzer import EmotionAnalyzer
        from emotional_space_model import EmotionCube, EmotionalPlane, EmotionalFunction
        from emotion_cube_modeler import EmotionCubeModeler
        from emotion_visualizer import EmotionVisualizer
        from emotion_analysis_demo import EmotionAnalysisDemo
        
        demo = EmotionAnalysisDemo()
        print("Demó osztály inicializálva")

@app.route('/')
def index():
    """Főoldal megjelenítése"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Szöveg elemzése"""
    try:
        # Modulok inicializálása, csak amikor szükséges
        init_demo()
        init_detector()
        
        # Szöveg beolvasása
        text = request.form.get('text', '')
        context_option = request.form.get('context_option', 'auto')
        
        if not text:
            return jsonify({'error': 'Nem adott meg szöveget!'}), 400
        
        # Kontextus meghatározása
        if context_option == 'auto':
            # Automatikus kontextus-felismerés
            context_info = context_detector.detect_context(text)
            context = context_info['context']
            context_detector.save_detection_result(text, context_info)
            print(f"Automatikusan felismert kontextus: {context} ({context_info['confidence']:.2f})")
        else:
            # Manuálisan megadott kontextus
            context = request.form.get('context', 'general')
            context_info = {'context': context, 'confidence': 1.0, 'method': 'manual'}
            print(f"Manuálisan megadott kontextus: {context}")
            
        # Időbélyeg generálása az outputhoz
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = f"{OUTPUT_DIR}/{timestamp}"
        os.makedirs(output_subdir, exist_ok=True)
        
        # Szöveg elemzése
        results = demo.analyze_text(text, context)
        
        # Kontextus információ hozzáadása az eredményekhez
        results['context_info'] = context_info
        
        # Vizualizáció létrehozása
        demo.visualize_text_analysis(results, output_dir=output_subdir)
        
        # Eredmények visszaadása
        return jsonify({
            'success': True,
            'result_dir': timestamp,
            'overall_category': results['overall_cube_info']['label'],
            'valence': round(results['overall_emotions']['valence'], 2),
            'arousal': round(results['overall_emotions']['arousal'], 2),
            'dominance': round(results['overall_emotions']['dominance'], 2),
            'context': context,
            'context_confidence': context_info.get('confidence', 1.0),
            'context_method': context_info.get('method', 'manual'),
            'report_url': f"/results/{timestamp}/analysis_report.html"
        })
        
    except Exception as e:
        # Hiba kezelése
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Hiba történt: {error_message}")
        print(stack_trace)
        return jsonify({
            'success': False,
            'error': error_message,
            'stack_trace': stack_trace
        }), 500

@app.route('/results/<path:result_dir>/<path:filename>')
def results(result_dir, filename):
    """Eredmények fájljainak kiszolgálása"""
    return send_from_directory(f"{OUTPUT_DIR}/{result_dir}", filename)

@app.route('/contexts', methods=['GET'])
def get_available_contexts():
    """Elérhető kontextusok lekérdezése"""
    init_detector()
        
    return jsonify({
        'success': True,
        'contexts': context_detector.get_all_contexts() if hasattr(context_detector, 'get_all_contexts') else 
                   ['general', 'business', 'personal', 'academic', 'social_media']
    })

@app.route('/sample')
def sample():
    """Minta elemzés futtatása"""
    try:
        # Modulok inicializálása, csak amikor szükséges
        init_demo()
        init_detector()
            
        # Példa szöveg
        sample_text = """
        A kocka-sík-függvényes modell egy izgalmas és újszerű megközelítés az érzelmek elemzésére.
        Ez a módszer lehetővé teszi számunkra, hogy az emberi érzelmek komplexitását háromdimenziós térben ábrázoljuk.
        Természetesen vannak korlátai is, de a hagyományos módszerekhez képest sokkal árnyaltabb képet kaphatunk.
        Nagyon örülök, hogy sikerült implementálni ezt a rendszert!
        Az érzelmi dimenziók közötti kapcsolatok feltárása különösen fontos lehet a pszichológiai kutatásokban és a mesterséges intelligencia fejlesztésében.
        Kíváncsi vagyok, hogy a jövőben milyen további fejlesztésekkel lehet majd tovább finomítani az elemzést.
        """
        
        # Időbélyeg generálása
        timestamp = "sample_analysis"
        output_subdir = f"{OUTPUT_DIR}/{timestamp}"
        os.makedirs(output_subdir, exist_ok=True)
        
        # Elemzés futtatása - automatikus kontextussal
        context_info = context_detector.detect_context(sample_text)
        context = context_info['context']
        
        # Elemzés futtatása
        results = demo.analyze_text(sample_text, context)
        results['context_info'] = context_info
        demo.visualize_text_analysis(results, output_dir=output_subdir)
        
        # Átirányítás az eredményekhez
        return redirect(f"/results/{timestamp}/analysis_report.html")
        
    except Exception as e:
        # Hiba kezelése
        error_message = str(e)
        print(f"Hiba történt a minta elemzés során: {error_message}")
        return f"Hiba történt a minta elemzés során: {error_message}", 500

# Ez a rész csak akkor hajtódik végre, ha közvetlenül ezt a fájlt futtatod,
# és nem importáláskor. Render.com-on a run.py fogja indítani az alkalmazást,
# így ez a rész nem fut le automatikusan.
if __name__ == '__main__':
    # Helyi fejlesztéshez
    app.run(debug=True)