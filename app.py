# ====================================================================
# FLASK ALKALMAZÁS - KOCKA-SÍK-FÜGGVÉNYES SZÖVEGÉRTELMEZŐ
# ====================================================================
# Fájlnév: app.py
# Verzió: 1.0
# Leírás: Flask webszerver a szövegértelmező rendszer futtatásához
# ====================================================================

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
import traceback
from datetime import datetime

# Importáljuk a saját moduljainkat
from text_preprocessor import TextPreprocessor
from emotion_analyzer import EmotionAnalyzer
from emotional_space_model import EmotionCube, EmotionalPlane, EmotionalFunction
from emotion_cube_modeler import EmotionCubeModeler
from emotion_visualizer import EmotionVisualizer
from emotion_analysis_demo import EmotionAnalysisDemo

app = Flask(__name__)

# Globális változók
OUTPUT_DIR = 'static/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demó osztály inicializálása
demo = None

@app.route('/')
def index():
    """Főoldal megjelenítése"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Szöveg elemzése"""
    try:
        # Demó inicializálása, ha még nem történt meg
        global demo
        if demo is None:
            demo = EmotionAnalysisDemo()
        
        # Szöveg beolvasása
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'Nem adott meg szöveget!'}), 400
            
        # Időbélyeg generálása az outputhoz
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = f"{OUTPUT_DIR}/{timestamp}"
        os.makedirs(output_subdir, exist_ok=True)
        
        # Szöveg elemzése
        results = demo.analyze_text(text)
        
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

@app.route('/sample')
def sample():
    """Minta elemzés futtatása"""
    try:
        # Demó inicializálása, ha még nem történt meg
        global demo
        if demo is None:
            demo = EmotionAnalysisDemo()
            
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
        
        # Elemzés futtatása
        results = demo.analyze_text(sample_text)
        demo.visualize_text_analysis(results, output_dir=output_subdir)
        
        # Átirányítás az eredményekhez
        return redirect(f"/results/{timestamp}/analysis_report.html")
        
    except Exception as e:
        # Hiba kezelése
        error_message = str(e)
        print(f"Hiba történt a minta elemzés során: {error_message}")
        return f"Hiba történt a minta elemzés során: {error_message}", 500

if __name__ == '__main__':
    try:
        print("Kocka-Sík-Függvényes Szövegértelmező alkalmazás indítása...")
        print("A webszerver a következő címen érhető el: http://127.0.0.1:5000")
        app.run(debug=True)
    except Exception as e:
        print(f"Hiba történt az alkalmazás indításakor: {str(e)}")
        print("Részletes hibajelentés:")
        traceback.print_exc()