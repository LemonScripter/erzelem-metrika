# ====================================================================
# 3. VIZUALIZÁCIÓS MODUL - DEMO ALKALMAZÁS
# ====================================================================
# Fájlnév: emotion_analysis_demo.py
# Verzió: 1.0
# Leírás: Komplett demó alkalmazás, amely demonstrálja a szövegelemzést 
#         és vizualizációt
# ====================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
import os
import json
from emotional_function_analyzer import EmotionalFunctionAnalyzer
from context_plane_system import ContextPlaneSystem
from intersection_analyzer import IntersectionAnalyzer
from datetime import datetime
from text_preprocessor import TextPreprocessor
from emotion_analyzer import EmotionAnalyzer
from emotional_space_model import EmotionCube, EmotionalPlane, EmotionalFunction
from emotion_cube_modeler import EmotionCubeModeler
from emotion_visualizer import EmotionVisualizer

class EmotionAnalysisDemo:
   """
   Szöveg érzelmi elemzését és vizualizációját bemutató demó alkalmazás
   """
   
   def __init__(self):
       """
       Inicializálja a demó alkalmazást minden szükséges komponenssel
       """
       print("Érzelmi tér elemző és vizualizáló rendszer inicializálása...")
       
       # Komponensek inicializálása
       self.text_preprocessor = TextPreprocessor()
       self.emotion_analyzer = EmotionAnalyzer()
       self.modeler = EmotionCubeModeler(cube_size=0.25)
       self.visualizer = EmotionVisualizer(self.modeler)
       self.function_analyzer = EmotionalFunctionAnalyzer()
       self.plane_system = ContextPlaneSystem()
       self.intersection_analyzer = IntersectionAnalyzer()
       
       # Érzelmi függvények regisztrálása
       sine_params = {
           'amplitudes': [0.2, 0.3, 0.15],
           'frequencies': [0.1, 0.05, 0.02],
           'phases': [0, np.pi/2, np.pi/4],
           'offsets': [0.5, 0.5, 0.5]
       }
       sine_function = EmotionalFunction(
           name="szinuszos_érzelmek",
           function=self._sine_wave_emotion,
           parameters=sine_params,
           description="Szinuszos érzelmi hullámok a három dimenzión"
       )
       
       gaussian_params = {
           'centers': [5, 5, 5],
           'widths': [2, 3, 4],
           'heights': [0.5, 0.4, 0.3],
           'baselines': [0.3, 0.4, 0.5]
       }
       gaussian_function = EmotionalFunction(
           name="gauss_érzelmek",
           function=self._gaussian_emotion,
           parameters=gaussian_params,
           description="Gauss-görbe alakú érzelmi változások"
       )
       
       self.modeler.register_emotion_function(sine_function)
       self.modeler.register_emotion_function(gaussian_function)
       
       print("Rendszer inicializálása kész!")
   
   def _sine_wave_emotion(self, t: float, params: Dict) -> Tuple[float, float, float]:
       """
       Szinuszos érzelmi hullám függvény
       
       Args:
           t: Időpillanat
           params: Paraméterek (amplitudes, frequencies, phases, offsets)
           
       Returns:
           (valence, arousal, dominance) az adott időpillanatban
       """
       amplitudes = params.get('amplitudes', [0.2, 0.3, 0.15])
       frequencies = params.get('frequencies', [0.1, 0.05, 0.02])
       phases = params.get('phases', [0, np.pi/2, np.pi/4])
       offsets = params.get('offsets', [0.5, 0.5, 0.5])
       
       valence = offsets[0] + amplitudes[0] * np.sin(frequencies[0] * t + phases[0])
       arousal = offsets[1] + amplitudes[1] * np.sin(frequencies[1] * t + phases[1])
       dominance = offsets[2] + amplitudes[2] * np.sin(frequencies[2] * t + phases[2])
       
       # Értékek korlátozása [0, 1] tartományba
       valence = max(0, min(1, valence))
       arousal = max(0, min(1, arousal))
       dominance = max(0, min(1, dominance))
       
       return (valence, arousal, dominance)

   def _gaussian_emotion(self, t: float, params: Dict) -> Tuple[float, float, float]:
       """
       Gauss görbe-szerű érzelmi függvény
       
       Args:
           t: Időpillanat
           params: Paraméterek (centers, widths, heights, baselines)
           
       Returns:
           (valence, arousal, dominance) az adott időpillanatban
       """
       centers = params.get('centers', [5, 5, 5])
       widths = params.get('widths', [2, 3, 4])
       heights = params.get('heights', [0.5, 0.4, 0.3])
       baselines = params.get('baselines', [0.3, 0.4, 0.5])
       
       valence = baselines[0] + heights[0] * np.exp(-((t - centers[0]) / widths[0])**2)
       arousal = baselines[1] + heights[1] * np.exp(-((t - centers[1]) / widths[1])**2)
       dominance = baselines[2] + heights[2] * np.exp(-((t - centers[2]) / widths[2])**2)
       
       # Értékek korlátozása [0, 1] tartományba
       valence = max(0, min(1, valence))
       arousal = max(0, min(1, arousal))
       dominance = max(0, min(1, dominance))
       
       return (valence, arousal, dominance)

   def analyze_text(self, text: str, context: str = "general") -> Dict:
       """
       Elemzi a szöveget és visszaadja az érzelmi eredményeket
    
       Args:
           text: Elemzendő szöveg
           context: Elemzési kontextus, alapértelmezetten "general"
        
       Returns:
           Elemzési eredmények szótára
       """
       print(f"Szöveg elemzése: '{text[:50]}...'")
       print(f"Elemzési kontextus: {context}")
    
       # Mondatokra bontás
       sentences = self.text_preprocessor.split_sentences(text)
       print(f"A szöveg {len(sentences)} mondatra lett bontva.")
       
       # Mondatonkénti érzelmi elemzés
       sentence_emotions = []
       emotion_points = []
       
       for i, sentence in enumerate(sentences):
           print(f"  {i+1}. mondat elemzése: '{sentence[:30]}...'")
           emotions = self.emotion_analyzer.analyze_emotions(sentence)
           
           # Kocka azonosítása és osztályozás
           cube_info = self.modeler.classify_emotion(
               emotions['valence'], 
               emotions['arousal'], 
               emotions['dominance']
           )
           
           # Eredmények összegyűjtése
           sentence_emotions.append({
               'sentence': sentence,
               'emotions': emotions,
               'cube_info': cube_info
           })
           
           # Érzelmi pont hozzáadása
           emotion_points.append((
               emotions['valence'],
               emotions['arousal'],
               emotions['dominance'],
               f"Mondat {i+1}"
           ))
           
           print(f"    Érzelmi dimenziók: V={emotions['valence']:.2f}, " 
                 f"A={emotions['arousal']:.2f}, D={emotions['dominance']:.2f}")
           print(f"    Kategória: {cube_info['label']} ({cube_info['confidence']:.2f})")
           
       # Teljes szöveg elemzése
       overall_emotions = self.emotion_analyzer.analyze_emotions(text)
       overall_cube_info = self.modeler.classify_emotion(
           overall_emotions['valence'],
           overall_emotions['arousal'],
           overall_emotions['dominance']
       )
       
       # Érzelmi trajektória létrehozása a mondatokból
       trajectory = []
       for i, sent_emotion in enumerate(sentence_emotions):
           emotions = sent_emotion['emotions']
           cube_info = sent_emotion['cube_info']
        
           trajectory.append({
               't': i,
               'valence': emotions['valence'],
               'arousal': emotions['arousal'],
               'dominance': emotions['dominance'],
               'label': cube_info['label'],
               'confidence': cube_info['confidence'],
               'sentence': sent_emotion['sentence'][:50] + ("..." if len(sent_emotion['sentence']) > 50 else "")
           })
    
       # Új: Érzelmi folyamat elemzése
       process_analysis = {}
       if len(trajectory) > 1:
           process_analysis = self.intersection_analyzer.analyze_emotional_trajectory(
               trajectory, context
           )
        
       # Eredmények összegyűjtése
       results = {
           'text': text,
           'context': context,
           'sentences': sentence_emotions,
           'overall_emotions': overall_emotions,
           'overall_cube_info': overall_cube_info,
           'emotion_points': emotion_points,
           'trajectory': trajectory,
           'process_analysis': process_analysis
       }
    
       print("Szövegelemzés kész!")
       print(f"Teljes szöveg érzelmi kategóriája: {overall_cube_info['label']}")
    
       return results
       
   def visualize_text_analysis(self, analysis_results: Dict, 
                              output_dir: str = './output') -> None:
       """
       Az elemzési eredmények vizualizálása
       
       Args:
           analysis_results: Az analyze_text() által visszaadott eredmények
           output_dir: Kimeneti könyvtár a képek mentéséhez
       """
       # Kimeneti könyvtár létrehozása
       os.makedirs(output_dir, exist_ok=True)
       
       # 1. Érzelmi kocka 3D megjelenítése a teljes szöveg érzelmi pontjával
       v = analysis_results['overall_emotions']['valence']
       a = analysis_results['overall_emotions']['arousal']
       d = analysis_results['overall_emotions']['dominance']
       
       print("1. Érzelmi kocka 3D megjelenítése...")
       self.visualizer.plot_emotion_cube_3d(
           emotion_point=(v, a, d),
           save_path=f"{output_dir}/emotion_cube_3d.png"
       )
       
       # 2. Interaktív 3D ábra létrehozása
       print("2. Interaktív 3D ábra készítése...")
       emotion_points = analysis_results['emotion_points']
       
       # Érzelmi trajektória automatikus generálása
       if len(analysis_results['trajectory']) > 1:
           trajectory = analysis_results['trajectory']
       else:
           # Ha csak egy mondat van, vagy nincs trajektória, akkor generálunk egy szinuszost
           trajectory = self.modeler.get_emotion_trajectory(
               "szinuszos_érzelmek", 0, 10, 100
           )
       
       fig = self.visualizer.create_interactive_3d_plot(
           emotion_points=emotion_points,
           trajectory=trajectory
       )
       fig.write_html(f"{output_dir}/interactive_emotion_plot.html")
       
       # 3. 2D projekciók létrehozása
       print("3. 2D projekciók létrehozása...")
       for dims in ['valence_arousal', 'valence_dominance', 'arousal_dominance']:
           self.visualizer.plot_emotion_trajectory_2d(
               trajectory=trajectory,
               dimensions=dims,
               save_path=f"{output_dir}/trajectory_2d_{dims}.png"
           )
           
       # 4. Idősor ábra létrehozása
       print("4. Idősor ábra létrehozása...")
       self.visualizer.plot_emotion_time_series(
           trajectory=trajectory,
           save_path=f"{output_dir}/emotion_time_series.png"
       )
       
       # 5. Mondatok érzelmi statisztikáinak táblázata
       print("5. Érzelmi statisztikák táblázat készítése...")
       stats_data = []
       for i, sentence_data in enumerate(analysis_results['sentences']):
           emotions = sentence_data['emotions']
           cube_info = sentence_data['cube_info']
           
           stats_data.append({
               'Sorszám': i+1,
               'Mondat': sentence_data['sentence'][:50] + "..." if len(sentence_data['sentence']) > 50 else sentence_data['sentence'],
               'Valencia': round(emotions['valence'], 2),
               'Arousal': round(emotions['arousal'], 2),
               'Dominancia': round(emotions['dominance'], 2),
               'Kategória': cube_info['label'],
               'Konfidencia': round(cube_info['confidence'], 2)
           })
       
       stats_df = pd.DataFrame(stats_data)
       stats_df.to_csv(f"{output_dir}/sentence_emotion_stats.csv", index=False)
       
       # 6. JSON formátumban is mentjük az eredményeket
       serializable_results = {
           'text': analysis_results['text'],
           'overall_emotions': {k: float(v) for k, v in analysis_results['overall_emotions'].items()},
           'overall_category': analysis_results['overall_cube_info']['label'],
           'overall_confidence': float(analysis_results['overall_cube_info']['confidence']),
           'sentences': []
       }
       
       for sent_data in analysis_results['sentences']:
           serializable_results['sentences'].append({
               'text': sent_data['sentence'],
               'emotions': {k: float(v) for k, v in sent_data['emotions'].items()},
               'category': sent_data['cube_info']['label'],
               'confidence': float(sent_data['cube_info']['confidence'])
           })
           
       with open(f"{output_dir}/analysis_results.json", 'w', encoding='utf-8') as f:
           json.dump(serializable_results, f, ensure_ascii=False, indent=2)
       
       # 7. HTML összefoglaló generálása
       self._generate_html_summary(analysis_results, output_dir)
       
       print(f"Vizualizáció kész! Az eredmények az '{output_dir}' könyvtárban találhatók.")
       
   def _generate_html_summary(self, analysis_results: Dict, output_dir: str) -> None:
       """
       HTML összefoglaló generálása az elemzésről
       
       Args:
           analysis_results: Elemzési eredmények
           output_dir: Kimeneti könyvtár
       """
       v = round(analysis_results['overall_emotions']['valence'], 2)
       a = round(analysis_results['overall_emotions']['arousal'], 2)
       d = round(analysis_results['overall_emotions']['dominance'], 2)
       category = analysis_results['overall_cube_info']['label']
       confidence = round(analysis_results['overall_cube_info']['confidence'], 2)
       
       # Kontextus információ hozzáadása
       context_info = analysis_results.get('context_info', {})
       context = context_info.get('context', 'general')
       context_confidence = context_info.get('confidence', 1.0)
       context_method = context_info.get('method', 'manual')
       
       context_html = ""
       if context != 'general':
           context_html = f"""
           <div class="context-info">
               <h3>Szöveg kontextusa</h3>
               <div class="context-badge">
                   {context.replace('_', ' ').capitalize()}
                   {f'<span class="confidence-pill">{int(context_confidence * 100)}%</span>' if context_method != 'manual' else ''}
               </div>
               <p class="detection-method">
                   Felismerési módszer: {
                       'Manuálisan megadva' if context_method == 'manual' else
                       'Kulcsszó-alapú felismerés' if context_method == 'keyword' else
                       'Gépi tanulás alapú felismerés'
                   }
               </p>
           </div>
           """
       
       # Mondatok táblázata
       sentences_html = """
       <table class="sentence-table">
           <tr>
               <th>Sorszám</th>
               <th>Mondat</th>
               <th>Valencia</th>
               <th>Arousal</th>
               <th>Dominancia</th>
               <th>Kategória</th>
               <th>Konfidencia</th>
           </tr>
       """
       
       for i, sent_data in enumerate(analysis_results['sentences']):
           emotions = sent_data['emotions']
           cube_info = sent_data['cube_info']
           
           sentences_html += f"""
           <tr>
               <td>{i+1}</td>
               <td>{sent_data['sentence'][:50] + '...' if len(sent_data['sentence']) > 50 else sent_data['sentence']}</td>
               <td>{round(emotions['valence'], 2)}</td>
               <td>{round(emotions['arousal'], 2)}</td>
               <td>{round(emotions['dominance'], 2)}</td>
               <td>{cube_info['label']}</td>
               <td>{round(cube_info['confidence'], 2)}</td>
           </tr>
           """
           
       sentences_html += "</table>"
       
       # Érzelmi méterek HTML kódja
       valence_pos = int(v * 100)
       arousal_pos = int(a * 100)
       dominance_pos = int(d * 100)
       
       html_content = f"""
       <!DOCTYPE html>
       <html>
       <head>
           <title>Kocka-Sík-Függvényes Szövegértelmezés Elemzési Jelentés</title>
           <meta charset="UTF-8">
           <style>
               body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }}
               h1, h2, h3 {{ color: #2c3e50; }}
               .container {{ max-width: 1200px; margin: 0 auto; }}
               .summary-box {{ background-color: #f9f9f9; border: 1px solid #ddd; 
                             padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
               .emotion-meter {{ height: 20px; background: linear-gradient(to right, #e74c3c, #f1c40f, #2ecc71);
                               border-radius: 10px; position: relative; margin-bottom: 40px; }}
               .emotion-marker {{ position: absolute; width: 10px; height: 30px; background-color: #2c3e50;
                                transform: translateX(-50%); top: -5px; }}
               .emotion-value {{ position: absolute; transform: translateX(-50%); top: 25px; font-weight: bold; }}
               .emotion-label {{ position: absolute; left: 0; top: -25px; font-weight: bold; }}
               .sentence-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
               .sentence-table th, .sentence-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
               .sentence-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
               .sentence-table th {{ background-color: #2c3e50; color: white; }}
               .visual-section {{ margin-bottom: 40px; }}
               .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }}
               .image-container {{ margin: 20px 0; text-align: center; }}
               .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
               .interactive-link {{ display: block; margin: 20px 0; padding: 10px; background-color: #3498db;
                                 color: white; text-align: center; border-radius: 5px; text-decoration: none; }}
               .interactive-link:hover {{ background-color: #2980b9; }}
               .category-badge {{ display: inline-block; padding: 5px 10px; background-color: #3498db;
                               color: white; border-radius: 15px; font-weight: bold; }}
               .context-info {{ margin-top: 20px; background-color: #eaf2f8; padding: 15px; border-radius: 5px; }}
               .confidence-pill {{ display: inline-block; font-size: 0.8rem; background-color: rgba(0,0,0,0.1);
                                padding: 2px 6px; border-radius: 10px; margin-left: 5px; }}
               .detection-method {{ font-size: 0.9rem; color: #555; margin-top: 5px; }}
           </style>
       </head>
       <body>
           <div class="container">
               <h1>Érzelmi térmetrika - bővebb elemzés</h1>
               
               <div class="summary-box">
                   <h2>Elemzési összefoglaló</h2>
                   <p><strong>Elemzés időpontja:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                   <p><strong>Elemzett szöveg:</strong> {analysis_results['text'][:200] + '...' if len(analysis_results['text']) > 200 else analysis_results['text']}</p>
                   <p><strong>Mondatok száma:</strong> {len(analysis_results['sentences'])}</p>
                   <p><strong>Érzelmi kategória:</strong> <span class="category-badge">{category}</span> <strong>Konfidencia:</strong> {confidence}</p>
                   
                   {context_html}
                   
                   <h3>Érzelmi dimenziók</h3>
                   
                   <div>
                       <div class="emotion-label">Valencia (pozitív-negatív dimenzió)</div>
                       <div class="emotion-meter">
                           <div class="emotion-marker" style="left: {valence_pos}%;"></div>
                           <div class="emotion-value" style="left: {valence_pos}%;">{v}</div>
                       </div>
                   </div>
                   
                   <div>
                       <div class="emotion-label">Arousal (aktivitás-passzivitás dimenzió)</div>
                       <div class="emotion-meter">
                           <div class="emotion-marker" style="left: {arousal_pos}%;"></div>
                           <div class="emotion-value" style="left: {arousal_pos}%;">{a}</div>
                       </div>
                   </div>
                   
                   <div>
                       <div class="emotion-label">Dominancia (uralom-alávetettség dimenzió)</div>
                       <div class="emotion-meter">
                           <div class="emotion-marker" style="left: {dominance_pos}%;"></div>
                           <div class="emotion-value" style="left: {dominance_pos}%;">{d}</div>
                       </div>
                   </div>
               </div>
               
               <div class="visual-section">
                   <h2>Érzelmi tér vizualizációja</h2>
                   <div class="image-container">
                       <img src="emotion_cube_3d.png" alt="Érzelmi kocka 3D megjelenítése">
                       <p>Érzelmi kocka 3D megjelenítése a szöveg teljes érzelmi pontjával</p>
                   </div>
                   
                   <a href="interactive_emotion_plot.html" class="interactive-link" target="_blank">
                       Interaktív 3D érzelmi térkép megnyitása
                   </a>
               </div>
               
               <div class="visual-section">
                   <h2>Érzelmi dimenziók vetületei</h2>
                   <div class="image-container">
                       <img src="trajectory_2d_valence_arousal.png" alt="Valencia-Arousal 2D vetület">
                       <p>Valencia-Arousal síkra vetített érzelmi változások</p>
                   </div>
                   
                   <div class="image-container">
                       <img src="trajectory_2d_valence_dominance.png" alt="Valencia-Dominancia 2D vetület">
                       <p>Valencia-Dominancia síkra vetített érzelmi változások</p>
                   </div>
                   
                   <div class="image-container">
                       <img src="trajectory_2d_arousal_dominance.png" alt="Arousal-Dominancia 2D vetület">
                       <p>Arousal-Dominancia síkra vetített érzelmi változások</p>
                   </div>
               </div>
               
               <div class="visual-section">
                   <h2>Érzelmi változások időben</h2>
                   <div class="image-container">
                       <img src="emotion_time_series.png" alt="Érzelmi idősoros ábra">
                       <p>Az érzelmi dimenziók időbeli változásának grafikonja</p>
                   </div>
               </div>
               
               <h2>Mondatonkénti érzelmi elemzés</h2>
               {sentences_html}
               
               <div class="footer">
                   <p>Az Érzelmi térmetrika algoritmus és vizualizációs logika szerint generálva</p>
                   <p>&copy; {datetime.now().year}</p>
               </div>
           </div>
       </body>
       </html>
       """
       
       with open(f"{output_dir}/analysis_report.html", 'w', encoding='utf-8') as f:
           f.write(html_content)


# Demó futtatása példaszöveggel
def run_demo():
   """Demó futtatása minta szöveggel"""
   
   # Példa szöveg
   sample_text = """
   A kocka-sík-függvényes modell egy izgalmas és újszerű megközelítés az érzelmek elemzésére.
   Ez a módszer lehetővé teszi számunkra, hogy az emberi érzelmek komplexitását háromdimenziós térben ábrázoljuk.
   Természetesen vannak korlátai is, de a hagyományos módszerekhez képest sokkal árnyaltabb képet kaphatunk.
   Nagyon örülök, hogy sikerült implementálni ezt a rendszert!
   Az érzelmi dimenziók közötti kapcsolatok feltárása különösen fontos lehet a pszichológiai kutatásokban és a mesterséges intelligencia fejlesztésében.
   Kíváncsi vagyok, hogy a jövőben milyen további fejlesztésekkel lehet majd tovább finomítani az elemzést.
   """
   
   # Demo indítása
   demo = EmotionAnalysisDemo()
   results = demo.analyze_text(sample_text)
   demo.visualize_text_analysis(results, output_dir='./emotion_analysis_output')
   
   print("\nDemó futtatása befejeződött. Tekintse meg az eredményeket az 'emotion_analysis_output' könyvtárban!")
   print("A teljes elemzési jelentés megtekinthető az 'emotion_analysis_output/analysis_report.html' fájlban.")


if __name__ == "__main__":
   run_demo()