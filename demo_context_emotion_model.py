# ====================================================================
# MODELL HASZNÁLATA - DEMONSTRÁCIÓ
# ====================================================================
# Fájlnév: demo_context_emotion_model.py
# Verzió: 1.0
# Leírás: Betanított kontextuális érzelmi modell használata
# ====================================================================

from train_context_emotion_model import ContextualEmotionModel
from emotion_cube_modeler import EmotionCubeModeler
from emotion_visualizer import EmotionVisualizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_with_context():
    """Kontextus-specifikus érzelmi elemzés demonstrálása"""
    
    # Modell betöltése
    model = ContextualEmotionModel.load("models/context_emotion_model")
    
    # Érzelmi kocka modellező inicializálása a kategorizáláshoz
    cube_modeler = EmotionCubeModeler()
    visualizer = EmotionVisualizer(cube_modeler)
    
    # Tesztszövegek különböző kontextussal
    test_examples = [
        {
            "text": "A projekt megvalósítása komoly kihívásokkal járt, de végül sikerült.",
            "contexts": ["business", "technical", "personal"]
        },
        {
            "text": "A befektetéseink jelentős veszteséget termeltek az elmúlt negyedévben.",
            "contexts": ["business", "personal"]
        },
        {
            "text": "Hosszú kutatómunka után végre áttörést értünk el.",
            "contexts": ["academic", "business", "personal"]
        }
    ]
    
    # Elemzés és eredmények összegyűjtése
    all_results = []
    
    for example in test_examples:
        text = example["text"]
        contexts = example["contexts"]
        
        # Minden kontextusban elemezzük ugyanazt a szöveget
        for context in contexts:
            results = model.predict([text], [context])[0]
            
            # Kategorizálás az érzelmi térben
            category_info = cube_modeler.classify_emotion(
                results['valence'], 
                results['arousal'], 
                results['dominance']
            )
            
            # Eredmények kiegészítése
            results['emotion_category'] = category_info['label']
            results['confidence'] = category_info['confidence']
            all_results.append(results)
    
    # Eredmények megjelenítése táblázatban
    df = pd.DataFrame(all_results)
    display_columns = ['text', 'context', 'emotion_category', 'valence', 'arousal', 'dominance']
    print(df[display_columns].to_string(index=False))
    
    # Eredmények vizualizálása grafikonon
    for text in set(df['text']):
        text_df = df[df['text'] == text]
        
        # Radar diagram az érzelmi dimenziókról kontextus szerint
        plot_emotion_radar(text_df)
        
        # 3D vizualizáció
        plot_3d_emotion_context(text_df, visualizer)
        
def plot_emotion_radar(data):
    """Radar diagram az érzelmi dimenziókról kontextus szerint"""
    
    contexts = data['context'].tolist()
    values = []
    
    for _, row in data.iterrows():
        values.append([
            row['valence'],
            row['arousal'],
            row['dominance'],
        ])
    
    # Radar diagram létrehozása
    labels = ['Valencia', 'Arousal', 'Dominancia']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Kör lezárása
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, context in enumerate(contexts):
        values_for_plot = values[i] + values[i][:1]  # Kör lezárása
        ax.plot(angles, values_for_plot, 'o-', linewidth=2, label=context)
        ax.fill(angles, values_for_plot, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Szöveg érzelmi profilja különböző kontextusokban\n\"{data.iloc[0]['text']}\"", 
                y=1.1, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.show()
    
def plot_3d_emotion_context(data, visualizer):
    """3D megjelenítés az érzelmi térben különböző kontextusokkal"""
    
    # Érzelmi pontok előkészítése
    emotion_points = []
    
    for _, row in data.iterrows():
        emotion_points.append((
            row['valence'],
            row['arousal'],
            row['dominance'],
            f"Kontextus: {row['context']}"
        ))
    
    # 3D ábra létrehozása
    fig = visualizer.create_interactive_3d_plot(emotion_points=emotion_points)
    fig.update_layout(title=f"Szöveg érzelmi pozíciója különböző kontextusokban<br>\"{data.iloc[0]['text']}\"")
    fig.show()

# Demonstráció futtatása
if __name__ == "__main__":
    analyze_with_context()