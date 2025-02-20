# ====================================================================
# MODELL BETANÍTÁS - ADATKÉSZLET LÉTREHOZÁSA
# ====================================================================
# Fájlnév: create_emotional_dataset.py
# Verzió: 1.0
# Leírás: Kontextuális érzelmi adatkészlet létrehozása betanításhoz
# ====================================================================

import pandas as pd
import json
from emotion_analysis_demo import EmotionAnalysisDemo
from tqdm import tqdm
import os

def create_training_dataset(input_texts_file, output_file, contexts=None):
    """
    Létrehoz egy annotált adatkészletet a modell betanításához
    
    Args:
        input_texts_file: JSON fájl a bemeneti szövegekkel
        output_file: Kimeneti CSV fájl
        contexts: Opcionális kontextus-címkék szótára
    """
    # Kontextusok definiálása, ha nincs megadva
    if contexts is None:
        contexts = {
            "business": ["üzleti", "céges", "vállalati", "gazdasági"],
            "personal": ["személyes", "magán", "családi"],
            "technical": ["technikai", "műszaki", "informatikai"],
            "academic": ["tudományos", "akadémiai", "oktatási"],
            "social_media": ["közösségi média", "poszt", "megosztás"]
        }
    
    # Bemeneti szövegek betöltése
    with open(input_texts_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Érzelmi elemző inicializálása
    analyzer = EmotionAnalysisDemo()
    
    # Adatkészlet létrehozása
    dataset = []
    
    for item in tqdm(input_data, desc="Szövegek feldolgozása"):
        text = item["text"]
        context_label = item.get("context", "unknown")
        
        # Szöveg érzelmi elemzése
        results = analyzer.analyze_text(text)
        
        # Érzelmi értékek kinyerése
        valence = results['overall_emotions']['valence']
        arousal = results['overall_emotions']['arousal']
        dominance = results['overall_emotions']['dominance']
        category = results['overall_cube_info']['label']
        
        # Adatsor hozzáadása
        dataset.append({
            "text": text,
            "context": context_label,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "emotion_category": category
        })
    
    # Adatkészlet mentése
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    print(f"Adatkészlet sikeresen létrehozva: {len(dataset)} példa, elmentve: {output_file}")
    
    # Statisztikák
    print("\nAdatkészlet statisztikák:")
    print(f"Kontextusok eloszlása:\n{df['context'].value_counts()}")
    print(f"Érzelmi kategóriák eloszlása:\n{df['emotion_category'].value_counts()}")
    
    return df