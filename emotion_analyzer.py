# ====================================================================
# 1. ADATFELDOLGOZÓ MODUL - ÉRZELMI DIMENZIÓ ANALÍZIS
# ====================================================================
# Fájlnév: emotion_analyzer.py
# Verzió: 1.0
# Leírás: Érzelmi dimenziók elemzése és finomhangolása
# ====================================================================

import numpy as np
from typing import List, Dict, Tuple, Union
from transformers import pipeline
from text_preprocessor import TextPreprocessor

class EmotionAnalyzer:
    """
    Érzelmi dimenziók elemzésére specializált osztály,
    amely a TextPreprocessor alaposztályt egészíti ki
    """
    
    def __init__(self, lexicon_path: str = None, use_transformer: bool = True):
        """
        Inicializálja az érzelmi elemzőt
        
        Args:
            lexicon_path: Opcionális érzelmi lexikon útvonala
            use_transformer: Transformer modell használata (ajánlott)
        """
        self.preprocessor = TextPreprocessor()
        self.use_transformer = use_transformer
        
        # Érzelmi lexikon betöltése, ha meg van adva
        self.emotion_lexicon = {}
        if lexicon_path:
            self._load_emotion_lexicon(lexicon_path)
            
        # Transformer alapú klasszifikációs pipeline
        if use_transformer:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except Exception as e:
                print(f"Hiba a sentiment pipeline inicializálásakor: {e}")
                print("Alternatív modell használata...")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis"
                )
            
    def _load_emotion_lexicon(self, lexicon_path: str) -> None:
        """
        Érzelmi lexikon betöltése
        
        Args:
            lexicon_path: A lexikon fájl elérési útja
        """
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        word, valence, arousal, dominance = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                        self.emotion_lexicon[word] = {
                            'valence': valence,
                            'arousal': arousal,
                            'dominance': dominance
                        }
            print(f"Érzelmi lexikon betöltve: {len(self.emotion_lexicon)} szó")
        except Exception as e:
            print(f"Hiba az érzelmi lexikon betöltésekor: {e}")
            
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Teljes érzelmi elemzés végrehajtása
        
        Args:
            text: Elemzendő szöveg
            
        Returns:
            Érzelmi dimenziók és értékek
        """
        # Alap érzelmi dimenziók kiszámítása a preprocessorral
        base_emotions = self.preprocessor.get_emotion_dimensions(text)
        
        # Lexikon alapú elemzés (ha van betöltve lexikon)
        lexicon_emotions = self._analyze_with_lexicon(text) if self.emotion_lexicon else None
        
        # Transformer alapú elemzés (ha engedélyezve van)
        transformer_emotions = self._analyze_with_transformer(text) if self.use_transformer else None
        
        # Az eredmények kombinálása (súlyozott átlag)
        combined_emotions = self._combine_emotion_sources(
            base=base_emotions, 
            lexicon=lexicon_emotions,
            transformer=transformer_emotions
        )
        
        return combined_emotions
        
    def _analyze_with_lexicon(self, text: str) -> Dict[str, float]:
        """
        Érzelmi lexikon alapú elemzés
        
        Args:
            text: Elemzendő szöveg
            
        Returns:
            Érzelmi dimenziók és értékek
        """
        tokens = self.preprocessor.tokenize(text)
        emotions = {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        word_count = 0
        
        for token in tokens:
            if token['is_stop'] or token['is_punct']:
                continue
                
            word = token['lemma'].lower()
            if word in self.emotion_lexicon:
                for dim in emotions.keys():
                    emotions[dim] += self.emotion_lexicon[word][dim]
                word_count += 1
                
        # Átlagolás, ha volt találat
        if word_count > 0:
            for dim in emotions.keys():
                emotions[dim] /= word_count
                
        return emotions
    
    def _analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """
        Transformer modell alapú érzelmi elemzés
        
        Args:
            text: Elemzendő szöveg
            
        Returns:
            Érzelmi dimenziók és értékek
        """
        try:
            # Mondatokra bontás és elemzés
            sentences = self.preprocessor.split_sentences(text)
            results = []
            
            for sentence in sentences:
                if sentence.strip():
                    sentiment = self.sentiment_pipeline(sentence)[0]
                    results.append({
                        'label': sentiment['label'],
                        'score': sentiment['score']
                    })
            
            # Eredmények aggregálása
            if not results:
                return None
                
            # Pozitív és negatív értékek számítása
            positive_scores = [r['score'] for r in results if r['label'] == 'POSITIVE']
            negative_scores = [r['score'] for r in results if r['label'] == 'NEGATIVE']
            neutral_scores = [r['score'] for r in results if r['label'] == 'NEUTRAL']
            
            # Valencia (pozitív-negatív dimenzió)
            if positive_scores and negative_scores:
                valence = np.mean(positive_scores) - np.mean(negative_scores)
                valence = (valence + 1) / 2  # Skálázás [0, 1] tartományba
            elif positive_scores:
                valence = np.mean(positive_scores)
            elif negative_scores:
                valence = 1 - np.mean(negative_scores)
            else:
                valence = 0.5  # Semleges
                
            # Arousal (érzelmi intenzitás) - magas confidence = magas arousal
            all_confidences = [r['score'] for r in results if r['label'] != 'NEUTRAL']
            arousal = np.mean(all_confidences) if all_confidences else 0.5
            
            # Dominancia (egyszerűsített becsléssel)
            dominance = 0.5 + (valence - 0.5) * 0.7  # Pozitív hangulat általában magasabb dominanciával jár
            
            return {
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance
            }
                
        except Exception as e:
            print(f"Hiba a transformer alapú érzelmi elemzés során: {e}")
            return None
            
    def _combine_emotion_sources(self, base: Dict[str, float], 
                                lexicon: Dict[str, float] = None,
                                transformer: Dict[str, float] = None) -> Dict[str, float]:
        """
        Különböző érzelmi elemzési források kombinálása
        
        Args:
            base: Alap (embedding) érzelmi értékek
            lexicon: Lexikon alapú érzelmi értékek
            transformer: Transformer alapú érzelmi értékek
            
        Returns:
            Kombinált érzelmi dimenziók
        """
        # Súlyok az egyes forrásokhoz
        weights = {
            'base': 0.3,
            'lexicon': 0.3 if lexicon else 0,
            'transformer': 0.4 if transformer else 0
        }
        
        # Súlyok normalizálása
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Kombinált értékek számítása
        combined = {}
        for dim in ['valence', 'arousal', 'dominance']:
            combined[dim] = (
                weights['base'] * base[dim] +
                (weights['lexicon'] * lexicon[dim] if lexicon else 0) +
                (weights['transformer'] * transformer[dim] if transformer else 0)
            )
            
        return combined