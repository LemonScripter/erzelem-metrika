# ====================================================================
# KONTEXTUS FELISMERŐ MODUL
# ====================================================================
# Fájlnév: context_detector.py
# Verzió: 1.0
# Leírás: Automatikus kontextus-felismerő rendszer
# ====================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import re
import json

class ContextDetector:
    """Szöveg kontextusának automatikus felismerése"""
    
    def __init__(self, model_path="models/context_detector.pkl", 
                default_contexts=None):
        """
        Inicializálja a kontextus-felismerő rendszert
        
        Args:
            model_path: Betanított modell útvonala (ha létezik)
            default_contexts: Alapértelmezett kontextusok szótára
        """
        # Alapértelmezett kontextusok és kulcsszavak
        self.default_contexts = default_contexts or {
            "general": ["általános", "vegyes", "hétköznapi"],
            "business": ["cég", "vállalat", "profit", "üzlet", "befektetés", "bevétel", 
                         "pénzügy", "részvény", "piac", "kereskedelem", "gazdaság", "stratégia"],
            "personal": ["család", "barát", "érzés", "szeretet", "magán", "életem", "otthon",
                         "emlék", "kapcsolat", "hobbi", "ünnep", "szerelem", "gyerek"],
            "academic": ["tanulmány", "kutatás", "tudomány", "egyetem", "elmélet", "publikáció",
                         "oktatás", "tanulás", "kísérlet", "hipotézis", "eredmény", "elemzés"],
            "social_media": ["poszt", "megosztás", "like", "követő", "komment", "hashtag",
                            "platform", "online", "közösségi", "profil", "tartalom", "influenszer"],
            "technical": ["rendszer", "szoftver", "hardver", "kód", "fejlesztés", "technológia",
                         "implementáció", "verzió", "bug", "javítás", "interfész", "funkció"],
            "medical": ["betegség", "gyógyszer", "kezelés", "orvos", "páciens", "diagnózis",
                       "tünet", "egészség", "kórház", "terápia", "gyógyulás", "vizsgálat"],
            "political": ["kormány", "választás", "párt", "politika", "szavazás", "parlament",
                         "törvény", "képviselő", "ellenzék", "vita", "kampány", "hatalom"]
        }
        
        # Betanított modell betöltése, ha létezik
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.model = model_data['model']
                    self.contexts = model_data.get('contexts', list(self.default_contexts.keys()))
                    self.use_ml_model = True
                    print(f"Kontextus-felismerő modell betöltve: {len(self.contexts)} kategória")
            except Exception as e:
                print(f"Hiba a kontextus-felismerő modell betöltésekor: {e}")
                self.use_ml_model = False
        else:
            self.use_ml_model = False
            self.contexts = list(self.default_contexts.keys())
            print("Kontextus-felismerő modell nem található, egyszerű kulcsszó-alapú felismerés lesz használva")
            
    def detect_context(self, text, confidence_threshold=0.3):
        """
        Szöveg kontextusának automatikus felismerése
        
        Args:
            text: Elemzendő szöveg
            confidence_threshold: Konfidencia küszöb ML modellhez
            
        Returns:
            dict: Felismert kontextus és konfidencia érték
        """
        if self.use_ml_model:
            return self._detect_with_ml(text, confidence_threshold)
        else:
            return self._detect_with_keywords(text)
            
    def _detect_with_ml(self, text, confidence_threshold):
        """ML modell alapú kontextus-felismerés"""
        X = self.vectorizer.transform([text])
        prediction_proba = self.model.predict_proba(X)[0]
        
        # Legvalószínűbb kontextus és annak konfidenciája
        max_idx = prediction_proba.argmax()
        confidence = prediction_proba[max_idx]
        
        if confidence >= confidence_threshold:
            context = self.contexts[max_idx]
        else:
            # Ha a konfidencia túl alacsony, használjunk kulcsszó-alapú felismerést
            keyword_result = self._detect_with_keywords(text)
            context = keyword_result['context']
            confidence = keyword_result['confidence']
            
        return {
            'context': context,
            'confidence': float(confidence),
            'method': 'machine_learning',
            'alternatives': self._get_alternative_contexts(prediction_proba)
        }
        
    def _detect_with_keywords(self, text):
        """Kulcsszó-alapú kontextus-felismerés"""
        text = text.lower()
        context_scores = {"general": 1}  # Alapértelmezett pontszám
        
        # Szavak kinyerése és tisztítása
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        
        # Kulcsszavak keresése
        for context, keywords in self.default_contexts.items():
            matches = sum(1 for word in words if word in keywords or any(kw in word for kw in keywords))
            if matches > 0:
                # Pontszám: találatok száma / összes szó, max 0.95
                context_scores[context] = min(0.95, (matches / total_words) * 3)
        
        # Legmagasabb pontszámú kontextus választása
        best_context = max(context_scores, key=context_scores.get)
        confidence = context_scores[best_context]
        
        # Alternatív kontextusok
        alternatives = []
        for context, score in sorted(context_scores.items(), key=lambda x: x[1], reverse=True):
            if context != best_context and score > 0.1:
                alternatives.append({'context': context, 'confidence': score})
        
        return {
            'context': best_context,
            'confidence': confidence,
            'method': 'keyword',
            'alternatives': alternatives[:3]  # Top 3 alternatíva
        }
    
    def _get_alternative_contexts(self, probabilities):
        """Alternatív kontextusok a valószínűségek alapján"""
        # Rendezzük a kontextusokat valószínűség szerint
        context_probs = [(self.contexts[i], float(prob)) for i, prob in enumerate(probabilities)]
        sorted_contexts = sorted(context_probs, key=lambda x: x[1], reverse=True)
        
        # Top 3 alternatíva (az első után)
        alternatives = []
        for context, confidence in sorted_contexts[1:4]:
            if confidence > 0.05:  # Csak azok, amik legalább 5% valószínűségűek
                alternatives.append({'context': context, 'confidence': confidence})
                
        return alternatives
        
    def save_detection_result(self, text, result, output_dir="context_analysis"):
        """Kontextus-felismerés eredményének mentése"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Egyedi fájlnév generálása
        import hashlib
        import time
        text_hash = hashlib.md5(text[:50].encode()).hexdigest()[:8]
        timestamp = int(time.time())
        filename = f"{output_dir}/context_{result['context']}_{text_hash}_{timestamp}.json"
        
        # Eredmény összeállítása
        output = {
            'text': text[:200] + ('...' if len(text) > 200 else ''),
            'text_length': len(text),
            'detection_result': result,
            'timestamp': timestamp
        }
        
        # Mentés
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        return filename