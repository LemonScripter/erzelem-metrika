# ====================================================================
# 1. ADATFELDOLGOZÓ MODUL - SZÖVEGELŐFELDOLGOZÁS
# ====================================================================
# Fájlnév: text_preprocessor.py
# Verzió: 1.1
# Leírás: Szöveges adatok előfeldolgozása és vektorizálása
# ====================================================================

import spacy
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

class TextPreprocessor:
    """Szövegek előfeldolgozása és vektorizálása a kocka-sík-függvényes modellhez"""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Inicializálja a szövegelőfeldolgozót
        
        Args:
            model_name: A használandó Hugging Face transzformer modell neve
        """
        # SpaCy magyar nyelvi modell betöltése vagy angol használata helyette
        try:
            # Először megpróbáljuk betölteni a kisebb magyar modellt
            self.nlp = spacy.load('hu_core_news_sm')
            print("Magyar nyelvi modell (hu_core_news_sm) sikeresen betöltve.")
        except OSError:
            try:
                # Ha nincs magyar, akkor megpróbáljuk betölteni az angol modellt
                print("Magyar nyelvi modell nem található, angol modell használata helyette...")
                self.nlp = spacy.load('en_core_web_sm')
                print("Angol nyelvi modell (en_core_web_sm) sikeresen betöltve.")
            except OSError:
                # Ha egyik sincs, akkor letöltjük az angol modellt
                print("Nyelvi modell letöltése (angol)...")
                spacy.cli.download('en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
                print("Angol nyelvi modell letöltve és betöltve.")
            
        # Transformer modell és tokenizer betöltése
        print(f"Transformer modell betöltése: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("Transformer modell sikeresen betöltve.")
        
        # Magyar stop szavak manuális definiálása (ha nincs magyar modell)
        self.stop_words = set([
            'a', 'az', 'és', 'hogy', 'van', 'volt', 'lesz', 'nem', 'igen',
            'de', 'ha', 'már', 'még', 'csak', 'el', 'be', 'ki', 'fel', 'le',
            'meg', 'át', 'rá', 'egy', 'kettő', 'három', 'négy', 'öt',
            'ezt', 'azt', 'ezek', 'azok', 'aki', 'ami', 'amely', 'mint',
            'vagy', 'illetve', 'ilyen', 'olyan', 'ezt', 'azt', 'is'
        ])
        
        # Ha van SpaCy modell, akkor frissítjük a stop szavakat
        if hasattr(self.nlp, 'Defaults') and hasattr(self.nlp.Defaults, 'stop_words'):
            self.stop_words.update(self.nlp.Defaults.stop_words)
        
    def clean_text(self, text: str) -> str:
        """
        Szöveg tisztítása (speciális karakterek, felesleges szóközök eltávolítása)
        
        Args:
            text: Bemeneti szöveg
            
        Returns:
            Tisztított szöveg
        """
        # Speciális karakterek és felesleges whitespace eltávolítása
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[Dict]:
        """
        Szöveg tokenizálása SpaCy segítségével
        
        Args:
            text: Bemeneti szöveg
            
        Returns:
            A tokenizált szöveg, gazdag nyelvi annotációkkal
        """
        doc = self.nlp(self.clean_text(text))
        tokens = []
        
        for token in doc:
            token_info = {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'is_stop': token.is_stop,
                'is_punct': token.is_punct,
                'sentiment': token.sentiment if hasattr(token, 'sentiment') else 0.0
            }
            tokens.append(token_info)
            
        return tokens
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Szöveg mondatokra bontása
        
        Args:
            text: Bemeneti szöveg
            
        Returns:
            Mondatok listája
        """
        doc = self.nlp(self.clean_text(text))
        return [sent.text for sent in doc.sents]
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Szöveg vektorizálása a Transformer modell segítségével
        
        Args:
            text: Bemeneti szöveg
            
        Returns:
            Szövegembeeding (vektorreprezentáció)
        """
        # Tisztított szöveg tokenizálása a transformer modellhez
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Embeddings kiszámítása
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # [CLS] token utolsó rejtett állapotának használata reprezentációként
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]  # Batch dimenzió eltávolítása
    
    def get_emotion_dimensions(self, text: str) -> Dict[str, float]:
        """
        Érzelmi dimenziók (valencia, arousal, dominancia) kiszámítása
        
        Args:
            text: Bemeneti szöveg
            
        Returns:
            Érzelmi dimenziók szótára
        """
        # Embedding alapú érzelmi értékek számítása
        # Ez egy leegyszerűsített implementáció, valós esetben érdemes
        # egy külön osztályozót betanítani erre a feladatra
        embedding = self.get_embeddings(text)
        
        # PCA-szerű leképezés a 3D érzelmi térbe
        # (implementációs egyszerűsítés)
        emotion_dims = {
            'valence': self._scale_to_range(np.mean(embedding[:100])),   # Első 100 dimenzió átlaga
            'arousal': self._scale_to_range(np.std(embedding[100:200])), # Következő 100 dimenzió szórása  
            'dominance': self._scale_to_range(np.max(embedding[200:300]))  # Következő 100 dimenzió maximuma
        }
        
        return emotion_dims
    
    def _scale_to_range(self, value: float, target_min: float = 0.0, target_max: float = 1.0) -> float:
        """
        Érték skálázása adott tartományba
        
        Args:
            value: Skálázandó érték
            target_min: Célskála minimuma
            target_max: Célskála maximuma
            
        Returns:
            Skálázott érték
        """
        # Heurisztikus skálázás a [-1, 1] tartományból [0, 1] tartományba
        scaled = (np.tanh(value) + 1) / 2
        
        # Célskálára igazítás
        return target_min + scaled * (target_max - target_min)