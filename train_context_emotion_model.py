# ====================================================================
# MODELL BETANÍTÁS - KONTEXTUÁLIS ÉRZELMI MODELL
# ====================================================================
# Fájlnév: train_context_emotion_model.py
# Verzió: 1.0
# Leírás: Kontextusfüggő érzelmi modell betanítása
# ====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
import pickle
import os

class ContextualEmotionModel:
    """
    Kontextus-specifikus érzelmi modell osztály
    """
    
    def __init__(self, base_model_name="bert-base-multilingual-cased"):
        """
        Inicializálja a modellt
        
        Args:
            base_model_name: Az alapul szolgáló nyelvi modell neve
        """
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.bert_model = TFAutoModel.from_pretrained(base_model_name)
        self.context_encoder = None
        self.model = None
        
    def prepare_data(self, dataset_file):
        """
        Adatok előkészítése a betanításhoz
        
        Args:
            dataset_file: Adatkészlet CSV fájl
            
        Returns:
            Betanításra előkészített adatok
        """
        # Adatkészlet betöltése
        df = pd.read_csv(dataset_file)
        
        # Kontextusok one-hot kódolása
        self.context_encoder = OneHotEncoder(sparse=False)
        context_encoded = self.context_encoder.fit_transform(df[['context']])
        
        # Szövegek tokenizálása
        texts = df['text'].tolist()
        encodings = self.tokenizer(texts, truncation=True, padding=True, 
                                  max_length=128, return_tensors="tf")
        
        # Célváltozók
        y_valence = df['valence'].values
        y_arousal = df['arousal'].values
        y_dominance = df['dominance'].values
        
        # Train-test split
        indices = np.arange(len(texts))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Adatok szétválasztása
        X_train = {
            'input_ids': encodings['input_ids'][train_indices],
            'attention_mask': encodings['attention_mask'][train_indices],
            'context': context_encoded[train_indices]
        }
        X_test = {
            'input_ids': encodings['input_ids'][test_indices],
            'attention_mask': encodings['attention_mask'][test_indices],
            'context': context_encoded[test_indices]
        }
        
        y_train = {
            'valence': y_valence[train_indices],
            'arousal': y_arousal[train_indices],
            'dominance': y_dominance[train_indices]
        }
        y_test = {
            'valence': y_valence[test_indices],
            'arousal': y_arousal[test_indices],
            'dominance': y_dominance[test_indices]
        }
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, context_dim):
        """
        Modell felépítése
        
        Args:
            context_dim: Kontextus dimenziója (one-hot kódolás után)
        """
        # BERT bemenetek
        input_ids = Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')
        
        # Kontextus bemenet
        context_input = Input(shape=(context_dim,), dtype=tf.float32, name='context')
        
        # BERT kimenet
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = bert_output[:, 0, :]  # [CLS] token állapotának használata
        
        # Szöveg és kontextus kombinálása
        combined = Concatenate()([sequence_output, context_input])
        
        # Közös rejtett rétegek
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Érzelmi dimenziók kimeneti rétegei
        valence_output = Dense(1, activation='sigmoid', name='valence')(x)
        arousal_output = Dense(1, activation='sigmoid', name='arousal')(x)
        dominance_output = Dense(1, activation='sigmoid', name='dominance')(x)
        
        # Modell összeállítása
        self.model = Model(
            inputs=[input_ids, attention_mask, context_input],
            outputs=[valence_output, arousal_output, dominance_output]
        )
        
        # Modell kompilálása
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss={
                'valence': 'mse',
                'arousal': 'mse',
                'dominance': 'mse'
            },
            metrics={
                'valence': ['mae', 'mse'],
                'arousal': ['mae', 'mse'],
                'dominance': ['mae', 'mse']
            }
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_test, y_test, 
             epochs=10, batch_size=16, model_save_path="context_emotion_model"):
        """
        Modell betanítása
        
        Args:
            X_train, y_train: Tanító adatok
            X_test, y_test: Validációs adatok
            epochs: Tanítási epochok száma
            batch_size: Batch méret
            model_save_path: Modell mentési útvonal
        """
        if self.model is None:
            self.build_model(X_train['context'].shape[1])
            
        # Korai leállítás
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        # Tanítás
        history = self.model.fit(
            x=[X_train['input_ids'], X_train['attention_mask'], X_train['context']],
            y=[y_train['valence'], y_train['arousal'], y_train['dominance']],
            validation_data=(
                [X_test['input_ids'], X_test['attention_mask'], X_test['context']],
                [y_test['valence'], y_test['arousal'], y_test['dominance']]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        # Modell mentése
        os.makedirs(model_save_path, exist_ok=True)
        self.model.save(f"{model_save_path}/keras_model")
        
        # Tokenizer és encoder mentése
        with open(f"{model_save_path}/tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        with open(f"{model_save_path}/context_encoder.pkl", 'wb') as f:
            pickle.dump(self.context_encoder, f)
            
        # Konfiguráció mentése
        config = {
            'base_model_name': self.base_model_name
        }
        with open(f"{model_save_path}/config.json", 'w') as f:
            json.dump(config, f)
            
        return history
    
    def predict(self, texts, contexts):
        """
        Érzelmi értékek előrejelzése
        
        Args:
            texts: Bementi szövegek listája
            contexts: Kontextusok listája
            
        Returns:
            Előrejelzett érzelmi értékek
        """
        # Szövegek tokenizálása
        encodings = self.tokenizer(texts, truncation=True, padding=True, 
                                  max_length=128, return_tensors="tf")
        
        # Kontextusok kódolása
        context_data = np.array([[c] for c in contexts])
        context_encoded = self.context_encoder.transform(context_data)
        
        # Előrejelzés
        valence_pred, arousal_pred, dominance_pred = self.model.predict(
            [encodings['input_ids'], encodings['attention_mask'], context_encoded]
        )
        
        # Eredmények összegyűjtése
        results = []
        for i in range(len(texts)):
            results.append({
                'text': texts[i],
                'context': contexts[i],
                'valence': float(valence_pred[i][0]),
                'arousal': float(arousal_pred[i][0]),
                'dominance': float(dominance_pred[i][0])
            })
            
        return results
    
    @classmethod
    def load(cls, model_path):
        """
        Betanított modell betöltése
        
        Args:
            model_path: Modell útvonal
            
        Returns:
            Betöltött ContextualEmotionModel
        """
        # Konfiguráció betöltése
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
            
        # Modell inicializálása
        instance = cls(base_model_name=config['base_model_name'])
        
        # Tokenizer betöltése
        with open(f"{model_path}/tokenizer.pkl", 'rb') as f:
            instance.tokenizer = pickle.load(f)
            
        # Context encoder betöltése
        with open(f"{model_path}/context_encoder.pkl", 'rb') as f:
            instance.context_encoder = pickle.load(f)
            
        # Keras modell betöltése
        instance.model = tf.keras.models.load_model(f"{model_path}/keras_model")
        
        return instance