# ====================================================================
# 2. MODELLEZŐ MODUL - ÉRZELMI KOCKA MODELLEZŐ
# ====================================================================
# Fájlnév: emotion_cube_modeler.py
# Verzió: 1.0
# Leírás: Érzelmek kategorizálása és modellezése a kocka-sík-függvény térben
# ====================================================================

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Set, Union
from emotional_space_model import EmotionCube, EmotionalPlane, EmotionalFunction
from sklearn.cluster import KMeans
import pickle
import os

class EmotionCubeModeler:
    """
    Érzelmi kockák kezelésére és modellezésére szolgáló osztály
    """
    
    def __init__(self, cube_size: float = 0.25):
        """
        Inicializálja az érzelmi kocka modellezőt
        
        Args:
            cube_size: Alapértelmezett kockaméret (1 egységnyi dimenzió hány részre legyen osztva)
        """
        self.cube_size = cube_size
        self.cubes: List[EmotionCube] = []
        self.planes: List[EmotionalPlane] = []
        self.functions: Dict[str, EmotionalFunction] = {}
        
        # Tensorflow modell a kockák osztályozásához
        self.model = None
        self.initialize_standard_cubes()
        self.initialize_standard_planes()
        
    def initialize_standard_cubes(self) -> None:
        """
        Standard érzelmi kockák inicializálása
        (8 kocka a tér 8 oktánsának megfelelően)
        """
        # Kockaméret alapján felosztjuk a [0,1]^3 teret
        steps = int(1.0 / self.cube_size)
        
        cube_id = 1
        labels = [
            "Ellenséges", "Stresszes", "Izgatott", "Lelkes",
            "Depresszív", "Nyugodt", "Elégedett", "Boldog"
        ]
        
        descriptions = [
            "Alacsony valencia, magas arousal, alacsony dominancia",
            "Alacsony valencia, magas arousal, magas dominancia",
            "Magas valencia, magas arousal, alacsony dominancia",
            "Magas valencia, magas arousal, magas dominancia",
            "Alacsony valencia, alacsony arousal, alacsony dominancia",
            "Alacsony valencia, alacsony arousal, magas dominancia",
            "Magas valencia, alacsony arousal, alacsony dominancia",
            "Magas valencia, alacsony arousal, magas dominancia"
        ]
        
        # A 8 fő érzelmi oktáns
        for i in range(2):  # Valence
            for j in range(2):  # Arousal
                for k in range(2):  # Dominance
                    idx = i*4 + j*2 + k
                    cube = EmotionCube(
                        id=f"base_{cube_id}",
                        valence_range=(i*0.5, (i+1)*0.5),
                        arousal_range=(j*0.5, (j+1)*0.5),
                        dominance_range=(k*0.5, (k+1)*0.5),
                        label=labels[idx],
                        description=descriptions[idx]
                    )
                    self.cubes.append(cube)
                    cube_id += 1
        
        # További részletesebb kockák hozzáadása
        for i in range(steps):
            for j in range(steps):
                for k in range(steps):
                    v_min, v_max = i*self.cube_size, (i+1)*self.cube_size
                    a_min, a_max = j*self.cube_size, (j+1)*self.cube_size
                    d_min, d_max = k*self.cube_size, (k+1)*self.cube_size
                    
                    # Csak ha nem része már a fő oktánsoknak
                    is_new = True
                    for base_cube in self.cubes[:8]:
                        v_mid, a_mid, d_mid = (v_min+v_max)/2, (a_min+a_max)/2, (d_min+d_max)/2
                        if base_cube.contains_point(v_mid, a_mid, d_mid):
                            is_new = False
                            break
                            
                    if is_new:
                        cube = EmotionCube(
                            id=f"detail_{cube_id}",
                            valence_range=(v_min, v_max),
                            arousal_range=(a_min, a_max),
                            dominance_range=(d_min, d_max)
                        )
                        self.cubes.append(cube)
                        cube_id += 1
    
    def initialize_standard_planes(self) -> None:
        """
        Standard érzelmi síkok inicializálása
        """
        # 1. Valencia-Arousal sík (z=0.5)
        va_plane = EmotionalPlane(
            name="valencia_arousal",
            coefficients=(0, 0, 1, -0.5),
            description="Valencia-Arousal sík (középső dominancia)"
        )
        
        # 2. Valencia-Dominancia sík (y=0.5)
        vd_plane = EmotionalPlane(
            name="valencia_dominance",
            coefficients=(0, 1, 0, -0.5),
            description="Valencia-Dominancia sík (középső arousal)"
        )
        
        # 3. Arousal-Dominancia sík (x=0.5)
        ad_plane = EmotionalPlane(
            name="arousal_dominance",
            coefficients=(1, 0, 0, -0.5),
            description="Arousal-Dominancia sík (középső valencia)"
        )
        
        # 4. Átlós sík (x + y + z = 1.5)
        diagonal_plane = EmotionalPlane(
            name="diagonal",
            coefficients=(1, 1, 1, -1.5),
            description="Átlós sík az érzelmi kocka középpontján keresztül"
        )
        
        self.planes.extend([va_plane, vd_plane, ad_plane, diagonal_plane])
    
    def find_cube_for_emotion(self, valence: float, arousal: float, dominance: float) -> EmotionCube:
        """
        Megkeresi az adott érzelmi pontot tartalmazó kockát
        
        Args:
            valence: Valencia érték [0, 1]
            arousal: Arousal érték [0, 1]
            dominance: Dominancia érték [0, 1]
            
        Returns:
            A pontot tartalmazó érzelmi kocka
        """
        # Értékek korlátozása [0, 1] tartományba
        valence = max(0, min(1, valence))
        arousal = max(0, min(1, arousal))
        dominance = max(0, min(1, dominance))
        
        # Először a 8 alapkockában keresünk
        for cube in self.cubes[:8]:
            if cube.contains_point(valence, arousal, dominance):
                return cube
                
        # Ha nincs találat, akkor a részletesebb kockákban keresünk
        for cube in self.cubes[8:]:
            if cube.contains_point(valence, arousal, dominance):
                return cube
                
        # Ha így sem találtunk, akkor a legközelebbi kockát adjuk vissza
        closest_cube = self.find_closest_cube(valence, arousal, dominance)
        return closest_cube
    
    def find_closest_cube(self, valence: float, arousal: float, dominance: float) -> EmotionCube:
        """
        Megkeresi a legközelebbi kockát egy adott érzelmi ponthoz
        
        Args:
            valence: Valencia érték [0, 1]
            arousal: Arousal érték [0, 1]
            dominance: Dominancia érték [0, 1]
            
        Returns:
            A ponthoz legközelebbi kocka
        """
        min_distance = float('inf')
        closest_cube = None
        
        for cube in self.cubes:
            center = cube.get_center()
            distance = np.sqrt(
                (valence - center[0])**2 +
                (arousal - center[1])**2 +
                (dominance - center[2])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_cube = cube
                
        return closest_cube
    
    def build_classifier_model(self) -> None:
        """
        Felépíti a TensorFlow osztályozó modellt az érzelmi kockákhoz
        """
        input_layer = tf.keras.layers.Input(shape=(3,))
        
        x = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        output_layer = tf.keras.layers.Dense(len(self.cubes), activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def create_synthetic_training_data(self, samples_per_cube: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Szintetikus adatok generálása a modell tanításához
        
        Args:
            samples_per_cube: Kockánként generálandó minták száma
            
        Returns:
            X_train, y_train adatok
        """
        X_train = []
        y_train = []
        
        for i, cube in enumerate(self.cubes):
            # Egyenletes eloszlású pontok generálása a kockában
            for _ in range(samples_per_cube):
                v = np.random.uniform(cube.valence_range[0], cube.valence_range[1])
                a = np.random.uniform(cube.arousal_range[0], cube.arousal_range[1])
                d = np.random.uniform(cube.dominance_range[0], cube.dominance_range[1])
                
                X_train.append([v, a, d])
                
                # One-hot encoding a címkékhez
                label = np.zeros(len(self.cubes))
                label[i] = 1
                y_train.append(label)
                
        return np.array(X_train), np.array(y_train)
    
    def train_classifier(self, epochs: int = 20, batch_size: int = 32) -> None:
        """
        Betanítja az osztályozó modellt
        
        Args:
            epochs: Tanítási epochok száma
            batch_size: Batch méret
        """
        if self.model is None:
            self.build_classifier_model()
            
        X_train, y_train = self.create_synthetic_training_data()
        
        # Train-validation split
        indices = np.random.permutation(len(X_train))
        train_size = int(0.8 * len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, X_val = X_train[train_indices], X_train[val_indices]
        y_train, y_val = y_train[train_indices], y_train[val_indices]
        
        # Tanítás
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
    def classify_emotion(self, valence: float, arousal: float, dominance: float) -> Dict:
        """
        Osztályozza az érzelmet a betanított modell segítségével
        
        Args:
            valence: Valencia érték [0, 1]
            arousal: Arousal érték [0, 1]
            dominance: Dominancia érték [0, 1]
            
        Returns:
            Osztályozási eredmények szótára
        """
        if self.model is None:
            # Ha nincs betanítva a modell, akkor az egyszerű kocka keresést használjuk
            cube = self.find_cube_for_emotion(valence, arousal, dominance)
            return {
                'cube_id': cube.id,
                'label': cube.label,
                'confidence': 1.0,
                'description': cube.description
            }
        
        # Osztályozás a modellel
        input_data = np.array([[valence, arousal, dominance]])
        predictions = self.model.predict(input_data)[0]
        
        # Legjobb találat
        best_idx = np.argmax(predictions)
        best_cube = self.cubes[best_idx]
        
        # Top 3 találat
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_cubes = [
            {
                'cube_id': self.cubes[idx].id,
                'label': self.cubes[idx].label,
                'confidence': float(predictions[idx]),
                'description': self.cubes[idx].description
            }
            for idx in top_indices
        ]
        
        return {
            'cube_id': best_cube.id,
            'label': best_cube.label,
            'confidence': float(predictions[best_idx]),
            'description': best_cube.description,
            'top_matches': top_cubes
        }
        
    def save_model(self, filepath: str) -> None:
        """
        Elmenti a modellt és a konfigurációt
        
        Args:
            filepath: Mentési útvonal
        """
        if self.model:
            # TensorFlow modell mentése
            self.model.save(f"{filepath}_tf_model")
            
        # Kockák és síkok mentése
        config = {
            'cubes': self.cubes,
            'planes': self.planes,
            'cube_size': self.cube_size
        }
        
        with open(f"{filepath}_config.pkl", 'wb') as f:
            pickle.dump(config, f)
            
    def load_model(self, filepath: str) -> None:
        """
        Betölti a modellt és a konfigurációt
        
        Args:
            filepath: Betöltési útvonal
        """
        # Konfiguráció betöltése
        config_path = f"{filepath}_config.pkl"
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                self.cubes = config['cubes']
                self.planes = config['planes']
                self.cube_size = config['cube_size']
                
        # TensorFlow modell betöltése
        model_path = f"{filepath}_tf_model"
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        
    def register_emotion_function(self, function: EmotionalFunction) -> None:
        """
        Regisztrál egy érzelmi függvényt
        
        Args:
            function: Az érzelmi függvény
        """
        self.functions[function.name] = function
        
    def get_emotion_trajectory(self, function_name: str, t_start: float, t_end: float, 
                              steps: int = 100) -> List[Dict]:
        """
        Kiszámítja egy érzelmi trajektóriát egy adott függvény mentén
        
        Args:
            function_name: A használandó függvény neve
            t_start: Kezdő időpont
            t_end: Befejező időpont
            steps: Időlépések száma
            
        Returns:
            Érzelmi pontok listája a trajektória mentén, kocka információkkal
        """
        if function_name not in self.functions:
            raise ValueError(f"Ismeretlen függvény: {function_name}")
            
        function = self.functions[function_name]
        raw_trajectory = function.evaluate_interval(t_start, t_end, steps)
        
        trajectory = []
        for t, valence, arousal, dominance in raw_trajectory:
            emotion_info = self.classify_emotion(valence, arousal, dominance)
            trajectory.append({
                't': t,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'cube_id': emotion_info['cube_id'],
                'label': emotion_info['label'],
                'confidence': emotion_info['confidence']
            })
            
        return trajectory