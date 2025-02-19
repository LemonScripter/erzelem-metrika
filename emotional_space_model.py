# ====================================================================
# 2. MODELLEZŐ MODUL - KOCKA-SÍK-FÜGGVÉNY MODELLEZÉS
# ====================================================================
# Fájlnév: emotional_space_model.py
# Verzió: 1.0
# Leírás: Érzelmek térbeli modellezése kockákkal, síkokkal és függvényekkel
# ====================================================================

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Union, Callable
from dataclasses import dataclass

@dataclass
class EmotionCube:
    """
    Érzelmi kocka definíciója a háromdimenziós térben
    """
    id: str
    valence_range: Tuple[float, float]  # (min, max)
    arousal_range: Tuple[float, float]  # (min, max)
    dominance_range: Tuple[float, float]  # (min, max)
    label: str = None
    description: str = None
    
    def contains_point(self, valence: float, arousal: float, dominance: float) -> bool:
        """
        Ellenőrzi, hogy egy érzelmi pont a kockán belül van-e
        
        Args:
            valence: Valencia érték
            arousal: Arousal érték
            dominance: Dominancia érték
            
        Returns:
            True, ha a pont a kockán belül van
        """
        return (self.valence_range[0] <= valence <= self.valence_range[1] and
                self.arousal_range[0] <= arousal <= self.arousal_range[1] and
                self.dominance_range[0] <= dominance <= self.dominance_range[1])
    
    def get_center(self) -> Tuple[float, float, float]:
        """
        Visszaadja a kocka középpontját
        
        Returns:
            (valence, arousal, dominance) középpontok
        """
        return (
            (self.valence_range[0] + self.valence_range[1]) / 2,
            (self.arousal_range[0] + self.arousal_range[1]) / 2,
            (self.dominance_range[0] + self.dominance_range[1]) / 2
        )
    
    def get_volume(self) -> float:
        """
        Kiszámítja a kocka térfogatát
        
        Returns:
            A kocka térfogata
        """
        return ((self.valence_range[1] - self.valence_range[0]) *
                (self.arousal_range[1] - self.arousal_range[0]) *
                (self.dominance_range[1] - self.dominance_range[0]))


class EmotionalPlane:
    """
    Érzelmi sík definíciója a háromdimenziós térben
    """
    def __init__(self, name: str, 
                 coefficients: Tuple[float, float, float, float],
                 description: str = None):
        """
        Inicializál egy érzelmi síkot ax + by + cz + d = 0 formában
        
        Args:
            name: A sík neve/azonosítója
            coefficients: (a, b, c, d) együtthatók
            description: A sík leírása
        """
        self.name = name
        self.a, self.b, self.c, self.d = coefficients
        self.description = description
        
    def distance_from_point(self, point: Tuple[float, float, float]) -> float:
        """
        Kiszámítja egy pont távolságát a síktól
        
        Args:
            point: (valence, arousal, dominance) pont
            
        Returns:
            A pont távolsága a síktól
        """
        x, y, z = point
        numerator = abs(self.a * x + self.b * y + self.c * z + self.d)
        denominator = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return numerator / denominator
    
    def project_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Egy pont merőleges vetítése a síkra
        
        Args:
            point: (valence, arousal, dominance) pont
            
        Returns:
            A pont vetülete a síkon
        """
        x, y, z = point
        distance = self.distance_from_point(point)
        
        # Normálvektor
        normal = np.array([self.a, self.b, self.c])
        normal = normal / np.linalg.norm(normal)
        
        # Pont vetítése a síkra
        projection_vector = distance * normal
        if self.a * x + self.b * y + self.c * z + self.d > 0:
            projection_vector = -projection_vector
            
        projected_point = np.array([x, y, z]) + projection_vector
        return tuple(projected_point)
    
    def get_z_value(self, x: float, y: float) -> float:
        """
        Kiszámítja a z értéket adott x, y koordinátákhoz
        
        Args:
            x: x koordináta (valencia)
            y: y koordináta (arousal)
            
        Returns:
            z koordináta (dominancia)
        """
        if self.c == 0:
            raise ValueError("A sík párhuzamos a z-tengellyel")
        return -(self.a * x + self.b * y + self.d) / self.c


class EmotionalFunction:
    """
    Érzelmi függvény definíciója időbeli érzelmi változások modellezésére
    """
    def __init__(self, name: str, 
                 function: Callable[[float, Dict], Tuple[float, float, float]],
                 parameters: Dict = None,
                 description: str = None):
        """
        Inicializálja az érzelmi függvényt
        
        Args:
            name: A függvény neve/azonosítója
            function: A függvény, ami t időpillanathoz érzelmi pontot rendel
            parameters: A függvény paraméterei
            description: A függvény leírása
        """
        self.name = name
        self.function = function
        self.parameters = parameters or {}
        self.description = description
        
    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """
        Kiértékeli a függvényt adott időpillanatban
        
        Args:
            t: Időpillanat
            
        Returns:
            (valence, arousal, dominance) pont az adott időpillanatban
        """
        return self.function(t, self.parameters)
    
    def evaluate_interval(self, t_start: float, t_end: float, 
                          steps: int = 100) -> List[Tuple[float, float, float, float]]:
        """
        Kiértékeli a függvényt egy időintervallumon
        
        Args:
            t_start: Kezdő időpillanat
            t_end: Befejező időpillanat
            steps: Időlépések száma
            
        Returns:
            [(t, valence, arousal, dominance)] lista
        """
        result = []
        for i in range(steps):
            t = t_start + (t_end - t_start) * i / (steps - 1)
            emotion = self.evaluate(t)
            result.append((t,) + emotion)
        return result


# Példa függvények

def sine_wave_emotion(t: float, params: Dict) -> Tuple[float, float, float]:
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


def gaussian_emotion(t: float, params: Dict) -> Tuple[float, float, float]:
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