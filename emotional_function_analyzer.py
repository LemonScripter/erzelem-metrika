# emotional_function_analyzer.py

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Union

class EmotionalFunctionAnalyzer:
    """Érzelmi folyamatok elemzése folytonos függvények segítségével"""
    
    def __init__(self):
        """Inicializálás"""
        self.fitted_functions = {}
        
    def fit_emotional_curve(self, emotion_points: List[Dict], dimension: str) -> np.ndarray:
        """
        Érzelmi adatpontokra illesztett folytonos görbe készítése
        
        Args:
            emotion_points: Érzelmi pontok listája
            dimension: Érzelmi dimenzió ('valence', 'arousal', 'dominance')
            
        Returns:
            Görbeillesztés eredménye
        """
        if len(emotion_points) < 2:
            return None
            
        # Időpontok és értékek kinyerése
        t_values = np.array([point['t'] for point in emotion_points])
        emotion_values = np.array([point[dimension] for point in emotion_points])
        
        # Ha csak két pont van, lineáris interpolációt használunk
        if len(emotion_points) == 2:
            def linear_function(t):
                # y = mx + b alakú egyenes
                slope = (emotion_values[1] - emotion_values[0]) / (t_values[1] - t_values[0])
                intercept = emotion_values[0] - slope * t_values[0]
                return slope * t + intercept
                
            self.fitted_functions[dimension] = linear_function
            return linear_function
            
        # Cubic spline illesztés a pontokra (természetes és sima átmenetek)
        cs = CubicSpline(t_values, emotion_values, bc_type='natural')
        self.fitted_functions[dimension] = cs
        
        return cs
    
    def get_emotional_derivatives(self, emotion_points: List[Dict]) -> Dict[str, List[float]]:
        """
        Érzelmi változások irányának és sebességének számítása
        
        Args:
            emotion_points: Érzelmi pontok listája
            
        Returns:
            Deriváltak dimenziónként
        """
        derivatives = {
            'valence_direction': [],
            'arousal_direction': [],
            'dominance_direction': []
        }
        
        # Minden dimenzióra kiszámítjuk a deriváltakat
        for dimension in ['valence', 'arousal', 'dominance']:
            # Görbeillesztés, ha még nem történt meg
            if dimension not in self.fitted_functions:
                self.fit_emotional_curve(emotion_points, dimension)
                
            # Ha nincs elég adat a görbeillesztéshez
            if dimension not in self.fitted_functions or self.fitted_functions[dimension] is None:
                derivatives[f'{dimension}_direction'] = [0.0] * len(emotion_points)
                continue
                
            # Időpontok
            t_values = [point['t'] for point in emotion_points]
            
            # Numerikus deriváltak számítása
            t_fine = np.linspace(min(t_values), max(t_values), 100)
            if isinstance(self.fitted_functions[dimension], CubicSpline):
                # CubicSpline esetén van beépített derivált számítás
                derivatives_fine = self.fitted_functions[dimension](t_fine, 1)  # Első derivált
                # Interpoláljuk vissza az eredeti időpontokra
                derivatives[f'{dimension}_direction'] = np.interp(t_values, t_fine, derivatives_fine)
            else:
                # Egyszerű numerikus derivált számítása
                h = 0.001  # Kis lépésköz
                derivatives[f'{dimension}_direction'] = [
                    (self.fitted_functions[dimension](t + h) - 
                     self.fitted_functions[dimension](t)) / h
                    for t in t_values
                ]
        
        return derivatives
    
    def classify_emotional_changes(self, derivatives: Dict[str, List[float]]) -> List[Dict]:
        """
        Érzelmi változások osztályozása
        
        Args:
            derivatives: Érzelmi dimenziók deriváltjai
            
        Returns:
            Érzelmi változás osztályozások
        """
        classifications = []
        
        for i in range(len(derivatives['valence_direction'])):
            # Minden érzelmi dimenzió irányának osztályozása
            valence_change = self._classify_direction(derivatives['valence_direction'][i])
            arousal_change = self._classify_direction(derivatives['arousal_direction'][i])
            dominance_change = self._classify_direction(derivatives['dominance_direction'][i])
            
            # Összetett érzelmi változás meghatározása
            # Példa: "javuló-élénkülő", "romló-nyugvó", stb.
            combined_change = self._determine_combined_change(
                valence_change, arousal_change, dominance_change
            )
            
            classifications.append({
                'valence_change': valence_change,
                'arousal_change': arousal_change,
                'dominance_change': dominance_change,
                'combined_change': combined_change,
                'intensity': self._calculate_change_intensity(
                    derivatives['valence_direction'][i],
                    derivatives['arousal_direction'][i],
                    derivatives['dominance_direction'][i]
                )
            })
            
        return classifications
    
    def _classify_direction(self, derivative: float) -> str:
        """Érzelmi változás irányának osztályozása"""
        threshold = 0.05  # Küszöbérték a jelentős változáshoz
        
        if derivative > threshold:
            return "improving"  # Javuló
        elif derivative < -threshold:
            return "worsening"  # Romló
        else:
            return "stable"     # Stabil
    
    def _determine_combined_change(self, valence: str, arousal: str, dominance: str) -> str:
        """Összetett érzelmi változás meghatározása"""
        # Egyszerűsített logika az összetett érzelmi változás meghatározásához
        # Valós esetben ez sokkal összetettebb lehet
        
        # Valenciát vesszük alapértelmezettnek
        if valence == "improving":
            return "positive_trend"
        elif valence == "worsening":
            return "negative_trend"
        elif arousal == "improving":
            return "activating"
        elif arousal == "worsening":
            return "calming"
        else:
            return "stable_emotion"
    
    def _calculate_change_intensity(self, v_deriv: float, a_deriv: float, d_deriv: float) -> float:
        """Érzelmi változás intenzitásának számítása"""
        # Az összes derivált abszolút értékének átlaga
        return (abs(v_deriv) + abs(a_deriv) + abs(d_deriv)) / 3.0