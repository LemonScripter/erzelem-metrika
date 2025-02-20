# intersection_analyzer.py

import numpy as np
from typing import List, Dict, Tuple
from emotional_function_analyzer import EmotionalFunctionAnalyzer
from context_plane_system import ContextPlaneSystem

class IntersectionAnalyzer:
    """
    Érzelmi görbe és kontextus-síkok metszéspontjainak elemzése
    """
    
    def __init__(self):
        """Inicializálás"""
        self.function_analyzer = EmotionalFunctionAnalyzer()
        self.plane_system = ContextPlaneSystem()
        
    def analyze_emotional_trajectory(self, emotion_points: List[Dict], context: str) -> Dict:
        """
        Teljes érzelmi trajektória elemzése adott kontextusban
        
        Args:
            emotion_points: Érzelmi pontok
            context: Elemzési kontextus
            
        Returns:
            Elemzési eredmények
        """
        # 1. Folytonos érzelmi görbék illesztése
        valence_curve = self.function_analyzer.fit_emotional_curve(emotion_points, 'valence')
        arousal_curve = self.function_analyzer.fit_emotional_curve(emotion_points, 'arousal')
        dominance_curve = self.function_analyzer.fit_emotional_curve(emotion_points, 'dominance')
        
        # Ha nincs elég pont a görbeillesztéshez
        if valence_curve is None:
            return {
                'intersections': [],
                'emotional_changes': [],
                'summary': "Nincs elég adat az elemzéshez"
            }
        
        # 2. Érzelmi változások irányának és sebességének számítása
        derivatives = self.function_analyzer.get_emotional_derivatives(emotion_points)
        emotional_changes = self.function_analyzer.classify_emotional_changes(derivatives)
        
        # 3. Diszkrét görbepontok generálása sűrűbben a metszéspontok megtalálásához
        t_fine = np.linspace(
            min(p['t'] for p in emotion_points),
            max(p['t'] for p in emotion_points),
            100  # 100 pont a pontos metszéspontkereséshez
        )
        
        curve_points = []
        for t in t_fine:
            # Minden időpontra kiszámítjuk a háromdimenziós érzelmi pontot
            if isinstance(valence_curve, np.ndarray):
                # Lineáris interpoláció
                v = float(np.interp(t, [p['t'] for p in emotion_points], [p['valence'] for p in emotion_points]))
                a = float(np.interp(t, [p['t'] for p in emotion_points], [p['arousal'] for p in emotion_points]))
                d = float(np.interp(t, [p['t'] for p in emotion_points], [p['dominance'] for p in emotion_points]))
            else:
                # Cubic Spline vagy más függvény
                v = float(valence_curve(t))
                a = float(arousal_curve(t))
                d = float(dominance_curve(t))
                
            curve_points.append(np.array([v, a, d]))
        
        # 4. Metszéspontok keresése a kontextus-síkkal
        intersections = self.plane_system.find_intersections(curve_points, context)
        
        # 5. Metszéspontok értelmezése
        intersection_meanings = self._interpret_intersections(
            intersections, emotional_changes, emotion_points
        )
        
        # 6. Összefoglaló készítése
        summary = self._create_analysis_summary(
            intersections, emotional_changes, context, emotion_points
        )
        
        return {
            'intersections': intersection_meanings,
            'emotional_changes': emotional_changes,
            'summary': summary
        }
        
    def _interpret_intersections(self, intersections: List[Dict], 
                              emotional_changes: List[Dict],
                              emotion_points: List[Dict]) -> List[Dict]:
        """
        Metszéspontok értelmezése
        
        Args:
            intersections: Nyers metszéspontok
            emotional_changes: Érzelmi változások elemzése
            emotion_points: Eredeti érzelmi pontok
            
        Returns:
            Értelmezett metszéspontok
        """
        interpreted = []
        
        for intersection in intersections:
            t = intersection['t']
            
            # A legközelebbi érzelmi változás megtalálása
            nearest_change_idx = min(
                range(len(emotional_changes)),
                key=lambda i: abs(emotion_points[i]['t'] - t)
            )
            nearest_change = emotional_changes[nearest_change_idx]
            
            # Érzelmi pont értékei a metszéspontban
            point = intersection['point']
            valence, arousal, dominance = point[0], point[1], point[2]
            
            # A metszéspontok jelentésének gazdagítása
            interpretation = {
                't': t,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'context': intersection['context'],
                'change_direction': nearest_change['combined_change'],
                'intensity': nearest_change['intensity'],
                'significance': self._calculate_significance(
                    point, nearest_change, intersection['context']
                )
            }
            
            # A metszéspont típusának meghatározása a változás alapján
            if nearest_change['valence_change'] == 'improving':
                interpretation['type'] = 'positive_transition'
            elif nearest_change['valence_change'] == 'worsening':
                interpretation['type'] = 'negative_transition'
            else:
                interpretation['type'] = 'neutral_crossing'
                
            interpreted.append(interpretation)
            
        return interpreted
    
    def _calculate_significance(self, point: np.ndarray, change: Dict, context: str) -> float:
        """
        Metszéspont jelentőségének kiszámítása
        
        Args:
            point: Metszéspont koordinátái
            change: Érzelmi változás a metszéspontnál
            context: Kontextus
            
        Returns:
            Jelentőség 0-1 skálán
        """
        # A jelentőség számításakor figyelembe vesszük:
        # 1. A változás intenzitását
        # 2. A metszéspontban lévő érzelmi értékek abszolút nagyságát
        # 3. A kontextus relevanciáját
        
        # Érzelmi értékek abszolút hatása (mennyire szélsőségesek)
        emotion_extremity = np.mean([
            abs(point[0] - 0.5),  # Valencia távolsága a semlegestől
            abs(point[1] - 0.5),  # Arousal távolsága a semlegestől
            abs(point[2] - 0.5)   # Dominancia távolsága a semlegestől
        ])
        
        # Változás intenzitása
        intensity = change['intensity']
        
        # Kontextus súlyozás (egyszerűsített)
        context_weight = 1.0
        
        # Végső jelentőség
        significance = (emotion_extremity * 0.4 + intensity * 0.6) * context_weight
        
        # Normalizálás 0-1 tartományra
        return min(1.0, max(0.0, significance))
    
    def _create_analysis_summary(self, intersections: List[Dict], 
                              emotional_changes: List[Dict],
                              context: str, emotion_points: List[Dict]) -> str:
        """
        Elemzési összefoglaló készítése
        
        Args:
            intersections: Metszéspontok
            emotional_changes: Érzelmi változások
            context: Kontextus
            emotion_points: Érzelmi pontok
            
        Returns:
            Szöveges összefoglaló
        """
        if not intersections:
            return f"Az érzelmi görbe nem metszi a(z) {context} kontextus síkját a vizsgált időszakban."
            
        # Metszéspontok száma és típusok
        positive_crossings = sum(1 for i in intersections if i.get('type') == 'positive_transition')
        negative_crossings = sum(1 for i in intersections if i.get('type') == 'negative_transition')
        neutral_crossings = sum(1 for i in intersections if i.get('type') == 'neutral_crossing')
        
        # Jelentős metszéspontok
        significant_intersections = [i for i in intersections if i.get('significance', 0) > 0.6]
        
        # Domináns érzelmi változás meghatározása
        if len(emotional_changes) > 0:
            change_types = [change['combined_change'] for change in emotional_changes]
            from collections import Counter
            dominant_change = Counter(change_types).most_common(1)[0][0]
        else:
            dominant_change = "stable_emotion"
        
        # Összefoglaló szöveg összeállítása
        summary = f"Az érzelmi görbe {len(intersections)} ponton metszi a(z) {context} kontextus síkját. "
        
        if positive_crossings > negative_crossings:
            summary += f"Az érzelmi változások többnyire pozitív irányúak ({positive_crossings} pozitív átmenet). "
        elif negative_crossings > positive_crossings:
            summary += f"Az érzelmi változások többnyire negatív irányúak ({negative_crossings} negatív átmenet). "
        else:
            summary += f"Az érzelmi változások kiegyensúlyozottak ({positive_crossings} pozitív, {negative_crossings} negatív átmenet). "
            
        if significant_intersections:
            summary += f"A szövegben {len(significant_intersections)} jelentős érzelmi váltás azonosítható. "
            
        summary += f"A domináns érzelmi trend: {self._translate_change_type(dominant_change)}."
        
        return summary
    
    def _translate_change_type(self, change_type: str) -> str:
        """Magyar fordítás az érzelmi változás típusához"""
        translations = {
            "positive_trend": "pozitív tendencia",
            "negative_trend": "negatív tendencia",
            "activating": "aktivizálódás",
            "calming": "megnyugvás",
            "stable_emotion": "érzelmi stabilitás"
        }
        return translations.get(change_type, change_type)