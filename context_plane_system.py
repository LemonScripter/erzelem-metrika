# context_plane_system.py

from context_plane import ContextPlane
from typing import Dict, List, Tuple
import numpy as np

class ContextPlaneSystem:
    """
    Különböző kontextusokhoz tartozó síkok rendszere
    """
    
    def __init__(self):
        """Inicializálás üres sík-rendszerrel"""
        self.planes = {}
        self._initialize_default_planes()
        
    def _initialize_default_planes(self):
        """Alapértelmezett síkok definiálása"""
        # Különböző kontextusok különböző orientációjú síkokat kapnak
        # az érzelmi térben
        
        default_planes = {
            # Üzleti kontextus: inkább a valencia és arousal számít
            "business": (0.7, 0.7, 0.1, -0.7),
            
            # Tudományos kontextus: inkább a dominancia és arousal számít
            "academic": (0.1, 0.5, 0.85, -0.7),
            
            # Személyes kontextus: mindhárom dimenzió egyformán fontos
            "personal": (0.6, 0.5, 0.6, -0.85),
            
            # Közösségi média: erős valencia és arousal hatás
            "social_media": (0.8, 0.6, 0.0, -0.7),
            
            # Általános: kiegyensúlyozott
            "general": (0.58, 0.58, 0.58, -0.85)
        }
        
        # Síkok létrehozása
        for context, coefficients in default_planes.items():
            self.add_plane(context, coefficients)
            
    def add_plane(self, context_name: str, coefficients: Tuple[float, float, float, float]):
        """
        Új sík hozzáadása
        
        Args:
            context_name: Kontextus neve
            coefficients: (a, b, c, d) sík-együtthatók
        """
        self.planes[context_name] = ContextPlane(context_name, coefficients)
        
    def get_plane(self, context_name: str) -> ContextPlane:
        """
        Adott nevű sík lekérdezése
        
        Args:
            context_name: Kontextus neve
            
        Returns:
            A kontextushoz tartozó sík
        """
        if context_name in self.planes:
            return self.planes[context_name]
        
        # Ha nincs ilyen nevű sík, visszaadjuk az általánosat
        if "general" in self.planes:
            return self.planes["general"]
            
        # Ha az általános sem létezik, kivételt dobunk
        raise KeyError(f"Nincs {context_name} nevű kontextus-sík")
        
    def get_all_contexts(self) -> List[str]:
        """
        Az összes kontextus nevének lekérdezése
        
        Returns:
            Kontextus nevek listája
        """
        return list(self.planes.keys())
    
    def find_intersections(self, curve_points: List[np.ndarray], context: str) -> List[Dict]:
        """
        Görbe és kontextus-sík metszéspontjainak keresése
        
        Args:
            curve_points: Görbepontok [x, y, z] formátumban
            context: Kontextus neve
            
        Returns:
            Metszéspontok listája
        """
        if len(curve_points) < 2:
            return []
            
        plane = self.get_plane(context)
        intersections = []
        
        for i in range(len(curve_points) - 1):
            start = curve_points[i]
            end = curve_points[i+1]
            
            intersection = plane.intersect_with_line(start, end)
            if intersection is not None:
                # Kiszámítjuk, hogy a metszéspont milyen t paraméternél van
                # a [i, i+1] intervallumon belül
                t_relative = np.linalg.norm(intersection - start) / np.linalg.norm(end - start)
                t_global = i + t_relative
                
                intersections.append({
                    'point': intersection,
                    't': t_global,
                    'segment_start': i,
                    'segment_end': i+1,
                    'context': context
                })
                
        return intersections