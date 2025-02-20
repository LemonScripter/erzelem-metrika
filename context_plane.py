# context_plane.py

import numpy as np
from typing import List, Dict, Tuple

class ContextPlane:
    """
    Kontextus-sík a 3D érzelmi térben
    
    A sík egyenlete: a*x + b*y + c*z + d = 0, ahol:
    x: valencia, y: arousal, z: dominancia
    """
    
    def __init__(self, context_name: str, coefficients: Tuple[float, float, float, float]):
        """
        Inicializál egy kontextus-síkot
        
        Args:
            context_name: Kontextus neve
            coefficients: (a, b, c, d) sík-együtthatók
        """
        self.context_name = context_name
        self.a, self.b, self.c, self.d = coefficients
        
        # Sík normálvektora
        self.normal = np.array([self.a, self.b, self.c])
        # Normalizáljuk
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal = self.normal / norm
        
    def get_point_on_plane(self) -> np.ndarray:
        """Visszaad egy pontot a síkon"""
        # Ha c nem nulla, akkor z = -d/c pontban x=0, y=0
        if abs(self.c) > 1e-6:
            return np.array([0, 0, -self.d / self.c])
        # Ha b nem nulla, akkor y = -d/b pontban x=0, z=0
        elif abs(self.b) > 1e-6:
            return np.array([0, -self.d / self.b, 0])
        # Ha a nem nulla, akkor x = -d/a pontban y=0, z=0
        elif abs(self.a) > 1e-6:
            return np.array([-self.d / self.a, 0, 0])
        else:
            raise ValueError("Érvénytelen síkegyütthatók")
            
    def distance_from_point(self, point: np.ndarray) -> float:
        """
        Kiszámítja egy pont távolságát a síktól
        
        Args:
            point: [x, y, z] formátumú pont
            
        Returns:
            A pont síktól való távolsága
        """
        numerator = abs(np.dot(self.normal, point) + self.d)
        denominator = np.linalg.norm(self.normal)
        return numerator / denominator if denominator > 0 else 0
    
    def project_point_to_plane(self, point: np.ndarray) -> np.ndarray:
        """
        Egy pont merőleges vetítése a síkra
        
        Args:
            point: [x, y, z] formátumú pont
            
        Returns:
            A pont vetülete a síkon
        """
        # A pont és a sík távolsága
        dist = self.distance_from_point(point)
        
        # A vetületi pont = eredeti pont - dist * normálvektor
        projection = point - dist * self.normal
        
        return projection
    
    def intersect_with_line(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        Egyenes és sík metszéspontjának kiszámítása
        
        Args:
            start: Egyenes kezdőpontja [x1, y1, z1]
            end: Egyenes végpontja [x2, y2, z2]
            
        Returns:
            Metszéspont [x, y, z] vagy None, ha nincs metszéspont
        """
        # Irányvektor
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return None  # Túl rövid szakasz
            
        direction = direction / direction_norm
        
        # Ellenőrizzük, hogy az egyenes és a sík párhuzamosak-e
        dot_product = np.dot(self.normal, direction)
        if abs(dot_product) < 1e-6:
            return None  # Párhuzamos vagy majdnem párhuzamos
            
        # Metszéspont t paramétere
        t = -(np.dot(self.normal, start) + self.d) / dot_product
        
        # Ellenőrizzük, hogy a metszéspont a szakaszon van-e
        if t < 0 or t > 1:
            return None  # A metszéspont a szakaszon kívül van
            
        # Metszéspont koordinátái
        intersection = start + t * direction
        
        return intersection