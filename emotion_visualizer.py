# ====================================================================
# 3. VIZUALIZÁCIÓS MODUL - ÉRZELMI TÉR VIZUALIZÁCIÓ
# ====================================================================
# Fájlnév: emotion_visualizer.py
# Verzió: 1.0
# Leírás: Érzelmi tér vizualizációja 3D-ben és időbeli változások ábrázolása
# ====================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from emotional_space_model import EmotionCube, EmotionalPlane, EmotionalFunction
from emotion_cube_modeler import EmotionCubeModeler

class EmotionVisualizer:
    """
    Érzelmi állapotok és változások vizualizációja
    """
    
    def __init__(self, modeler: EmotionCubeModeler):
        """
        Inicializálja a vizualizáló osztályt
        
        Args:
            modeler: Érzelmi kocka modellező példány
        """
        self.modeler = modeler
        self.color_map = {
            'Ellenséges': '#E74C3C',    # piros
            'Stresszes': '#9B59B6',     # lila
            'Izgatott': '#F39C12',      # narancs
            'Lelkes': '#F1C40F',        # sárga
            'Depresszív': '#34495E',    # sötétkék
            'Nyugodt': '#2980B9',       # kék
            'Elégedett': '#27AE60',     # zöld
            'Boldog': '#2ECC71'         # világoszöld
        }
        
    def plot_emotion_cube_3d(self, emotion_point: Optional[Tuple[float, float, float]] = None, 
                             save_path: Optional[str] = None) -> None:
        """
        Érzelmi kockák megjelenítése 3D térben
        
        Args:
            emotion_point: Opcionális érzelmi pont (valence, arousal, dominance)
            save_path: Opcionális mentési útvonal
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Tengelyek beállítása
        ax.set_xlabel('Valencia')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominancia')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title('Érzelmi kockák 3D térben', fontsize=16)
        
        # Kockák megjelenítése
        for cube in self.modeler.cubes[:8]:  # Csak az alapkockákat jelenítjük meg
            v_min, v_max = cube.valence_range
            a_min, a_max = cube.arousal_range
            d_min, d_max = cube.dominance_range
            
            # Kocka csúcspontjainak kiszámítása
            vertices = [
                (v_min, a_min, d_min), (v_max, a_min, d_min),
                (v_min, a_max, d_min), (v_max, a_max, d_min),
                (v_min, a_min, d_max), (v_max, a_min, d_max),
                (v_min, a_max, d_max), (v_max, a_max, d_max)
            ]
            
            # Kocka éleinek meghatározása
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 3),
                (0, 4), (1, 5), (2, 6), (3, 7),
                (4, 5), (4, 6), (5, 7), (6, 7)
            ]
            
            # Kocka színének beállítása
            color = self.color_map.get(cube.label, '#CCCCCC')
            
            # Kocka éleinek kirajzolása
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color=color, alpha=0.7)
                
            # Kocka címkéjének megjelenítése a középpontban
            center = cube.get_center()
            ax.text(center[0], center[1], center[2], cube.label, 
                   color='black', fontsize=8, ha='center', va='center')
        
        # Érzelmi pont megjelenítése, ha meg van adva
        if emotion_point:
            v, a, d = emotion_point
            emotion_cube = self.modeler.find_cube_for_emotion(v, a, d)
            point_color = self.color_map.get(emotion_cube.label, '#CCCCCC')
            ax.scatter([v], [a], [d], color=point_color, s=100, edgecolor='black', alpha=1.0)
            ax.text(v+0.02, a+0.02, d+0.02, f"({v:.2f}, {a:.2f}, {d:.2f})", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def create_interactive_3d_plot(self, emotion_points: List[Tuple[float, float, float, str]] = None,
                                  trajectory: List[Dict] = None) -> go.Figure:
        """
        Interaktív 3D ábra készítése Plotly-val
        
        Args:
            emotion_points: Lista érzelmi pontokról (valence, arousal, dominance, label)
            trajectory: Érzelmi trajektória pontok listája
            
        Returns:
            Plotly Figure objektum
        """
        fig = go.Figure()
        
        # Kockák megjelenítése
        for cube in self.modeler.cubes[:8]:  # Csak az alapkockákat jelenítjük meg
            v_min, v_max = cube.valence_range
            a_min, a_max = cube.arousal_range
            d_min, d_max = cube.dominance_range
            color = self.color_map.get(cube.label, '#CCCCCC')
            
            # Kocka megjelenítése átlátszó 3D alakzatként
            vertices = [
                [v_min, a_min, d_min], [v_max, a_min, d_min],
                [v_max, a_max, d_min], [v_min, a_max, d_min],
                [v_min, a_min, d_max], [v_max, a_min, d_max],
                [v_max, a_max, d_max], [v_min, a_max, d_max]
            ]
            
            i = [0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3, 4, 5, 6, 7]
            j = [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7, 0, 1, 2, 3]
            k = [2, 3, 0, 1, 6, 7, 4, 5, 3, 2, 1, 0, 7, 6, 5, 4]
            
            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in vertices],
                y=[v[1] for v in vertices],
                z=[v[2] for v in vertices],
                i=i, j=j, k=k,
                opacity=0.2,
                color=color,
                hoverinfo='text',
                text=cube.label
            ))
            
            # Kocka középpontjának és címkéjének megjelenítése
            center = cube.get_center()
            fig.add_trace(go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]],
                mode='text',
                text=[cube.label],
                textposition='middle center',
                textfont=dict(size=10, color='black'),
                hoverinfo='text',
                hovertext=cube.description
            ))
            
        # Érzelmi pontok megjelenítése
        if emotion_points:
            x, y, z, labels = zip(*emotion_points)
            colors = [self.color_map.get(self.modeler.find_cube_for_emotion(x[i], y[i], z[i]).label, '#CCCCCC')
                     for i in range(len(x))]
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=6,
                    color=colors,
                    opacity=0.8,
                    line=dict(color='black', width=1)
                ),
                text=labels,
                hoverinfo='text'
            ))
            
        # Érzelmi trajektória megjelenítése
        if trajectory:
            t_values = [point['t'] for point in trajectory]
            valence = [point['valence'] for point in trajectory]
            arousal = [point['arousal'] for point in trajectory]
            dominance = [point['dominance'] for point in trajectory]
            labels = [point['label'] for point in trajectory]
            hovertext = [f"t={t:.2f}, V={v:.2f}, A={a:.2f}, D={d:.2f}<br>{label}" 
                        for t, v, a, d, label in zip(t_values, valence, arousal, dominance, labels)]
            
            fig.add_trace(go.Scatter3d(
                x=valence, y=arousal, z=dominance,
                mode='lines+markers',
                marker=dict(
                    size=4,
                    colorscale='Viridis',
                    color=t_values,
                    colorbar=dict(title='Idő'),
                    opacity=0.8
                ),
                line=dict(
                    color='black',
                    width=3
                ),
                hovertext=hovertext,
                hoverinfo='text',
                name='Érzelmi trajektória'
            ))
            
        # Tengelyek és címkék beállítása
        fig.update_layout(
            scene=dict(
                xaxis_title='Valencia',
                yaxis_title='Arousal',
                zaxis_title='Dominancia',
                xaxis=dict(range=[0, 1], dtick=0.2),
                yaxis=dict(range=[0, 1], dtick=0.2),
                zaxis=dict(range=[0, 1], dtick=0.2),
                aspectratio=dict(x=1, y=1, z=1)
            ),
            width=1000,
            height=800,
            title='Érzelmi tér interaktív 3D megjelenítése',
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
        
    def plot_emotion_trajectory_2d(self, trajectory: List[Dict], 
                                 dimensions: str = 'valence_arousal',
                                 save_path: Optional[str] = None) -> None:
        """
        Érzelmi trajektória 2D ábrázolása
        
        Args:
            trajectory: Érzelmi trajektória pontok listája
            dimensions: Melyik két dimenziót jelenítsük meg ('valence_arousal', 
                       'valence_dominance', 'arousal_dominance')
            save_path: Opcionális mentési útvonal
        """
        plt.figure(figsize=(12, 8))
        
        # Dimenziók kiválasztása
        if dimensions == 'valence_arousal':
            x_key, y_key = 'valence', 'arousal'
            plt.xlabel('Valencia')
            plt.ylabel('Arousal')
            title = 'Érzelmi trajektória a Valencia-Arousal síkon'
        elif dimensions == 'valence_dominance':
            x_key, y_key = 'valence', 'dominance'
            plt.xlabel('Valencia')
            plt.ylabel('Dominancia')
            title = 'Érzelmi trajektória a Valencia-Dominancia síkon'
        elif dimensions == 'arousal_dominance':
            x_key, y_key = 'arousal', 'dominance'
            plt.xlabel('Arousal')
            plt.ylabel('Dominancia')
            title = 'Érzelmi trajektória a Arousal-Dominancia síkon'
        else:
            raise ValueError("Érvénytelen dimenzió-pár")
        
        plt.title(title, fontsize=16)
        
        # Tengelyek beállítása
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Érzelmek síkjának felosztása
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Érzelmi negyedek címkézése
        quadrant_labels = {
            'valence_arousal': ['Negatív aktív', 'Pozitív aktív', 'Negatív passzív', 'Pozitív passzív'],
            'valence_dominance': ['Negatív alárendelt', 'Pozitív alárendelt', 'Negatív uralkodó', 'Pozitív uralkodó'],
            'arousal_dominance': ['Passzív alárendelt', 'Aktív alárendelt', 'Passzív uralkodó', 'Aktív uralkodó']
        }
        
        labels = quadrant_labels[dimensions]
        plt.text(0.25, 0.75, labels[0], ha='center', va='center', fontsize=12, alpha=0.7)
        plt.text(0.75, 0.75, labels[1], ha='center', va='center', fontsize=12, alpha=0.7)
        plt.text(0.25, 0.25, labels[2], ha='center', va='center', fontsize=12, alpha=0.7)
        plt.text(0.75, 0.25, labels[3], ha='center', va='center', fontsize=12, alpha=0.7)
        
        # Trajektória pontok kinyerése
        x = [point[x_key] for point in trajectory]
        y = [point[y_key] for point in trajectory]
        t = [point['t'] for point in trajectory]
        
        # Adatok színezése idő szerint
        plt.scatter(x, y, c=t, cmap='viridis', s=50, edgecolor='black', alpha=0.7)
        plt.colorbar(label='Idő')
        
        # Adatok összekötése vonallal
        plt.plot(x, y, 'k-', alpha=0.5)
        
        # Kezdő és végpontok kiemelése
        plt.plot(x[0], y[0], 'go', markersize=10, label='Kezdőpont')
        plt.plot(x[-1], y[-1], 'ro', markersize=10, label='Végpont')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_emotion_time_series(self, trajectory: List[Dict], save_path: Optional[str] = None) -> None:
        """
        Érzelmi dimenziók idősoraként való ábrázolása
        
        Args:
            trajectory: Érzelmi trajektória pontok listája
            save_path: Opcionális mentési útvonal
        """
        plt.figure(figsize=(14, 8))
        
        # Adatok kinyerése
        t = [point['t'] for point in trajectory]
        valence = [point['valence'] for point in trajectory]
        arousal = [point['arousal'] for point in trajectory]
        dominance = [point['dominance'] for point in trajectory]
        
        # Érzelmi dimenziók rajzolása
        plt.plot(t, valence, 'r-', linewidth=2, label='Valencia')
        plt.plot(t, arousal, 'g-', linewidth=2, label='Arousal')
        plt.plot(t, dominance, 'b-', linewidth=2, label='Dominancia')
        
        # Kockák váltásának jelölése
        prev_label = trajectory[0]['label']
        change_points = []
        change_labels = []
        
        for i, point in enumerate(trajectory):
            if point['label'] != prev_label:
                change_points.append(t[i])
                change_labels.append(point['label'])
                prev_label = point['label']
                
        for cp in change_points:
            plt.axvline(x=cp, color='black', linestyle=':', alpha=0.5)
            
        for i, (cp, label) in enumerate(zip(change_points, change_labels)):
            plt.text(cp, 1.05, label, rotation=45, ha='center', fontsize=8)
        
        # Tengelyek és címkék beállítása
        plt.title('Érzelmi dimenziók időbeli változása', fontsize=16)
        plt.xlabel('Idő')
        plt.ylabel('Érték')
        plt.ylim(0, 1.1)  # Helyet biztosítunk a felső címkéknek
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()