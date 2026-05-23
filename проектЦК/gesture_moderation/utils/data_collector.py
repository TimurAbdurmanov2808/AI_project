"""
Утилита для сбора датасета жестов
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from ..core.detector import HandDetector

class DataCollector:
    """
    Сбор данных с веб-камеры для обучения модели
    
    Пример использования:
        collector = DataCollector()
        collector.start_collection()
        # В цикле: collector.record_frame(frame, label)
        collector.save("data/gestures.json")
    """
    
    def __init__(self):
        self.detector = HandDetector()
        self.data = []  # list of {"points": [...], "label": int}
        
    def process_frame(self, frame: np.ndarray, label: int) -> bool:
        """
        Обрабатывает кадр и добавляет в датасет, если рука найдена
        
        Args:
            frame: кадр из веб-камеры
            label: 0 - нейтральный, 1 - оскорбительный
            
        Returns:
            True если запись добавлена, False если рука не найдена
        """
        points = self.detector.detect(frame)
        if points is not None:
            self.data.append({
                "points": points.tolist(),
                "label": label
            })
            return True
        return False
    
    def add_sample(self, points: List[float], label: int):
        """
        Добавляет образец напрямую (без кадра)
        
        Args:
            points: список из 42 значений
            label: 0 или 1
        """
        self.data.append({
            "points": points,
            "label": label
        })
    
    def save(self, path: str):
        """Сохраняет датасет в JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Saved {len(self.data)} samples to {path}")
    
    def load(self, path: str):
        """Загружает датасет из JSON"""
        with open(path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from {path}")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику датасета"""
        labels = [item["label"] for item in self.data]
        return {
            "total": len(self.data),
            "neutral": labels.count(0),
            "obscene": labels.count(1)
        }
    
    def release(self):
        self.detector.release()
