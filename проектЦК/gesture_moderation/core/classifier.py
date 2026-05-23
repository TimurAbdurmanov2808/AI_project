"""
Классификатор жестов на основе обученной модели
"""

import pickle
import numpy as np
from typing import Union, Optional
from pathlib import Path

class GestureClassifier:
    """
    Классификатор жестов (обёртка над ML моделью)
    
    Пример использования:
        classifier = GestureClassifier.load("models/model.pkl")
        prediction = classifier.predict(points)  # 0 или 1
    """
    
    def __init__(self, model):
        self.model = model
        
    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "GestureClassifier":
        """
        Загружает модель из файла
        
        Args:
            model_path: путь к .pkl файлу
            
        Returns:
            GestureClassifier
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return cls(model)
    
    def predict(self, points: np.ndarray) -> int:
        """
        Предсказывает класс жеста
        
        Args:
            points: массив из 42 значений (ключевые точки руки)
            
        Returns:
            0 - нейтральный жест
            1 - оскорбительный жест
        """
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        return int(self.model.predict(points)[0])
    
    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности классов
        
        Returns:
            массив [вероятность_нейтрального, вероятность_оскорбительного]
        """
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        return self.model.predict_proba(points)[0]
    
    def save(self, path: Union[str, Path]):
        """Сохраняет модель"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
