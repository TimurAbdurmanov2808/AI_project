"""
Обучение модели классификации жестов
"""

import numpy as np
import json
from typing import Tuple, Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
from pathlib import Path

class ModelTrainer:
    """
    Обучение и оценка модели классификации жестов
    
    Пример использования:
        trainer = ModelTrainer()
        trainer.load_data("data/gestures.json")
        metrics = trainer.train()
        trainer.save_model("models/model.pkl")
    """
    
    def __init__(self):
        self.X = None
        self.y = None
        self.model = None
        self.metrics = {}
        
    def load_data(self, data_path: str, augment: bool = False):
        """
        Загружает данные из JSON файла
        
        Args:
            data_path: путь к JSON с ключами "points" и "label"
            augment: применять ли аугментацию
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if augment:
            from .augmentations import augment_dataset
            data = augment_dataset(data, augmentations_per_sample=2)
        
        self.X = np.array([item["points"] for item in data])
        self.y = np.array([item["label"] for item in data])
        
        print(f"Loaded {len(self.X)} samples (pos: {sum(self.y)}, neg: {len(self.y)-sum(self.y)})")
        
    def train(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 15
    ) -> Dict[str, float]:
        """
        Обучает модель RandomForest
        
        Returns:
            Словарь с метриками (accuracy, f1)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "test_size": len(X_test),
            "train_size": len(X_train)
        }
        
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"F1-score: {self.metrics['f1_score']:.4f}")
        
        return self.metrics
    
    def cross_validate(self, cv: int = 5) -> Dict[str, float]:
        """
        Кросс-валидация модели
        """
        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=15)
            
        acc_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='f1')
        
        return {
            "cv_accuracy_mean": acc_scores.mean(),
            "cv_accuracy_std": acc_scores.std(),
            "cv_f1_mean": f1_scores.mean(),
            "cv_f1_std": f1_scores.std()
        }
    
    def save_model(self, path: str):
        """Сохраняет обученную модель"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Загружает модель из файла"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
