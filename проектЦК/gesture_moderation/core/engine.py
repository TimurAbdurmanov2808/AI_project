"""
Основной движок системы модерации
Объединяет детектор и классификатор в единый пайплайн
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
from .detector import HandDetector
from .classifier import GestureClassifier

class GestureModerationEngine:
    """
    Главный класс для встраивания в приложения
    
    Пример использования:
        engine = GestureModerationEngine(model_path="models/model.pkl")
        
        # В цикле обработки кадров:
        result = engine.process_frame(frame)
        if result["is_obscene"]:
            # Блокируем пользователя
            send_alert_to_moderator()
            
        # Получить только статус без обработки кадра:
        status = engine.get_status()
    """
    
    def __init__(
        self,
        model_path: str = "models/gesture_model.pkl",
        required_frames: int = 8,      # количество кадров для подтверждения
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: путь к обученной модели
            required_frames: сколько кадров нужно для подтверждения жеста
            confidence_threshold: порог уверенности (0-1)
        """
        self.detector = HandDetector()
        self.classifier = GestureClassifier.load(model_path)
        
        self.required_frames = required_frames
        self.confidence_threshold = confidence_threshold
        
        # Буфер для сглаживания предсказаний
        self.prediction_buffer = deque(maxlen=required_frames)
        self.current_gesture = "neutral"
        self.hand_detected = False
        self.last_points = None
        self.last_confidence = 0.0
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает один кадр видео
        
        Args:
            frame: изображение в формате BGR (OpenCV)
            
        Returns:
            Словарь с результатами:
                - "is_obscene": bool - обнаружен ли оскорбительный жест
                - "is_neutral": bool - нейтральный жест
                - "hand_detected": bool - найдена ли рука
                - "confidence": float - уверенность (0-1)
                - "points": np.array - ключевые точки руки (или None)
                - "frame": np.array - кадр с визуализацией (опционально)
        """
        result = {
            "is_obscene": False,
            "is_neutral": False,
            "hand_detected": False,
            "confidence": 0.0,
            "points": None,
            "frame": frame
        }
        
        # Детекция руки
        points, landmarks = self.detector.detect_with_landmarks(frame)
        
        if points is not None:
            result["hand_detected"] = True
            result["points"] = points
            self.hand_detected = True
            
            # Классификация жеста
            proba = self.classifier.predict_proba(points)
            confidence = proba[1]  # вероятность оскорбительного жеста
            result["confidence"] = confidence
            
            # Добавляем в буфер
            is_obscene = confidence >= self.confidence_threshold
            self.prediction_buffer.append(1 if is_obscene else 0)
            
            # Проверяем, достаточно ли кадров для уверенного решения
            if len(self.prediction_buffer) == self.required_frames:
                obscene_ratio = sum(self.prediction_buffer) / self.required_frames
                if obscene_ratio >= 0.6:  # большинство кадров - оскорбительные
                    self.current_gesture = "obscene"
                    result["is_obscene"] = True
                else:
                    self.current_gesture = "neutral"
                    result["is_neutral"] = True
            else:
                # Ещё недостаточно данных
                result["is_neutral"] = (self.current_gesture == "neutral")
                result["is_obscene"] = (self.current_gesture == "obscene")
                
            # Визуализация (если нужна)
            if landmarks is not None:
                result["frame"] = self.detector.draw_landmarks(frame, landmarks)
                
        else:
            self.hand_detected = False
            result["is_neutral"] = True
            # Сбрасываем буфер при потере руки
            self.prediction_buffer.clear()
            
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус без обработки нового кадра
        Удобно для API или отображения в UI
        
        Returns:
            {
                "gesture": "neutral" | "obscene" | "no_hand",
                "hand_detected": bool,
                "confidence": float
            }
        """
        if not self.hand_detected:
            gesture = "no_hand"
        else:
            gesture = self.current_gesture
            
        return {
            "gesture": gesture,
            "hand_detected": self.hand_detected,
            "confidence": self.last_confidence
        }
    
    def reset(self):
        """Сбрасывает состояние (при смене пользователя)"""
        self.prediction_buffer.clear()
        self.current_gesture = "neutral"
        self.hand_detected = False
        self.last_confidence = 0.0
    
    def release(self):
        """Освобождает ресурсы"""
        self.detector.release()
