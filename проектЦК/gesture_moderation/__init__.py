"""
Gesture Moderation System - Core Library
Автоматическое распознавание и блокировка оскорбительных жестов
"""

from .core.engine import GestureModerationEngine
from .core.detector import HandDetector
from .core.classifier import GestureClassifier
from .utils.data_collector import DataCollector
from .utils.trainer import ModelTrainer

__version__ = "1.0.0"
__all__ = [
    "GestureModerationEngine",
    "HandDetector", 
    "GestureClassifier",
    "DataCollector",
    "ModelTrainer"
]
