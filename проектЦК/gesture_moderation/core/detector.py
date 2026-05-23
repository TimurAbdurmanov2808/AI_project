"""
Детектор руки на основе MediaPipe
Возвращает ключевые точки руки (21 точка x,y)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Optional, Tuple, Any

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


class HandDetector:
    """
    Детектор руки с использованием MediaPipe Hand Landmarker (новый API)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            return None

        lms = result.hand_landmarks[0]
        points = []
        for lm in lms:
            points.extend([lm.x, lm.y, lm.z])
        return np.array(points, dtype=np.float32)

    def detect_with_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        if not result.hand_landmarks:
            return None, None

        lms = result.hand_landmarks[0]
        points = []
        for lm in lms:
            points.extend([lm.x, lm.y, lm.z])
        return np.array(points, dtype=np.float32), lms

    def draw_landmarks(self, frame: np.ndarray, hand_landmarks: Any) -> np.ndarray:
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
        return frame

    def release(self):
        self._landmarker.close()
