"""
Аугментация данных для увеличения датасета
"""

import numpy as np
import random
from typing import List, Tuple

def augment_points(
    points: List[float],
    noise_std: float = 0.01,
    rotation_deg: float = 5.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    shift_range: float = 0.02
) -> List[float]:
    """
    Аугментация ключевых точек руки
    
    Args:
        points: список из 42 значений (x1,y1,...,x21,y21)
        noise_std: стандартное отклонение шума
        rotation_deg: максимальный угол поворота (градусы)
        scale_range: диапазон масштабирования (min, max)
        shift_range: максимальное смещение
        
    Returns:
        Аугментированные точки
    """
    points = np.array(points).reshape(21, 2)
    center = np.mean(points, axis=0)
    
    # Сдвиг
    if shift_range > 0:
        shift = np.random.uniform(-shift_range, shift_range, 2)
        points = points + shift
    
    # Масштабирование
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        scale = np.random.uniform(*scale_range)
        points = (points - center) * scale + center
    
    # Поворот
    if rotation_deg > 0:
        angle = np.radians(np.random.uniform(-rotation_deg, rotation_deg))
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = (points - center) @ rot_matrix.T + center
    
    # Шум
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, points.shape)
        points = points + noise
    
    return points.flatten().tolist()

def augment_dataset(
    data: List[dict],
    augmentations_per_sample: int = 3,
    **augment_kwargs
) -> List[dict]:
    """
    Увеличивает датасет путём аугментации
    
    Args:
        data: список словарей {"points": [...], "label": int}
        augmentations_per_sample: количество аугментированных версий на один образец
        **augment_kwargs: параметры для augment_points()
        
    Returns:
        Расширенный датасет
    """
    augmented = []
    for sample in data:
        # Оригинальный образец
        augmented.append(sample)
        points = sample["points"]
        label = sample["label"]
        
        # Аугментированные версии
        for _ in range(augmentations_per_sample):
            new_points = augment_points(points, **augment_kwargs)
            augmented.append({
                "points": new_points,
                "label": label
            })
    return augmented
