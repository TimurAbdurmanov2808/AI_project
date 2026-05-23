#!/usr/bin/env python3
"""
Скрипт для обучения модели на собранных данных
"""

import sys
import os
import csv
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_moderation import ModelTrainer

CSV_PATH = os.path.join(os.path.dirname(__file__), "../../dataset.csv")

def load_csv_as_json(csv_path):
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points = [float(row[k]) for k in row if k != "label"]
            label = 1 if row["label"] == "offensive" else 0
            data.append({"points": points, "label": label})
    return data

def main():
    trainer = ModelTrainer()

    data = load_csv_as_json(CSV_PATH)
    trainer.X = np.array([d["points"] for d in data])
    trainer.y = np.array([d["label"] for d in data])
    print(f"Загружено: {len(data)} примеров (offensive: {sum(trainer.y)}, neutral: {len(trainer.y)-sum(trainer.y)})")
    
    # Обучаем
    metrics = trainer.train()
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.3f} (target: 0.95)")
    print(f"F1-score: {metrics['f1_score']:.3f} (target: 0.93)")
    
    if metrics['accuracy'] >= 0.95 and metrics['f1_score'] >= 0.93:
        print("\n✅ ЦЕЛЕВЫЕ МЕТРИКИ ДОСТИГНУТЫ!")
    else:
        print("\n⚠️ НУЖНО СОБРАТЬ БОЛЬШЕ ДАННЫХ")
        print(f"   Current neutral: {trainer.y.tolist().count(0)}")
        print(f"   Current obscene: {trainer.y.tolist().count(1)}")
        print("   Рекомендуется: 200+ примеров на класс")
    
    # Сохраняем
    trainer.save_model("models/gesture_model.pkl")
    
    # Кросс-валидация
    print("\n" + "=" * 50)
    print("КРОСС-ВАЛИДАЦИЯ")
    print("=" * 50)
    cv_metrics = trainer.cross_validate()
    print(f"CV Accuracy: {cv_metrics['cv_accuracy_mean']:.3f} (+/- {cv_metrics['cv_accuracy_std']:.3f})")
    print(f"CV F1: {cv_metrics['cv_f1_mean']:.3f} (+/- {cv_metrics['cv_f1_std']:.3f})")

if __name__ == "__main__":
    main()
