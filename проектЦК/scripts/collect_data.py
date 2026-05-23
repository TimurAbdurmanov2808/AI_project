#!/usr/bin/env python3
"""
Скрипт для сбора данных с веб-камеры
Запустите и показывайте жесты, нажимая 1 (нейтральный) или 2 (оскорбительный)
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_moderation import DataCollector

def main():
    collector = DataCollector()
    cap = cv2.VideoCapture(0)
    
    print("=" * 50)
    print("СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 50)
    print("1 - Нейтральный жест")
    print("2 - Оскорбительный жест")
    print("q - Выход и сохранение")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Показываем текущий счёт
        stats = collector.get_stats()
        cv2.putText(frame, f"Neutral: {stats['neutral']}  Obscene: {stats['obscene']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            if collector.process_frame(frame, 0):
                print(f"[+] Neutral #{stats['neutral'] + 1}")
            else:
                print("[-] Hand not found")
        elif key == ord('2'):
            if collector.process_frame(frame, 1):
                print(f"[+] Obscene #{stats['obscene'] + 1}")
            else:
                print("[-] Hand not found")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Сохраняем данные
    output_path = "data/gestures.json"
    collector.save(output_path)
    print(f"\nSaved {collector.get_stats()['total']} samples to {output_path}")

if __name__ == "__main__":
    main()
