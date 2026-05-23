#!/usr/bin/env python3
"""
Скрипт для тестирования модели в реальном времени
"""

import cv2
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_moderation import GestureModerationEngine

def draw_stats(frame, stats, confidence):
    h, w = frame.shape[:2]

    # Фон для панели статистики
    cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)

    elapsed = int(time.time() - stats["start_time"])
    mins, secs = divmod(elapsed, 60)

    cv2.putText(frame, f"Время: {mins:02d}:{secs:02d}", (15, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Offensive: {stats['offensive_count']}", (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    cv2.putText(frame, f"Neutral: {stats['neutral_count']}", (200, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

    # Шкала уверенности
    bar_x, bar_y, bar_w, bar_h = w - 220, h - 70, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    color = (0, int(255 * (1 - confidence)), int(255 * confidence))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (bar_x, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def show_graph(confidence_history, timestamps):
    if len(confidence_history) < 2:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, confidence_history, color="royalblue", linewidth=1.5, label="Уверенность")
    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Порог (0.5)")
    plt.fill_between(timestamps, confidence_history, 0.5,
                     where=[c >= 0.5 for c in confidence_history],
                     alpha=0.3, color="red", label="Offensive зона")
    plt.fill_between(timestamps, confidence_history, 0.5,
                     where=[c < 0.5 for c in confidence_history],
                     alpha=0.3, color="green", label="Neutral зона")
    plt.ylim(0, 1)
    plt.xlabel("Время (сек)")
    plt.ylabel("Уверенность модели")
    plt.title("График уверенности модели за сессию")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    model_path = os.path.join(os.path.dirname(__file__), "../models/gesture_model.pkl")
    engine = GestureModerationEngine(model_path=model_path)
    cap = cv2.VideoCapture(0)

    stats = {
        "start_time": time.time(),
        "offensive_count": 0,
        "neutral_count": 0,
    }
    confidence_history = []
    timestamps = []

    print("=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 50)
    print("Показывайте жесты перед камерой")
    print("q - Выход и показ графика")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = engine.process_frame(frame)
        confidence = result["confidence"]

        # Статус
        if result["is_obscene"]:
            color = (0, 0, 255)
            status = "OFFENSIVE GESTURE!"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 8)
            stats["offensive_count"] += 1
        elif not result["hand_detected"]:
            color = (128, 128, 128)
            status = "Рука не найдена"
        else:
            color = (0, 200, 80)
            status = "Neutral"
            stats["neutral_count"] += 1

        cv2.putText(frame, status, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # Записываем историю уверенности
        if result["hand_detected"]:
            confidence_history.append(confidence)
            timestamps.append(time.time() - stats["start_time"])

        frame = draw_stats(frame, stats, confidence)

        cv2.imshow("Gesture Moderation Test  |  Q - выход", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.release()

    total = stats["offensive_count"] + stats["neutral_count"]
    print("\n" + "=" * 50)
    print("ИТОГИ СЕССИИ")
    print("=" * 50)
    print(f"Всего кадров с рукой: {total}")
    print(f"Offensive: {stats['offensive_count']} ({stats['offensive_count']/max(total,1)*100:.1f}%)")
    print(f"Neutral:   {stats['neutral_count']} ({stats['neutral_count']/max(total,1)*100:.1f}%)")
    if confidence_history:
        print(f"Средняя уверенность: {sum(confidence_history)/len(confidence_history):.2f}")

    show_graph(confidence_history, timestamps)


if __name__ == "__main__":
    main()
