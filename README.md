# Система модерации жестов — запуск и тест

## Что это

Программа определяет через веб-камеру — показывает человек неприличный жест или нет. Работает в реальном времени.

---

## Установка

Открой терминал в папке проекта и выполни по порядку:

**1. Создай виртуальное окружение**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Установи библиотеки**
```bash
pip install mediapipe==0.10.35 opencv-python==4.13.0.92 scikit-learn==1.8.0 numpy==2.4.6 matplotlib==3.10.9
```

**3. Скачай модель MediaPipe** (~8MB)
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Запуск теста

```bash
source venv/bin/activate
cd проектЦК
python scripts/test_model.py
```

Откроется окно с камерой. Показывай жесты — программа покажет:
- Зелёный `Neutral` — нейтральный жест
- Красная рамка `OFFENSIVE GESTURE!` — неприличный жест

Нажми `Q` чтобы выйти. После выхода откроется график уверенности модели за сессию.

---

> Требования: Python 3.11 или 3.12, MacBook с Apple Silicon (M1/M2/M3), веб-камера
