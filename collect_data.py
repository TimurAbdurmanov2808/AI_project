import cv2
import mediapipe as mp
import csv
import os
import time
import urllib.request

# ─── Скачиваем модель если нет ───────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Скачиваю модель MediaPipe (~8MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Модель скачана!")

# ─── Настройки ────────────────────────────────────────────────────────────────
GESTURE_LABEL = "neutral"
OUTPUT_CSV    = "dataset.csv"
HOLD_TIME     = 0.5

# ─── MediaPipe новый API ──────────────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode       = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ─── CSV ──────────────────────────────────────────────────────────────────────
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{ax}{i}" for i in range(21) for ax in ("x","y","z")]
        header.append("label")
        writer.writerow(header)

# ─── Соединения пальцев для отрисовки ────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

cap          = cv2.VideoCapture(0)
sample_count = 0
holding      = False
hold_start   = None

print("=== Сбор датасета ===")
print(f"Жест: {GESTURE_LABEL}")
print("Показывай жест → держи 0.5 сек → автосохранение")
print("Нажми Q для выхода\n")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_img)

        status_text = "Рука не найдена"
        color       = (0, 0, 255)

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]

            # Рисуем точки и линии
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)

            # Координаты для CSV
            landmarks = [coord for lm in lms for coord in (lm.x, lm.y, lm.z)]

            # Логика удержания
            if not holding:
                holding    = True
                hold_start = time.time()
                status_text = "Держи жест..."
                color       = (0, 200, 255)
            else:
                elapsed = time.time() - hold_start
                if elapsed >= HOLD_TIME:
                    with open(OUTPUT_CSV, "a", newline="") as f:
                        csv.writer(f).writerow(landmarks + [GESTURE_LABEL])
                    sample_count += 1
                    hold_start   = time.time()
                    status_text  = f"Сохранено: {sample_count}"
                    color        = (0, 255, 0)
                else:
                    status_text = f"Держи... {HOLD_TIME - elapsed:.1f}с"
                    color       = (0, 200, 255)
        else:
            holding    = False
            hold_start = None

        # UI
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Samples: {sample_count}", (w - 210, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Сбор датасета — Q для выхода", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nГотово! Сохранено {sample_count} samples → {OUTPUT_CSV}")
