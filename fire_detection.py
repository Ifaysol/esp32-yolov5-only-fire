import cv2
import time
import os
import pygame
import torch
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

pygame.mixer.init()

STREAM_URL = "http://192.168.167.235:81/stream"
SAVE_DIR = "frames"
SIREN_FILE = "yolov5/siren.mp3"

CONF_TH = 0.3
RECONNECT_DELAY = 2
SIREN_DURATION = 300

os.makedirs(SAVE_DIR, exist_ok=True)

model = torch.hub.load(
    "yolov5",
    "custom",
    path="yolov5/runs/train/exp8/weights/best.pt",
    source="local"
)

model.conf = CONF_TH
names = model.names

def connect_camera():
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def estimate_temp(area):
    return min(200 + area * 0.02, 1000)

cap = connect_camera()
siren_on = False
siren_end = None

print("[INFO] Fire Detection Started")

while True:
    if cap is None or not cap.isOpened():
        print("[WARN] Reconnecting camera...")
        time.sleep(RECONNECT_DELAY)
        cap = connect_camera()
        continue

    ret, frame = cap.read()

    if not ret or frame is None:
        cap.release()
        time.sleep(RECONNECT_DELAY)
        cap = connect_camera()
        continue

    img = cv2.resize(frame, (640, 480))
    results = model(img, size=640)

    fire = False

    if results.xyxy[0] is not None:
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det

            if conf > CONF_TH and names[int(cls)] == "fire":
                fire = True
                area = (x2 - x1) * (y2 - y1)
                temp = estimate_temp(area)

                cv2.rectangle(img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0,0,255), 2)

                cv2.putText(img, f"FIRE {int(temp)}C",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

                cv2.imwrite(f"{SAVE_DIR}/{int(time.time())}.jpg", img)

    if fire:
        if not siren_on:
            pygame.mixer.music.load(SIREN_FILE)
            pygame.mixer.music.play(-1)
            siren_end = datetime.now() + timedelta(seconds=SIREN_DURATION)
            siren_on = True
    else:
        if siren_on and datetime.now() > siren_end:
            pygame.mixer.music.stop()
            siren_on = False

    cv2.imshow("ESP32-CAM Fire Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()
