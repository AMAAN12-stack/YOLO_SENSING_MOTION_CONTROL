import cv2
import threading
import torch
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# -----------------------------
# Load YOLOv5 model
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

# -----------------------------
# Initialize Pi Camera
# -----------------------------
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration(main={"size": (640, 480)}))
picam2.start()

frame = None
running = True

# -----------------------------
# Camera thread (fast capture)
# -----------------------------
def camera_thread():
    global frame, running
    while running:
        frame = picam2.capture_array()
        time.sleep(0.001)

threading.Thread(target=camera_thread, daemon=True).start()

# -----------------------------
# Servo setup
# -----------------------------
GPIO.setmode(GPIO.BCM)

PAN_PIN = 18
TILT_PIN = 19

GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_servo = GPIO.PWM(PAN_PIN, 50)
tilt_servo = GPIO.PWM(TILT_PIN, 50)

pan_servo.start(7.5)   # Center
tilt_servo.start(7.5)

pan_angle = 90
tilt_angle = 90

def angle_to_duty(angle):
    return 2.5 + (angle / 18.0)

# -----------------------------
# Object tracking loop
# -----------------------------
FRAME_W = 640
FRAME_H = 480
center_x_target = FRAME_W // 2
center_y_target = FRAME_H // 2

while True:
    if frame is None:
        continue

    results = model(frame)
    detections = results.xyxy[0]  # bounding boxes

    if len(detections) > 0:
        # Pick the largest detected object (phone)
        det = max(detections, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))

        x1, y1, x2, y2, conf, cls = det

        # Calculate object center
        obj_x = int((x1 + x2) / 2)
        obj_y = int((y1 + y2) / 2)

        # -----------------------------
        # PAN (left-right)
        # -----------------------------
        if obj_x < center_x_target - 30:
            pan_angle += 1
        elif obj_x > center_x_target + 30:
            pan_angle -= 1

        pan_angle = max(0, min(180, pan_angle))
        pan_servo.ChangeDutyCycle(angle_to_duty(pan_angle))

        # -----------------------------
        # TILT (up-down)
        # -----------------------------
        if obj_y < center_y_target - 30:
            tilt_angle -= 1
        elif obj_y > center_y_target + 30:
            tilt_angle += 1

        tilt_angle = max(0, min(180, tilt_angle))
        tilt_servo.ChangeDutyCycle(angle_to_duty(tilt_angle))

        annotated = results.render()[0]
    else:
        annotated = frame

    cv2.imshow("YOLOv5 Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
picam2.stop()
pan_servo.stop()
tilt_servo.stop()
GPIO.cleanup()
