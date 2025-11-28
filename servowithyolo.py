import cv2
import threading
import torch
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# -------------------------------
# SERVO SETUP (PAN = 18, TILT = 19)
# -------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Pan
GPIO.setup(19, GPIO.OUT)  # Tilt

pan = GPIO.PWM(18, 50)   # 50Hz
tilt = GPIO.PWM(19, 50)

pan_angle = 90
tilt_angle = 90

pan.start(7.5)   # Center position
tilt.start(7.5)

def set_servo(pwm, angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)

# --------------------------------
# YOLO MODEL
# --------------------------------
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
TARGET_CLASS = "cell phone"

# --------------------------------
# CAMERA SETUP
# --------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

frame_lock = threading.Lock()
latest_frame = None

def capture_frames():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        with frame_lock:
            latest_frame = frame

threading.Thread(target=capture_frames, daemon=True).start()


# --------------------------------
# MAIN LOOP (UPDATED FOR "MOVE ONLY IF PHONE")
# --------------------------------
def main():
    global latest_frame, pan_angle, tilt_angle

    frame_w, frame_h = 640, 480
    tracking_enabled = False  # <--- New: start with servo disabled

    while True:
        if latest_frame is None:
            continue

        with frame_lock:
            frame = latest_frame.copy()

        results = model(frame)

        phone_found = False

        # -----------------------------
        # YOLO Detection
        # -----------------------------
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == TARGET_CLASS:
                phone_found = True
                tracking_enabled = True  # <--- Enable tracking ONLY after first phone detection

                x1, y1, x2, y2 = map(int, xyxy)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Center of phone
                phone_x = (x1 + x2) // 2
                phone_y = (y1 + y2) // 2

                center_x = frame_w // 2
                center_y = frame_h // 2

                # -----------------------------------------------------
                # MOVE SERVOS ONLY IF tracking_enabled = TRUE
                # -----------------------------------------------------
                if tracking_enabled:
                    error_x = phone_x - center_x
                    error_y = phone_y - center_y

                    # Servo adjustments
                    pan_angle -= error_x * 0.03
                    tilt_angle += error_y * 0.03

                    # Limit
                    pan_angle = max(0, min(180, pan_angle))
                    tilt_angle = max(0, min(180, tilt_angle))

                    # MOVE SERVOS
                    set_servo(pan, pan_angle)
                    set_servo(tilt, tilt_angle)

        # ----------------------------------------------------------------
        # If NO phone detected → do nothing (servos freeze & stay still)
        # ----------------------------------------------------------------
        if not phone_found:
            pass  # Do not move servos

        cv2.imshow("Phone Tracking Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pan.stop()
    tilt.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
