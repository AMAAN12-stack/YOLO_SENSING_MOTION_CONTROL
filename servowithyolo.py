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

pan.start(7.5)   # center
tilt.start(7.5)  # center

def set_servo(pwm, angle):
    duty = 2 + (angle / 18)   # convert angle° → duty cycle
    pwm.ChangeDutyCycle(duty)

pan_angle = 90
tilt_angle = 90

# -------------------------------
# YOLO MODEL
# -------------------------------
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
TARGET_CLASS = "cell phone"

# -------------------------------
# CAMERA SETUP
# -------------------------------
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

# -------------------------------
# MAIN TRACKING LOOP
# -------------------------------
def main():
    global latest_frame, pan_angle, tilt_angle

    frame_w, frame_h = 640, 480

    while True:
        if latest_frame is None:
            continue

        with frame_lock:
            frame = latest_frame.copy()

        results = model(frame)

        phone_detected = False

        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]

            if label == TARGET_CLASS:
                phone_detected = True
                x1, y1, x2, y2 = map(int, xyxy)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ----------------------
                # CENTER OF PHONE
                # ----------------------
                phone_x = (x1 + x2) // 2
                phone_y = (y1 + y2) // 2

                # SCREEN CENTER
                center_x = frame_w // 2
                center_y = frame_h // 2

                # ----------------------
                # SERVO CONTROL
                # ----------------------
                error_x = phone_x - center_x
                error_y = phone_y - center_y

                # Adjust angles (gain factor 0.03)
                pan_angle -= error_x * 0.03
                tilt_angle += error_y * 0.03

                # Limit angles to servo range
                pan_angle = max(0, min(180, pan_angle))
                tilt_angle = max(0, min(180, tilt_angle))

                # Move servos
                set_servo(pan, pan_angle)
                set_servo(tilt, tilt_angle)

        cv2.imshow("Phone Tracking Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pan.stop()
    tilt.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
