import cv2
import threading
import torch
import time
import pigpio
from picamera2 import Picamera2

# -------------------------------
# SERVO SETUP (PAN = 18, TILT = 19)
# -------------------------------
pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

PAN_PIN = 18
TILT_PIN = 19

pi.set_mode(PAN_PIN, pigpio.OUTPUT)
pi.set_mode(TILT_PIN, pigpio.OUTPUT)

# Your custom starting angles
pan_angle = 168.9
tilt_angle = 131.8

# -------------------------------
# FIXED SERVO FUNCTION (Arduino-compatible range)
# -------------------------------
def set_servo(pin, angle):
    angle = max(0, min(180, angle))  # ensure safe range
    pulse = 544 + (angle / 180.0) * (2400 - 544)  # Arduino Servo.h mapping
    pi.set_servo_pulsewidth(pin, pulse)

# Initialize servo positions
set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)

# --------------------------------
# YOLO MODEL
# --------------------------------
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
TARGET_CLASS = "cell phone"

# --------------------------------
# CAMERA SETUP
# --------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
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
# MAIN LOOP
# --------------------------------
def main():
    global latest_frame, pan_angle, tilt_angle

    frame_w, frame_h = 640, 480
    tracking_enabled = False

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
                tracking_enabled = True

                x1, y1, x2, y2 = map(int, xyxy)

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Center of detected phone
                phone_x = (x1 + x2) // 2
                phone_y = (y1 + y2) // 2
                center_x = frame_w // 2
                center_y = frame_h // 2

                # -----------------------------------------
                # SERVO MOVEMENT
                # -----------------------------------------
                if tracking_enabled:
                    error_x = phone_x - center_x
                    error_y = phone_y - center_y

                    # PAN normal direction
                    pan_angle -= error_x * 0.03

                    # TILT inverted
                    tilt_angle -= error_y * 0.03

                    # Clamp safe limits
                    pan_angle = max(0, min(180, pan_angle))
                    tilt_angle = max(30, min(150, tilt_angle))

                    # Apply servo commands
                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)

                    # Print angles
                    print(f"PAN: {pan_angle:.1f}°,  TILT: {tilt_angle:.1f}°")

        # -----------------------------
        # DISPLAY ANGLES ON SCREEN
        # -----------------------------
        cv2.putText(frame,
                    f"PAN: {pan_angle:.1f}  |  TILT: {tilt_angle:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.imshow("Phone Tracking Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
