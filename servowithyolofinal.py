import cv2
import threading
import torch
import time
import pigpio
from picamera2 import Picamera2

# -------------------------------
# SERVO SETUP
# -------------------------------
pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

PAN_PIN = 18
TILT_PIN = 19

pan_angle = 90
tilt_angle = 131

def set_servo(pin, angle):
    angle = max(0, min(180, angle))
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)

# -------------------------------
# YOLOv5
# -------------------------------
print("Loading YOLOv5 model...")
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

# Enable FP16 if supported (much faster)
if torch.cuda.is_available():
    model.half()

TARGET_CLASS = "bottle"
print("Model loaded!")

# -------------------------------
# CAMERA SETUP
# -------------------------------
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
        latest_frame = picam2.capture_array()

threading.Thread(target=capture_frames, daemon=True).start()

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    global latest_frame, pan_angle, tilt_angle

    frame_w, frame_h = 640, 480

    last_detect = None
    detect_interval = 3  # run YOLO every 3 frames
    frame_counter = 0

    prev_time = time.time()
    fps = 0

    print("Tracking started...")

    while True:
        if latest_frame is None:
            continue

        frame = latest_frame

        small_frame = cv2.resize(frame, (320, 240))

        # -------------------------------
        # Run YOLO only every N frames
        # -------------------------------
        if frame_counter % detect_interval == 0:
            with torch.no_grad():
                results = model(small_frame)
                detections = results.xyxy[0]
                last_detect = None

                for *xyxy, conf, cls in detections:
                    label = model.names[int(cls)]
                    if label == TARGET_CLASS:
                        last_detect = xyxy
                        break

        frame_counter += 1

        bottle_found = last_detect is not None

        if bottle_found:
            x1, y1, x2, y2 = last_detect

            # scale back to 640x480
            x1, y1, x2, y2 = [int(v * 2) for v in (x1, y1, x2, y2)]

            # bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, "bottle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            bottle_x = (x1 + x2) // 2
            bottle_y = (y1 + y2) // 2

            center_x = 320
            center_y = 240

            # Servo PID-like movement
            error_x = bottle_x - center_x
            error_y = bottle_y - center_y

            pan_angle -= error_x * 0.02
            tilt_angle -= error_y * 0.02

            pan_angle = max(0, min(180, pan_angle))
            tilt_angle = max(30, min(150, tilt_angle))

            set_servo(PAN_PIN, pan_angle)
            set_servo(TILT_PIN, tilt_angle)

        else:
            cv2.putText(frame, "Bottle not found",
                        (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw angles and FPS
        cv2.putText(frame, f"PAN:{pan_angle:.1f}  TILT:{tilt_angle:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Bottle Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
