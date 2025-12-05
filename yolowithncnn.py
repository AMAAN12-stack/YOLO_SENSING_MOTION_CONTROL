import cv2
import time
import pigpio
from picamera2 import Picamera2
from ultralytics import YOLO

# -------------------------------
# SERVO SETUP
# -------------------------------
pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

PAN_PIN = 18
TILT_PIN = 19

# Start pointed roughly centre
pan_angle = 90.0
tilt_angle = 131.0

def set_servo(pin, angle):
    """Set servo to angle in degrees (0–180)."""
    angle = max(0.0, min(180.0, float(angle)))
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)

# -------------------------------
# LOAD NCNN MODEL
# -------------------------------
# You must first run (offline, once):
#   model = YOLO("yolov5n.pt")
#   model.export(format="ncnn")
# This creates ./yolov5n_ncnn_model/
print("Loading NCNN YOLO model...")
model = YOLO("/home/yolo/yolo/yolov5/yolo11n_ncnn_model")
print("Model loaded!")

TARGET_CLASS_ID = 39  # bottle in COCO

# -------------------------------
# CAMERA SETUP
# -------------------------------
FRAME_W = 640
FRAME_H =  480

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
)
picam2.configure(config)
picam2.start()

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    global pan_angle, tilt_angle

    # Detection + tracking parameters
    YOLO_INTERVAL = 0.20      # run YOLO at most every 200 ms
    DETECT_TIMEOUT = 0.7      # forget target if not seen for 700 ms
    SERVO_INTERVAL = 0.05     # update servos at most every 50 ms (20 Hz)

    # PID-like gains
    KP_PAN = 0.04
    KP_TILT = 0.04
    DEADZONE_PIXELS = 5       # ignore very small errors

    last_yolo_time = 0.0
    last_detect_time = 0.0
    last_servo_update = 0.0

    last_detect = None  # (x1, y1, x2, y2) in 320x240 space

    prev_time = time.time()
    frame_counter = 0

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            # Get latest frame from camera (blocking, but cheap at 320x240)
            frame = picam2.capture_array()  # RGB888, 320x240

            now = time.time()

            # -------------------------------
            # Run YOLO at limited rate
            # -------------------------------
            if now - last_yolo_time >= YOLO_INTERVAL:
                results = model(frame, verbose=False)
                last_yolo_time = now

                last_detect = None

                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        if cls == TARGET_CLASS_ID:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            # Clamp to frame bounds just in case
                            x1 = max(0, min(FRAME_W - 1, int(x1)))
                            x2 = max(0, min(FRAME_W - 1, int(x2)))
                            y1 = max(0, min(FRAME_H - 1, int(y1)))
                            y2 = max(0, min(FRAME_H - 1, int(y2)))
                            last_detect = (x1, y1, x2, y2)
                            last_detect_time = now
                            break  # use first bottle

            # If last detection is too old, consider target lost
            if last_detect is not None and now - last_detect_time > DETECT_TIMEOUT:
                last_detect = None

            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            display_frame = frame.copy()
            bottle_found = last_detect is not None

            center_x = FRAME_W // 2
            center_y = FRAME_H // 2

            if bottle_found:
                x1, y1, x2, y2 = last_detect

                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(display_frame, "bottle", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                bottle_x = (x1 + x2) // 2
                bottle_y = (y1 + y2) // 2

                # Draw crosshair for target
                cv2.circle(display_frame, (bottle_x, bottle_y), 4,
                           (0, 255, 0), -1)

                error_x = bottle_x - center_x
                error_y = bottle_y - center_y

                # Optional: small deadzone to avoid jitter
                if abs(error_x) < DEADZONE_PIXELS:
                    error_x = 0
                if abs(error_y) < DEADZONE_PIXELS:
                    error_y = 0

                # Update servo angles at limited rate
                if now - last_servo_update >= SERVO_INTERVAL:
                    # Pan left/right: screen x -> servo pan
                    pan_angle -= error_x * KP_PAN
                    # Tilt up/down: screen y -> servo tilt
                    tilt_angle -= error_y * KP_TILT

                    # Clamp angles
                    pan_angle = max(0.0, min(180.0, pan_angle))
                    tilt_angle = max(30.0, min(150.0, tilt_angle))

                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)

                    last_servo_update = now

            else:
                cv2.putText(display_frame, "Bottle not found",
                            (60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            # -------------------------------
            # FPS counter
            # -------------------------------
            frame_counter += 1
            now2 = time.time()
            fps = 1.0 / (now2 - prev_time) if now2 != prev_time else 0.0
            prev_time = now2

            cv2.putText(display_frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                        (10, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"FPS:{fps:.1f}",
                        (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            # -------------------------------
            # Show video (this itself can be a bottleneck)
            # -------------------------------
            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        # Cleanup
        print("Stopping, cleaning up...")
        pi.set_servo_pulsewidth(PAN_PIN, 0)
        pi.set_servo_pulsewidth(TILT_PIN, 0)
        pi.stop()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
