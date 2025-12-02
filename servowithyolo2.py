import cv2
import threading
import torch
import time
import pigpio
from picamera2 import Picamera2

# ----------------------------------------
# SERVO SETUP
# ----------------------------------------
PAN_PIN = 18
TILT_PIN = 19

pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

pi.set_mode(PAN_PIN, pigpio.OUTPUT)
pi.set_mode(TILT_PIN, pigpio.OUTPUT)

# initial angles
pan_angle = 168.9
tilt_angle = 131.8

def set_servo(pin, angle):
    angle = max(0, min(180, angle))
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)


# apply initial angles
set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)

# ----------------------------------------
# YOLOV5N (TorchHub version for Pi4)
# ----------------------------------------
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
TARGET_CLASS = "cell phone"

# reduce work
model.conf = 0.35
model.iou = 0.45

# ----------------------------------------
# CAMERA SETUP (FAST MODE)
# ----------------------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (320, 240)},  # SMALLER = MUCH FASTER
    controls={"FrameDurationLimits": (33333, 33333)},  # 30 fps
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


# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
def main():
    global latest_frame, pan_angle, tilt_angle

    tracking_enabled = False
    frame_count = 0

    while True:
        if latest_frame is None:
            continue

        frame_count += 1

        with frame_lock:
            frame = latest_frame.copy()

        # skip YOLO every 2 frames (massive speed boost)
        if frame_count % 3 != 0:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # YOLO DETECTION
        results = model(frame)
        phone_found = False

        x_center = 160  # half of 320
        y_center = 120  # half of 240

        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]

            if label == TARGET_CLASS:
                phone_found = True
                tracking_enabled = True

                x1, y1, x2, y2 = map(int, xyxy)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # ------------------------------------
                # SERVO CONTROL (smooth & stable)
                # ------------------------------------
                error_x = cx - x_center
                error_y = cy - y_center

                # small smoothing factor (PID-like)
                pan_angle = 0.8 * pan_angle + 0.2 * (pan_angle - error_x * 0.08)
                tilt_angle = 0.8 * tilt_angle + 0.2 * (tilt_angle - error_y * 0.08)

                # clamp
                pan_angle = max(0, min(180, pan_angle))
                tilt_angle = max(30, min(150, tilt_angle))

                # apply
                set_servo(PAN_PIN, pan_angle)
                set_servo(TILT_PIN, tilt_angle)

        # display angles
        cv2.putText(frame, f"PAN: {pan_angle:.1f} | TILT: {tilt_angle:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # cleanup
    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
