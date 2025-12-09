import cv2
import time
import threading
import pigpio
from picamera2 import Picamera2
from ultralytics import YOLO

pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

PAN_PIN = 18
TILT_PIN = 19

pan_angle = 90.0
tilt_angle = 131.0

def set_servo(pin, angle):
    angle = max(0.0, min(180.0, float(angle)))
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)

print("Loading NCNN YOLO model...")
model = YOLO("/home/yolo/yolo/yolo11n_ncnn_model")
print("Model loaded!")

TARGET_CLASS_ID = 39

FRAME_W = 640
FRAME_H = 480

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
)
picam2.configure(config)
picam2.start()

# THREAD SAFE DETECTION DATA
latest_boxes = []
last_detect = None
last_detect_time = 0.0

stop_thread = False


# -------------------------------------------------------------------
# YOLO THREAD – runs in background so video NEVER lags
# -------------------------------------------------------------------
def yolo_thread():
    global latest_boxes, last_detect, last_detect_time, stop_thread

    while not stop_thread:
        frame = picam2.capture_array()

        results = model(frame, verbose=False)

        boxes_out = []
        det_bottle = None

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            cls_list = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
            xyxy_list = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()

            for cls_id, box_xyxy in zip(cls_list, xyxy_list):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = [int(v) for v in box_xyxy]

                boxes_out.append((cls_id, x1, y1, x2, y2))

                if cls_id == TARGET_CLASS_ID and det_bottle is None:
                    det_bottle = (x1, y1, x2, y2)

        latest_boxes = boxes_out

        if det_bottle:
            last_detect = det_bottle
            last_detect_time = time.time()
        else:
            if time.time() - last_detect_time > 0.7:
                last_detect = None


# start background detector
threading.Thread(target=yolo_thread, daemon=True).start()


# -------------------------------------------------------------------
# MAIN LOOP (Only drawing + servo control => smooth)
# -------------------------------------------------------------------
def main():
    global pan_angle, tilt_angle, stop_thread

    YOLO_INTERVAL = 0.20
    DETECT_TIMEOUT = 0.7
    SERVO_INTERVAL = 0.05

    KP_PAN = 0.04
    KP_TILT = 0.04
    DEADZONE_PIXELS = 5

    last_servo_update = 0.0

    prev_time = time.time()

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            display_frame = frame

            now = time.time()

            # Draw ALL detections (red), bottle is overwritten in green below
            for cls_id, x1, y1, x2, y2 in latest_boxes:
                if cls_id != TARGET_CLASS_ID:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(display_frame, str(cls_id), (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                print(f"Object ID {cls_id} at {x1},{y1},{x2},{y2}")

            bottle_found = last_detect is not None

            center_x = FRAME_W // 2
            center_y = FRAME_H // 2

            if bottle_found:
                x1, y1, x2, y2 = last_detect

                cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(display_frame, "bottle", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                bottle_x = (x1 + x2) // 2
                bottle_y = (y1 + y2) // 2

                cv2.circle(display_frame, (bottle_x, bottle_y), 4,
                           (0, 255, 0), -1)

                error_x = bottle_x - center_x
                error_y = bottle_y - center_y

                if abs(error_x) < DEADZONE_PIXELS:
                    error_x = 0
                if abs(error_y) < DEADZONE_PIXELS:
                    error_y = 0

                if now - last_servo_update >= SERVO_INTERVAL:
                    pan_angle -= error_x * KP_PAN
                    tilt_angle -= error_y * KP_TILT

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

            # FPS COUNTER
            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()

            cv2.putText(display_frame, f"FPS:{fps:.1f}",
                        (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        print("Stopping...")
        stop_thread = True
        pi.set_servo_pulsewidth(PAN_PIN, 0)
        pi.set_servo_pulsewidth(TILT_PIN, 0)
        pi.stop()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
