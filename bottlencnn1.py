import cv2
import time
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
    """Set servo to angle in degrees (0–180)."""
    angle = max(0.0, min(180.0, float(angle)))
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)


print("Loading NCNN YOLO model...")
model = YOLO("/home/yolo/yolo/yolo11n_ncnn_model")
print("Model loaded!")

TARGET_CLASS_ID = 39  # bottle in COCO


FRAME_W = 640
FRAME_H = 480

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

    last_detect = None  # (x1, y1, x2, y2) in 640x480 space

    # Store last detections (for drawing & printing)
    last_detections = []  # list of dicts: {"cls_id", "name", "x1","y1","x2","y2"}

    prev_time = time.time()
    frame_counter = 0

    # For smoother tracking: low-pass filter on error
    filtered_error_x = 0.0
    filtered_error_y = 0.0
    ERROR_SMOOTHING = 0.5  # 0=no smoothing, 1=full smoothing

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            # Get latest frame from camera
            frame = picam2.capture_array()  # RGB888, 640x480

            now = time.time()

            # -------------------------------
            # Run YOLO at limited rate (on smaller frame for speed)
            # -------------------------------
            if now - last_yolo_time >= YOLO_INTERVAL:
                last_yolo_time = now

                # Downscale for faster inference
                small_w = FRAME_W // 2
                small_h = FRAME_H // 2
                small_frame = cv2.resize(frame, (small_w, small_h))

                results = model(small_frame, verbose=False)

                last_detect = None
                last_detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes

                    # Get class IDs and boxes
                    cls_list = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
                    xyxy_list = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()

                    names = getattr(model, "names", None)

                    scale_x = FRAME_W / float(small_w)
                    scale_y = FRAME_H / float(small_h)

                    # Collect detections for printing and drawing
                    for cls_id, box_xyxy in zip(cls_list, xyxy_list):
                        cls_id = int(cls_id)
                        x1, y1, x2, y2 = box_xyxy

                        # Map back to full 640x480 coordinates
                        x1 = int(x1 * scale_x)
                        x2 = int(x2 * scale_x)
                        y1 = int(y1 * scale_y)
                        y2 = int(y2 * scale_y)

                        # Clamp to frame bounds
                        x1 = max(0, min(FRAME_W - 1, x1))
                        x2 = max(0, min(FRAME_W - 1, x2))
                        y1 = max(0, min(FRAME_H - 1, y1))
                        y2 = max(0, min(FRAME_H - 1, y2))

                        if names is not None and cls_id in names:
                            cls_name = str(names[cls_id])
                        else:
                            cls_name = f"class_{cls_id}"

                        det = {
                            "cls_id": cls_id,
                            "name": cls_name,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                        last_detections.append(det)

                    # Print detections line by line in terminal
                    if last_detections:
                        print("Detections:")
                        for det in last_detections:
                            print(
                                f"  {det['name']} (id={det['cls_id']}) "
                                f"at [{det['x1']}, {det['y1']}, {det['x2']}, {det['y2']}]"
                            )
                    else:
                        print("No objects detected.")

                    # Select bottle for tracking (first bottle)
                    for det in last_detections:
                        if det["cls_id"] == TARGET_CLASS_ID:
                            last_detect = (
                                det["x1"],
                                det["y1"],
                                det["x2"],
                                det["y2"],
                            )
                            last_detect_time = now
                            break

            # If last detection is too old, consider target lost
            if last_detect is not None and now - last_detect_time > DETECT_TIMEOUT:
                last_detect = None

            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            # Draw directly on the captured frame (avoid extra copy for speed)
            display_frame = frame

            bottle_found = last_detect is not None

            center_x = FRAME_W // 2
            center_y = FRAME_H // 2

            # First draw non-bottle objects (red)
            for det in last_detections:
                if det["cls_id"] != TARGET_CLASS_ID:
                    x1 = det["x1"]
                    y1 = det["y1"]
                    x2 = det["x2"]
                    y2 = det["y2"]
                    name = det["name"]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)  # red
                    cv2.putText(display_frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

            if bottle_found:
                x1, y1, x2, y2 = last_detect

                # Draw bounding box and label for bottle (green)
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

                # Smooth errors for smoother tracking
                filtered_error_x = (
                    ERROR_SMOOTHING * filtered_error_x
                    + (1.0 - ERROR_SMOOTHING) * error_x
                )
                filtered_error_y = (
                    ERROR_SMOOTHING * filtered_error_y
                    + (1.0 - ERROR_SMOOTHING) * error_y
                )

                # Update servo angles at limited rate
                if now - last_servo_update >= SERVO_INTERVAL:
                    # Pan left/right: screen x -> servo pan
                    pan_angle -= filtered_error_x * KP_PAN
                    # Tilt up/down: screen y -> servo tilt
                    tilt_angle -= filtered_error_y * KP_TILT

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
            # Show video
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
