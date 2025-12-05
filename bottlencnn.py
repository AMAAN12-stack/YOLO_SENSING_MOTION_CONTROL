import cv2
import threading
import time
import pigpio
import numpy as np
import ncnn
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
# NCNN YOLOv5n WRAPPER
# -------------------------------
class YOLOv5_NCNN:
    def __init__(self, param_path, bin_path):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # CPU only (stable on Pi)
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # COCO ID for "bottle"
        self.target_class_id = 39

    def detect(self, frame):
        h, w = frame.shape[:2]

        # Convert BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 640×640 (YOLOv5 input)
        img_resized = cv2.resize(img, (640, 640))

        # NCNN input
        mat_in = ncnn.Mat.from_pixels(
            img_resized, ncnn.Mat.PixelType.PIXEL_RGB, 640, 640
        )

        # Normalize
        mat_in.substract_mean_normalize(
            mean_vals=[0.0, 0.0, 0.0],
            norm_vals=[1/255.0, 1/255.0, 1/255.0]
        )

        ex = self.net.create_extractor()
        ret = ex.input("images", mat_in)

        ret, mat_out = ex.extract("output")
        if ret != 0:
            return None

        best_det = None
        best_score = 0

        for i in range(mat_out.h):
            values = mat_out.row(i)

            x1, y1, x2, y2, score, class_id = values

            if int(class_id) == self.target_class_id and score > best_score:
                best_score = score
                # Scale back to original resolution
                sx = w / 640
                sy = h / 640
                best_det = (
                    int(x1 * sx),
                    int(y1 * sy),
                    int(x2 * sx),
                    int(y2 * sy)
                )

        return best_det

# Load NCNN model
print("Loading YOLOv5n NCNN model...")
yolo = YOLOv5_NCNN("yolov5n.param", "yolov5n.bin")
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

    detect_interval = 2
    frame_counter = 0

    prev_time = time.time()

    print("Tracking started...")

    last_detect = None

    while True:
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        small_frame = cv2.resize(frame, (320, 240))

        # Run detection every N frames
        if frame_counter % detect_interval == 0:
            last_detect = yolo.detect(small_frame)

            # Scale small-frame detections → full 640×480
            if last_detect is not None:
                x1, y1, x2, y2 = last_detect
                x1 *= 2
                y1 *= 2
                x2 *= 2
                y2 *= 2
                last_detect = (x1, y1, x2, y2)

        frame_counter += 1

        bottle_found = last_detect is not None

        if bottle_found:
            x1, y1, x2, y2 = last_detect

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, "bottle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            bottle_x = (x1 + x2) // 2
            bottle_y = (y1 + y2) // 2

            center_x = 320
            center_y = 240

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

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        cv2.imshow("Bottle Tracking (NCNN)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pi.set_servo_pulsewidth(PAN_PIN, 0)
    pi.set_servo_pulsewidth(TILT_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
