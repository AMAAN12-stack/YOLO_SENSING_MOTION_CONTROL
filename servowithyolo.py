import cv2
import threading
import torch
import time
from picamera2 import Picamera2
from pynput import keyboard
import RPi.GPIO as GPIO

# ===============================
# YOLO MODEL
# ===============================
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.5
PHONE_CLASS = 67

# ===============================
# CAMERA SETUP
# ===============================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

current_frame = None
annotated_frame = None
frame_lock = threading.Lock()
skip_counter = 0
running = True

# ===============================
# SERVO SETUP (GPIO)
# ===============================
PAN_PIN = 18
TILT_PIN = 19

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_pwm = GPIO.PWM(PAN_PIN, 50)
tilt_pwm = GPIO.PWM(TILT_PIN, 50)

pan_pwm.start(0)
tilt_pwm.start(0)

pan_angle = 90
tilt_angle = 90

def move_servo(pwm, angle):
    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10.0
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    pwm.ChangeDutyCycle(0)
    return angle

# ===============================
# YOLO INFERENCE THREAD
# ===============================
def yolo_thread():
    global annotated_frame, skip_counter, running

    while running:
        time.sleep(0.01)

        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Skip some frames to reduce lag
        if skip_counter % 3 == 0:
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == PHONE_CLASS:
                    print(f"[PHONE DETECTED] conf={conf:.2f}")

            rendered = results.render()[0]  # already BGR
            annotated_frame = rendered

        skip_counter += 1

# ===============================
# KEYBOARD CONTROL THREAD
# ===============================
def on_press(key):
    global pan_angle, tilt_angle, running

    try:
        if key == keyboard.Key.left:
            pan_angle -= 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        elif key == keyboard.Key.right:
            pan_angle += 5
            pan_angle = move_servo(pan_pwm, pan_angle)
            print("Pan:", pan_angle)

        elif key == keyboard.Key.up:
            tilt_angle += 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        elif key == keyboard.Key.down:
            tilt_angle -= 5
            tilt_angle = move_servo(tilt_pwm, tilt_angle)
            print("Tilt:", tilt_angle)

        elif key.char == 'q':
            print("Exiting…")
            running = False
            return False

    except AttributeError:
        pass

# ===============================
# START THREADS
# ===============================
threading.Thread(target=yolo_thread, daemon=True).start()
listener = keyboard.Listener(on_press=on_press)
listener.start()

# ===============================
# MAIN LOOP (DISPLAY)
# ===============================
try:
    while running:
        frame = picam2.capture_array()
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        with frame_lock:
            current_frame = rgb

        if annotated_frame is not None:
            cv2.imshow("YOLOv5 + Servo Control", annotated_frame)
        else:
            cv2.imshow("YOLOv5 + Servo Control", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

except KeyboardInterrupt:
    running = False

# ===============================
# CLEANUP
# ===============================
listener.stop()
pan_pwm.stop()
tilt_pwm.stop()
GPIO.cleanup()
picam2.stop()
cv2.destroyAllWindows()

print("Shutdown complete.")
