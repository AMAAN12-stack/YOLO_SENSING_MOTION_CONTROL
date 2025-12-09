import cv2
import time
import pigpio
from picamera2 import Picamera2
from ultralytics import YOLO
import threading
from collections import deque

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
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)  # Let camera warm up

# Global variables for threaded detection
detection_lock = threading.Lock()
shared_detections = []
shared_last_detect = None
shared_last_detect_time = 0.0
stop_detection_thread = False

# Kalman-like prediction for smoother tracking
class TargetPredictor:
    def __init__(self, smoothing=0.7):
        self.smoothing = smoothing
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        
    def update(self, new_x, new_y):
        if self.x is None:
            self.x = float(new_x)
            self.y = float(new_y)
        else:
            # Update velocity
            self.vx = (new_x - self.x) * (1.0 - self.smoothing) + self.vx * self.smoothing
            self.vy = (new_y - self.y) * (1.0 - self.smoothing) + self.vy * self.smoothing
            # Update position
            self.x = self.smoothing * self.x + (1.0 - self.smoothing) * new_x
            self.y = self.smoothing * self.y + (1.0 - self.smoothing) * new_y
    
    def predict(self, dt=0.05):
        if self.x is None:
            return None, None
        return self.x + self.vx * dt, self.y + self.vy * dt
    
    def get_position(self):
        return self.x, self.y

predictor = TargetPredictor(smoothing=0.65)

def detection_thread():
    """Background thread for YOLO detection"""
    global shared_detections, shared_last_detect, shared_last_detect_time, stop_detection_thread
    
    YOLO_INTERVAL = 0.15  # Faster detection updates
    last_yolo_time = 0.0
    
    # Smaller inference size for speed
    small_w = FRAME_W // 2
    small_h = FRAME_H // 2
    scale_x = FRAME_W / float(small_w)
    scale_y = FRAME_H / float(small_h)
    
    while not stop_detection_thread:
        now = time.time()
        
        if now - last_yolo_time >= YOLO_INTERVAL:
            last_yolo_time = now
            
            # Capture and resize frame
            frame = picam2.capture_array()
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            
            # Run YOLO
            results = model(small_frame, verbose=False, conf=0.5)
            
            temp_detections = []
            temp_last_detect = None
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                cls_list = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
                xyxy_list = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
                names = getattr(model, "names", None)
                
                for cls_id, box_xyxy in zip(cls_list, xyxy_list):
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = box_xyxy
                    
                    # Scale to full resolution
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    
                    # Clamp
                    x1 = max(0, min(FRAME_W - 1, x1))
                    x2 = max(0, min(FRAME_W - 1, x2))
                    y1 = max(0, min(FRAME_H - 1, y1))
                    y2 = max(0, min(FRAME_H - 1, y2))
                    
                    cls_name = str(names[cls_id]) if names and cls_id in names else f"class_{cls_id}"
                    
                    det = {
                        "cls_id": cls_id,
                        "name": cls_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                    temp_detections.append(det)
                    
                    # Track first bottle found
                    if cls_id == TARGET_CLASS_ID and temp_last_detect is None:
                        temp_last_detect = (x1, y1, x2, y2)
            
            # Update shared state
            with detection_lock:
                shared_detections = temp_detections
                if temp_last_detect is not None:
                    shared_last_detect = temp_last_detect
                    shared_last_detect_time = now
        
        time.sleep(0.01)  # Small sleep to prevent busy loop

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    global pan_angle, tilt_angle, stop_detection_thread

    # Start detection thread
    det_thread = threading.Thread(target=detection_thread, daemon=True)
    det_thread.start()

    # Detection + tracking parameters
    DETECT_TIMEOUT = 0.8
    SERVO_INTERVAL = 0.03  # Faster servo updates (33 Hz)

    # Improved PID gains for smoother tracking
    KP_PAN = 0.055
    KP_TILT = 0.055
    DEADZONE_PIXELS = 8

    last_servo_update = 0.0
    last_detect = None

    prev_time = time.time()
    frame_counter = 0

    # Enhanced smoothing for servo control
    filtered_error_x = 0.0
    filtered_error_y = 0.0
    ERROR_SMOOTHING = 0.65

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            # Get latest frame (non-blocking)
            frame = picam2.capture_array()
            now = time.time()

            # Get latest detections from thread
            with detection_lock:
                last_detections = shared_detections.copy()
                if shared_last_detect is not None and now - shared_last_detect_time <= DETECT_TIMEOUT:
                    last_detect = shared_last_detect
                elif now - shared_last_detect_time > DETECT_TIMEOUT:
                    last_detect = None
                    predictor.x = None  # Reset predictor

            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            display_frame = frame
            bottle_found = last_detect is not None

            center_x = FRAME_W // 2
            center_y = FRAME_H // 2

            # Draw non-bottle objects
            for det in last_detections:
                if det["cls_id"] != TARGET_CLASS_ID:
                    cv2.rectangle(display_frame, (det["x1"], det["y1"]), 
                                (det["x2"], det["y2"]), (0, 0, 255), 2)
                    cv2.putText(display_frame, det["name"], 
                              (det["x1"], det["y1"] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if bottle_found:
                x1, y1, x2, y2 = last_detect

                # Draw bottle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "bottle", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                bottle_x = (x1 + x2) // 2
                bottle_y = (y1 + y2) // 2

                # Update predictor
                predictor.update(bottle_x, bottle_y)
                
                # Use predicted position for smoother tracking
                pred_x, pred_y = predictor.get_position()
                track_x = int(pred_x) if pred_x is not None else bottle_x
                track_y = int(pred_y) if pred_y is not None else bottle_y

                cv2.circle(display_frame, (track_x, track_y), 5, (0, 255, 0), -1)

                error_x = track_x - center_x
                error_y = track_y - center_y

                # Deadzone
                if abs(error_x) < DEADZONE_PIXELS:
                    error_x = 0
                if abs(error_y) < DEADZONE_PIXELS:
                    error_y = 0

                # Smooth errors
                filtered_error_x = ERROR_SMOOTHING * filtered_error_x + (1.0 - ERROR_SMOOTHING) * error_x
                filtered_error_y = ERROR_SMOOTHING * filtered_error_y + (1.0 - ERROR_SMOOTHING) * error_y

                # Update servos
                if now - last_servo_update >= SERVO_INTERVAL:
                    pan_angle -= filtered_error_x * KP_PAN
                    tilt_angle -= filtered_error_y * KP_TILT

                    pan_angle = max(0.0, min(180.0, pan_angle))
                    tilt_angle = max(30.0, min(150.0, tilt_angle))

                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)

                    last_servo_update = now
            else:
                cv2.putText(display_frame, "Bottle not found", (60, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # -------------------------------
            # FPS counter
            # -------------------------------
            frame_counter += 1
            now2 = time.time()
            fps = 1.0 / (now2 - prev_time) if now2 != prev_time else 0.0
            prev_time = now2

            cv2.putText(display_frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                      (10, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display_frame, f"FPS:{fps:.1f}",
                      (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show video
            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        # Cleanup
        print("Stopping, cleaning up...")
        stop_detection_thread = True
        det_thread.join(timeout=2.0)
        pi.set_servo_pulsewidth(PAN_PIN, 0)
        pi.set_servo_pulsewidth(TILT_PIN, 0)
        pi.stop()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
