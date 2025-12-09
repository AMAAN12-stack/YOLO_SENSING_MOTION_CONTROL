import cv2
import time
import pigpio
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

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
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

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
    DETECT_TIMEOUT = 1.0      # forget target if not seen for 1000 ms
    SERVO_INTERVAL = 0.033    # update servos at 30 Hz (33ms)
    
    # PID-like gains
    KP_PAN = 0.035
    KP_TILT = 0.035
    KI_PAN = 0.001
    KI_TILT = 0.001
    KD_PAN = 0.015
    KD_TILT = 0.015
    
    DEADZONE_PIXELS = 8       # ignore very small errors
    
    # Tracking history for smoother movement
    ERROR_HISTORY_SIZE = 3
    error_x_history = []
    error_y_history = []
    
    # Kalman filter-like smoothing for position
    predicted_x = CENTER_X
    predicted_y = CENTER_Y
    position_variance = 50.0
    process_variance = 1.0
    measurement_variance = 25.0

    last_yolo_time = 0.0
    last_detect_time = 0.0
    last_servo_update = 0.0

    last_detect = None  # (x1, y1, x2, y2) in 640x480 space
    
    # Store last detections (for drawing & printing)
    last_detections = []  # list of dicts: {"cls_id", "name", "x1","y1","x2","y2"}
    
    # For integral and derivative terms
    integral_x = 0.0
    integral_y = 0.0
    prev_error_x = 0.0
    prev_error_y = 0.0
    
    # FPS calculation
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0
    
    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            loop_start = time.time()
            
            # Get latest frame from camera
            frame = picam2.capture_array()  # RGB888, 640x480
            display_frame = frame.copy() if last_detect is not None else frame
            
            now = time.time()

            # -------------------------------
            # Run YOLO at limited rate
            # -------------------------------
            if now - last_yolo_time >= YOLO_INTERVAL:
                last_yolo_time = now
                
                # Downscale for faster inference (balanced speed/accuracy)
                small_w = FRAME_W // 2
                small_h = FRAME_H // 2
                small_frame = cv2.resize(frame, (small_w, small_h))
                
                # Run inference with minimal preprocessing
                results = model(small_frame, verbose=False, conf=0.5)
                
                last_detect = None
                last_detections = []
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    # Get class IDs and boxes
                    cls_list = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()
                    xyxy_list = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
                    conf_list = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
                    
                    names = getattr(model, "names", None)
                    
                    scale_x = FRAME_W / float(small_w)
                    scale_y = FRAME_H / float(small_h)
                    
                    # Collect detections for printing and drawing
                    for cls_id, box_xyxy, conf in zip(cls_list, xyxy_list, conf_list):
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
                            "conf": conf
                        }
                        last_detections.append(det)
                    
                    # Print detections line by line in terminal
                    if last_detections:
                        print(f"\nDetections (FPS: {current_fps:.1f}):")
                        for det in last_detections[:3]:  # Limit to top 3 detections
                            print(f"  {det['name']} ({det['conf']:.2f}) at [{det['x1']}, {det['y1']}, {det['x2']}, {det['y2']}]")
                    
                    # Select bottle for tracking (highest confidence bottle)
                    bottle_detections = [d for d in last_detections if d["cls_id"] == TARGET_CLASS_ID]
                    if bottle_detections:
                        # Choose bottle with highest confidence
                        best_bottle = max(bottle_detections, key=lambda x: x["conf"])
                        last_detect = (
                            best_bottle["x1"],
                            best_bottle["y1"],
                            best_bottle["x2"],
                            best_bottle["y2"],
                        )
                        last_detect_time = now
                        
                        # Update Kalman prediction
                        bottle_x = (best_bottle["x1"] + best_bottle["x2"]) // 2
                        bottle_y = (best_bottle["y1"] + best_bottle["y2"]) // 2
                        
                        # Kalman update step
                        kalman_gain = position_variance / (position_variance + measurement_variance)
                        predicted_x = predicted_x + kalman_gain * (bottle_x - predicted_x)
                        predicted_y = predicted_y + kalman_gain * (bottle_y - predicted_y)
                        position_variance = (1 - kalman_gain) * position_variance
            
            # If last detection is too old, consider target lost
            if last_detect is not None and now - last_detect_time > DETECT_TIMEOUT:
                last_detect = None
                # Reset integral term when target lost
                integral_x = 0.0
                integral_y = 0.0
            
            # Update Kalman prediction (process step)
            position_variance += process_variance
            
            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            bottle_found = last_detect is not None
            
            # Draw all detections first
            for det in last_detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                name = det["name"]
                conf = det["conf"]
                
                color = (0, 255, 0) if det["cls_id"] == TARGET_CLASS_ID else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if bottle_found:
                x1, y1, x2, y2 = last_detect
                bottle_x = (x1 + x2) // 2
                bottle_y = (y1 + y2) // 2
                
                # Draw bottle center and predicted position
                cv2.circle(display_frame, (bottle_x, bottle_y), 4, (0, 255, 0), -1)
                cv2.circle(display_frame, (int(predicted_x), int(predicted_y)), 6, (255, 255, 0), 2)
                
                # Use Kalman-filtered position for error calculation
                error_x = predicted_x - CENTER_X
                error_y = predicted_y - CENTER_Y
                
                # Add to error history for smoothing
                error_x_history.append(error_x)
                error_y_history.append(error_y)
                if len(error_x_history) > ERROR_HISTORY_SIZE:
                    error_x_history.pop(0)
                    error_y_history.pop(0)
                
                # Use average of recent errors for smoother tracking
                if error_x_history:
                    smooth_error_x = sum(error_x_history) / len(error_x_history)
                    smooth_error_y = sum(error_y_history) / len(error_y_history)
                else:
                    smooth_error_x = error_x
                    smooth_error_y = error_y
                
                # Deadzone
                if abs(smooth_error_x) < DEADZONE_PIXELS:
                    smooth_error_x = 0
                if abs(smooth_error_y) < DEADZONE_PIXELS:
                    smooth_error_y = 0
                
                # PID control
                # Proportional term
                p_term_x = smooth_error_x * KP_PAN
                p_term_y = smooth_error_y * KP_TILT
                
                # Integral term (with anti-windup)
                integral_x += smooth_error_x
                integral_y += smooth_error_y
                integral_x = max(-100, min(100, integral_x))  # Clamp integral
                integral_y = max(-100, min(100, integral_y))
                i_term_x = integral_x * KI_PAN
                i_term_y = integral_y * KI_TILT
                
                # Derivative term
                d_term_x = (smooth_error_x - prev_error_x) * KD_PAN
                d_term_y = (smooth_error_y - prev_error_y) * KD_TILT
                
                # Update previous errors
                prev_error_x = smooth_error_x
                prev_error_y = smooth_error_y
                
                # Combined control signal
                control_x = p_term_x + i_term_x - d_term_x
                control_y = p_term_y + i_term_y - d_term_y
                
                # Update servo angles at limited rate
                if now - last_servo_update >= SERVO_INTERVAL:
                    # Pan left/right: screen x -> servo pan
                    pan_angle -= control_x
                    # Tilt up/down: screen y -> servo tilt
                    tilt_angle -= control_y
                    
                    # Clamp angles with safety margins
                    pan_angle = max(10.0, min(170.0, pan_angle))
                    tilt_angle = max(40.0, min(140.0, tilt_angle))
                    
                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)
                    
                    last_servo_update = now
            else:
                cv2.putText(display_frame, "Bottle not found", (60, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Reset predictions when no bottle
                predicted_x = CENTER_X
                predicted_y = CENTER_Y
                position_variance = 50.0
            
            # -------------------------------
            # FPS counter
            # -------------------------------
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # Draw center crosshair
            cv2.drawMarker(display_frame, (CENTER_X, CENTER_Y), (255, 0, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Draw status information
            cv2.putText(display_frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                       (10, FRAME_H - 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"FPS:{current_fps:.1f}",
                       (10, FRAME_H - 40), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
            if bottle_found:
                cv2.putText(display_frame, "TRACKING", (10, FRAME_H - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "SEARCHING", (10, FRAME_H - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # -------------------------------
            # Show video with minimal delay
            # -------------------------------
            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame)
            
            # Adaptive waitKey for better FPS
            loop_time = time.time() - loop_start
            wait_time = max(1, int(1000/30 - loop_time*1000))  # Target 30 FPS
            
            key = cv2.waitKey(wait_time)
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
