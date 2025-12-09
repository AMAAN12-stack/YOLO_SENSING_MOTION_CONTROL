import cv2
import time
import pigpio
import numpy as np
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
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
    controls={"FrameRate": 30}
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
    SERVO_INTERVAL = 0.033    # update servos at 30 Hz
    
    # Improved PID-like gains with derivative term for smoother motion
    KP_PAN = 0.035
    KP_TILT = 0.035
    KD_PAN = 0.01
    KD_TILT = 0.01
    DEADZONE_PIXELS = 8       # increased to reduce micro-adjustments

    last_yolo_time = 0.0
    last_detect_time = 0.0
    last_servo_update = 0.0

    last_detect = None  # (x1, y1, x2, y2) in 640x480 space
    last_bottle_pos = None  # For velocity calculation

    # Store last detections (for drawing & printing)
    last_detections = []  # list of dicts: {"cls_id", "name", "x1","y1","x2","y2"}

    prev_time = time.time()
    frame_counter = 0
    fps_time = prev_time

    # For smoother tracking: improved low-pass filter
    filtered_error_x = 0.0
    filtered_error_y = 0.0
    ERROR_SMOOTHING = 0.6  # Increased smoothing for better stability
    
    # Derivative terms
    prev_error_x = 0.0
    prev_error_y = 0.0

    # Kalman filter-like prediction for lost frames
    velocity_x = 0.0
    velocity_y = 0.0
    last_prediction_time = 0.0
    
    # Buffer for display frames to reduce latency
    display_buffer = None
    frame_ready_event = False

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            # Get latest frame from camera
            frame = picam2.capture_array()  # RGB888, 640x480

            now = time.time()

            # -------------------------------
            # Run YOLO at limited rate
            # -------------------------------
            if now - last_yolo_time >= YOLO_INTERVAL:
                last_yolo_time = now

                # Use even smaller frame for faster inference
                small_w = 320  # Fixed size for consistency
                small_h = 240
                small_frame = cv2.resize(frame, (small_w, small_h))

                # Run inference with optimized settings
                results = model(small_frame, verbose=False, conf=0.3, iou=0.4)

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
                        confidence = float(conf)

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
                            "conf": confidence
                        }
                        last_detections.append(det)

                    # Select bottle for tracking (highest confidence bottle)
                    bottle_dets = [d for d in last_detections if d["cls_id"] == TARGET_CLASS_ID]
                    if bottle_dets:
                        # Select bottle with highest confidence
                        best_bottle = max(bottle_dets, key=lambda x: x["conf"])
                        last_detect = (
                            best_bottle["x1"],
                            best_bottle["y1"],
                            best_bottle["x2"],
                            best_bottle["y2"],
                        )
                        
                        # Calculate velocity for prediction
                        if last_bottle_pos is not None:
                            current_center = ((last_detect[0] + last_detect[2]) // 2, 
                                            (last_detect[1] + last_detect[3]) // 2)
                            last_center = last_bottle_pos
                            dt = now - last_detect_time
                            if dt > 0:
                                velocity_x = (current_center[0] - last_center[0]) / dt
                                velocity_y = (current_center[1] - last_center[1]) / dt
                        
                        last_bottle_pos = ((last_detect[0] + last_detect[2]) // 2, 
                                         (last_detect[1] + last_detect[3]) // 2)
                        last_detect_time = now
                        last_prediction_time = now

            # If last detection is too old, consider target lost
            if last_detect is not None and now - last_detect_time > DETECT_TIMEOUT:
                last_detect = None
                velocity_x = 0.0
                velocity_y = 0.0
            elif last_detect is None and last_bottle_pos is not None and now - last_prediction_time < 0.3:
                # Predict position for short time after losing detection
                dt = now - last_prediction_time
                predicted_x = last_bottle_pos[0] + velocity_x * dt
                predicted_y = last_bottle_pos[1] + velocity_y * dt
                
                # Create a virtual detection for prediction
                if 0 <= predicted_x <= FRAME_W and 0 <= predicted_y <= FRAME_H:
                    size = 40  # Default size for predicted box
                    last_detect = (
                        int(predicted_x - size/2),
                        int(predicted_y - size/2),
                        int(predicted_x + size/2),
                        int(predicted_y + size/2)
                    )

            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            # Create display frame
            display_frame = frame.copy()
            
            # Draw center crosshair
            cv2.line(display_frame, (FRAME_W//2 - 10, FRAME_H//2), 
                    (FRAME_W//2 + 10, FRAME_H//2), (255, 255, 0), 1)
            cv2.line(display_frame, (FRAME_W//2, FRAME_H//2 - 10), 
                    (FRAME_W//2, FRAME_H//2 + 10), (255, 255, 0), 1)

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
                    conf = det["conf"]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)  # red
                    cv2.putText(display_frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)

            if bottle_found:
                x1, y1, x2, y2 = last_detect

                # Draw bounding box and label for bottle (green)
                box_color = (0, 255, 0) if now - last_detect_time < DETECT_TIMEOUT else (0, 255, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                              box_color, 2)
                label = "bottle (pred)" if now - last_detect_time >= DETECT_TIMEOUT else "bottle"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            box_color, 2)

                bottle_x = (x1 + x2) // 2
                bottle_y = (y1 + y2) // 2

                # Draw crosshair for target
                cv2.circle(display_frame, (bottle_x, bottle_y), 4,
                           box_color, -1)
                cv2.line(display_frame, (bottle_x, center_y), 
                        (bottle_x, bottle_y), (255, 100, 0), 1)
                cv2.line(display_frame, (center_x, bottle_y), 
                        (bottle_x, bottle_y), (255, 100, 0), 1)

                error_x = bottle_x - center_x
                error_y = bottle_y - center_y

                # Normalize errors to frame size
                norm_error_x = error_x / FRAME_W
                norm_error_y = error_y / FRAME_H

                # Apply non-linear scaling for smoother response
                if abs(norm_error_x) < 0.1:  # Center region
                    norm_error_x *= 0.5
                if abs(norm_error_y) < 0.1:
                    norm_error_y *= 0.5

                # Optional: small deadzone to avoid jitter
                if abs(error_x) < DEADZONE_PIXELS:
                    error_x = 0
                    norm_error_x = 0
                if abs(error_y) < DEADZONE_PIXELS:
                    error_y = 0
                    norm_error_y = 0

                # Smooth errors for smoother tracking
                filtered_error_x = (
                    ERROR_SMOOTHING * filtered_error_x
                    + (1.0 - ERROR_SMOOTHING) * error_x
                )
                filtered_error_y = (
                    ERROR_SMOOTHING * filtered_error_y
                    + (1.0 - ERROR_SMOOTHING) * error_y
                )

                # Calculate derivative terms
                dt_servo = now - last_servo_update if last_servo_update > 0 else SERVO_INTERVAL
                if dt_servo > 0:
                    derivative_x = (filtered_error_x - prev_error_x) / dt_servo
                    derivative_y = (filtered_error_y - prev_error_y) / dt_servo
                else:
                    derivative_x = 0
                    derivative_y = 0
                
                prev_error_x = filtered_error_x
                prev_error_y = filtered_error_y

                # Update servo angles at limited rate
                if now - last_servo_update >= SERVO_INTERVAL:
                    # Calculate control outputs with derivative damping
                    pan_adjust = -filtered_error_x * KP_PAN - derivative_x * KD_PAN
                    tilt_adjust = -filtered_error_y * KP_TILT - derivative_y * KD_TILT
                    
                    # Apply limits to prevent overshoot
                    pan_adjust = max(-2.0, min(2.0, pan_adjust))
                    tilt_adjust = max(-2.0, min(2.0, tilt_adjust))
                    
                    pan_angle += pan_adjust
                    tilt_angle += tilt_adjust

                    # Clamp angles with tighter bounds for safety
                    pan_angle = max(20.0, min(160.0, pan_angle))
                    tilt_angle = max(40.0, min(140.0, tilt_angle))

                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)

                    last_servo_update = now
            else:
                cv2.putText(display_frame, "Bottle not found",
                            (FRAME_W//2 - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            # -------------------------------
            # FPS counter
            # -------------------------------
            frame_counter += 1
            if now - fps_time >= 1.0:
                fps = frame_counter / (now - fps_time)
                frame_counter = 0
                fps_time = now
            
                cv2.putText(display_frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                            (10, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, f"FPS:{fps:.1f}",
                            (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            # -------------------------------
            # Show video with reduced latency
            # -------------------------------
            # Use a smaller window for faster display
            display_frame_small = cv2.resize(display_frame, (FRAME_W, FRAME_H))
            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame_small)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset to center
                pan_angle = 90.0
                tilt_angle = 131.0
                set_servo(PAN_PIN, pan_angle)
                set_servo(TILT_PIN, tilt_angle)

    finally:
        # Cleanup
        print("\nStopping, cleaning up...")
        pi.set_servo_pulsewidth(PAN_PIN, 0)
        pi.set_servo_pulsewidth(TILT_PIN, 0)
        pi.stop()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
