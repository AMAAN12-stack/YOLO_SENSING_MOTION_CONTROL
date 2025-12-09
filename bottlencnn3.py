import cv2
import time
import pigpio
from picamera2 import Picamera2
from ultralytics import YOLO
import threading
import queue

# --- PIGPIO & SERVO SETUP ---
pi = pigpio.pi()
if not pi.connected:
    print("Unable to connect to pigpio daemon")
    exit()

PAN_PIN = 18
TILT_PIN = 19

# Current servo angles
pan_angle = 90.0
tilt_angle = 131.0

def set_servo(pin, angle):
    """Set servo to angle in degrees (0–180)."""
    # Angle clamping is already done in the main loop, but here for safety.
    angle = max(0.0, min(180.0, float(angle)))
    # Calculate pulse width: 544 us (0 deg) to 2400 us (180 deg)
    pulse = 544 + (angle / 180.0) * (2400 - 544)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(PAN_PIN, pan_angle)
set_servo(TILT_PIN, tilt_angle)


print("Loading NCNN YOLO model...")
# Model loading is fine, assuming the path is correct
model = YOLO("/home/yolo/yolo/yolo11n_ncnn_model")
print("Model loaded!")

TARGET_CLASS_ID = 39  # bottle in COCO

FRAME_W = 640
FRAME_H = 480

# --- CAMERA SETUP ---
# Use a separate thread and queue for frame capture to minimize main loop block time.
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
)
picam2.configure(config)
picam2.start()

# Queue for frames
frame_queue = queue.Queue(maxsize=1)
camera_running = threading.Event()
camera_running.set()

def camera_thread_func():
    """Continuously capture frames and put them into the queue."""
    # Use a faster, less complex capture method if possible, or stick to capture_array
    # It's better to process the latest frame than to wait for the next, so we empty the queue
    while camera_running.is_set():
        try:
            frame = picam2.capture_array()
            # Drop old frames if queue is full (always want the latest one)
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        except Exception as e:
            print(f"Camera thread error: {e}")
            break
        # Optional: brief sleep to avoid 100% CPU on fast camera, but often not needed
        # time.sleep(0.001)

camera_thread = threading.Thread(target=camera_thread_func)
camera_thread.start()

# --- PID CONTROLLER GLOBAL VARIABLES ---
# Global variables for I and D terms. We'll add these to the tracking.
integral_pan = 0.0
integral_tilt = 0.0
last_error_x = 0.0
last_error_y = 0.0

# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    global pan_angle, tilt_angle
    global integral_pan, integral_tilt, last_error_x, last_error_y

    # Detection + tracking parameters
    YOLO_INTERVAL = 0.15          # **Faster YOLO: 200ms -> 150ms** (for better responsiveness)
    DETECT_TIMEOUT = 0.7          # forget target if not seen for 700 ms
    SERVO_INTERVAL = 0.03         # **Faster servo update: 50ms -> 30ms** (for smoother movement, up to 33Hz)

    # PID-like gains - **Increased gains and added I/D terms**
    KP_PAN = 0.04
    KP_TILT = 0.04
    KI_PAN = 0.0005     # Integral gain for reducing steady-state error
    KI_TILT = 0.0005
    KD_PAN = 0.005      # Derivative gain for damping oscillations
    KD_TILT = 0.005
    
    # Anti-windup clamping for integral term
    INTEGRAL_CLAMP = 5.0

    DEADZONE_PIXELS = 10          # **Increased deadzone** to reduce jitter from noise
    
    # Error Smoothing: **Increased smoothing** for a slower, more deliberate response
    # 0.5 is already a high level of smoothing. Keep it at 0.5 or increase slightly.
    ERROR_SMOOTHING = 0.6  # 0=no smoothing, 1=full smoothing

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

    print("Tracking started... press 'q' to quit.")

    try:
        while True:
            # Check for a new frame from the camera thread queue
            try:
                # Use get_nowait() to prevent blocking the main loop
                frame = frame_queue.get_nowait()
            except queue.Empty:
                # If no new frame, just continue to the next iteration
                # This ensures the main loop is not blocked by a slow camera/read
                continue
            
            # This is the frame captured_time, for better timing calculations
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

                # Use results as-is, no need to scale down the model.
                results = model(small_frame, verbose=False) 

                last_detect = None
                last_detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes

                    # Efficiently handle tensor to list conversion
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

                    # --- LOGGING (Reduced to reduce lag) ---
                    # Only print if detections changed, or less frequently, to reduce I/O bottleneck
                    # The original print is kept, but be aware that console I/O can be a bottleneck on SBCs
                    # Print detections line by line in terminal
                    if last_detections:
                        # print("Detections:") # Commented out to reduce console I/O lag
                        for det in last_detections:
                             if det["cls_id"] == TARGET_CLASS_ID:
                                 # print(f" Bottle detected at [{det['x1']}, {det['y1']}, {det['x2']}, {det['y2']}]") # Keep only target printing
                                 pass
                    else:
                        # print("No objects detected.") # Commented out

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
                    
                    # Reset PID integral term if we lost and regained the target
                    if last_detect is not None and last_error_x == 0.0:
                        integral_pan = 0.0
                        integral_tilt = 0.0
                    
                    # Reset errors if target lost
                    if last_detect is None:
                        last_error_x = 0.0
                        last_error_y = 0.0


            # If last detection is too old, consider target lost
            if last_detect is not None and now - last_detect_time > DETECT_TIMEOUT:
                last_detect = None
                integral_pan = 0.0
                integral_tilt = 0.0
                last_error_x = 0.0
                last_error_y = 0.0


            # -------------------------------
            # Tracking / servo control
            # -------------------------------
            display_frame = frame

            bottle_found = last_detect is not None

            center_x = FRAME_W // 2
            center_y = FRAME_H // 2
            
            # --- DRAWING (Moved up) ---
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
                
                # --- PID CONTROL ---

                # Proportional Term is the current error
                P_pan = error_x
                P_tilt = error_y

                # Optional: small deadzone to ignore very small errors/noise
                if abs(P_pan) < DEADZONE_PIXELS:
                    P_pan = 0
                if abs(P_tilt) < DEADZONE_PIXELS:
                    P_tilt = 0

                # Smooth errors for smoother tracking
                filtered_error_x = (
                    ERROR_SMOOTHING * filtered_error_x
                    + (1.0 - ERROR_SMOOTHING) * P_pan
                )
                filtered_error_y = (
                    ERROR_SMOOTHING * filtered_error_y
                    + (1.0 - ERROR_SMOOTHING) * P_tilt
                )
                
                # Use smoothed error for control
                P_pan = filtered_error_x
                P_tilt = filtered_error_y

                # Update servo angles at limited rate
                if now - last_servo_update >= SERVO_INTERVAL:
                    
                    time_delta = now - last_servo_update # Time since last update
                    
                    # Integral Term
                    integral_pan += P_pan * time_delta
                    integral_tilt += P_tilt * time_delta

                    # Anti-windup clamping (prevents integral build-up when servos max out)
                    integral_pan = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral_pan))
                    integral_tilt = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral_tilt))

                    # Derivative Term (Rate of change of error)
                    # Use the difference between the current P and the last P term.
                    D_pan = (P_pan - last_error_x) / time_delta if time_delta > 0 else 0.0
                    D_tilt = (P_tilt - last_error_y) / time_delta if time_delta > 0 else 0.0
                    
                    last_error_x = P_pan
                    last_error_y = P_tilt
                    
                    # Total control output: PID Sum
                    output_pan = (KP_PAN * P_pan) + (KI_PAN * integral_pan) + (KD_PAN * D_pan)
                    output_tilt = (KP_TILT * P_tilt) + (KI_TILT * integral_tilt) + (KD_TILT * D_tilt)

                    # Update servo angles
                    # Pan left/right: screen x error -> servo pan. Note the negative sign to move *towards* the target center.
                    pan_angle -= output_pan
                    # Tilt up/down: screen y error -> servo tilt. Note the negative sign.
                    tilt_angle -= output_tilt 

                    # Clamp angles (crucial for physical limits)
                    pan_angle = max(0.0, min(180.0, pan_angle))
                    # Tilt limits adjusted for common camera setups (e.g., 30 deg up to 150 deg down)
                    tilt_angle = max(30.0, min(150.0, tilt_angle)) 

                    set_servo(PAN_PIN, pan_angle)
                    set_servo(TILT_PIN, tilt_angle)

                    last_servo_update = now

            else:
                cv2.putText(display_frame, "Bottle not found",
                            (60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                # Reset PID terms when target is lost
                integral_pan = 0.0
                integral_tilt = 0.0
                last_error_x = 0.0
                last_error_y = 0.0
                
            # -------------------------------
            # FPS counter
            # -------------------------------
            frame_counter += 1
            now2 = time.time()
            # Calculate FPS based on the time between captured frames
            fps = 1.0 / (now2 - prev_time) if now2 != prev_time else 0.0
            prev_time = now2

            # Display info
            cv2.putText(display_frame, f"PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                        (10, FRAME_H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"FPS:{fps:.1f}",
                        (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            # -------------------------------
            # Show video
            # -------------------------------
            # The use of a separate thread for capture ensures that cv2.imshow
            # is called with the latest frame, minimizing perceived video lag.
            cv2.imshow("Bottle Tracking (NCNN-YOLO)", display_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        # Cleanup
        print("Stopping, cleaning up...")
        camera_running.clear()
        camera_thread.join() # Wait for camera thread to finish
        pi.set_servo_pulsewidth(PAN_PIN, 0)
        pi.set_servo_pulsewidth(TILT_PIN, 0)
        pi.stop()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
