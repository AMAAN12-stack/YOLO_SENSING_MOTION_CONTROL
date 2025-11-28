from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import threading
import RPi.GPIO as GPIO
from time import sleep

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)

# PWM setup for servos
pwm_17 = GPIO.PWM(17, 50)  # 50Hz frequency
pwm_27 = GPIO.PWM(27, 50)
pwm_17.start(7.5)  # Center position (90 degrees)
pwm_27.start(7.5)

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize Raspberry Pi camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (480, 360)})
picam2.configure(config)
picam2.start()

frame_ready = False
annotated_frame = None
current_frame = None

# Smoothing variables
prev_servo_x = 7.5
prev_servo_y = 7.5
smoothing_factor = 0.3  # Lower = smoother (0-1)
deadzone = 20  # Pixels

# Class names - 67 is cell phone in COCO dataset
PHONE_CLASS = 67

def inference_thread():
    global frame_ready, annotated_frame, prev_servo_x, prev_servo_y
    while True:
        if frame_ready:
            results = model(current_frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            
            # Track only mobile phone (class 67)
            phone_detected = False
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    if int(box.cls[0]) == PHONE_CLASS:
                        phone_detected = True
                        x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                        y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                        
                        # Center of frame
                        frame_center_x = 240
                        frame_center_y = 180
                        
                        # Calculate offset from center with deadzone
                        offset_x = x_center - frame_center_x
                        offset_y = y_center - frame_center_y
                        
                        if abs(offset_x) > deadzone or abs(offset_y) > deadzone:
                            # Map coordinates to servo angles
                            servo_x = 7.5 + (offset_x / 240) * 4.5
                            servo_y = 7.5 + (offset_y / 180) * 4.5
                            
                            # Clamp to valid range
                            servo_x = max(3, min(12, servo_x))
                            servo_y = max(3, min(12, servo_y))
                            
                            # Apply smoothing
                            servo_x = prev_servo_x + (servo_x - prev_servo_x) * smoothing_factor
                            servo_y = prev_servo_y + (servo_y - prev_servo_y) * smoothing_factor
                            
                            prev_servo_x = servo_x
                            prev_servo_y = servo_y
                            
                            pwm_17.ChangeDutyCycle(servo_x)
                            pwm_27.ChangeDutyCycle(servo_y)
                        break
            
            frame_ready = False

# Start inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

try:
    while True:
        # Capture frame from RPi camera
        frame = picam2.capture_array()
        
        # Convert RGBA to RGB
        current_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame_ready = True
        
        # Display the frame
        if annotated_frame is not None:
            cv2.imshow('YOLO Live Detection', annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    pwm_17.stop()
    pwm_27.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()