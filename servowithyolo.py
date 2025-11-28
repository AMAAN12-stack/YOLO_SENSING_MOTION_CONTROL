import cv2
import threading
import torch
import time
from picamera2 import Picamera2

# Load YOLO model (you must have yolov8n.pt downloaded)
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
TARGET_CLASS = "cell phone"

# Setup PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
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

# Start capture thread
threading.Thread(target=capture_frames, daemon=True).start()

def main():
    global latest_frame
    while True:
        if latest_frame is None:
            continue

        with frame_lock:
            frame = latest_frame.copy()

        # Run YOLO model
        results = model(frame)

        # Filter only "cell phone"
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Phone Detection (YOLO + Picamera2)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
