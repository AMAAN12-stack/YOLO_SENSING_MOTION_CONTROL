import cv2
import threading
import torch
import time
from picamera2 import Picamera2

# -----------------------------
# Load YOLOv5 model
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

# -----------------------------
# Initialize Pi camera
# -----------------------------
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Frame variable shared with thread
frame = None
running = True

# -----------------------------
# Camera thread (faster capture)
# -----------------------------
def camera_thread():
    global frame, running
    while running:
        frame = picam2.capture_array()
        time.sleep(0.001)

threading.Thread(target=camera_thread, daemon=True).start()

# -----------------------------
# YOLOv5 detection loop
# -----------------------------
while True:
    if frame is None:
        continue

    # Run detection
    results = model(frame)
    annotated = results.render()[0]

    # Show output
    cv2.imshow("YOLOv5 - PiCamera2 Live Detection", annotated)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
picam2.stop()
