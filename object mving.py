from picamera2 import Picamera2
import cv2
import threading
import torch
import numpy as np
from time import sleep

# --------------------------
# Load YOLOv5 model properly
# --------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
PHONE_CLASS = 67   # COCO index for cellphone

# --------------------------
# Camera setup
# --------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (480, 360)})
picam2.configure(config)
picam2.start()

# Shared variables
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None
run_inference = True
skip_counter = 0

# --------------------------
# Inference Thread
# --------------------------
def inference_thread():
    global annotated_frame, skip_counter

    while run_inference:
        sleep(0.01)  # reduce CPU load

        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        # Run inference only on every 2nd frame
        if skip_counter % 2 == 0:
            results = model(frame)

            # Detection parsing
            detections = results.xyxy[0].cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == PHONE_CLASS:
                    print(f"[DETECTED] Phone, conf={conf:.2f}")

            # results.render() returns BGR images already
            rendered = results.render()[0]
            annotated_frame = rendered

        skip_counter += 1

# Start inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# --------------------------
# Main Loop
# --------------------------
try:
    while True:
        frame = picam2.capture_array()  # RGBA
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        with frame_lock:
            current_frame = rgb

        # Display
        if annotated_frame is not None:
            cv2.imshow('YOLOv5 Live', annotated_frame)
        else:
            cv2.imshow('YOLOv5 Live', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    run_inference = False
    picam2.stop()
    cv2.destroyAllWindows()
