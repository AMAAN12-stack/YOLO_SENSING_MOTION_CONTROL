from picamera2 import Picamera2
import cv2
import threading
import torch
from time import sleep

# Load a pretrained YOLOv5 model via torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold

# Initialize Raspberry Pi camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (480, 360)})
picam2.configure(config)
picam2.start()

frame_ready = False
annotated_frame = None
current_frame = None
inference_skip = 0

# COCO class index for cell phone
PHONE_CLASS = 67

def inference_thread():
    global frame_ready, annotated_frame, inference_skip
    while True:
        if frame_ready:
            # run inference on every 2nd frame to keep UI smooth
            if inference_skip % 2 == 0:
                # model expects RGB numpy array
                results = model(current_frame)  # returns a Results object
                # detect phone(s)
                xyxy = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else []
                for det in xyxy:
                    cls = int(det[5])
                    conf = float(det[4])
                    if cls == PHONE_CLASS:
                        print(f"Phone detected - conf: {conf:.2f}")
                        break
                # render annotated image (RGB)
                rendered = results.render()  # list of annotated images (numpy)
                if rendered:
                    # convert RGB -> BGR for OpenCV display
                    annotated_frame = cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR)
                    # store to global
                    globals()['annotated_frame'] = annotated_frame
            inference_skip += 1
            frame_ready = False

# Start inference thread
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

try:
    while True:
        frame = picam2.capture_array()  # RGBA
        # Convert RGBA -> RGB for model
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        current_frame = rgb
        frame_ready = True

        # show annotated if available, else show raw
        if annotated_frame is not None:
            cv2.imshow('YOLOv5 Live', annotated_frame)
        else:
            cv2.imshow('YOLOv5 Live', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    cv2.destroyAllWindows()
