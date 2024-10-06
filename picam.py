import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("best.pt")

while True:
    frame = picam2.capture_array()
    results = model(frame)
    annotated_frame = results[0].plot()
    if results[0].probs.data[0].item() < results[0].probs.data[1].item():
        annotated_frame = cv2.copyMakeBorder(annotated_frame,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,255])
        cv2.imshow("Camera", annotated_frame)
    else:
        cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
