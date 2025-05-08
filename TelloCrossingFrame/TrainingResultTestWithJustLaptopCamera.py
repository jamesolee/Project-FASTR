from ultralytics import YOLO
import torch
import cv2
from pathlib import Path


MODEL_WEIGHTS_PATH = Path(r"D:\AAAUAVdata\Yolo8nTrainingBasicFiles\bestBy11s.pt")

CONFIDENCE_THRESHOLD = 0.4
WEBCAM_INDEX = 0

print("Loading model from:", MODEL_WEIGHTS_PATH)
model = YOLO(str(MODEL_WEIGHTS_PATH))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f" Cannot open webcam index {WEBCAM_INDEX}")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0].cpu().numpy())
            label = f"{model.names.get(cls_id, str(cls_id))}: {conf:.2f}"

            cv2.rectangle(frame, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]-th-bl), (xyxy[0]+tw, xyxy[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (xyxy[0], xyxy[1]-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv8 Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
