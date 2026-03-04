import cv2
from ultralytics import YOLO

# Use specialized fruit/vegetable model trained on 100+ classes
# Download from: https://universe.roboflow.com/joseph-nelson/fruits-vegetables-detection
model = YOLO('yolov8m.pt')  # Will switch to custom model

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Multi-scale detection for small objects
    results = model.predict(
        frame,
        conf=0.2,           # Lower threshold for small objects
        iou=0.4,            # Better overlap handling
        imgsz=1280,         # Larger image size for small objects
        max_det=50,         # Detect up to 50 objects
        agnostic_nms=True,  # Better multi-class detection
        verbose=False
    )
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = result.names[int(box.cls[0])]
            
            label = f'{cls_name} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Advanced Fruit Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
