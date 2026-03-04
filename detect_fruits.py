import cv2
from ultralytics import YOLO
import json
import os

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load active models only
models = []
for model_cfg in config['models']:
    if model_cfg['active']:
        try:
            models.append((model_cfg['name'], YOLO(model_cfg['path'])))
            print(f"✓ Loaded {model_cfg['name']}: {model_cfg['description']}")
        except:
            print(f"✗ Failed to load {model_cfg['name']} ({model_cfg['path']})")

# Load active food classes from COCO
FOOD_CLASSES = {int(k): v['name'] for k, v in config['food_classes']['custom'].items() if v['active']}
settings = config['detection_settings']

def remove_duplicates(detections, iou_thresh=0.5):
    """Remove overlapping detections from multiple models"""
    if not detections:
        return []
    
    detections.sort(key=lambda x: x[4], reverse=True)
    keep = []
    
    for det in detections:
        x1, y1, x2, y2, conf, name = det
        duplicate = False
        
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept[0], kept[1], kept[2], kept[3]
            
            # Calculate IoU
            xi1, yi1 = max(x1, kx1), max(y1, ky1)
            xi2, yi2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            
            if inter / union > iou_thresh:
                duplicate = True
                break
        
        if not duplicate:
            keep.append(det)
    
    return keep

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['camera_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['camera_height'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run all models and collect detections
    all_detections = []
    
    for model_name, model in models:
        results = model(frame, conf=settings['confidence_threshold'], 
                       iou=settings['iou_threshold'], 
                       imgsz=settings['image_size'], verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                
                # COCO: only food classes, Specialized: all classes
                if model_name == 'coco' and cls not in FOOD_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                
                all_detections.append((x1, y1, x2, y2, conf, name))
    
    # Remove duplicates from multiple models
    final_detections = remove_duplicates(all_detections)
    
    # Draw all detections
    for x1, y1, x2, y2, conf, name in final_detections:
        label = f'{name} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, f'Items: {len(final_detections)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Fruit & Vegetable Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
