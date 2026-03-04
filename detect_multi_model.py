import cv2
from ultralytics import YOLO
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load active models only
models = {}
for model_cfg in config['models']:
    if model_cfg['active']:
        try:
            models[model_cfg['name']] = YOLO(model_cfg['path'])
            print(f"✓ {model_cfg['name']}: {model_cfg['description']}")
        except:
            print(f"✗ Failed: {model_cfg['name']} ({model_cfg['path']})")

# Load active food classes
FOOD_CLASSES = {int(k): v['name'] for k, v in config['food_classes']['coco'].items() if v['active']}
settings = config['detection_settings']

def non_max_suppression_custom(detections, iou_threshold=0.5):
    """Remove duplicate detections from multiple models"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []
    
    for det in detections:
        x1, y1, x2, y2, conf, name, src = det
        is_duplicate = False
        
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept[0], kept[1], kept[2], kept[3]
            
            # Calculate IoU
            xi1, yi1 = max(x1, kx1), max(y1, ky1)
            xi2, yi2 = min(x2, kx2), min(y2, ky2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (kx2 - kx1) * (ky2 - ky1)
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(det)
    
    return keep

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['camera_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['camera_height'])

print(f"Loaded {len(models)} models for detection")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    all_detections = []
    
    # Run all active models
    for model_name, model in models.items():
        results = model(frame, conf=settings['confidence_threshold'], 
                       iou=settings['iou_threshold'], 
                       imgsz=settings['image_size'], verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                
                # Filter: only food from COCO, all from specialized models
                if model_name == 'coco' and cls not in FOOD_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls]
                
                all_detections.append((x1, y1, x2, y2, conf, cls_name, model_name))
    
    # Remove duplicates
    final_detections = non_max_suppression_custom(all_detections, iou_threshold=0.5)
    
    # Draw detections
    for det in final_detections:
        x1, y1, x2, y2, conf, cls_name, source = det
        
        # Color by source: green=COCO, yellow=specialized
        color = (0, 255, 0) if source == 'coco' else (0, 255, 255)
        label = f'{cls_name} {conf:.2f}'
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display count
    cv2.putText(frame, f'Detected: {len(final_detections)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Multi-Model Fruit Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
