#!/usr/bin/env python3
"""
Custom Configuration Example

This example demonstrates how to customize detection settings and model weights
without modifying the main config.json file.
"""

import cv2
from ultralytics import YOLO
import json
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    union = (x2-x1)*(y2-y1) + (x2b-x1b)*(y2b-y1b) - inter
    return inter / union if union > 0 else 0

def fusion_detection(all_detections, iou_threshold=0.5, min_votes=1):
    """Fusion algorithm for combining detections"""
    if not all_detections:
        return []
    
    clusters = []
    
    for det in all_detections:
        x1, y1, x2, y2, conf, name, model_name, weight = det
        box = (x1, y1, x2, y2)
        
        matched = False
        for cluster in clusters:
            rep_box = cluster['boxes'][0][:4]
            if calculate_iou(box, rep_box) > iou_threshold:
                cluster['boxes'].append(det)
                cluster['votes'] += 1
                cluster['total_weight'] += weight
                matched = True
                break
        
        if not matched:
            clusters.append({
                'boxes': [det],
                'votes': 1,
                'total_weight': weight
            })
    
    final_detections = []
    
    for cluster in clusters:
        if cluster['votes'] < min_votes:
            continue
        
        total_weight = cluster['total_weight']
        avg_x1 = avg_y1 = avg_x2 = avg_y2 = avg_conf = 0
        class_votes = defaultdict(float)
        
        for det in cluster['boxes']:
            x1, y1, x2, y2, conf, name, model_name, weight = det
            w = weight / total_weight
            
            avg_x1 += x1 * w
            avg_y1 += y1 * w
            avg_x2 += x2 * w
            avg_y2 += y2 * w
            avg_conf += conf * weight
            class_votes[name] += weight
        
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        final_detections.append({
            'box': (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)),
            'confidence': avg_conf / total_weight,
            'class': best_class,
            'votes': cluster['votes']
        })
    
    return final_detections

# Load base configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Custom settings - override defaults
CUSTOM_SETTINGS = {
    'confidence_threshold': 0.3,      # Higher threshold for fewer false positives
    'fusion_iou': 0.6,                # Stricter IoU for fusion
    'min_model_votes': 2,             # Require at least 2 models to agree
    'camera_width': 1920,             # Higher resolution
    'camera_height': 1080,
    'image_size': 640
}

# Custom model weights - prioritize specialized models
CUSTOM_WEIGHTS = {
    'coco': 0.8,
    'fruit_360': 2.0,      # Higher weight for fruit specialist
    'grocery': 1.5,
    'yolov8n': 0.5,
    'yolov26x': 0.5
}

# Load models with custom weights
models = []
for model_cfg in config['models']:
    if model_cfg['active']:
        try:
            model = YOLO(model_cfg['path'])
            custom_weight = CUSTOM_WEIGHTS.get(model_cfg['name'], 1.0)
            models.append({
                'name': model_cfg['name'],
                'model': model,
                'weight': custom_weight
            })
            print(f"✓ {model_cfg['name']}: weight={custom_weight}")
        except Exception as e:
            print(f"✗ Failed: {model_cfg['name']} - {e}")

# Open camera with custom resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CUSTOM_SETTINGS['camera_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CUSTOM_SETTINGS['camera_height'])

print(f"\n🚀 Custom Configuration Active")
print(f"   Confidence: {CUSTOM_SETTINGS['confidence_threshold']}")
print(f"   Min Votes: {CUSTOM_SETTINGS['min_model_votes']}")
print(f"   Fusion IoU: {CUSTOM_SETTINGS['fusion_iou']}")
print("Press 'q' to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    all_detections = []
    
    # Collect detections from all models
    for model_info in models:
        results = model_info['model'](
            frame, 
            conf=CUSTOM_SETTINGS['confidence_threshold'],
            imgsz=CUSTOM_SETTINGS['image_size'],
            verbose=False
        )
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = result.names[int(box.cls[0])]
                
                all_detections.append((
                    x1, y1, x2, y2, conf, cls_name,
                    model_info['name'], model_info['weight']
                ))
    
    # Apply fusion with custom settings
    final_detections = fusion_detection(
        all_detections,
        iou_threshold=CUSTOM_SETTINGS['fusion_iou'],
        min_votes=CUSTOM_SETTINGS['min_model_votes']
    )
    
    # Draw detections
    for det in final_detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        cls_name = det['class']
        votes = det['votes']
        
        color = (0, 255, 0) if votes >= 3 else (0, 200, 200)
        label = f"{cls_name} {conf:.2f} [{votes}M]"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display stats
    cv2.putText(frame, f'Detected: {len(final_detections)} | Min Votes: {CUSTOM_SETTINGS["min_model_votes"]}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Custom Configuration Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
