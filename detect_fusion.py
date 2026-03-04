import cv2
from ultralytics import YOLO
import json
import numpy as np
from collections import defaultdict

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load active models
models = []
for model_cfg in config['models']:
    if model_cfg['active']:
        try:
            model = YOLO(model_cfg['path'])
            models.append({
                'name': model_cfg['name'],
                'model': model,
                'weight': model_cfg.get('weight', 1.0)
            })
            print(f"✓ {model_cfg['name']}: {model_cfg['description']} (weight: {model_cfg.get('weight', 1.0)})")
        except Exception as e:
            print(f"✗ Failed: {model_cfg['name']} - {e}")

FOOD_CLASSES = {int(k): v['name'] for k, v in config['food_classes']['custom'].items() if v['active']}
settings = config['detection_settings']

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
    """
    Fusion algorithm: Combines detections from multiple models
    - Groups overlapping boxes from different models
    - Weighted voting for final confidence
    - Averages box coordinates
    """
    if not all_detections:
        return []
    
    # Group detections by spatial proximity
    clusters = []
    
    for det in all_detections:
        x1, y1, x2, y2, conf, name, model_name, weight = det
        box = (x1, y1, x2, y2)
        
        # Find matching cluster
        matched = False
        for cluster in clusters:
            # Check IoU with cluster representative
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
    
    # Process clusters into final detections
    final_detections = []
    
    for cluster in clusters:
        if cluster['votes'] < min_votes:
            continue
        
        # Weighted average of boxes and confidence
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
        
        # Select class with highest weighted vote
        best_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        final_detections.append({
            'box': (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)),
            'confidence': avg_conf / total_weight,
            'class': best_class,
            'votes': cluster['votes'],
            'models': len(cluster['boxes'])
        })
    
    return final_detections

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['camera_width'])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['camera_height'])

print(f"\n🚀 Fusion Detection System Active ({len(models)} models)\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    all_detections = []
    
    # Collect detections from all models
    for model_info in models:
        model_name = model_info['name']
        model = model_info['model']
        weight = model_info['weight']
        
        results = model(frame, conf=settings['confidence_threshold'], 
                       iou=settings['iou_threshold'], 
                       imgsz=settings['image_size'], verbose=False)
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                
                # Filter COCO to food only
                if model_name == 'coco' and cls not in FOOD_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls]
                
                all_detections.append((x1, y1, x2, y2, conf, cls_name, model_name, weight))
    
    # Apply fusion algorithm
    final_detections = fusion_detection(all_detections, 
                                        iou_threshold=settings.get('fusion_iou', 0.5),
                                        min_votes=settings.get('min_model_votes', 1))
    
    # Draw detections
    for det in final_detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        cls_name = det['class']
        votes = det['votes']
        
        # Color intensity based on votes (more models = brighter green)
        intensity = min(255, 100 + (votes * 50))
        color = (0, intensity, 0)
        
        label = f"{cls_name} {conf:.2f} [{votes}M]"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display stats
    cv2.putText(frame, f'Detected: {len(final_detections)} | Models: {len(models)}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Fusion Detection System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
