#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to process multiple images in batch mode,
saving annotated results to disk instead of displaying them in real-time.
"""

import cv2
from ultralytics import YOLO
import json
from pathlib import Path
from collections import defaultdict
import time

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

def process_image(image_path, models, settings):
    """Process a single image and return annotated frame with detections"""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None, []
    
    all_detections = []
    
    # Collect detections from all models
    for model_info in models:
        results = model_info['model'](
            frame,
            conf=settings['confidence_threshold'],
            imgsz=settings['image_size'],
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
    
    # Apply fusion
    final_detections = fusion_detection(
        all_detections,
        iou_threshold=settings.get('fusion_iou', 0.5),
        min_votes=settings.get('min_model_votes', 1)
    )
    
    # Draw detections
    for det in final_detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        cls_name = det['class']
        votes = det['votes']
        
        intensity = min(255, 100 + (votes * 50))
        color = (0, intensity, 0)
        label = f"{cls_name} {conf:.2f} [{votes}M]"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, final_detections

# Configuration
INPUT_DIR = Path('input_images')      # Directory with images to process
OUTPUT_DIR = Path('output_images')    # Directory for annotated results
RESULTS_FILE = 'detection_results.txt'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load models
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
            print(f"✓ {model_cfg['name']}: {model_cfg['description']}")
        except Exception as e:
            print(f"✗ Failed: {model_cfg['name']} - {e}")

settings = config['detection_settings']

print(f"\n🚀 Batch Processing Mode")
print(f"   Input: {INPUT_DIR}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Models: {len(models)}\n")

# Check if input directory exists
if not INPUT_DIR.exists():
    print(f"⚠️  Input directory '{INPUT_DIR}' not found!")
    print(f"   Creating example directory...")
    INPUT_DIR.mkdir(exist_ok=True)
    print(f"   Please add images to '{INPUT_DIR}' and run again.")
    exit(0)

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [f for f in INPUT_DIR.iterdir() 
               if f.suffix.lower() in image_extensions]

if not image_files:
    print(f"⚠️  No images found in '{INPUT_DIR}'")
    print(f"   Supported formats: {', '.join(image_extensions)}")
    exit(0)

print(f"Found {len(image_files)} images to process\n")

# Process images
results_summary = []
start_time = time.time()

for idx, image_path in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...", end=' ')
    
    frame, detections = process_image(image_path, models, settings)
    
    if frame is None:
        print("❌ Failed to load")
        continue
    
    # Save annotated image
    output_path = OUTPUT_DIR / f"annotated_{image_path.name}"
    cv2.imwrite(str(output_path), frame)
    
    # Record results
    results_summary.append({
        'filename': image_path.name,
        'detections': len(detections),
        'objects': [f"{d['class']} ({d['confidence']:.2f})" for d in detections]
    })
    
    print(f"✓ {len(detections)} objects detected")

elapsed_time = time.time() - start_time

# Save results summary
with open(OUTPUT_DIR / RESULTS_FILE, 'w') as f:
    f.write("Batch Processing Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Images: {len(image_files)}\n")
    f.write(f"Processing Time: {elapsed_time:.2f}s\n")
    f.write(f"Average Time: {elapsed_time/len(image_files):.2f}s per image\n")
    f.write(f"Models Used: {len(models)}\n\n")
    f.write("=" * 60 + "\n\n")
    
    for result in results_summary:
        f.write(f"File: {result['filename']}\n")
        f.write(f"Detections: {result['detections']}\n")
        if result['objects']:
            f.write("Objects:\n")
            for obj in result['objects']:
                f.write(f"  - {obj}\n")
        f.write("\n")

print(f"\n✅ Batch processing complete!")
print(f"   Processed: {len(image_files)} images")
print(f"   Time: {elapsed_time:.2f}s ({elapsed_time/len(image_files):.2f}s per image)")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Results: {OUTPUT_DIR / RESULTS_FILE}")
