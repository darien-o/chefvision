#!/usr/bin/env python3
"""
Basic Detection Example

This example demonstrates the simplest way to use the fusion detection system
with default configuration settings.
"""

import cv2
from ultralytics import YOLO
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load a single model for basic detection
model_cfg = config['models'][0]  # Use first active model
model = YOLO(model_cfg['path'])

print(f"✓ Loaded: {model_cfg['name']} - {model_cfg['description']}")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🚀 Basic Detection Active")
print("Press 'q' to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.25, verbose=False)
    
    # Draw results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Basic Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
