# API Reference

This document provides a comprehensive reference for the Python API of the Multi-Model Fusion Detection System. Use this API to integrate fruit and vegetable detection capabilities into your own applications.

## Table of Contents

- [Core Functions](#core-functions)
  - [calculate_iou](#calculate_iou)
  - [fusion_detection](#fusion_detection)
  - [non_max_suppression_custom](#non_max_suppression_custom)
- [Configuration](#configuration)
  - [Loading Configuration](#loading-configuration)
  - [Configuration Structure](#configuration-structure)
- [Model Management](#model-management)
  - [Loading Models](#loading-models)
  - [Model Information](#model-information)
- [Usage Examples](#usage-examples)
  - [Basic Detection](#basic-detection)
  - [Custom Fusion Parameters](#custom-fusion-parameters)
  - [Batch Processing](#batch-processing)

---

## Core Functions

### calculate_iou

Calculate the Intersection over Union (IoU) between two bounding boxes.

**Signature:**
```python
def calculate_iou(box1: tuple, box2: tuple) -> float
```

**Parameters:**
- `box1` (tuple): First bounding box as `(x1, y1, x2, y2)` where:
  - `x1`, `y1`: Top-left corner coordinates
  - `x2`, `y2`: Bottom-right corner coordinates
- `box2` (tuple): Second bounding box in the same format

**Returns:**
- `float`: IoU value between 0.0 and 1.0, where:
  - `0.0` = No overlap
  - `1.0` = Perfect overlap

**Example:**
```python
from detect_fusion import calculate_iou

box1 = (100, 100, 200, 200)  # 100x100 box
box2 = (150, 150, 250, 250)  # 100x100 box, partially overlapping

iou = calculate_iou(box1, box2)
print(f"IoU: {iou:.2f}")  # Output: IoU: 0.14
```

**Notes:**
- Boxes must have valid coordinates where `x1 < x2` and `y1 < y2`
- Returns 0.0 if boxes don't overlap or have zero area

---

### fusion_detection

Combine detections from multiple models using weighted voting and spatial clustering.

**Signature:**
```python
def fusion_detection(
    all_detections: list,
    iou_threshold: float = 0.5,
    min_votes: int = 1
) -> list
```

**Parameters:**
- `all_detections` (list): List of detections, where each detection is a tuple:
  ```python
  (x1, y1, x2, y2, confidence, class_name, model_name, weight)
  ```
  - `x1, y1, x2, y2` (int): Bounding box coordinates
  - `confidence` (float): Detection confidence (0.0 to 1.0)
  - `class_name` (str): Detected class name (e.g., "apple", "banana")
  - `model_name` (str): Source model identifier
  - `weight` (float): Model weight for voting (higher = more influence)

- `iou_threshold` (float, optional): IoU threshold for clustering overlapping detections. Default: `0.5`
  - Higher values (0.7-0.9) = Stricter clustering, fewer merges
  - Lower values (0.3-0.5) = More aggressive clustering, more merges

- `min_votes` (int, optional): Minimum number of models that must agree for a detection to be kept. Default: `1`
  - `1` = Keep all detections
  - `2` = Require at least 2 models to agree
  - `3+` = Higher confidence, fewer false positives

**Returns:**
- `list`: List of fused detections, where each detection is a dictionary:
  ```python
  {
      'box': (x1, y1, x2, y2),      # Averaged bounding box coordinates
      'confidence': float,           # Weighted average confidence
      'class': str,                  # Class with highest weighted vote
      'votes': int,                  # Number of models that detected this object
      'models': int                  # Number of model detections in cluster
  }
  ```

**Example:**
```python
from detect_fusion import fusion_detection

# Simulated detections from 3 models
detections = [
    # Model 1: Apple detection
    (100, 100, 200, 200, 0.85, "apple", "model1", 1.0),
    # Model 2: Apple detection (similar location)
    (105, 105, 205, 205, 0.90, "apple", "model2", 1.5),
    # Model 3: Banana detection (different location)
    (300, 300, 400, 400, 0.75, "banana", "model3", 1.0),
]

# Apply fusion with default parameters
fused = fusion_detection(detections, iou_threshold=0.5, min_votes=1)

for det in fused:
    print(f"{det['class']}: {det['confidence']:.2f} ({det['votes']} models)")
# Output:
# apple: 0.88 (2 models)
# banana: 0.75 (1 models)
```

**Algorithm:**
1. **Spatial Clustering**: Groups detections with IoU > threshold
2. **Weighted Voting**: Selects class with highest weighted votes
3. **Coordinate Averaging**: Computes weighted average of box coordinates
4. **Confidence Fusion**: Calculates weighted average confidence
5. **Filtering**: Removes detections with votes < min_votes

**Notes:**
- Empty input returns empty list
- Detections from the same model at the same location are treated as separate votes
- Higher model weights have more influence on final class and confidence

---

### non_max_suppression_custom

Remove duplicate detections from multiple models using Non-Maximum Suppression.

**Signature:**
```python
def non_max_suppression_custom(
    detections: list,
    iou_threshold: float = 0.5
) -> list
```

**Parameters:**
- `detections` (list): List of detections, where each detection is a tuple:
  ```python
  (x1, y1, x2, y2, confidence, class_name, source)
  ```
  - `x1, y1, x2, y2` (int): Bounding box coordinates
  - `confidence` (float): Detection confidence (0.0 to 1.0)
  - `class_name` (str): Detected class name
  - `source` (str): Source model identifier

- `iou_threshold` (float, optional): IoU threshold for considering boxes as duplicates. Default: `0.5`

**Returns:**
- `list`: Filtered list of detections with duplicates removed, maintaining the same tuple format

**Example:**
```python
from detect_multi_model import non_max_suppression_custom

# Detections with duplicates
detections = [
    (100, 100, 200, 200, 0.90, "apple", "model1"),
    (105, 105, 205, 205, 0.85, "apple", "model2"),  # Duplicate
    (300, 300, 400, 400, 0.80, "banana", "model1"),
]

# Remove duplicates
filtered = non_max_suppression_custom(detections, iou_threshold=0.5)

print(f"Original: {len(detections)}, Filtered: {len(filtered)}")
# Output: Original: 3, Filtered: 2
```

**Algorithm:**
1. Sort detections by confidence (descending)
2. Keep highest confidence detection
3. Remove overlapping detections with IoU > threshold
4. Repeat for remaining detections

**Notes:**
- Keeps the detection with highest confidence when duplicates are found
- Does not merge or average boxes like `fusion_detection`
- Simpler alternative to fusion when you don't need weighted voting

---

## Configuration

### Loading Configuration

The system uses a JSON configuration file (`config.json`) to manage models, classes, and detection settings.

**Example:**
```python
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Access model configurations
models = config['models']
settings = config['detection_settings']
food_classes = config['food_classes']
```

### Configuration Structure

**Model Configuration:**
```python
{
    "models": [
        {
            "name": str,           # Model identifier
            "path": str,           # Path to model weights file
            "active": bool,        # Whether to use this model
            "weight": float,       # Model weight for fusion (default: 1.0)
            "description": str     # Human-readable description
        }
    ]
}
```

**Detection Settings:**
```python
{
    "detection_settings": {
        "confidence_threshold": float,  # Minimum confidence (0.0-1.0)
        "iou_threshold": float,         # IoU threshold for NMS
        "image_size": int,              # Input image size for models
        "camera_width": int,            # Camera capture width
        "camera_height": int,           # Camera capture height
        "fusion_iou": float,            # IoU threshold for fusion clustering
        "min_model_votes": int          # Minimum models required to agree
    }
}
```

**Food Classes:**
```python
{
    "food_classes": {
        "coco": {                       # COCO dataset classes
            "46": {"name": "banana", "active": true},
            "47": {"name": "apple", "active": true}
        },
        "custom": {                     # Custom model classes
            "0": {"name": "strawberry", "active": true}
        }
    }
}
```

**Example - Modifying Configuration:**
```python
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Disable a model
config['models'][0]['active'] = False

# Adjust fusion parameters
config['detection_settings']['fusion_iou'] = 0.6
config['detection_settings']['min_model_votes'] = 2

# Save modified config
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

## Model Management

### Loading Models

**Example - Load Active Models:**
```python
from ultralytics import YOLO
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load only active models
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
            print(f"✓ Loaded: {model_cfg['name']}")
        except Exception as e:
            print(f"✗ Failed: {model_cfg['name']} - {e}")

print(f"\nTotal models loaded: {len(models)}")
```

### Model Information

**Get Model Details:**
```python
# Access model properties
for model_info in models:
    model = model_info['model']
    print(f"Model: {model_info['name']}")
    print(f"  Classes: {len(model.names)}")
    print(f"  Weight: {model_info['weight']}")
    print(f"  Names: {list(model.names.values())[:5]}...")  # First 5 classes
```

**Run Inference:**
```python
import cv2

# Load image
image = cv2.imread('test_image.jpg')

# Run detection on single model
model = models[0]['model']
results = model(image, conf=0.25, iou=0.45, verbose=False)

# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        print(f"{class_name}: {confidence:.2f} at ({x1}, {y1}, {x2}, {y2})")
```

---

## Usage Examples

### Basic Detection

Detect objects in a single image using fusion:

```python
import cv2
from ultralytics import YOLO
import json
from detect_fusion import fusion_detection, calculate_iou

# Load configuration and models
with open('config.json', 'r') as f:
    config = json.load(f)

models = []
for model_cfg in config['models']:
    if model_cfg['active']:
        model = YOLO(model_cfg['path'])
        models.append({
            'name': model_cfg['name'],
            'model': model,
            'weight': model_cfg.get('weight', 1.0)
        })

# Load image
image = cv2.imread('fruits.jpg')

# Collect detections from all models
all_detections = []
for model_info in models:
    results = model_info['model'](image, conf=0.25, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            detection = (
                x1, y1, x2, y2,
                conf,
                cls_name,
                model_info['name'],
                model_info['weight']
            )
            all_detections.append(detection)

# Apply fusion
fused = fusion_detection(all_detections, iou_threshold=0.5, min_votes=2)

# Draw results
for det in fused:
    x1, y1, x2, y2 = det['box']
    label = f"{det['class']} {det['confidence']:.2f} [{det['votes']}M]"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save result
cv2.imwrite('result.jpg', image)
print(f"Detected {len(fused)} objects")
```

### Custom Fusion Parameters

Adjust fusion behavior for different scenarios:

```python
from detect_fusion import fusion_detection

# Scenario 1: High precision (require multiple models to agree)
high_precision = fusion_detection(
    all_detections,
    iou_threshold=0.6,    # Stricter clustering
    min_votes=3           # Require 3+ models
)

# Scenario 2: High recall (accept all detections)
high_recall = fusion_detection(
    all_detections,
    iou_threshold=0.4,    # More aggressive clustering
    min_votes=1           # Accept single model detections
)

# Scenario 3: Balanced
balanced = fusion_detection(
    all_detections,
    iou_threshold=0.5,    # Moderate clustering
    min_votes=2           # Require 2+ models
)

print(f"High Precision: {len(high_precision)} detections")
print(f"High Recall: {len(high_recall)} detections")
print(f"Balanced: {len(balanced)} detections")
```

### Batch Processing

Process multiple images efficiently:

```python
import cv2
import os
from pathlib import Path
from detect_fusion import fusion_detection

# Setup (load models as shown in Basic Detection)
# ... model loading code ...

# Process directory of images
input_dir = Path('input_images')
output_dir = Path('output_images')
output_dir.mkdir(exist_ok=True)

results_summary = []

for image_path in input_dir.glob('*.jpg'):
    print(f"Processing: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    
    # Collect detections
    all_detections = []
    for model_info in models:
        results = model_info['model'](image, conf=0.25, verbose=False)
        
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
    fused = fusion_detection(all_detections, iou_threshold=0.5, min_votes=2)
    
    # Draw and save
    for det in fused:
        x1, y1, x2, y2 = det['box']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save result
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), image)
    
    # Record summary
    results_summary.append({
        'image': image_path.name,
        'detections': len(fused),
        'classes': list(set(d['class'] for d in fused))
    })

# Print summary
print("\n=== Batch Processing Summary ===")
for result in results_summary:
    print(f"{result['image']}: {result['detections']} objects")
    print(f"  Classes: {', '.join(result['classes'])}")
```

---

## Error Handling

### Common Exceptions

**FileNotFoundError:**
```python
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found")
    # Use default configuration or exit
```

**Model Loading Errors:**
```python
from ultralytics import YOLO

try:
    model = YOLO('models/yolov8m.pt')
except Exception as e:
    print(f"Failed to load model: {e}")
    # Handle missing model file
```

**Invalid Detection Data:**
```python
from detect_fusion import fusion_detection

# Handle empty detections
detections = []
result = fusion_detection(detections)  # Returns []

# Validate detection format
def validate_detection(det):
    if len(det) != 8:
        raise ValueError(f"Invalid detection format: expected 8 elements, got {len(det)}")
    x1, y1, x2, y2, conf, name, model, weight = det
    if not (0 <= conf <= 1):
        raise ValueError(f"Invalid confidence: {conf}")
    if weight <= 0:
        raise ValueError(f"Invalid weight: {weight}")
```

---

## Performance Tips

1. **Model Selection**: Use fewer, higher-quality models for faster inference
2. **Image Size**: Reduce `image_size` in config for faster processing (trade-off: accuracy)
3. **Confidence Threshold**: Increase to reduce false positives and processing time
4. **Batch Processing**: Process multiple images in sequence rather than loading/unloading models
5. **GPU Acceleration**: Ensure CUDA is available for YOLOv8 models

**Check GPU Availability:**
```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")
```

---

## See Also

- [Usage Guide](USAGE.md) - Comprehensive usage examples and tutorials
- [Fusion Algorithm](FUSION_ALGORITHM.md) - Detailed algorithm explanation
- [Architecture](ARCHITECTURE.md) - System design and components
- [Installation](INSTALLATION.md) - Setup instructions
