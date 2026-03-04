# Usage Guide

This guide provides comprehensive examples and instructions for using the Fruit & Vegetable Detection System. Whether you're running real-time detection, processing batch images, or integrating the system into your application, this guide covers all common workflows.

## Table of Contents

- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Configuration Options](#configuration-options)
- [Python API Usage](#python-api-usage)
- [Common Workflows](#common-workflows)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Real-Time Detection

The fastest way to start detecting fruits and vegetables:

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run fusion detection
python detect_fusion.py
```

**Controls:**
- Press `q` to quit
- The camera window shows real-time detections with bounding boxes
- Detection labels show: `[class_name] [confidence] [number_of_models]`

### Single Image Detection

Process a single image file:

```bash
python detect_fruits.py --source path/to/image.jpg
```

### Batch Processing

Process multiple images in a directory:

```bash
python detect_fruits_advanced.py --source path/to/images/ --save
```

## Command-Line Interface

### detect_fusion.py

Multi-model fusion detection with real-time camera feed.

**Usage:**
```bash
python detect_fusion.py
```

**Features:**
- Loads all active models from `config.json`
- Applies fusion algorithm to combine predictions
- Displays vote counts for each detection
- Color-coded boxes (brighter = more model agreement)

**Output:**
- Real-time camera window with annotated detections
- Console output showing loaded models and their weights
- Detection statistics (count, active models)

**Example Output:**
```
✓ coco: Base COCO model with 5 fruits/vegetables (weight: 1.0)
✓ fruit_360: Fruit-360 specialized model (weight: 1.5)
✓ grocery: Grocery items detection model (weight: 1.3)

🚀 Fusion Detection System Active (3 models)
```

### detect_fruits.py

Basic single-model detection script.

**Usage:**
```bash
python detect_fruits.py [OPTIONS]
```

**Options:**
- `--source`: Input source (camera index, image file, or directory)
  - Default: `0` (default camera)
  - Examples: `--source 1`, `--source image.jpg`, `--source images/`
- `--model`: Path to YOLO model file
  - Default: `models/yolov8m.pt`
  - Example: `--source models/yolov8n.pt`
- `--conf`: Confidence threshold (0.0-1.0)
  - Default: `0.25`
  - Example: `--conf 0.5`

**Examples:**
```bash
# Use default camera
python detect_fruits.py

# Process specific image
python detect_fruits.py --source photo.jpg

# Use different model with higher confidence
python detect_fruits.py --model models/yolov8n.pt --conf 0.5

# Process all images in directory
python detect_fruits.py --source images/
```

### detect_fruits_advanced.py

Advanced detection with additional options and output control.

**Usage:**
```bash
python detect_fruits_advanced.py [OPTIONS]
```

**Options:**
- `--source`: Input source (camera, image, video, directory)
- `--model`: Model path
- `--conf`: Confidence threshold
- `--iou`: IoU threshold for NMS
- `--imgsz`: Input image size (default: 640)
- `--save`: Save annotated results
- `--save-txt`: Save detection results as text files
- `--save-conf`: Include confidence in saved results
- `--nosave`: Don't save images/videos
- `--view-img`: Display results in window
- `--device`: Device to run on (cpu, 0, 1, etc.)

**Examples:**
```bash
# Process video file and save results
python detect_fruits_advanced.py --source video.mp4 --save

# High-confidence detections only
python detect_fruits_advanced.py --source images/ --conf 0.7 --save

# Use GPU device 0
python detect_fruits_advanced.py --source 0 --device 0

# Save detection coordinates to text files
python detect_fruits_advanced.py --source images/ --save-txt --save-conf
```

### detect_multi_model.py

Run multiple models in parallel without fusion.

**Usage:**
```bash
python detect_multi_model.py
```

**Features:**
- Loads all active models from config
- Displays separate detections from each model
- Useful for comparing model performance
- No fusion algorithm applied

### download_models.py

Download pre-trained models.

**Usage:**
```bash
python download_models.py
```

**Features:**
- Downloads base YOLOv8 models
- Places models in `models/` directory
- Verifies model integrity
- Shows download progress

## Configuration Options

All detection scripts use `config.json` for configuration. Edit this file to customize behavior.

### Model Configuration

Configure which models to use and their influence:

```json
{
  "models": [
    {
      "name": "coco",
      "path": "models/yolov8m.pt",
      "active": true,
      "weight": 1.0,
      "description": "Base COCO model"
    }
  ]
}
```

**Parameters:**
- `name`: Identifier for the model (string)
- `path`: File path to model weights (string)
- `active`: Enable/disable model (boolean)
  - `true`: Model will be loaded and used
  - `false`: Model will be skipped
- `weight`: Influence factor in fusion (float, 0.5-2.0 recommended)
  - `1.0`: Standard influence
  - `> 1.0`: Increased influence (more reliable models)
  - `< 1.0`: Decreased influence (less reliable models)
- `description`: Human-readable description (string)

**Example - Disable a model:**
```json
{
  "name": "yolov8n",
  "path": "models/yolov8n.pt",
  "active": false,
  "weight": 0.8,
  "description": "Lightweight model (disabled)"
}
```

**Example - Prioritize a model:**
```json
{
  "name": "fruit_360",
  "path": "models/fruit_360.pt",
  "active": true,
  "weight": 1.8,
  "description": "High-accuracy fruit model"
}
```

### Detection Settings

Configure detection parameters:

```json
{
  "detection_settings": {
    "confidence_threshold": 0.15,
    "iou_threshold": 0.4,
    "image_size": 640,
    "camera_width": 1280,
    "camera_height": 720,
    "fusion_iou": 0.45,
    "min_model_votes": 1
  }
}
```

**Parameters:**

- `confidence_threshold` (float, 0.0-1.0): Minimum confidence for detections
  - Lower = more detections (may include false positives)
  - Higher = fewer detections (may miss objects)
  - Recommended: 0.15-0.25

- `iou_threshold` (float, 0.0-1.0): IoU threshold for Non-Maximum Suppression
  - Controls duplicate detection removal within each model
  - Lower = more aggressive suppression
  - Higher = more lenient (may keep duplicates)
  - Recommended: 0.4-0.5

- `image_size` (int): Input size for YOLO models
  - Common values: 320, 640, 1280
  - Larger = more accurate but slower
  - Smaller = faster but less accurate
  - Recommended: 640

- `camera_width` (int): Camera capture width in pixels
  - Recommended: 1280 or 1920

- `camera_height` (int): Camera capture height in pixels
  - Recommended: 720 or 1080

- `fusion_iou` (float, 0.0-1.0): IoU threshold for clustering detections across models
  - Controls when detections from different models are considered the same object
  - Lower = more aggressive clustering (may merge distinct objects)
  - Higher = more conservative (may create duplicates)
  - Recommended: 0.45-0.55

- `min_model_votes` (int): Minimum number of models that must detect an object
  - `1`: Accept all detections (no filtering)
  - `2`: Require at least 2 models to agree (recommended)
  - `3+`: More conservative, reduces false positives
  - Recommended: 1-2

### Class Configuration

Enable/disable specific object classes:

```json
{
  "food_classes": {
    "coco": {
      "46": {"name": "banana", "active": true},
      "47": {"name": "apple", "active": true},
      "49": {"name": "orange", "active": false}
    },
    "custom": {
      "0": {"name": "strawberry", "active": true},
      "1": {"name": "grape", "active": true}
    }
  }
}
```

**Parameters:**
- Class ID (string): Numeric identifier from model
- `name`: Class name (string)
- `active`: Enable/disable detection of this class (boolean)

**Example - Detect only specific fruits:**
```json
{
  "food_classes": {
    "custom": {
      "0": {"name": "strawberry", "active": true},
      "1": {"name": "grape", "active": false},
      "2": {"name": "watermelon", "active": true}
    }
  }
}
```

## Python API Usage

### Basic Detection

Simple detection using a single model:

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/yolov8m.pt')

# Load image
image = cv2.imread('path/to/image.jpg')

# Run detection
results = model(image, conf=0.25)

# Process results
for result in results:
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        print(f"Detected: {class_name} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display result
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Fusion Detection API

Use the fusion algorithm programmatically:

```python
import json
from ultralytics import YOLO
import cv2

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load active models
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
image = cv2.imread('path/to/image.jpg')

# Collect detections from all models
all_detections = []
settings = config['detection_settings']

for model_info in models:
    model_name = model_info['name']
    model = model_info['model']
    weight = model_info['weight']
    
    results = model(image, conf=settings['confidence_threshold'], 
                   iou=settings['iou_threshold'], verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            
            all_detections.append((x1, y1, x2, y2, conf, cls_name, model_name, weight))

# Apply fusion algorithm (see fusion_detection function in detect_fusion.py)
from detect_fusion import fusion_detection
final_detections = fusion_detection(all_detections, 
                                    iou_threshold=settings['fusion_iou'],
                                    min_votes=settings['min_model_votes'])

# Process fused results
for det in final_detections:
    x1, y1, x2, y2 = det['box']
    confidence = det['confidence']
    class_name = det['class']
    votes = det['votes']
    
    print(f"Fused: {class_name} ({confidence:.2f}) with {votes} model votes")
```

### Custom Detection Class

Create a reusable detection class:

```python
import json
from ultralytics import YOLO
import cv2
from detect_fusion import fusion_detection

class FruitDetector:
    def __init__(self, config_path='config.json'):
        """Initialize detector with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.models = []
        for model_cfg in self.config['models']:
            if model_cfg['active']:
                model = YOLO(model_cfg['path'])
                self.models.append({
                    'name': model_cfg['name'],
                    'model': model,
                    'weight': model_cfg.get('weight', 1.0)
                })
        
        self.settings = self.config['detection_settings']
    
    def detect(self, image):
        """Run detection on image"""
        all_detections = []
        
        for model_info in self.models:
            model_name = model_info['name']
            model = model_info['model']
            weight = model_info['weight']
            
            results = model(image, conf=self.settings['confidence_threshold'], 
                           iou=self.settings['iou_threshold'], verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = result.names[cls]
                    
                    all_detections.append((x1, y1, x2, y2, conf, cls_name, model_name, weight))
        
        return fusion_detection(all_detections, 
                               iou_threshold=self.settings['fusion_iou'],
                               min_votes=self.settings['min_model_votes'])
    
    def detect_file(self, image_path):
        """Detect objects in image file"""
        image = cv2.imread(image_path)
        return self.detect(image)
    
    def annotate(self, image, detections):
        """Draw detections on image"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            cls_name = det['class']
            votes = det['votes']
            
            # Color intensity based on votes
            intensity = min(255, 100 + (votes * 50))
            color = (0, intensity, 0)
            
            label = f"{cls_name} {conf:.2f} [{votes}M]"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image

# Usage
detector = FruitDetector()
detections = detector.detect_file('image.jpg')
print(f"Found {len(detections)} objects")

# Annotate and display
image = cv2.imread('image.jpg')
annotated = detector.annotate(image, detections)
cv2.imshow('Detections', annotated)
cv2.waitKey(0)
```

## Common Workflows

### Workflow 1: Real-Time Camera Detection

Monitor camera feed with real-time detection:

```bash
# 1. Ensure camera is connected
# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run fusion detection
python detect_fusion.py

# 4. Press 'q' to quit when done
```

**Tips:**
- Adjust `camera_width` and `camera_height` in config for better performance
- Reduce number of active models if FPS is low
- Use `min_model_votes: 2` to reduce false positives

### Workflow 2: Batch Image Processing

Process multiple images and save results:

```bash
# 1. Place images in a directory (e.g., input_images/)
# 2. Run detection with save option
python detect_fruits_advanced.py --source input_images/ --save --save-txt

# 3. Results saved to runs/detect/exp/
# - Annotated images in runs/detect/exp/
# - Detection coordinates in runs/detect/exp/labels/
```

**Output Structure:**
```
runs/detect/exp/
├── image1.jpg          # Annotated image
├── image2.jpg
├── labels/
│   ├── image1.txt      # Detection coordinates
│   └── image2.txt
```

**Detection Text Format:**
```
class_id x_center y_center width height confidence
0 0.5 0.5 0.2 0.3 0.85
```

### Workflow 3: Video Processing

Process video file with detection:

```bash
# Process video
python detect_fruits_advanced.py --source video.mp4 --save

# Output video saved to runs/detect/exp/video.mp4
```

**Tips:**
- Use lighter models (yolov8n) for faster processing
- Reduce `image_size` to 320 or 480 for speed
- Consider frame skipping for long videos

### Workflow 4: Custom Model Integration

Add your own trained model:

```bash
# 1. Place model file in models/ directory
cp my_custom_model.pt models/

# 2. Edit config.json
```

```json
{
  "models": [
    {
      "name": "my_model",
      "path": "models/my_custom_model.pt",
      "active": true,
      "weight": 1.2,
      "description": "My custom trained model"
    }
  ]
}
```

```bash
# 3. Run detection
python detect_fusion.py
```

### Workflow 5: Performance Tuning

Optimize for speed or accuracy:

**For Speed (High FPS):**
```json
{
  "models": [
    {
      "name": "yolov8n",
      "path": "models/yolov8n.pt",
      "active": true,
      "weight": 1.0
    }
  ],
  "detection_settings": {
    "confidence_threshold": 0.3,
    "image_size": 320,
    "camera_width": 640,
    "camera_height": 480,
    "min_model_votes": 1
  }
}
```

**For Accuracy (Better Detection):**
```json
{
  "models": [
    {
      "name": "yolov8m",
      "path": "models/yolov8m.pt",
      "active": true,
      "weight": 1.0
    },
    {
      "name": "fruit_360",
      "path": "models/fruit_360.pt",
      "active": true,
      "weight": 1.5
    }
  ],
  "detection_settings": {
    "confidence_threshold": 0.15,
    "image_size": 640,
    "fusion_iou": 0.45,
    "min_model_votes": 2
  }
}
```

### Workflow 6: Filtering Specific Classes

Detect only specific fruits/vegetables:

```json
{
  "food_classes": {
    "custom": {
      "0": {"name": "strawberry", "active": true},
      "1": {"name": "grape", "active": false},
      "2": {"name": "watermelon", "active": true},
      "3": {"name": "pineapple", "active": false}
    }
  }
}
```

Run detection - only strawberries and watermelons will be detected.

## Advanced Usage

### Multi-Camera Setup

Run detection on multiple cameras:

```python
import cv2
from detect_fusion import fusion_detection
# ... (load models as shown above)

# Open multiple cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if ret1:
        # Process camera 1
        detections1 = detect(frame1)  # Your detection function
        annotated1 = annotate(frame1, detections1)
        cv2.imshow('Camera 1', annotated1)
    
    if ret2:
        # Process camera 2
        detections2 = detect(frame2)
        annotated2 = annotate(frame2, detections2)
        cv2.imshow('Camera 2', annotated2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
```

### Detection with Tracking

Add object tracking to maintain IDs across frames:

```python
from ultralytics import YOLO

# Use YOLO's built-in tracking
model = YOLO('models/yolov8m.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Track objects (maintains IDs across frames)
    results = model.track(frame, persist=True, conf=0.25)
    
    # Draw tracked results
    annotated = results[0].plot()
    cv2.imshow('Tracking', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Export Detection Results

Save detection results to JSON:

```python
import json
from detect_fusion import FruitDetector

detector = FruitDetector()
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']

results = {}

for img_path in image_files:
    detections = detector.detect_file(img_path)
    
    results[img_path] = [
        {
            'box': det['box'],
            'confidence': det['confidence'],
            'class': det['class'],
            'votes': det['votes']
        }
        for det in detections
    ]

# Save to JSON
with open('detection_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved results for {len(image_files)} images")
```

### Integration with Web API

Create a simple Flask API:

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from detect_fusion import FruitDetector

app = Flask(__name__)
detector = FruitDetector()

@app.route('/detect', methods=['POST'])
def detect():
    # Get image from request
    file = request.files['image']
    
    # Convert to OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Run detection
    detections = detector.detect(image)
    
    # Format response
    response = {
        'count': len(detections),
        'detections': [
            {
                'class': det['class'],
                'confidence': det['confidence'],
                'box': det['box'],
                'votes': det['votes']
            }
            for det in detections
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
# Start server
python api.py

# Send request
curl -X POST -F "image=@photo.jpg" http://localhost:5000/detect
```

## Troubleshooting

### Issue: Low FPS / Slow Detection

**Symptoms:**
- Camera feed is laggy
- Detection takes too long

**Solutions:**
1. Reduce number of active models in config
2. Use lighter model (yolov8n instead of yolov8m)
3. Decrease image size: `"image_size": 320`
4. Lower camera resolution: `"camera_width": 640, "camera_height": 480`
5. Disable unnecessary models: `"active": false`

### Issue: Too Many False Positives

**Symptoms:**
- Detecting objects that aren't there
- Incorrect classifications

**Solutions:**
1. Increase confidence threshold: `"confidence_threshold": 0.3`
2. Increase minimum votes: `"min_model_votes": 2`
3. Disable less accurate models
4. Adjust model weights (lower weight for unreliable models)

### Issue: Missing Detections

**Symptoms:**
- Objects not being detected
- Low detection count

**Solutions:**
1. Decrease confidence threshold: `"confidence_threshold": 0.15`
2. Decrease minimum votes: `"min_model_votes": 1`
3. Enable more models
4. Check if object class is active in config
5. Ensure good lighting and camera focus

### Issue: Duplicate Detections

**Symptoms:**
- Multiple boxes around same object
- Overlapping detections

**Solutions:**
1. Increase fusion IoU: `"fusion_iou": 0.6`
2. Increase minimum votes: `"min_model_votes": 2`
3. Decrease IoU threshold: `"iou_threshold": 0.3`

### Issue: Camera Not Opening

**Symptoms:**
- Black screen or error message
- "Cannot open camera" error

**Solutions:**
1. Check camera permissions (macOS: System Preferences → Security & Privacy → Camera)
2. Try different camera index: `cv2.VideoCapture(1)` instead of `0`
3. Close other applications using camera
4. Verify camera works with other applications
5. Check camera connection

### Issue: Model Not Loading

**Symptoms:**
- "Failed to load model" error
- Model file not found

**Solutions:**
1. Verify model file exists: `ls models/`
2. Check file path in config.json
3. Download models: `python download_models.py`
4. Verify model file isn't corrupted (check file size)
5. Ensure model is in correct format (.pt file)

### Issue: Out of Memory

**Symptoms:**
- Program crashes
- "Out of memory" error

**Solutions:**
1. Reduce number of active models
2. Use smaller models (yolov8n)
3. Decrease image size
4. Lower camera resolution
5. Close other applications
6. Process images one at a time instead of batch

## Next Steps

- **Training Custom Models**: See [TRAINING.md](../TRAINING.md) for model training guide
- **Algorithm Details**: See [FUSION_ALGORITHM.md](FUSION_ALGORITHM.md) for fusion algorithm explanation
- **System Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- **API Reference**: See [API.md](API.md) for detailed API documentation
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute to the project

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [YOLO Object Detection Guide](https://pjreddie.com/darknet/yolo/)

---

**Need Help?** If you encounter issues not covered here, please:
1. Check [GitHub Issues](https://github.com/darien-o/chefvision/issues)
2. Create a new issue with details about your problem
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) for community support channels
