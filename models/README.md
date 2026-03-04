# Models Directory

This directory contains YOLO model files used by the multi-model fusion detection system. The system supports multiple models running in parallel, combining their predictions through weighted ensemble learning for improved accuracy.

## Directory Structure

```
models/
├── README.md           # This file
├── yolov8m.pt         # YOLOv8 medium COCO model
├── yolov8n.pt         # YOLOv8 nano model (lightweight)
├── yolo26x.pt         # YOLO26x model
├── fruit_360.pt       # Fruit-360 specialized model
└── grocery.pt         # Grocery detection model
```

## Model Management

### Current Models

The repository is configured to use the following models:

1. **yolov8m.pt** - YOLOv8 medium model trained on COCO dataset
   - 80 object classes including common fruits and vegetables
   - Good balance between speed and accuracy
   - Auto-downloads on first run if missing

2. **yolov8n.pt** - YOLOv8 nano model
   - Lightweight, fast inference
   - Lower accuracy but excellent for real-time applications

3. **yolo26x.pt** - YOLO26x model
   - Extended model with additional capabilities
   - Higher accuracy for specific object classes

4. **fruit_360.pt** - Specialized fruit detection model
   - Trained on Fruit-360 dataset
   - Optimized for fruit and vegetable recognition

5. **grocery.pt** - Grocery-specific detection model
   - Trained on grocery store items
   - Enhanced detection for packaged and fresh produce

## Downloading Models

### Method 1: Automatic Download (Recommended)

The system automatically downloads missing COCO models on first run:

```python
from ultralytics import YOLO

# These models auto-download if not present
model = YOLO('yolov8m.pt')
model = YOLO('yolov8n.pt')
```

### Method 2: Manual Download

For specialized models or custom weights:

1. **From Roboflow Universe**:
   ```bash
   # Visit https://universe.roboflow.com
   # Search for "fruit vegetable detection" or "grocery detection"
   # Download in YOLOv8 PyTorch format
   # Place .pt file in models/ directory
   ```

2. **From Ultralytics Hub**:
   ```bash
   # Visit https://hub.ultralytics.com
   # Browse pre-trained models
   # Download and place in models/ directory
   ```

3. **Using download_models.py script**:
   ```bash
   python download_models.py
   ```

### Method 3: Training Custom Models

Train your own models using the provided training scripts:

```bash
# See TRAINING.md for detailed instructions
python train_model.py --data your_dataset.yaml --epochs 100
```

## Organizing Models

### File Naming Conventions

- Use descriptive names: `fruit_360.pt`, `grocery.pt`, `yolov8m.pt`
- Avoid generic names like `best.pt` or `model.pt`
- Include version numbers for custom models: `grocery_v2.pt`

### Model Configuration

Models are configured in `config.json`:

```json
{
  "models": [
    {
      "name": "YOLOv8m COCO",
      "path": "models/yolov8m.pt",
      "weight": 1.0,
      "active": true
    },
    {
      "name": "Fruit-360 Model",
      "path": "models/fruit_360.pt",
      "weight": 1.5,
      "active": true
    }
  ]
}
```

### Weight Configuration

Model weights determine influence in the fusion algorithm:

- **Default weight**: 1.0 (standard influence)
- **Specialized models**: 1.5-2.0 (higher trust for domain-specific models)
- **Large models**: 1.2-1.5 (better accuracy, higher confidence)
- **Lightweight models**: 0.8-1.0 (faster but less accurate)

**Guidelines**:
- Increase weight for models trained on similar data to your use case
- Decrease weight for general-purpose models
- Balance weights to avoid over-reliance on a single model

## Model Storage Best Practices

### Version Control

Model files are excluded from git (see `.gitignore`):
- Models are binary files and can be very large (100MB-500MB)
- Store models separately using Git LFS or external storage
- Document model sources and download instructions instead

### Storage Locations

**Local Development**:
```
models/          # Local model storage (gitignored)
```

**Production Deployment**:
- Use cloud storage (S3, GCS, Azure Blob)
- Download models during deployment
- Cache models for faster startup

### Model Versioning

Track model versions in `config.json` or separate metadata:

```json
{
  "name": "Fruit-360 Model",
  "path": "models/fruit_360.pt",
  "version": "2.1.0",
  "trained_date": "2024-01-15",
  "dataset": "Fruit-360 v2",
  "accuracy": 0.94
}
```

## Fusion Algorithm Overview

The system combines predictions from multiple models:

1. **Parallel Inference**: All active models process each frame simultaneously
2. **Spatial Clustering**: Groups overlapping detections using IoU threshold (default: 0.5)
3. **Weighted Voting**: Combines predictions using model weights
4. **Coordinate Averaging**: Averages bounding box coordinates weighted by confidence
5. **Minimum Votes**: Filters detections requiring agreement from multiple models

See `docs/FUSION_ALGORITHM.md` for detailed algorithm documentation.

## Troubleshooting

### Model Not Found

```
Error: Model file not found: models/yolov8m.pt
```

**Solution**: Run the detection script once to auto-download, or manually download:
```python
from ultralytics import YOLO
YOLO('yolov8m.pt')  # Downloads to default location
```

### Out of Memory

```
Error: CUDA out of memory
```

**Solution**: 
- Reduce number of active models in `config.json`
- Use smaller models (yolov8n instead of yolov8x)
- Reduce input image resolution

### Incompatible Model Format

```
Error: Unable to load model weights
```

**Solution**:
- Ensure model is in PyTorch (.pt) format
- Verify model is compatible with installed ultralytics version
- Re-download or re-train the model

## Model Sources and Licensing

Always verify model licensing before use:

- **COCO models**: Apache 2.0 License (commercial use allowed)
- **Custom models**: Check training data license
- **Roboflow models**: Varies by dataset (check individual licenses)

See `docs/MODEL_SOURCES.md` for detailed provenance information.

## Performance Considerations

### Model Selection

- **Real-time applications**: Use 2-3 lightweight models (yolov8n, yolov8s)
- **High accuracy**: Use 4-5 models including large variants (yolov8m, yolov8l)
- **Balanced**: Use 3-4 medium models with specialized weights

### Inference Speed

Approximate inference times on M1 Mac (per model):
- yolov8n: ~15ms
- yolov8s: ~25ms
- yolov8m: ~45ms
- yolov8l: ~80ms
- yolov8x: ~150ms

**Total system latency** = (slowest model time) + fusion overhead (~5ms)

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com)
- [YOLOv8 Model Zoo](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com)
- [Model Training Guide](../TRAINING.md)
- [Custom Model Integration](../docs/MODEL_SOURCES.md)
