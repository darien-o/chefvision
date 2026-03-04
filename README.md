# Fruit & Vegetable Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Real-time multi-model fusion detection system for identifying groceries, vegetables, and fruits using YOLOv8 with ensemble learning.

## Overview

This project implements an advanced object detection system that combines predictions from multiple YOLOv8 models through intelligent weighted voting and spatial clustering. By leveraging ensemble learning, the system achieves higher accuracy and robustness compared to single-model approaches, making it ideal for grocery identification, inventory management, and nutritional tracking applications.

The fusion algorithm uses Intersection over Union (IoU) for spatial clustering and weighted voting for class prediction, allowing different models to contribute based on their reliability and specialization.

## Key Features

- **Multi-Model Ensemble**: Combines up to 5 specialized YOLOv8 models for comprehensive detection
- **Weighted Voting System**: Configurable model weights allow prioritizing more reliable models
- **Intelligent Fusion Algorithm**: IoU-based spatial clustering with weighted confidence aggregation
- **Real-Time Performance**: Optimized for 15-30 FPS on Apple Silicon (M1/M2)
- **Flexible Configuration**: JSON-based configuration for easy model and detection parameter tuning
- **Extensive Class Support**: Detects 50+ fruits, vegetables, and grocery items
- **Local Processing**: All detection happens on-device with no external API calls
- **Easy Integration**: Simple Python API for embedding in larger applications

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS (M1/M2 optimized) or Linux
- 8GB RAM minimum (16GB recommended)
- Camera device (for real-time detection)
- 2GB disk space for models

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/darien-o/chefvision.git
cd chefvision
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download models (optional - base model auto-downloads):
```bash
python download_models.py
```

5. Run detection:
```bash
python detect_fusion.py
```

Press 'q' to quit the camera view.

For detailed platform-specific installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Usage Examples

### Basic Detection

Simple single-model detection:

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/yolov8m.pt')

# Capture from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Run detection
results = model(frame)

# Display results
annotated = results[0].plot()
cv2.imshow('Detection', annotated)
cv2.waitKey(0)
```

### Fusion Detection (Recommended)

Multi-model ensemble detection with weighted voting:

```python
python detect_fusion.py
```

This script:
- Loads all active models from `config.json`
- Runs parallel inference on each frame
- Applies the fusion algorithm to combine predictions
- Displays results with model vote counts

### Custom Configuration

Modify `config.json` to customize detection:

```python
{
  "models": [
    {
      "name": "coco",
      "path": "models/yolov8m.pt",
      "active": true,
      "weight": 1.0  # Adjust model influence (0.5-2.0)
    }
  ],
  "detection_settings": {
    "confidence_threshold": 0.15,  # Lower = more detections
    "iou_threshold": 0.4,          # IoU for NMS
    "fusion_iou": 0.45,            # IoU for clustering
    "min_model_votes": 1           # Minimum models required
  }
}
```

### Batch Processing

Process multiple images:

```python
from detect_fusion import FusionDetector

detector = FusionDetector('config.json')

images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for img_path in images:
    results = detector.detect(img_path)
    print(f"{img_path}: {len(results)} objects detected")
```

For more examples, see the [examples/](examples/) directory.

## Examples

The [examples/](examples/) directory contains ready-to-run scripts demonstrating various use cases:

- **[basic_detection.py](examples/basic_detection.py)**: Simple single-model detection from camera or image file
- **[custom_config.py](examples/custom_config.py)**: Demonstrates how to customize detection configuration programmatically
- **[batch_processing.py](examples/batch_processing.py)**: Process multiple images efficiently with the fusion detector

Each example is self-contained and includes comments explaining the code. Run any example with:

```bash
python examples/basic_detection.py
```

## Fusion Algorithm

The fusion algorithm combines predictions from multiple models through a two-phase process:

### Phase 1: Spatial Clustering

Detections from all models are grouped by spatial proximity using Intersection over Union (IoU):

```
For each detection:
  - Calculate IoU with existing clusters
  - If IoU > threshold: add to cluster
  - Otherwise: create new cluster
```

### Phase 2: Weighted Voting

Within each cluster, predictions are combined using weighted averaging:

```
For each cluster:
  - Average bounding box coordinates (weighted by model weight)
  - Sum confidence scores (weighted by model weight)
  - Select class with highest weighted vote
  - Filter by minimum vote threshold
```

**Key Parameters:**
- `fusion_iou` (default: 0.45): IoU threshold for clustering detections
- `min_model_votes` (default: 1): Minimum models required to accept detection
- Model `weight` (default: 1.0): Influence factor for each model (0.5-2.0 recommended)

The algorithm ensures that:
- Multiple models agreeing on an object increases confidence
- More reliable models have greater influence
- Spatial precision is maintained through weighted coordinate averaging
- False positives are reduced through vote thresholding

For detailed algorithm explanation with pseudocode and diagrams, see [docs/FUSION_ALGORITHM.md](docs/FUSION_ALGORITHM.md).

## Configuration Guide

### Model Configuration

Each model in `config.json` has the following properties:

- **name**: Identifier for the model
- **path**: File path to the model weights
- **active**: Enable/disable the model (true/false)
- **weight**: Influence factor in fusion (0.5-2.0 recommended)
- **description**: Human-readable description

### Detection Settings

- **confidence_threshold**: Minimum confidence for individual model detections (0.0-1.0)
- **iou_threshold**: IoU threshold for Non-Maximum Suppression within each model
- **fusion_iou**: IoU threshold for clustering detections across models
- **min_model_votes**: Minimum number of models that must detect an object
- **image_size**: Input size for YOLO models (640 recommended)
- **camera_width/height**: Camera resolution settings

### Class Configuration

Enable/disable specific fruits and vegetables in the `food_classes` section:

```json
"food_classes": {
  "coco": {
    "46": {"name": "banana", "active": true},
    "47": {"name": "apple", "active": true}
  }
}
```

## Project Structure

```
chefvision/
├── .github/              # GitHub templates and workflows
├── docs/                 # Extended documentation
│   ├── ARCHITECTURE.md
│   ├── FUSION_ALGORITHM.md
│   ├── MODEL_SOURCES.md
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   └── API.md
├── examples/             # Usage examples
│   ├── basic_detection.py
│   ├── custom_config.py
│   └── batch_processing.py
├── models/               # Model weights directory
│   └── README.md
├── tests/                # Test suite
│   ├── test_fusion_algorithm.py
│   └── fixtures/
├── config.json           # Configuration file
├── detect_fusion.py      # Main fusion detection script
├── detect_fruits.py      # Basic detection script
├── download_models.py    # Model download utility
├── train_model.py        # Training script
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py              # Package installation
├── README.md             # This file
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Community standards
└── CHANGELOG.md          # Version history
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design and component interaction
- **[Fusion Algorithm](docs/FUSION_ALGORITHM.md)**: Detailed algorithm explanation with diagrams
- **[Model Sources](docs/MODEL_SOURCES.md)**: Model provenance and training data
- **[Installation](docs/INSTALLATION.md)**: Platform-specific setup instructions
- **[Usage Guide](docs/USAGE.md)**: Comprehensive usage examples
- **[API Reference](docs/API.md)**: Python API documentation
- **[Training Guide](TRAINING.md)**: Custom model training instructions

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up the development environment
- Code style and standards
- Testing requirements
- Pull request process

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ultralytics**: YOLOv8 implementation and pre-trained models
- **COCO Dataset**: Base model training data
- **Fruit-360 Dataset**: Specialized fruit detection training
- **Roboflow**: Dataset management and model hosting
- **OpenCV**: Computer vision and camera access
- **Community Contributors**: Thank you to all who have contributed to this project

## Security

### Data Privacy

This system is designed with privacy as a core principle:

- **Local Processing Only**: All image processing and object detection happens entirely on your local device. No images, video frames, or detection results are transmitted to external servers or cloud services.
- **No Network Requests**: The detection pipeline operates completely offline once models are downloaded. No API calls or network connections are made during detection.
- **No Data Storage**: By default, the system does not save or log any camera frames or detection results. All processing is done in-memory and discarded after display.
- **Camera Access**: Camera access is only used when explicitly running detection scripts. The system does not access the camera in the background.

### Model Security

When using pre-trained models, consider the following security practices:

- **Trusted Sources**: Only download models from official and verified sources:
  - Ultralytics official repository (YOLOv8 base models)
  - Roboflow verified datasets and models
  - Official model repositories linked in [docs/MODEL_SOURCES.md](docs/MODEL_SOURCES.md)
- **Checksum Verification**: Verify model file integrity using checksums when available. The `download_models.py` script includes checksum validation for supported models.
- **Model Provenance**: Review model training data and provenance documentation in [docs/MODEL_SOURCES.md](docs/MODEL_SOURCES.md) to understand potential biases or limitations.
- **Sandboxed Execution**: Models run within the Python/PyTorch runtime environment. Avoid loading models from untrusted sources as they could potentially execute malicious code during deserialization.
- **Model File Permissions**: Store model files with appropriate file system permissions to prevent unauthorized modification.

### Dependency Security

Maintain security through proper dependency management:

- **Pinned Versions**: All dependencies in `requirements.txt` use version pinning to ensure reproducible builds and prevent unexpected updates.
- **Regular Updates**: Periodically update dependencies to receive security patches. Check for vulnerabilities using:
  ```bash
  pip install pip-audit
  pip-audit
  ```
- **Minimal Dependencies**: The project maintains a minimal dependency footprint to reduce attack surface.
- **Virtual Environments**: Always use virtual environments (`.venv`) to isolate project dependencies from system packages.
- **Development Dependencies**: Security-sensitive development tools are separated in `requirements-dev.txt` and should not be installed in production environments.

### Reporting Security Issues

If you discover a security vulnerability, please report it privately by emailing the maintainers directly rather than opening a public issue. Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes (if any)

We will respond to security reports within 48 hours and work to address confirmed vulnerabilities promptly.

## Citation

If you use this project in your research or application, please cite:

```bibtex
@software{chefvision,
  title = {Fruit & Vegetable Detection System - Chef Vision},
  author = {Darien Osorno, Diego Echavarria, Tomas Atehortua},
  year = {2026},
  url = {https://github.com/darien-o/chefvision}
}
```

---

**Note**: This system requires 2+ active models for optimal fusion results. Model weights should be tuned based on your specific use case and model reliability.
