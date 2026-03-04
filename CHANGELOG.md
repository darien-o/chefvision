# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository setup with multi-model fusion detection system
- YOLOv8-based ensemble learning for grocery, vegetable, and fruit detection
- Weighted voting fusion algorithm for combining predictions from multiple models
- Real-time detection support with camera integration
- Configuration system for model management (config.json)
- Model download utility script
- Training scripts and notebooks for custom model development
- Support for multiple pre-trained models (YOLOv8m COCO, Fruit-360, Grocery, YOLOv8n, YOLO26x)

### Changed

### Fixed

### Removed

## [0.1.0] - 2024-01-15

### Added
- Initial release of the multi-model fusion detection system
- Core detection scripts (detect_fruits.py, detect_fusion.py, detect_multi_model.py)
- Basic documentation (README.md, TRAINING.md, CUSTOM_MODEL.md)
- Configuration file for model weights and parameters
- Example detection scripts with various configurations

[Unreleased]: https://github.com/yourusername/chefvision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/chefvision/releases/tag/v0.1.0
