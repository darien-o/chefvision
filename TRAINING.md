# Training Custom Fruit & Vegetable Detection Model

## Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook train_model.ipynb
```

### Option 2: Python Script
```bash
python train_model.py
```

## Dataset Options

### A. Roboflow (Easiest)
1. Sign up at https://roboflow.com
2. Get API key from account settings
3. Update API key in notebook/script
4. Dataset auto-downloads (1000+ images)

### B. Custom Dataset
1. Collect images of fruits/vegetables
2. Label using Roboflow or LabelImg
3. Organize in YOLO format:
```
datasets/custom/
├── train/
│   ├── images/  (put .jpg here)
│   └── labels/  (put .txt here)
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Label Format (YOLO)

Each image needs a .txt file with same name:
```
class_id x_center y_center width height
```

Example (apple.txt):
```
0 0.5 0.5 0.3 0.4
```

Values are normalized (0-1).

## Training Parameters

### Model Size
- `yolov8n` - Fastest, least accurate (~6MB)
- `yolov8s` - Small, fast (~22MB)
- `yolov8m` - Medium, balanced (~50MB) ⭐ Recommended
- `yolov8l` - Large, accurate (~87MB)
- `yolov8x` - Largest, most accurate (~136MB)

### Epochs
- Quick test: 50 epochs
- Production: 100-200 epochs
- Large dataset: 300+ epochs

### Batch Size
- M1 Mac: 16-32
- NVIDIA GPU: 32-64
- CPU: 8-16

## Training Time Estimates

| Model | Dataset | Epochs | M1 Mac | NVIDIA GPU |
|-------|---------|--------|--------|------------|
| yolov8n | 1000 imgs | 100 | ~2 hrs | ~30 min |
| yolov8m | 1000 imgs | 100 | ~4 hrs | ~1 hr |
| yolov8x | 1000 imgs | 100 | ~8 hrs | ~2 hrs |

## After Training

1. Best model saved to: `runs/train/fruit_vegetable_detector/weights/best.pt`
2. Copy to: `models/best.pt`
3. Update `config.json`:
```json
{
  "name": "custom",
  "path": "models/best.pt",
  "active": true,
  "weight": 1.5
}
```
4. Run: `python detect_fusion.py`

## Tips for Better Results

### Data Quality
- ✅ 100+ images per class minimum
- ✅ Varied lighting conditions
- ✅ Different angles and backgrounds
- ✅ Multiple objects per image
- ✅ Balanced classes

### Training
- ✅ Use data augmentation
- ✅ Monitor validation loss
- ✅ Early stopping (patience=20)
- ✅ Save checkpoints
- ✅ Use pre-trained weights

### If Results Are Poor
- 📈 Collect more data
- 📈 Increase epochs
- 📈 Use larger model
- 📈 Adjust learning rate
- 📈 Check label quality

## Public Datasets

1. **Fruits-360**: 90,000+ images, 131 classes
   - https://www.kaggle.com/moltean/fruits

2. **Grocery Store Dataset**: 5,000+ images
   - https://github.com/marcusklasson/GroceryStoreDataset

3. **Roboflow Universe**: 1000s of datasets
   - https://universe.roboflow.com

## Troubleshooting

**Out of Memory**
- Reduce batch size
- Use smaller model
- Reduce image size

**Training Too Slow**
- Use GPU if available
- Reduce workers
- Use smaller model

**Poor Accuracy**
- More training data
- Longer training
- Better labels
- Data augmentation
