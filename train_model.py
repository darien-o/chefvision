"""
Fine-tune YOLOv8 for Fruit & Vegetable Detection
Run: python train_model.py
"""

import os
from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path

def setup_dataset():
    """Download dataset from Roboflow"""
    print("📦 Downloading dataset...")
    
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="YOUR_API_KEY")
        project = rf.workspace("joseph-nelson").project("fruits-vegetables-detection")
        dataset = project.version(2).download("yolov8", location="./datasets/fruits")
        print(f"✓ Dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"✗ Roboflow download failed: {e}")
        print("Using manual dataset setup...")
        return create_manual_dataset()

def create_manual_dataset():
    """Create dataset structure for manual labeling"""
    base = Path('datasets/custom')
    for split in ['train', 'valid', 'test']:
        (base / split / 'images').mkdir(parents=True, exist_ok=True)
        (base / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    classes = [
        'apple', 'banana', 'orange', 'strawberry', 'grape', 'watermelon', 'pineapple',
        'mango', 'kiwi', 'peach', 'pear', 'cherry', 'plum', 'lemon', 'lime',
        'blueberry', 'raspberry', 'blackberry', 'papaya', 'coconut', 'avocado',
        'pomegranate', 'cantaloupe', 'tomato', 'cucumber', 'carrot', 'broccoli',
        'cauliflower', 'lettuce', 'cabbage', 'spinach', 'kale', 'celery',
        'bell pepper', 'onion', 'garlic', 'potato', 'sweet potato', 'corn',
        'peas', 'green beans', 'eggplant', 'zucchini', 'pumpkin', 'mushroom',
        'asparagus', 'radish', 'beet', 'turnip', 'ginger'
    ]
    
    data_yaml = {
        'path': str(base.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    with open(base / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"✓ Dataset structure created at {base}")
    print(f"✓ Add images to train/images and labels to train/labels")
    return str(base)

def train_model(data_path, model_size='m', epochs=100):
    """Train YOLO model"""
    print(f"\n🚀 Starting training with YOLOv8{model_size}...")
    
    # Load pre-trained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train
    results = model.train(
        data=f'{data_path}/data.yaml',
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='fruit_vegetable_detector',
        patience=20,
        save=True,
        device='mps',  # M1 Mac
        workers=4,
        project='runs/train',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    print("\n✓ Training completed!")
    return model

def evaluate_model(model):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")
    metrics = model.val()
    
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")

def export_model():
    """Export trained model"""
    print("\n📤 Exporting model...")
    
    best_model = YOLO('runs/train/fruit_vegetable_detector/weights/best.pt')
    
    # Copy to models folder
    os.makedirs('models', exist_ok=True)
    shutil.copy(
        'runs/train/fruit_vegetable_detector/weights/best.pt',
        'models/best.pt'
    )
    
    print("✓ Model exported to models/best.pt")
    print("\nUpdate config.json:")
    print('  "path": "models/best.pt"')

if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8 Fine-tuning for Fruit & Vegetable Detection")
    print("=" * 60)
    
    # Setup dataset
    data_path = setup_dataset()
    
    # Train model
    model = train_model(data_path, model_size='m', epochs=100)
    
    # Evaluate
    evaluate_model(model)
    
    # Export
    export_model()
    
    print("\n" + "=" * 60)
    print("Training complete! Run detect_fusion.py to test.")
    print("=" * 60)
