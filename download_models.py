#!/usr/bin/env python3
"""
Download specialized fruit/vegetable detection models
"""
import os
import urllib.request

MODELS = {
    'fruit_360': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'file': 'fruit_360.pt',
        'classes': 131  # 131 fruit types
    },
    'grocery': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'file': 'grocery.pt',
        'classes': 80
    }
}

def download_model(name, info):
    """Download model if not exists"""
    if os.path.exists(info['file']):
        print(f"✓ {name} model already exists")
        return
    
    print(f"Downloading {name} model...")
    try:
        urllib.request.urlretrieve(info['url'], info['file'])
        print(f"✓ Downloaded {name} ({info['classes']} classes)")
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")

if __name__ == '__main__':
    print("Downloading specialized fruit/vegetable models...\n")
    
    for name, info in MODELS.items():
        download_model(name, info)
    
    print("\n" + "="*50)
    print("IMPORTANT: These are placeholder URLs")
    print("For real specialized models, use:")
    print("1. Roboflow Universe: https://universe.roboflow.com")
    print("2. Hugging Face: https://huggingface.co/models?other=yolo")
    print("3. Train your own with custom dataset")
    print("="*50)
