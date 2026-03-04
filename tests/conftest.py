"""
Pytest configuration and shared fixtures.

This file contains pytest configuration and fixtures that are shared
across all test modules.
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for testing."""
    return {
        "models": [
            {
                "name": "test_model_1",
                "path": "models/test1.pt",
                "active": True,
                "weight": 1.0,
                "description": "Test model 1"
            },
            {
                "name": "test_model_2",
                "path": "models/test2.pt",
                "active": True,
                "weight": 1.5,
                "description": "Test model 2"
            }
        ],
        "food_classes": {
            "custom": {
                "0": {"name": "apple", "active": True},
                "1": {"name": "banana", "active": True},
                "2": {"name": "orange", "active": False}
            }
        },
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


@pytest.fixture
def sample_detections():
    """Fixture providing sample detection data for testing."""
    return [
        (10, 10, 50, 50, 0.9, "apple", "model1", 1.0),
        (15, 15, 55, 55, 0.85, "apple", "model2", 1.0),
        (100, 100, 150, 150, 0.8, "banana", "model1", 1.0)
    ]


@pytest.fixture
def fixtures_dir():
    """Fixture providing path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
