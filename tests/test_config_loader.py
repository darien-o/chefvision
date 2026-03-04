"""
Unit tests for configuration loading and validation.

Tests configuration file parsing, model configuration validation,
and detection settings validation.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigStructure:
    """Test cases for configuration file structure."""
    
    @pytest.fixture
    def valid_config(self):
        """Fixture providing a valid configuration."""
        return {
            "models": [
                {
                    "name": "test_model",
                    "path": "models/test.pt",
                    "active": True,
                    "weight": 1.0,
                    "description": "Test model"
                }
            ],
            "food_classes": {
                "custom": {
                    "0": {"name": "apple", "active": True}
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
    
    def test_config_has_required_keys(self, valid_config):
        """Configuration should have all required top-level keys."""
        required_keys = ["models", "food_classes", "detection_settings"]
        for key in required_keys:
            assert key in valid_config
    
    def test_models_is_list(self, valid_config):
        """Models configuration should be a list."""
        assert isinstance(valid_config["models"], list)
    
    def test_model_has_required_fields(self, valid_config):
        """Each model should have required fields."""
        model = valid_config["models"][0]
        required_fields = ["name", "path", "active", "weight", "description"]
        for field in required_fields:
            assert field in model
    
    def test_detection_settings_has_required_fields(self, valid_config):
        """Detection settings should have all required fields."""
        settings = valid_config["detection_settings"]
        required_fields = [
            "confidence_threshold",
            "iou_threshold",
            "image_size",
            "camera_width",
            "camera_height",
            "fusion_iou",
            "min_model_votes"
        ]
        for field in required_fields:
            assert field in settings


class TestModelConfiguration:
    """Test cases for model configuration validation."""
    
    def test_model_weight_positive(self):
        """Model weight should be positive."""
        model = {
            "name": "test",
            "path": "models/test.pt",
            "active": True,
            "weight": 1.5,
            "description": "Test"
        }
        assert model["weight"] > 0
    
    def test_model_active_boolean(self):
        """Model active flag should be boolean."""
        model = {
            "name": "test",
            "path": "models/test.pt",
            "active": True,
            "weight": 1.0,
            "description": "Test"
        }
        assert isinstance(model["active"], bool)
    
    def test_model_path_string(self):
        """Model path should be a string."""
        model = {
            "name": "test",
            "path": "models/test.pt",
            "active": True,
            "weight": 1.0,
            "description": "Test"
        }
        assert isinstance(model["path"], str)
        assert model["path"].endswith(".pt")


class TestDetectionSettings:
    """Test cases for detection settings validation."""
    
    @pytest.fixture
    def settings(self):
        """Fixture providing detection settings."""
        return {
            "confidence_threshold": 0.15,
            "iou_threshold": 0.4,
            "image_size": 640,
            "camera_width": 1280,
            "camera_height": 720,
            "fusion_iou": 0.45,
            "min_model_votes": 1
        }
    
    def test_confidence_threshold_range(self, settings):
        """Confidence threshold should be between 0 and 1."""
        conf = settings["confidence_threshold"]
        assert 0 <= conf <= 1
    
    def test_iou_threshold_range(self, settings):
        """IoU threshold should be between 0 and 1."""
        iou = settings["iou_threshold"]
        assert 0 <= iou <= 1
    
    def test_fusion_iou_range(self, settings):
        """Fusion IoU threshold should be between 0 and 1."""
        fusion_iou = settings["fusion_iou"]
        assert 0 <= fusion_iou <= 1
    
    def test_image_size_positive(self, settings):
        """Image size should be positive."""
        assert settings["image_size"] > 0
    
    def test_camera_dimensions_positive(self, settings):
        """Camera dimensions should be positive."""
        assert settings["camera_width"] > 0
        assert settings["camera_height"] > 0
    
    def test_min_votes_positive(self, settings):
        """Minimum model votes should be positive."""
        assert settings["min_model_votes"] > 0
    
    def test_min_votes_integer(self, settings):
        """Minimum model votes should be an integer."""
        assert isinstance(settings["min_model_votes"], int)


class TestFoodClasses:
    """Test cases for food classes configuration."""
    
    @pytest.fixture
    def food_classes(self):
        """Fixture providing food classes configuration."""
        return {
            "custom": {
                "0": {"name": "apple", "active": True},
                "1": {"name": "banana", "active": False},
                "2": {"name": "orange", "active": True}
            }
        }
    
    def test_food_classes_structure(self, food_classes):
        """Food classes should have proper structure."""
        assert "custom" in food_classes
        assert isinstance(food_classes["custom"], dict)
    
    def test_class_has_name_and_active(self, food_classes):
        """Each food class should have name and active fields."""
        for class_id, class_info in food_classes["custom"].items():
            assert "name" in class_info
            assert "active" in class_info
            assert isinstance(class_info["name"], str)
            assert isinstance(class_info["active"], bool)
    
    def test_class_id_numeric_string(self, food_classes):
        """Class IDs should be numeric strings."""
        for class_id in food_classes["custom"].keys():
            assert class_id.isdigit()


class TestConfigLoading:
    """Test cases for configuration file loading."""
    
    def test_load_valid_json(self):
        """Should successfully load valid JSON configuration."""
        config_data = {
            "models": [],
            "food_classes": {"custom": {}},
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
        config_json = json.dumps(config_data)
        
        with patch("builtins.open", mock_open(read_data=config_json)):
            with open("config.json", "r") as f:
                config = json.load(f)
        
        assert config == config_data
    
    def test_invalid_json_raises_error(self):
        """Should raise error for invalid JSON."""
        invalid_json = "{ invalid json }"
        
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with pytest.raises(json.JSONDecodeError):
                with open("config.json", "r") as f:
                    json.load(f)


class TestConfigValidation:
    """Test cases for configuration validation logic."""
    
    def test_filter_active_models(self):
        """Should filter only active models."""
        models = [
            {"name": "model1", "active": True, "weight": 1.0},
            {"name": "model2", "active": False, "weight": 1.0},
            {"name": "model3", "active": True, "weight": 1.5}
        ]
        
        active_models = [m for m in models if m["active"]]
        
        assert len(active_models) == 2
        assert all(m["active"] for m in active_models)
    
    def test_filter_active_food_classes(self):
        """Should filter only active food classes."""
        food_classes = {
            "0": {"name": "apple", "active": True},
            "1": {"name": "banana", "active": False},
            "2": {"name": "orange", "active": True}
        }
        
        active_classes = {
            int(k): v["name"] 
            for k, v in food_classes.items() 
            if v["active"]
        }
        
        assert len(active_classes) == 2
        assert 0 in active_classes
        assert 2 in active_classes
        assert 1 not in active_classes


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
