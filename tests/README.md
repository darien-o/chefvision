# Test Suite

This directory contains the test suite for the chefvision system.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures and configuration
├── test_fusion_algorithm.py # Tests for fusion detection algorithm
├── test_config_loader.py    # Tests for configuration loading
├── fixtures/                # Test data and fixtures
└── README.md               # This file
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_fusion_algorithm.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests
- `test_fusion_algorithm.py`: Tests for IoU calculation, detection clustering, weighted voting
- `test_config_loader.py`: Tests for configuration validation and loading

### Test Fixtures
Shared fixtures are defined in `conftest.py`:
- `sample_config`: Sample configuration for testing
- `sample_detections`: Sample detection data
- `fixtures_dir`: Path to fixtures directory

## Writing New Tests

When adding new tests:

1. Create test files with `test_` prefix
2. Use descriptive test class and function names
3. Add docstrings explaining what is being tested
4. Use fixtures from `conftest.py` when possible
5. Follow the existing test structure

Example:
```python
class TestNewFeature:
    """Test cases for new feature."""
    
    def test_basic_functionality(self):
        """Feature should work with basic input."""
        # Test implementation
        assert True
```

## Configuration

Test configuration is defined in `pytest.ini` at the project root.

Key settings:
- Test discovery patterns
- Output verbosity
- Test markers for categorization
