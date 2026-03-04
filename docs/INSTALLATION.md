# Installation Guide

This guide provides detailed, platform-specific instructions for setting up the Fruit & Vegetable Detection System on macOS, Linux, and Windows.

## Table of Contents

- [System Requirements](#system-requirements)
- [Python Installation](#python-installation)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
- [Project Setup](#project-setup)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Dependency Installation](#dependency-installation)
- [Model Setup](#model-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for optimal performance)
- **Disk Space**: 2GB for models and dependencies
- **Camera**: USB or built-in camera for real-time detection
- **Operating System**: 
  - macOS 10.15 (Catalina) or later
  - Ubuntu 18.04 or later / Debian 10 or later
  - Windows 10 or later

### Recommended Specifications

- **Python**: 3.10 or 3.11
- **RAM**: 16GB
- **Processor**: 
  - macOS: Apple Silicon (M1/M2/M3) for optimal performance
  - Linux/Windows: Multi-core CPU (4+ cores recommended)
- **GPU**: Optional but recommended for faster inference
  - NVIDIA GPU with CUDA support (Linux/Windows)
  - Apple Silicon GPU (macOS)

## Python Installation

### macOS

#### Option 1: Using Homebrew (Recommended)

1. Install Homebrew if not already installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python:
```bash
brew install python@3.11
```

3. Verify installation:
```bash
python3 --version
```

#### Option 2: Using Official Installer

1. Download Python from [python.org](https://www.python.org/downloads/macos/)
2. Run the installer package (.pkg file)
3. Follow the installation wizard
4. Verify installation:
```bash
python3 --version
```

### Linux

#### Ubuntu/Debian

1. Update package list:
```bash
sudo apt update
```

2. Install Python and pip:
```bash
sudo apt install python3.11 python3.11-venv python3-pip
```

3. Verify installation:
```bash
python3 --version
pip3 --version
```

#### Fedora/RHEL/CentOS

1. Install Python:
```bash
sudo dnf install python3.11 python3-pip
```

2. Verify installation:
```bash
python3 --version
pip3 --version
```

#### Arch Linux

1. Install Python:
```bash
sudo pacman -S python python-pip
```

2. Verify installation:
```bash
python --version
pip --version
```

### Windows

#### Option 1: Using Official Installer (Recommended)

1. Download Python from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer (.exe file)
3. **Important**: Check "Add Python to PATH" during installation
4. Select "Install Now" or customize installation location
5. Verify installation by opening Command Prompt:
```cmd
python --version
pip --version
```

#### Option 2: Using Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install
4. Verify installation in Command Prompt:
```cmd
python --version
```

#### Option 3: Using Chocolatey

1. Install Chocolatey if not already installed (run as Administrator):
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

2. Install Python:
```powershell
choco install python311
```

3. Verify installation:
```cmd
python --version
```

## Project Setup

### 1. Clone the Repository

#### Using Git

```bash
git clone https://github.com/darien-o/chefvision.git
cd chefvision
```

#### Using GitHub CLI

```bash
gh repo clone darien-o/chefvision
cd chefvision
```

#### Download ZIP

1. Visit the repository on GitHub
2. Click "Code" → "Download ZIP"
3. Extract the archive
4. Navigate to the extracted directory

### 2. Verify Project Structure

Ensure you have the following key files:

```bash
ls -la  # macOS/Linux
dir     # Windows
```

You should see:
- `config.json`
- `detect_fusion.py`
- `requirements.txt`
- `models/` directory

## Virtual Environment Setup

Using a virtual environment is **strongly recommended** to isolate project dependencies and avoid conflicts with system packages.

### macOS/Linux

#### Create Virtual Environment

```bash
python3 -m venv .venv
```

#### Activate Virtual Environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` prefix in your terminal prompt.

#### Deactivate (when done)

```bash
deactivate
```

### Windows

#### Create Virtual Environment

```cmd
python -m venv .venv
```

#### Activate Virtual Environment

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You should see `(.venv)` prefix in your prompt.

#### Deactivate (when done)

```cmd
deactivate
```

## Dependency Installation

### 1. Upgrade pip (Recommended)

**macOS/Linux:**
```bash
pip install --upgrade pip
```

**Windows:**
```cmd
python -m pip install --upgrade pip
```

### 2. Install Project Dependencies

**All Platforms:**
```bash
pip install -r requirements.txt
```

This will install:
- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Computer vision and camera access
- `numpy` - Numerical operations
- `jupyter` - Interactive notebooks (optional)
- `roboflow` - Dataset management (optional)

### 3. Verify Installation

```bash
pip list
```

Check that the following packages are installed:
- ultralytics
- opencv-python
- numpy

### 4. Install Development Dependencies (Optional)

For contributors and developers:

```bash
pip install -r requirements-dev.txt
```

This includes:
- pytest - Testing framework
- hypothesis - Property-based testing
- black - Code formatting
- flake8 - Linting
- mypy - Type checking

## Model Setup

### Option 1: Automatic Download (Recommended)

The base COCO model (`yolov8m.pt`) will download automatically on first run.

Run the detection script:
```bash
python detect_fusion.py
```

The model will download to the `models/` directory (~50MB).

### Option 2: Manual Download

Use the provided download script to get all models:

```bash
python download_models.py
```

This will download:
- `yolov8m.pt` - Base COCO model (~50MB)
- `yolov8n.pt` - Lightweight nano model (~6MB)
- Additional specialized models (if configured)

### Option 3: Custom Models

If you have custom-trained models:

1. Place model files (`.pt` format) in the `models/` directory
2. Update `config.json` to reference your models:

```json
{
  "models": [
    {
      "name": "my_custom_model",
      "path": "models/my_model.pt",
      "active": true,
      "weight": 1.0,
      "description": "My custom trained model"
    }
  ]
}
```

### Verify Model Files

**macOS/Linux:**
```bash
ls -lh models/
```

**Windows:**
```cmd
dir models\
```

You should see `.pt` files in the models directory.

## Verification

### 1. Test Python Import

```bash
python -c "import cv2, numpy, ultralytics; print('All imports successful')"
```

Expected output: `All imports successful`

### 2. Test Camera Access

**macOS/Linux:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

**Windows:**
```cmd
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

Expected output: `Camera OK`

### 3. Run Detection

```bash
python detect_fusion.py
```

Expected behavior:
- Camera window opens
- Real-time detection starts
- Bounding boxes appear around detected objects
- Press 'q' to quit

### 4. Check Configuration

```bash
python -c "import json; config = json.load(open('config.json')); print(f'Loaded {len(config[\"models\"])} models')"
```

Expected output: `Loaded X models` (where X is the number of models in your config)

## Troubleshooting

### Python Version Issues

**Problem**: `python: command not found` or wrong version

**Solution**:
- macOS/Linux: Use `python3` instead of `python`
- Windows: Ensure Python is added to PATH during installation
- Check version: `python --version` or `python3 --version`
- Reinstall Python with PATH option enabled

### Virtual Environment Issues

**Problem**: Cannot activate virtual environment

**Solution (Windows PowerShell)**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Solution (macOS/Linux)**:
- Ensure you have execute permissions: `chmod +x .venv/bin/activate`
- Use correct activation command: `source .venv/bin/activate`

**Problem**: `venv` module not found

**Solution (Linux)**:
```bash
sudo apt install python3-venv
```

### Dependency Installation Issues

**Problem**: `pip: command not found`

**Solution**:
- macOS/Linux: Use `pip3` instead of `pip`
- Windows: Use `python -m pip` instead of `pip`
- Install pip: `python -m ensurepip --upgrade`

**Problem**: Permission denied during installation

**Solution**:
- **Don't use sudo** - use virtual environment instead
- If you must install globally (not recommended):
  - macOS/Linux: `pip install --user -r requirements.txt`
  - Windows: Run Command Prompt as Administrator

**Problem**: SSL certificate errors

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**Problem**: Slow download speeds

**Solution**:
- Use a mirror: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
- Or upgrade pip: `pip install --upgrade pip`

### OpenCV Issues

**Problem**: `ImportError: libGL.so.1: cannot open shared object file` (Linux)

**Solution**:
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

**Problem**: `ImportError: DLL load failed` (Windows)

**Solution**:
- Install Visual C++ Redistributable: [Download from Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Reinstall opencv-python: `pip uninstall opencv-python && pip install opencv-python`

**Problem**: OpenCV cannot find camera

**Solution**:
- macOS: Grant camera permissions in System Preferences → Security & Privacy → Camera
- Linux: Add user to video group: `sudo usermod -a -G video $USER` (logout/login required)
- Windows: Check camera permissions in Settings → Privacy → Camera

### Camera Access Issues

**Problem**: Camera not detected or permission denied

**Solution (macOS)**:
1. Open System Preferences → Security & Privacy → Camera
2. Enable camera access for Terminal or your IDE
3. Restart the application

**Solution (Linux)**:
```bash
# Check camera device
ls -l /dev/video*

# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

**Solution (Windows)**:
1. Open Settings → Privacy → Camera
2. Enable "Allow apps to access your camera"
3. Enable access for Python or your IDE

**Problem**: Camera already in use

**Solution**:
- Close other applications using the camera (Zoom, Skype, etc.)
- Restart your computer
- Try a different camera index: Edit `detect_fusion.py` and change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Model Download Issues

**Problem**: Model download fails or times out

**Solution**:
- Check internet connection
- Download manually from [Ultralytics](https://github.com/ultralytics/assets/releases)
- Place `.pt` files in `models/` directory
- Verify file integrity (check file size matches expected)

**Problem**: `FileNotFoundError: models/yolov8m.pt`

**Solution**:
```bash
# Create models directory if missing
mkdir -p models  # macOS/Linux
mkdir models     # Windows

# Download base model
python download_models.py
```

### Performance Issues

**Problem**: Low FPS or slow detection

**Solution**:
- Reduce camera resolution in `config.json`:
  ```json
  "camera_width": 640,
  "camera_height": 480
  ```
- Disable some models in `config.json` (set `"active": false`)
- Use lighter model: `yolov8n.pt` instead of `yolov8m.pt`
- Close other resource-intensive applications
- Ensure you're using GPU acceleration (if available)

**Problem**: High memory usage

**Solution**:
- Reduce number of active models
- Use smaller models (yolov8n instead of yolov8m)
- Reduce image size in config
- Close unnecessary applications

### macOS Specific Issues

**Problem**: "Python" cannot be opened because the developer cannot be verified

**Solution**:
1. Open System Preferences → Security & Privacy
2. Click "Open Anyway" for Python
3. Or disable Gatekeeper temporarily: `sudo spctl --master-disable`

**Problem**: xcrun error on macOS

**Solution**:
```bash
xcode-select --install
```

### Windows Specific Issues

**Problem**: Long path errors

**Solution**:
- Enable long paths in Windows:
  1. Open Registry Editor (regedit)
  2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
  3. Set `LongPathsEnabled` to 1
- Or use shorter installation path

**Problem**: Antivirus blocking installation

**Solution**:
- Temporarily disable antivirus during installation
- Add Python and project directory to antivirus exclusions
- Use Windows Defender exclusions: Settings → Update & Security → Windows Security → Virus & threat protection → Exclusions

### Linux Specific Issues

**Problem**: Missing system libraries

**Solution (Ubuntu/Debian)**:
```bash
sudo apt install -y python3-dev build-essential libssl-dev libffi-dev
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

**Solution (Fedora)**:
```bash
sudo dnf install python3-devel gcc gcc-c++ make openssl-devel
sudo dnf install mesa-libGL glib2 libSM libXext libXrender
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check existing issues**: [GitHub Issues](https://github.com/darien-o/chefvision/issues)
2. **Create a new issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - Error message (full traceback)
   - Steps to reproduce
3. **Community support**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for communication channels

## Next Steps

After successful installation:

1. **Read the Usage Guide**: [docs/USAGE.md](USAGE.md)
2. **Explore Examples**: Check the `examples/` directory
3. **Customize Configuration**: Edit `config.json` for your use case
4. **Train Custom Models**: See [TRAINING.md](../TRAINING.md)

---

**Installation complete!** You're ready to start detecting fruits and vegetables. Run `python detect_fusion.py` to begin.
