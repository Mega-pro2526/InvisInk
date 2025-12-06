# InvisInk Setup Instructions

## Python Version Compatibility Issue

**IMPORTANT:** This project requires **Python 3.12** (or 3.9-3.11). The current environment uses Python 3.13, which is not yet supported by MediaPipe 0.10.21.

## Solution Options:

### Option 1: Install Python 3.12 (Recommended)

1. Download Python 3.12 from: https://www.python.org/downloads/release/python-3120/
2. Install Python 3.12 (make sure to check "Add Python to PATH")
3. Create a new virtual environment:
   ```powershell
   cd "C:\invisINk (2)\invisINk"
   py -3.12 -m venv tf_env_py312
   .\tf_env_py312\Scripts\Activate.ps1
   pip install -r requirements.txt
   python invisink_app.py
   ```

### Option 2: Use Existing Wheel File

If you have Python 3.12 available via `py -3.12`:
```powershell
cd "C:\invisINk (2)\invisINk"
py -3.12 -m venv tf_env_new
.\tf_env_new\Scripts\Activate.ps1
pip install mediapipe-0.10.21-cp312-cp312-win_amd64.whl
pip install -r requirements.txt
python invisink_app.py
```

### Option 3: Wait for MediaPipe Python 3.13 Support

MediaPipe will eventually add Python 3.13 support. You can check for updates at:
https://github.com/google/mediapipe/releases

## Current Status

- ✅ OpenCV: Installed
- ✅ TensorFlow: Installed  
- ❌ MediaPipe: Incompatible with Python 3.13
- ❌ Application: Cannot run due to MediaPipe issue



