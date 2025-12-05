# InvisInk - Air Drawing Calculator 🖐️✏️

A real-time computer vision application that recognizes hand-drawn mathematical equations in the air using fingertip tracking and solves them instantly. Built with MediaPipe, TensorFlow, and OpenCV.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [File Descriptions](#file-descriptions)
- [Folder Structure & Data](#folder-structure--data)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

InvisInk is an innovative air-drawing calculator that combines hand gesture recognition with machine learning to solve mathematical expressions in real-time. Users draw digits and operators in the air using their index finger, and the system recognizes and evaluates the expression.

### How It Works

1. **Hand Tracking**: Uses MediaPipe to track hand landmarks in real-time
2. **Gesture Recognition**: Detects specific hand gestures for control
3. **Symbol Recognition**: CNN model classifies drawn symbols (0-9, +, -, *, /, (, ))
4. **Expression Solving**: Evaluates and displays the mathematical result

## ✨ Features

- 🎮 **Gesture-Based Control**: No keyboard or mouse needed
- 🖐️ **Real-Time Hand Tracking**: Uses MediaPipe for accurate fingertip detection
- 🧠 **Custom CNN Model**: Trained on your own handwritten dataset
- 🔢 **Mathematical Expression Recognition**: Supports digits and basic operators
- ⚡ **Real-Time Processing**: 30+ FPS performance
- 📊 **Visual Feedback**: Live drawing canvas and gesture display
- 🌐 **Web Interface**: Optional Streamlit web application

## 📦 Requirements

### System Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12 (⚠️ **NOT Python 3.13** - MediaPipe compatibility issue)
- **Operating System**: Windows, Linux, or macOS
- **Camera**: Webcam or built-in camera
- **RAM**: Minimum 4GB (8GB recommended)

### Library Versions

All required libraries are specified in `requirements.txt`:

```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy<2.0
tensorflow>=2.15.0
mediapipe==0.10.21
matplotlib
protobuf<5.0,>=4.25.3
absl-py
attrs>=19.1.0
flatbuffers>=2.0
jax
jaxlib
sounddevice>=0.4.4
sentencepiece
```

**Key Dependencies:**
- **OpenCV** (4.8.0+): Computer vision and image processing
- **MediaPipe** (0.10.21): Hand tracking and gesture recognition
- **TensorFlow** (2.15.0+): Deep learning framework for CNN model
- **NumPy** (<2.0): Numerical computations

> **Note**: For Streamlit web interface, you'll also need:
> - `streamlit`
> - `streamlit-webrtc`
> - `av` (PyAV)

## 🚀 Installation

### Method 1: Automated Setup (Windows - PowerShell)

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/Mega-pro2526/InvisInk.git
   cd InvisInk
   ```

2. **Run the setup script**:
   ```powershell
   .\setup_py312.ps1
   ```
   This script will:
   - Check for Python 3.12
   - Create a virtual environment
   - Install all dependencies
   - Set up MediaPipe from wheel file if available

3. **Activate the virtual environment**:
   ```powershell
   .\tf_env_py312\Scripts\Activate.ps1
   ```

### Method 2: Manual Setup

1. **Install Python 3.12** (if not already installed):
   - Download from: https://www.python.org/downloads/release/python-3120/
   - Make sure to check "Add Python to PATH" during installation

2. **Create a virtual environment**:
   ```powershell
   # Windows
   py -3.12 -m venv tf_env_py312
   .\tf_env_py312\Scripts\Activate.ps1
   
   # Linux/macOS
   python3.12 -m venv tf_env_py312
   source tf_env_py312/bin/activate
   ```

3. **Install MediaPipe** (using wheel file if available):
   ```bash
   pip install mediapipe-0.10.21-cp312-cp312-win_amd64.whl
   ```
   Or from PyPI:
   ```bash
   pip install mediapipe==0.10.21
   ```

4. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, tensorflow; print('All libraries installed successfully!')"
   ```

## 📁 Project Structure

```
InvisInk/
│
├── invisink_app.py              # Main desktop application
├── streamlit_app.py             # Web application (Streamlit)
├── data_collector.py            # Data collection tool
├── model_trainer.py             # CNN model training script
├── check_images.py              # Utility to verify dataset
│
├── requirements.txt             # Python dependencies
├── setup_py312.ps1              # Automated setup script (Windows)
├── README.md                    # This file
├── README_SETUP.md              # Additional setup instructions
│
├── data1/                       # Training dataset (excluded from git)
│   ├── 0/                       # Digit 0 samples
│   ├── 1/                       # Digit 1 samples
│   ├── ...                      # Digits 2-9
│   ├── +/                       # Addition operator
│   ├── -/                       # Subtraction operator
│   ├── x/                       # Multiplication operator
│   ├── slash/                   # Division operator
│   ├── (/                       # Left parenthesis
│   └── )/                       # Right parenthesis
│
├── invisink_model.h5            # Trained CNN model (excluded from git)
├── .gitignore                   # Git ignore rules
│
└── venv_py311/                  # Virtual environment (excluded from git)
```

## 🎮 Usage Guide

### Running the Main Application

1. **Ensure the model exists**:
   - If you don't have `invisink_model.h5`, you need to train the model first (see [Model Training](#model-training))

2. **Activate virtual environment**:
   ```powershell
   .\tf_env_py312\Scripts\Activate.ps1
   ```

3. **Run the application**:
   ```bash
   python invisink_app.py
   ```

4. **Use the gestures**:
   - **👆 Index Finger Extended**: Draw symbols in the air
   - **✊ Closed Fist**: Recognize the drawn symbol (wait 1.5 seconds between recognitions)
   - **👍 Thumbs Up**: Solve the equation and display result
   - **✋ Open Hand**: Clear the canvas and reset
   - **Q Key**: Quit the application

### Running the Web Application (Streamlit)

1. **Install Streamlit dependencies** (if not already installed):
   ```bash
   pip install streamlit streamlit-webrtc av
   ```

2. **Run Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access in browser**: The app will open automatically at `http://localhost:8501`

### Data Collection

To collect your own training data:

1. **Run the data collector**:
   ```bash
   python data_collector.py
   ```

2. **Follow the prompts**:
   - The script will cycle through all 16 classes (0-9, +, -, *, /, (, ))
   - Press **'d'** to start/stop drawing
   - Press **'s'** to save the current drawing
   - Press **'q'** to quit

3. **Data will be saved to** `./data/` directory (you may need to rename to `data1/` for training)

> **Note**: Collect at least 50 samples per class for good model performance.

### Model Training

1. **Prepare your dataset**:
   - Ensure `data1/` folder exists with class subdirectories
   - Each class folder should contain PNG images (30x30 or 28x28 grayscale)

2. **Verify your dataset**:
   ```bash
   python check_images.py
   ```
   This will display information about images in your dataset.

3. **Train the model**:
   ```bash
   python model_trainer.py
   ```

4. **Training parameters** (can be modified in `model_trainer.py`):
   - **EPOCHS**: 20 (default)
   - **BATCH_SIZE**: 32 (default)
   - **IMAGE_SIZE**: (30, 30)
   - **TEST_SPLIT**: 0.2 (20% for validation)

5. **Model will be saved as** `invisink_model.h5`

## 📄 File Descriptions

### Core Application Files

#### `invisink_app.py`
**Main desktop application** - The primary interface for using InvisInk.

- **Purpose**: Real-time air-drawing calculator using webcam
- **Features**:
  - Hand tracking with MediaPipe
  - Gesture recognition (fingertip, fist, thumbs up, open hand)
  - Symbol recognition using trained CNN
  - Mathematical expression evaluation
  - Visual feedback and UI overlay
- **Dependencies**: OpenCV, MediaPipe, TensorFlow, NumPy
- **Output**: Live camera feed with drawing overlay and results

#### `streamlit_app.py`
**Web-based application** - Streamlit interface for browser-based usage.

- **Purpose**: Web version of InvisInk using Streamlit and WebRTC
- **Features**: Same functionality as desktop app, but accessible via web browser
- **Dependencies**: All core dependencies + Streamlit, streamlit-webrtc, av
- **Usage**: `streamlit run streamlit_app.py`

### Data & Training Files

#### `data_collector.py`
**Dataset collection tool** - Interactive tool for gathering training data.

- **Purpose**: Collect handwritten symbol samples using fingertip drawing
- **Features**:
  - Guided collection process for 16 classes
  - Real-time fingertip tracking
  - Automatic image preprocessing (28x28 grayscale)
  - Saves images to class-specific folders
- **Output**: PNG images in `./data/` directory (organized by class)
- **Note**: Windows folder naming - `*` becomes `STAR`, `/` becomes `SLASH`

#### `model_trainer.py`
**CNN model training script** - Trains the classification model.

- **Purpose**: Train a Convolutional Neural Network for symbol recognition
- **Model Architecture**:
  - 2 Convolutional layers (32 and 64 filters)
  - MaxPooling layers
  - Dense layer (128 neurons) with Dropout (0.5)
  - Output layer (16 classes with softmax)
- **Features**:
  - Automatic data loading from `data1/` folder
  - Train/test split (80/20)
  - Training visualization (accuracy and loss plots)
  - Model evaluation and saving
- **Output**: `invisink_model.h5` (saved model file)

#### `check_images.py`
**Dataset verification utility** - Quick tool to inspect your dataset.

- **Purpose**: Check and verify images in the dataset
- **Features**: Displays image counts, shapes, and sample information
- **Usage**: Run before training to ensure data is properly formatted

### Configuration & Setup Files

#### `requirements.txt`
**Python dependencies** - List of all required packages with versions.

- Contains all library dependencies needed for the project
- Use with: `pip install -r requirements.txt`

#### `setup_py312.ps1`
**Automated setup script** - Windows PowerShell script for easy installation.

- **Purpose**: Automates Python 3.12 environment setup
- **Features**:
  - Checks for Python 3.12
  - Creates virtual environment
  - Installs all dependencies
  - Handles MediaPipe wheel file installation
- **Usage**: `.\setup_py312.ps1`

#### `.gitignore`
**Git ignore rules** - Specifies files/folders to exclude from version control.

- Excludes: virtual environments, model files, datasets, wheel files, etc.

## 📂 Folder Structure & Data

### Training Dataset: `data1/`

The `data1/` folder contains the training dataset organized by class. Each class has its own subdirectory with sample images.

**Folder Structure:**
```
data1/
├── 0/              # Digit 0 - Contains PNG images (28x28 or 30x30 grayscale)
├── 1/              # Digit 1
├── 2/              # Digit 2
├── 3/              # Digit 3
├── 4/              # Digit 4
├── 5/              # Digit 5
├── 6/              # Digit 6
├── 7/              # Digit 7
├── 8/              # Digit 8
├── 9/              # Digit 9
├── +/              # Addition operator
├── -/              # Subtraction operator
├── x/              # Multiplication operator (folder name "x" for Windows compatibility)
├── slash/          # Division operator (folder name "slash" for Windows compatibility)
├── (/              # Left parenthesis
└── )/              # Right parenthesis
```

**Data Format:**
- **Image Type**: PNG files
- **Dimensions**: 28x28 or 30x30 pixels
- **Color**: Grayscale (single channel)
- **Recommendation**: Minimum 50 samples per class for good accuracy

**Note**: The `data1/` folder is excluded from Git (see `.gitignore`) due to its large size. You need to collect your own data or obtain a dataset separately.

### Data Collection Folder: `data/`

The `data_collector.py` script saves collected samples to `./data/` directory. You may need to rename or copy this to `data1/` for training.

**Data Collection Process:**
1. Run `data_collector.py`
2. Draw symbols using your index fingertip
3. Images are automatically:
   - Extracted from drawing canvas
   - Resized to 28x28 pixels
   - Converted to grayscale
   - Saved to appropriate class folder

### Virtual Environment Folders

- `venv_py311/` - Python 3.11 virtual environment (if created)
- `tf_env_py312/` - Python 3.12 virtual environment (created by setup script)
- `tf_env_new/` - Alternative environment folder

All virtual environment folders are excluded from Git.

## 🔄 Workflow

### Complete Project Workflow

```
1. Setup Environment
   └─> Install Python 3.12
   └─> Create virtual environment
   └─> Install dependencies

2. Collect Training Data
   └─> Run data_collector.py
   └─> Draw 50+ samples for each class (0-9, +, -, *, /, (, ))
   └─> Verify data with check_images.py

3. Train Model
   └─> Organize data in data1/ folder
   └─> Run model_trainer.py
   └─> Wait for training to complete (20 epochs)
   └─> Model saved as invisink_model.h5

4. Run Application
   └─> Ensure invisink_model.h5 exists
   └─> Run invisink_app.py (desktop) or streamlit_app.py (web)
   └─> Draw mathematical expressions in the air!
```

### Example Usage Workflow

1. **Start the application**: `python invisink_app.py`
2. **Position yourself**: Sit 2-3 feet from the camera with good lighting
3. **Draw a number**: Extend your index finger and draw "2" in the air
4. **Recognize**: Make a fist to recognize the symbol
5. **Draw operator**: Draw "+" with your finger
6. **Recognize again**: Make a fist
7. **Draw another number**: Draw "3"
8. **Recognize**: Make a fist
9. **Solve**: Show thumbs up to calculate and display "2 + 3 = 5"
10. **Clear**: Show open hand to reset

## 🐛 Troubleshooting

### Common Issues

#### 1. **MediaPipe Installation Error**
```
Error: Could not install mediapipe
```
**Solution**:
- Ensure you're using Python 3.9-3.12 (NOT 3.13)
- Try installing from wheel file: `pip install mediapipe-0.10.21-cp312-cp312-win_amd64.whl`
- Or install specific version: `pip install mediapipe==0.10.21`

#### 2. **Camera Not Detected**
```
Error: Could not open any camera
```
**Solution**:
- Check if camera is connected and working
- Ensure no other application is using the camera
- Try different camera indices (0, 1, 2, etc.)
- On Windows: Check camera permissions in Settings

#### 3. **Model Not Found**
```
Error loading model: invisink_model.h5
```
**Solution**:
- Ensure `invisink_model.h5` exists in the project directory
- Train the model first: `python model_trainer.py`
- Check if the file path is correct

#### 4. **Hand Not Detected**
```
No hand detected
```
**Solution**:
- Ensure good lighting conditions
- Position yourself 2-3 feet from the camera
- Show your palm clearly to the camera
- Try different angles

#### 5. **Poor Recognition Accuracy**
```
Symbols not recognized correctly
```
**Solution**:
- Collect more training data (100+ samples per class)
- Ensure training data quality (clear, centered symbols)
- Retrain the model with more epochs
- Check if drawing style matches training data

#### 6. **Import Errors**
```
ModuleNotFoundError: No module named 'X'
```
**Solution**:
- Ensure virtual environment is activated
- Install missing package: `pip install X`
- Reinstall all dependencies: `pip install -r requirements.txt`

#### 7. **TensorFlow/GPU Issues**
```
TensorFlow not using GPU or errors
```
**Solution**:
- CPU-only TensorFlow should work fine
- For GPU support, install CUDA and cuDNN
- Check TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`

### Performance Tips

- **FPS Optimization**: Lower camera resolution if experiencing lag
- **Recognition Accuracy**: 
  - Draw slowly and clearly
  - Wait 1.5 seconds between recognitions (debounce time)
  - Ensure consistent drawing style
- **Lighting**: Use even, front-facing lighting for best results
- **Distance**: Maintain 2-3 feet distance from camera

## 📝 Notes

- **Model File**: The trained model (`invisink_model.h5`) is excluded from Git due to file size. You'll need to train your own model.
- **Dataset**: The training dataset (`data1/`) is also excluded from Git. Collect your own data using `data_collector.py`.
- **Python Version**: This project is tested with Python 3.12. Python 3.9-3.11 should also work. Python 3.13 is NOT supported due to MediaPipe compatibility.
- **Security Note**: The application uses `eval()` for mathematical evaluation. For production use, consider implementing a safer math parser.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 👤 Author

**Mega-pro2526**
- GitHub: [@Mega-pro2526](https://github.com/Mega-pro2526)
- Email: megapro202526@gmail.com

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools

---

**Enjoy drawing in the air! 🎨✨**

