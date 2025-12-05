# InvisInk Python 3.12 Setup Script
# This script helps set up Python 3.12 environment for InvisInk

Write-Host "=== InvisInk Python 3.12 Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if Python 3.12 is available
Write-Host "Checking for Python 3.12..." -ForegroundColor Yellow
$py312 = Get-Command python3.12 -ErrorAction SilentlyContinue
if (-not $py312) {
    $py312 = Get-Command py -ErrorAction SilentlyContinue
    if ($py312) {
        Write-Host "Trying 'py -3.12'..." -ForegroundColor Yellow
        $version = & py -3.12 --version 2>&1
        if ($version -match "3\.12") {
            $PYTHON_CMD = "py -3.12"
        }
    }
} else {
    $PYTHON_CMD = "python3.12"
}

if (-not $PYTHON_CMD) {
    Write-Host "ERROR: Python 3.12 not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.12 from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3120/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "After installation, run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found Python 3.12: $PYTHON_CMD" -ForegroundColor Green
Write-Host ""

# Create virtual environment
$VENV_NAME = "tf_env_py312"
Write-Host "Creating virtual environment: $VENV_NAME..." -ForegroundColor Yellow

if ($PYTHON_CMD -match "^py ") {
    & py -3.12 -m venv $VENV_NAME
} else {
    & python3.12 -m venv $VENV_NAME
}

if (-not (Test-Path "$VENV_NAME\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& "$VENV_NAME\Scripts\Activate.ps1"

# Upgrade pip first
python -m pip install --upgrade pip

# Install MediaPipe from wheel if available
if (Test-Path "mediapipe-0.10.21-cp312-cp312-win_amd64.whl") {
    Write-Host "Installing MediaPipe from wheel file..." -ForegroundColor Yellow
    python -m pip install mediapipe-0.10.21-cp312-cp312-win_amd64.whl
} else {
    Write-Host "Installing MediaPipe from PyPI..." -ForegroundColor Yellow
    python -m pip install mediapipe==0.10.21
}

# Install other requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installing other dependencies from requirements.txt..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
} else {
    Write-Host "Installing core dependencies..." -ForegroundColor Yellow
    python -m pip install opencv-python opencv-contrib-python "numpy<2.0" tensorflow
}

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "  .\$VENV_NAME\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python invisink_app.py" -ForegroundColor White
Write-Host ""



