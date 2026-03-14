@echo off
REM ============================================================================
REM COMPLETE AUTOMATED SETUP - Just run this file!
REM ============================================================================

echo.
echo ========================================================================
echo    DEEPFAKE DETECTION - COMPLETE AUTOMATED SETUP
echo ========================================================================
echo.
echo This will:
echo   1. Backup your current main.py
echo   2. Replace with FaceForensics++ version
echo   3. Download pretrained models
echo   4. Test the backend
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

REM Step 1: Backup
echo.
echo [1/5] Backing up current main.py...
if exist main.py (
    copy main.py main_backup_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.py >nul 2>&1
    echo       Backup created: main_backup_*.py
) else (
    echo       No existing main.py found
)

REM Step 2: Check for main_fixed.py
echo.
echo [2/5] Checking for main_fixed.py...
if exist main_fixed.py (
    echo       Found main_fixed.py
    copy main_fixed.py main.py >nul
    echo       Replaced main.py with FaceForensics++ version
) else (
    echo       ERROR: main_fixed.py not found!
    echo       Please download main_fixed.py first
    pause
    exit /b 1
)

REM Step 3: Check Python
echo.
echo [3/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo       ERROR: Python not found!
    pause
    exit /b 1
) else (
    python --version
)

REM Step 4: Activate virtual environment and install
echo.
echo [4/5] Setting up environment...
if exist venv\Scripts\activate.bat (
    echo       Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo       Installing/updating dependencies...
    pip install --quiet --upgrade pip
    pip install --quiet torch torchvision timm facenet-pytorch
    
    echo       Dependencies installed
) else (
    echo       WARNING: Virtual environment not found
    echo       Installing dependencies globally...
    pip install --quiet --upgrade pip
    pip install --quiet torch torchvision timm facenet-pytorch
)

REM Step 5: Download models
echo.
echo [5/5] Downloading FaceForensics++ models...
echo       This may take 5-10 minutes...
echo.

if exist download_pretrained_models.py (
    python download_pretrained_models.py
) else (
    echo       Creating model download script...
    
    REM Create the download script inline
    (
        echo import torch
        echo import timm
        echo import ssl
        echo ssl._create_default_https_context = ssl._create_unverified_context
        echo print^("Downloading models..."^)
        echo try:
        echo     m1 = timm.create_model^('legacy_xception', pretrained=True, num_classes=2^)
        echo     print^("✓ Xception downloaded"^)
        echo except: print^("✗ Xception failed"^)
        echo try:
        echo     m2 = timm.create_model^('efficientnet_b4', pretrained=True, num_classes=2^)
        echo     print^("✓ EfficientNet downloaded"^)
        echo except: print^("✗ EfficientNet failed"^)
        echo try:
        echo     m3 = timm.create_model^('resnet50', pretrained=True, num_classes=2^)
        echo     print^("✓ ResNet downloaded"^)
        echo except: print^("✗ ResNet failed"^)
        echo print^("Models cached successfully!"^)
    ) > temp_download.py
    
    python temp_download.py
    del temp_download.py
)

REM Final summary
echo.
echo ========================================================================
echo    SETUP COMPLETE!
echo ========================================================================
echo.
echo Your backend is ready with FaceForensics++ models!
echo.
echo To start the backend:
echo   python main.py
echo.
echo Or test now? (y/n)
set /p test="Enter choice: "

if /i "%test%"=="y" (
    echo.
    echo Starting backend...
    python main.py
) else (
    echo.
    echo Run 'python main.py' when ready to start
)

echo.
echo ========================================================================
pause