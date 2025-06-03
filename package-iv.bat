@echo off
setlocal enabledelayedexpansion

echo Creating deployment package for IV...

:: Configuration
set DEPLOY_DIR=iv-deploy
set BUILD_DIR=target\release
set EXE_NAME=iv.exe
set MODELS_DIR=models

:: Clean up any existing deploy directory
if exist "%DEPLOY_DIR%" (
    echo Cleaning up existing deployment directory...
    rmdir /s /q "%DEPLOY_DIR%"
)

:: Create deployment directory
echo Creating deployment directory...
mkdir "%DEPLOY_DIR%"

:: Copy the main executable
if exist "%BUILD_DIR%\%EXE_NAME%" (
    echo Copying %EXE_NAME%...
    copy "%BUILD_DIR%\%EXE_NAME%" "%DEPLOY_DIR%\"
) else (
    echo ERROR: %EXE_NAME% not found in %BUILD_DIR%
    echo Please run "cargo build --release --features cuda" first
    pause
    exit /b 1
)

:: Copy model files
if exist "%MODELS_DIR%" (
    echo Copying model files...
    xcopy "%MODELS_DIR%" "%DEPLOY_DIR%\%MODELS_DIR%\" /e /i /q
) else (
    echo WARNING: Models directory not found - app may not work without ONNX model
)

:: Create the smart launcher batch file that sets up environment paths
echo Creating smart launcher script...
if exist "smart-launcher-template.bat" (
    copy "smart-launcher-template.bat" "%DEPLOY_DIR%\run-iv.bat" > nul
    echo Smart launcher copied from template.
) else (
    echo WARNING: Template file not found. Creating basic launcher...
    echo @echo off > "%DEPLOY_DIR%\run-iv.bat"
    echo set "APP_DIR=%%~dp0" >> "%DEPLOY_DIR%\run-iv.bat"
    echo "%%APP_DIR%%\iv.exe" %%* >> "%DEPLOY_DIR%\run-iv.bat"
)

:: Create a simple direct launcher for advanced users
echo Creating direct launcher...
(
echo @echo off
echo :: Direct IV Launcher - No environment setup
echo :: Use this if you have all dependencies properly installed
echo.
echo set "APP_DIR=%%~dp0"
echo.
echo if not exist "%%APP_DIR%%\iv.exe" ^(
echo     echo ERROR: iv.exe not found
echo     pause
echo     exit /b 1
echo ^)
echo.
echo "%%APP_DIR%%\iv.exe" %%*
) > "%DEPLOY_DIR%\iv-direct.bat"

:: Create a README file
echo Creating README...
(
echo IV - Image Viewer with AI Rating
echo ================================
echo.
echo This package contains everything needed to run IV on Windows.
echo.
echo Files included:
echo - iv.exe          : Main application
echo - run-iv.bat      : Smart launcher ^(RECOMMENDED^) - auto-detects dependencies
echo - iv-direct.bat   : Direct launcher ^(advanced users^) - no environment setup
echo - models/         : AI model files
echo.
echo Usage:
echo 1. Extract this zip file to any folder
echo 2. Run: run-iv.bat ^<path-to-your-images^>
echo.
echo Example:
echo   run-iv.bat "C:\Users\YourName\Pictures"
echo.
echo Launcher Options:
echo - run-iv.bat      : Automatically finds and sets up CUDA/MSVC paths
echo - iv-direct.bat   : Uses system PATH only ^(faster startup^)
echo.
echo Requirements:
echo - Windows 10/11 x64
echo - For GPU acceleration: NVIDIA GPU with CUDA support and drivers
echo - Visual Studio C++ Redistributables ^(usually pre-installed^)
echo.
echo Dependencies ^(auto-detected by smart launcher^):
echo - CUDA Toolkit ^(for GPU acceleration^)
echo - MSVC Runtime Libraries
echo - Windows SDK ^(optional^)
echo.
echo The smart launcher will automatically find these if installed and add
echo them to the PATH for the duration of the application run.
echo.
echo Controls:
echo - Arrow keys or mouse: Navigate images
echo - Number keys 1-5: Rate images
echo - Plus key: Add to favorites
echo - Delete key: Move to deleted folder
echo - Escape: Exit
echo.
echo Note: The AI rating system becomes active after rating 500 images.
echo.
echo Troubleshooting:
echo - If you get DLL errors, install/reinstall CUDA Toolkit
echo - If colors look wrong, update your NVIDIA drivers
echo - Use iv-direct.bat if run-iv.bat is too slow
) > "%DEPLOY_DIR%\README.txt"

:: Create the zip file using PowerShell (available on Windows 10+)
echo Creating zip file...
powershell -command "Compress-Archive -Path '%DEPLOY_DIR%\*' -DestinationPath '%DEPLOY_DIR%.zip' -Force"

if exist "%DEPLOY_DIR%.zip" (
    echo.
    echo ======================================
    echo SUCCESS: Deployment package created!
    echo ======================================
    echo.
    echo Package location: %DEPLOY_DIR%.zip
    echo Package contents: %DEPLOY_DIR%\
    echo.
    echo The zip file contains everything needed to run IV on another Windows machine.
    echo Just extract and run "run-iv.bat" with a folder path as argument.
    echo.
) else (
    echo ERROR: Failed to create zip file
    pause
    exit /b 1
)

:: Ask if user wants to clean up the directory
set /p CLEANUP="Clean up deployment directory? (y/n): "
if /i "%CLEANUP%"=="y" (
    rmdir /s /q "%DEPLOY_DIR%"
    echo Deployment directory cleaned up.
)

echo.
echo Packaging complete!
pause
