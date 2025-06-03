@echo off
setlocal enabledelayedexpansion
:: IV Smart Launcher - Dynamically sets up environment paths
:: This finds and adds required DLL paths without copying files

set "APP_DIR=%~dp0"
set "ORIGINAL_PATH=%PATH%"

:: Check if executable exists
if not exist "%APP_DIR%\iv.exe" (
    echo ERROR: iv.exe not found in %APP_DIR%
    pause
    exit /b 1
)

:: Initialize enhanced PATH
set "NEW_PATH=%ORIGINAL_PATH%"

:: Try to find CUDA installation
echo Looking for CUDA installation...
for %%v in (v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0 v11.8 v11.7 v11.6) do (
    set "CUDA_PATH_TEST=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v\bin"
    if exist "!CUDA_PATH_TEST!" (
        echo Found CUDA at !CUDA_PATH_TEST!
        set "NEW_PATH=!CUDA_PATH_TEST!;!NEW_PATH!"
        goto :cuda_found
    )
)
echo WARNING: CUDA not found in standard locations
:cuda_found

:: Try to find MSVC runtime
echo Looking for MSVC runtime...
set "MSVC_BASE=C:\Program Files\Microsoft Visual Studio"
for %%y in (2022 2019 2017) do (
    for %%e in (BuildTools Community Professional Enterprise) do (
        set "MSVC_TEST=!MSVC_BASE!\%%y\%%e\VC\Redist\MSVC"
        if exist "!MSVC_TEST!" (
            for /f "delims=" %%i in ('dir "!MSVC_TEST!" /b /ad /o-n') do (
                set "MSVC_DLL_PATH=!MSVC_TEST!\%%i\x64\Microsoft.VC143.CRT"
                if exist "!MSVC_DLL_PATH!" (
                    echo Found MSVC runtime at !MSVC_DLL_PATH!
                    set "NEW_PATH=!MSVC_DLL_PATH!;!NEW_PATH!"
                    goto :msvc_found
                )
            )
        )
    )
)
echo WARNING: MSVC runtime not found in standard locations
:msvc_found

:: Set the enhanced PATH
set "PATH=!NEW_PATH!"

:: Show what we're using
echo Environment setup complete:
echo - App directory: %APP_DIR%
echo - Enhanced PATH with detected dependencies
echo.

:: Run the application with all arguments
echo Starting IV Image Viewer...
"%APP_DIR%\iv.exe" %*

:: Check exit code
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo This might be due to missing dependencies.
    echo.
    echo Troubleshooting:
    echo 1. Make sure you have NVIDIA drivers installed
    echo 2. Install CUDA Toolkit if using GPU features
    echo 3. Install Visual Studio C++ Redistributables
    echo.
    pause
)