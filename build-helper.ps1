# Image Viewer Build Helper Script (PowerShell)
# Automatically detects the best build configuration for your platform

Write-Host "Detecting platform and build configuration..." -ForegroundColor Cyan
Write-Host

# Detect Rust toolchain
$rustTarget = (rustc -vV | Select-String "host:").ToString().Split(' ')[1]

Write-Host "Platform: Windows"
Write-Host "Rust target: $rustTarget"
Write-Host

# Provide recommendations based on toolchain
switch -Wildcard ($rustTarget) {
    "*windows-gnu*" {
        Write-Host "Windows GNU toolchain detected" -ForegroundColor Yellow
        Write-Host "CUDA is not supported due to linking issues with GNU toolchain."
        Write-Host
        Write-Host "Recommended options:"
        Write-Host "1. Switch to MSVC for CUDA support:" -ForegroundColor Green
        Write-Host "   rustup default stable-x86_64-pc-windows-msvc"
        Write-Host "   cargo build --release --features cuda"
        Write-Host
        Write-Host "2. Use CPU-only mode (current toolchain):" -ForegroundColor Blue
        Write-Host "   cargo build --release"
        Write-Host "3. Use GPU interface without CUDA:" -ForegroundColor Magenta
        Write-Host "   cargo build --release --features gpu"
        Write-Host
    }
    "*windows-msvc*" {
        Write-Host "Windows MSVC toolchain detected" -ForegroundColor Green
        Write-Host "Full CUDA support available!"
        Write-Host
        Write-Host "Recommended build commands:"
        Write-Host "With CUDA: Use the CUDA build function below" -ForegroundColor Green
        Write-Host "CPU-only:  cargo build --release" -ForegroundColor Blue
        Write-Host
        Write-Host "Note: CUDA builds require Visual Studio environment setup."
        Write-Host "   Use the Build-WithCuda function below for automatic setup."
        Write-Host
    }
    default {
        Write-Host "Unknown toolchain: $rustTarget" -ForegroundColor Yellow
        Write-Host "Falling back to CPU-only build"
        Write-Host
        Write-Host "Safe build command:"
        Write-Host "CPU-only: cargo build --release" -ForegroundColor Blue
        Write-Host
    }
}

# Check for NVIDIA GPU
try {
    $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
    if ($gpu) {
        Write-Host "NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green
        Write-Host "   CUDA acceleration recommended!"
    } else {
        Write-Host "No NVIDIA GPU detected" -ForegroundColor Yellow
        Write-Host "   CPU-only or basic GPU features recommended"
    }
} catch {
    Write-Host "Could not detect GPU information" -ForegroundColor Yellow
}

Write-Host
Write-Host "For more information, see README.md"
Write-Host "To run: cargo run --release [--features <feature>] <image_directory>"
Write-Host

# Function to build with CUDA (requires MSVC environment)
function Build-WithCuda {
    Write-Host "Setting up MSVC environment for CUDA build..." -ForegroundColor Cyan

    # Import Visual Studio environment
    $vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vsPath) {
        Write-Host "Found Visual Studio 2022 Community" -ForegroundColor Green

        # Get environment variables from vcvars64.bat
        $tempFile = [System.IO.Path]::GetTempFileName()
        cmd /c "`"$vsPath`" `&`& set" > $tempFile

        # Parse and set environment variables
        Get-Content $tempFile | ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$") {
                $name = $matches[1]
                $value = $matches[2]
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
        Remove-Item $tempFile

        Write-Host "MSVC environment configured" -ForegroundColor Green

        # Verify cl.exe is available
        try {
            $null = Get-Command cl.exe -ErrorAction Stop
            Write-Host "cl.exe found in PATH" -ForegroundColor Green
        } catch {
            Write-Host "cl.exe not found in PATH" -ForegroundColor Red
            return
        }

        # Build with CUDA
        Write-Host "Building with CUDA support..." -ForegroundColor Cyan
        cargo build --release --features cuda

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Build successful!" -ForegroundColor Green
        } else {
            Write-Host "Build failed!" -ForegroundColor Red
        }
    } else {
        Write-Host "Visual Studio 2022 Community not found at expected location" -ForegroundColor Red
        Write-Host "   Please install Visual Studio 2022 with C++ development tools" -ForegroundColor Yellow
    }
}

Write-Host "Quick start commands:" -ForegroundColor Cyan
Write-Host "   Build-WithCuda          # Build with CUDA support (auto-setup)"
Write-Host "   cargo build --release   # Build CPU-only version"
Write-Host "   cargo run --release .   # Run with current directory"