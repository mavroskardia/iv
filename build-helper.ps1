# Image Viewer Build Helper Script (PowerShell)
# Automatically detects the best build configuration for your platform

Write-Host "🔍 Detecting platform and build configuration..." -ForegroundColor Cyan
Write-Host

# Detect Rust toolchain
$rustTarget = (rustc -vV | Select-String "host:").ToString().Split(' ')[1]

Write-Host "Platform: Windows"
Write-Host "Rust target: $rustTarget"
Write-Host

# Provide recommendations based on toolchain
switch -Wildcard ($rustTarget) {
    "*windows-gnu*" {
        Write-Host "⚠️  Windows GNU toolchain detected" -ForegroundColor Yellow
        Write-Host "CUDA is not supported due to linking issues with GNU toolchain."
        Write-Host
        Write-Host "Recommended options:"
        Write-Host "1. 🚀 Switch to MSVC for CUDA support:" -ForegroundColor Green
        Write-Host "   rustup default stable-x86_64-pc-windows-msvc"
        Write-Host "   cargo build --release --features cuda"
        Write-Host
        Write-Host "2. 💻 Use CPU-only mode (current toolchain):" -ForegroundColor Blue
        Write-Host "   cargo build --release"
        Write-Host "3. 🔧 Use GPU interface without CUDA:" -ForegroundColor Magenta
        Write-Host "   cargo build --release --features gpu"
        Write-Host
    }
    "*windows-msvc*" {
        Write-Host "✅ Windows MSVC toolchain detected" -ForegroundColor Green
        Write-Host "Full CUDA support available!"
        Write-Host
        Write-Host "Recommended build commands:"
        Write-Host "🚀 With CUDA: cargo build --release --features cuda" -ForegroundColor Green
        Write-Host "💻 CPU-only:  cargo build --release" -ForegroundColor Blue
        Write-Host
    }
    default {
        Write-Host "❓ Unknown toolchain: $rustTarget" -ForegroundColor Yellow
        Write-Host "Falling back to CPU-only build"
        Write-Host
        Write-Host "Safe build command:"
        Write-Host "💻 CPU-only: cargo build --release" -ForegroundColor Blue
        Write-Host
    }
}

# Check for NVIDIA GPU
try {
    $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
    if ($gpu) {
        Write-Host "🎮 NVIDIA GPU detected: $($gpu.Name)" -ForegroundColor Green
        Write-Host "   CUDA acceleration recommended!"
    } else {
        Write-Host "ℹ️  No NVIDIA GPU detected" -ForegroundColor Yellow
        Write-Host "   CPU-only or basic GPU features recommended"
    }
} catch {
    Write-Host "ℹ️  Could not detect GPU information" -ForegroundColor Yellow
}

Write-Host
Write-Host "📖 For more information, see README.md"
Write-Host "🚀 To run: cargo run --release [--features <feature>] <image_directory>"