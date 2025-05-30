#!/bin/bash

# Image Viewer Build Helper Script
# Automatically detects the best build configuration for your platform

echo "🔍 Detecting platform and build configuration..."
echo

# Detect OS
OS=$(uname -s)
ARCH=$(uname -m)

# Detect Rust toolchain
RUST_TARGET=$(rustc -vV | grep "host:" | cut -d' ' -f2)

echo "Platform: $OS $ARCH"
echo "Rust target: $RUST_TARGET"
echo

# Provide recommendations based on platform
case "$RUST_TARGET" in
    *"windows-gnu"*)
        echo "⚠️  Windows GNU toolchain detected"
        echo "CUDA is not supported due to linking issues with GNU toolchain."
        echo
        echo "Recommended options:"
        echo "1. 🚀 Switch to MSVC for CUDA support:"
        echo "   rustup default stable-x86_64-pc-windows-msvc"
        echo "   cargo build --release --features cuda"
        echo
        echo "2. 💻 Use CPU-only mode (current toolchain):"
        echo "   cargo build --release"
        echo
        echo "3. 🔧 Use GPU interface without CUDA:"
        echo "   cargo build --release --features gpu"
        echo
        ;;
    *"windows-msvc"*)
        echo "✅ Windows MSVC toolchain detected"
        echo "Full CUDA support available!"
        echo
        echo "Recommended build commands:"
        echo "🚀 With CUDA: cargo build --release --features cuda"
        echo "💻 CPU-only:  cargo build --release"
        echo
        ;;
    *"linux"*)
        echo "✅ Linux detected"
        echo "CUDA support available (if NVIDIA drivers installed)"
        echo
        echo "Recommended build commands:"
        echo "🚀 With CUDA: cargo build --release --features cuda"
        echo "🔧 With GPU:  cargo build --release --features gpu"
        echo "💻 CPU-only:  cargo build --release"
        echo
        ;;
    *"darwin"*)
        echo "✅ macOS detected"
        echo "Metal GPU acceleration available"
        echo
        echo "Recommended build commands:"
        echo "🚀 With Metal: cargo build --release --features gpu"
        echo "💻 CPU-only:   cargo build --release"
        echo
        ;;
    *)
        echo "❓ Unknown platform: $RUST_TARGET"
        echo "Falling back to CPU-only build"
        echo
        echo "Safe build command:"
        echo "💻 CPU-only: cargo build --release"
        echo
        ;;
esac

# Check for NVIDIA GPU on Linux
if [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
        echo "   CUDA acceleration recommended!"
    else
        echo "ℹ️  No NVIDIA GPU detected or nvidia-smi not available"
        echo "   CPU-only or basic GPU features recommended"
    fi
    echo
fi

echo "📖 For more information, see README.md"
echo "🚀 To run: cargo run --release [--features <feature>] <image_directory>"