# Image Viewer with AI Rating System

An intelligent image viewer that learns your preferences and suggests ratings for new images.

## Features

### Basic Image Viewing
- Navigate through images with arrow keys or mouse clicks
- Support for JPG, PNG, GIF, BMP, and WebP formats
- Recursive directory scanning
- Progress tracking and state persistence
- Copy favorites and delete unwanted images

### AI-Powered Rating System
- Rate images from 1-5 using number keys (1-5)
- **Global rating system**: Ratings persist across all directories
- Visual feedback with color-coded flash effects:
  - 1 (Red) - Poor
  - 2 (Orange) - Below Average
  - 3 (Yellow) - Average
  - 4 (Light Green) - Good
  - 5 (Green) - Excellent
- After rating 500+ images, AI will suggest ratings for new images
- Override AI suggestions by pressing any number key
- Continuous learning from your feedback across all image collections

## Usage

```bash
cargo run <directory_path>
```

### Controls
- **Arrow Keys / Mouse**: Navigate between images
- **Number Keys 1-5**: Rate current image (automatically advances to next)
- **Plus (+)**: Copy to favorites folder
- **Delete**: Move to deleted folder
- **D**: Debug - Show feature analysis for current image
- **Escape**: Exit application

### AI Training
1. Start rating images using number keys 1-5
2. The app tracks your progress (shown in top bar)
3. After 500 ratings, AI suggestions activate
4. Yellow text shows AI suggestions
5. Green text shows your confirmed ratings
6. Continue rating to improve AI accuracy

## Technical Details

### Machine Learning
- Uses a simple neural network implemented with ndarray
- Simple feedforward network: 512 → 128 → 64 → 5 classes
- **Advanced feature extraction** analyzing:
  - Color histograms and saturation
  - Brightness, contrast, and exposure
  - Edge density and sharpness (Sobel operators)
  - Texture patterns (Local Binary Patterns)
  - Noise estimation and image quality
  - Composition analysis (rule of thirds)
  - Resolution and aspect ratio
  - **Shape and geometry analysis**:
    - Corner detection (Harris corner detector)
    - Line detection and orientation analysis
    - Symmetry analysis (horizontal, vertical, diagonal)
    - Geometric shape detection (circles, rectangles, triangles)
    - Structural balance and geometric harmony
- **Adaptive learning system**:
  - Dynamic learning rate adjustment based on performance
  - Incremental training for continuous improvement
  - Validation-based accuracy monitoring
  - Smart epoch scheduling (more for new models, fewer for updates)
- **Background processing**: Feature extraction and AI predictions run in separate threads
- **Intelligent caching**: Features are cached to avoid recomputation
- **GPU acceleration** (optional): Uses Candle framework for CUDA/Metal acceleration
- Automatic retraining when new ratings are added
- Pure Rust implementation with optional GPU dependencies

### Performance Optimizations
- **Non-blocking UI**: Expensive computations moved to background threads
- **Feature caching**: Computed features are cached by image hash
- **Asynchronous predictions**: AI suggestions appear when ready, don't block navigation
- **Responsive interface**: Rating and navigation remain fast even with complex analysis
- **GPU acceleration**: Optional CUDA/Metal support for faster training and inference
- **Adaptive training**: Lighter incremental updates vs full retraining based on model maturity

### Data Storage
- `app_state.json`: Application state and directory positions
- `~/.iv_ratings/ratings.json`: Global rating data with features
- `~/.iv_ratings/model.json`: Trained neural network weights
- `favorites/`: Copied favorite images
- `deleted/`: Moved deleted images

## Dependencies

- **eframe**: GUI framework
- **image**: Image processing
- **ndarray**: Numerical arrays for ML
- **serde**: Serialization
- **walkdir**: Directory traversal
- **sha2**: Image hashing
- **candle-core** (optional): GPU acceleration framework
- **candle-nn** (optional): Neural network operations on GPU

## Installation

1. Install Rust and Cargo
2. Clone and build:
   ```bash
   git clone <repository>
   cd image-viewer

   # Use the build helper to detect the best configuration
   ./build-helper.sh

   # Or manually choose:

   # CPU-only build (default)
   cargo build --release

   # GPU-accelerated build (requires CUDA or Metal)
   cargo build --release --features gpu

   # CUDA-specific build (Linux/Windows with NVIDIA GPU)
   cargo build --release --features cuda
   ```
3. Run with:
   ```bash
   # CPU-only
   cargo run --release <directory_path>

   # With GPU acceleration
   cargo run --release --features gpu <directory_path>

   # With CUDA acceleration
   cargo run --release --features cuda <directory_path>
   ```

**Note**:
- Use `--release` for optimal performance, especially with AI features enabled
- GPU features require appropriate drivers (CUDA for NVIDIA, Metal for Apple Silicon)
- The app automatically falls back to CPU if GPU initialization fails
- GPU acceleration significantly improves training speed for large datasets

### Windows GNU Toolchain Limitation

**Important**: CUDA support is not available when using the GNU toolchain on Windows (`x86_64-pc-windows-gnu`) due to linking incompatibilities between NVIDIA's CUDA libraries and the GNU linker.

**Solutions**:

1. **Use MSVC toolchain for CUDA** (Recommended):
   ```bash
   # Switch to MSVC toolchain
   rustup default stable-x86_64-pc-windows-msvc

   # Build with CUDA support
   cargo build --release --features cuda
   ```

2. **Use CPU-only mode with GNU**:
   ```bash
   # CPU-only build (works fine with GNU)
   cargo build --release

   # GPU interface without CUDA (CPU fallback)
   cargo build --release --features gpu
   ```

3. **Cross-compile from Linux/WSL**:
   ```bash
   # Install Windows MSVC target
   rustup target add x86_64-pc-windows-msvc

   # Cross-compile with CUDA
   cargo build --release --features cuda --target x86_64-pc-windows-msvc
   ```

The application will automatically detect the platform and provide appropriate error messages if CUDA is not available.

## Notes

- The AI model analyzes actual image content for quality assessment
- **Global rating system**: Your preferences learned from one directory apply to all others
- **Sophisticated feature extraction**: Analyzes sharpness, contrast, color quality, composition, and noise
- Model automatically saves and loads between sessions
- Requires at least 500 ratings before AI suggestions activate
- Pure Rust implementation with no external dependencies
- **Improved AI accuracy**: New feature extraction should provide much better rating suggestions
- **Performance optimized**: Background processing keeps the UI responsive during AI analysis
- Rating data stored in `~/.iv_ratings/` for cross-directory persistence