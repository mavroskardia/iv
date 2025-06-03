#!/bin/bash
# Shell packaging script for iv (Unix/Linux/macOS)

OUTPUT_DIR="${1:-./package}"
APP_NAME="iv"

echo "Packaging $APP_NAME to $OUTPUT_DIR..."

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy executable
echo "Copying executable..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    cp "./target/release/iv" "$OUTPUT_DIR/"
else
    # Linux
    cp "./target/release/iv" "$OUTPUT_DIR/"
fi

# Create models directory and copy model
echo "Copying model files..."
mkdir -p "$OUTPUT_DIR/models"
cp "./models/resnet50-v1-7-compatible.onnx" "$OUTPUT_DIR/models/"

# Create a simple readme
echo "Creating README..."
cat > "$OUTPUT_DIR/README.txt" << 'EOF'
# IV - Image Viewer

## Usage
./iv [path_to_images_directory]

## Example
./iv "/home/user/Pictures"

## Features
- Navigate images with arrow keys or mouse clicks
- Rate images with number keys 1-5
- Copy favorites with + key
- Delete images with Delete key
- AI-powered rating suggestions after rating 500+ images

## Requirements
The models folder must be in the same directory as iv executable for AI features to work.
EOF

echo "Package created in: $OUTPUT_DIR"
echo ""
echo "To test the packaged version:"
echo "  cd $OUTPUT_DIR"
echo "  ./iv \"path_to_your_images\""

# Make executable
chmod +x "$OUTPUT_DIR/iv"