# PowerShell packaging script for iv.exe
# This script packages the executable with its dependencies

param(
    [string]$OutputDir = ".\package",
    [string]$AppName = "iv"
)

Write-Host "Packaging $AppName to $OutputDir..."

# Create output directory
if (Test-Path $OutputDir) {
    Remove-Item $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

# Copy executable
Write-Host "Copying executable..."
Copy-Item ".\target\release\iv.exe" "$OutputDir\"

# Create models directory and copy model
Write-Host "Copying model files..."
New-Item -ItemType Directory -Path "$OutputDir\models" | Out-Null
Copy-Item ".\models\resnet50-v1-7-compatible.onnx" "$OutputDir\models\"

# Create a simple readme
Write-Host "Creating README..."
@"
# IV - Image Viewer

## Usage
.\iv.exe [path_to_images_directory]

## Example
.\iv.exe "C:\Users\YourName\Pictures"

## Features
- Navigate images with arrow keys or mouse clicks
- Rate images with number keys 1-5
- Copy favorites with + key
- Delete images with Delete key
- AI-powered rating suggestions after rating 500+ images

## Requirements
The models folder must be in the same directory as iv.exe for AI features to work.
"@ | Out-File "$OutputDir\README.txt" -Encoding UTF8

Write-Host "Package created in: $OutputDir"
Write-Host ""
Write-Host "To test the packaged version:"
Write-Host "  cd $OutputDir"
Write-Host "  .\iv.exe `"path_to_your_images`""