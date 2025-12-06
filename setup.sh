#!/bin/bash
# Setup script for VastAI instance - Complete setup with model downloads
set -e

# Redirect all output to log file
exec >> /workspace/log.txt 2>&1

echo "========================================"
echo "🚀 VastAI Instance Setup - $(date)"
echo "========================================"

# Update system
echo "📦 Updating system packages..."
apt-get update -qq

# Install essential tools
echo "🔧 Installing essential tools..."
apt-get install -y -qq \
    git \
    wget \
    curl \
    rsync \
    vim \
    htop \
    tmux \
    screen

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip install -q --upgrade pip
pip install -q \
    huggingface_hub \
    requests \
    tqdm

# Install ComfyUI dependencies
echo "📦 Installing ComfyUI dependencies..."
apt-get install -y -qq \
    libgl1-mesa-glx \
    libglib2.0-0

# Verify installations
echo "✅ Verifying installations..."
python --version
git --version
pip --version

# Create workspace structure
echo "📁 Creating workspace structure..."
mkdir -p /workspace/logs
mkdir -p /workspace/models
mkdir -p /workspace/outputs

# Download additional setup files
echo "📥 Downloading configuration files..."
GITHUB_BASE="https://raw.githubusercontent.com/phuongtuan1803/vast-ai/refs/heads/master"

wget -q -O /workspace/models_config.json "${GITHUB_BASE}/models_config.json"
wget -q -O /workspace/setup_comfyui.py "${GITHUB_BASE}/setup_comfyui.py"

echo "✅ Configuration files downloaded"

# Run ComfyUI setup
echo "🚀 Setting up ComfyUI..."
cd /workspace

# Check if MODEL_TYPE is set, otherwise default to flux
if [ -z "$MODEL_TYPE" ]; then
    export MODEL_TYPE="flux"
    echo "⚠️  MODEL_TYPE not set, defaulting to: flux"
else
    echo "📋 Using MODEL_TYPE: $MODEL_TYPE"
fi

python3 /workspace/setup_comfyui.py --workspace /workspace --model-type "$MODEL_TYPE"

# Set permissions
chmod -R 755 /workspace

echo "========================================"
echo "✅ Setup completed successfully!"
echo "📊 ComfyUI is running in background"
echo "🌐 Access at: http://localhost:8188"
echo "========================================"
