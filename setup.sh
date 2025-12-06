#!/bin/bash
# Setup script for VastAI instance - Install required software
set -e

echo "========================================"
echo "🚀 VastAI Instance Setup"
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

# Set permissions
chmod -R 755 /workspace

echo "========================================"
echo "✅ Setup completed successfully!"
echo "========================================"
