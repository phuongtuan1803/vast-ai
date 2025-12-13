#!/bin/bash
# Setup script for VastAI instance - Complete setup with model downloads
# Supports: workflow-based, comfyui_only, or legacy model_type mode
set -e

# Redirect all output to log file
exec >> /workspace/log.txt 2>&1

echo "========================================"
echo "🚀 VastAI Instance Setup - $(date)"
echo "========================================"

# Detect provisioning mode
if [ -n "$PROVISION_MODE" ]; then
    echo "📋 Provision mode: $PROVISION_MODE"
else
    # Legacy support: if MODEL_TYPE is set, use model_type mode
    if [ -n "$MODEL_TYPE" ]; then
        PROVISION_MODE="model_type"
    else
        PROVISION_MODE="comfyui_only"
    fi
    echo "📋 Provision mode (auto-detected): $PROVISION_MODE"
fi

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

pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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
mkdir -p /workspace/workflows

# Download additional setup files
echo "📥 Downloading configuration files..."
GITHUB_BASE="https://raw.githubusercontent.com/phuongtuan1803/vast-ai/refs/heads/master"

wget -q -O /workspace/models_config.json "${GITHUB_BASE}/models_config.json"
wget -q -O /workspace/setup_comfyui.py "${GITHUB_BASE}/setup_comfyui.py"
wget -q -O /workspace/workflow_parser.py "${GITHUB_BASE}/workflow_parser.py"
wget -q -O /workspace/download_models.py "${GITHUB_BASE}/download_models.py"

echo "✅ Configuration files downloaded"

# Run ComfyUI setup based on provision mode
echo "🚀 Setting up ComfyUI..."
cd /workspace

case "$PROVISION_MODE" in
    "comfyui_only")
        echo "📦 ComfyUI only mode - no models will be downloaded"
        if ! python3 /workspace/setup_comfyui.py --workspace /workspace; then
            echo "❌ Failed to setup ComfyUI"
            exit 1
        fi
        ;;
    
    "workflow")
        echo "📦 Workflow-based provisioning"
        if [ -n "$WORKFLOWS" ]; then
            echo "📄 Workflows: $WORKFLOWS"
            # Process each workflow
            IFS=';' read -ra WORKFLOW_ARRAY <<< "$WORKFLOWS"
            for wf in "${WORKFLOW_ARRAY[@]}"; do
                if [ -f "/workspace/workflows/$wf" ]; then
                    echo "🔄 Processing workflow: $wf"
                    if ! python3 /workspace/setup_comfyui.py --workspace /workspace --workflow "/workspace/workflows/$wf"; then
                        echo "❌ Failed to process workflow: $wf"
                        exit 1
                    fi
                else
                    echo "⚠️  Workflow file not found: $wf (will be processed when uploaded)"
                fi
            done
        else
            echo "⚠️  No workflows specified, running ComfyUI only"
            if ! python3 /workspace/setup_comfyui.py --workspace /workspace; then
                echo "❌ Failed to setup ComfyUI"
                exit 1
            fi
        fi
        ;;
    
    "model_type")
        echo "📦 Legacy model type mode"
        if [ -z "$MODEL_TYPE" ]; then
            export MODEL_TYPE="flux"
            echo "⚠️  MODEL_TYPE not set, defaulting to: flux"
        else
        echo "📍 Using MODEL_TYPE: $MODEL_TYPE"
        fi
        if ! python3 /workspace/setup_comfyui.py --workspace /workspace --model-type "$MODEL_TYPE"; then
            echo "❌ Failed to setup with model type: $MODEL_TYPE"
            exit 1
        fi
        ;;
    
    *)
        echo "⚠️  Unknown provision mode: $PROVISION_MODE, defaulting to comfyui_only"
        if ! python3 /workspace/setup_comfyui.py --workspace /workspace; then
            echo "❌ Failed to setup ComfyUI"
            exit 1
        fi
        ;;
esac

# Set permissions
chmod -R 755 /workspace

# Final status check
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "✅ Setup completed successfully!"
    echo "✅ ComfyUI is ready and running"
    echo "📊 Access at: http://localhost:8188"
    echo "========================================"
    exit 0
else
    echo "========================================"
    echo "❌ Setup failed - check logs above"
    echo "========================================"
    exit 1
fi
echo "🌐 Access at: http://localhost:8188"
echo "========================================"
