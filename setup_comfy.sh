#!/bin/bash

# 1. Cài đặt các công cụ cần thiết & Tối ưu mạng
apt-get update && apt-get install -y rsync tmux aria2
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Setup Workspace
mkdir -p /workspace
cd /workspace

# 3. Clone ComfyUI (Nếu chưa có)
if [! -d "ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install -r requirements.txt
    
    # Cài ComfyUI Manager tự động
    cd custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd..
else
    cd ComfyUI
fi

# 4. Tải Models (Sử dụng hf_transfer để max băng thông server)
# Định nghĩa thư mục
CKPT_DIR=/workspace/ComfyUI/models/checkpoints
CLIP_DIR=/workspace/ComfyUI/models/clip
VAE_DIR=/workspace/ComfyUI/models/vae
LORA_DIR=/workspace/ComfyUI/models/loras

# --- Tải SDXL ---
echo "Downloading SDXL..."
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir $CKPT_DIR --local-dir-use-symlinks False

# --- Tải Flux.1 Dev (Phiên bản FP8 để chạy mượt trên 24GB VRAM) ---
# YÊU CẦU: Bạn phải có Token HuggingFace trong biến môi trường HF_TOKEN
echo "Downloading Flux.1 Dev FP8..."
huggingface-cli download kijai/flux-fp8 flux1-dev-fp8.safetensors --local-dir $CKPT_DIR --local-dir-use-symlinks False

# --- Tải các Encoder cần thiết cho Flux ---
echo "Downloading Clips..."
huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir $CLIP_DIR --local-dir-use-symlinks False
huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir $CLIP_DIR --local-dir-use-symlinks False
huggingface-cli download comfyanonymous/flux_text_encoders ae.safetensors --local-dir $VAE_DIR --local-dir-use-symlinks False

# 5. Khởi chạy ComfyUI trong Tmux (để không bị tắt khi ngắt kết nối SSH)
tmux new-session -d -s comfy 'python main.py --listen --port 8188'