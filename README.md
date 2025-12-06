# VastAI Setup Scripts

This folder contains the setup scripts that are downloaded and executed on VastAI instances during provisioning.

## 📁 Files

### 1. setup.sh
**Purpose:** Install required system software on Linux  
**Runs:** First, after instance creation  
**Installs:**
- Essential tools (git, wget, curl, rsync, vim, htop, tmux, screen)
- Python packages (pip, huggingface_hub, requests, tqdm)
- ComfyUI dependencies (libgl1-mesa-glx, libglib2.0-0)

### 2. models_config.json
**Purpose:** Define available AI models and their download sources  
**Used by:** setup_comfyui.py  
**Contains:**
- Model configurations (Flux, SDXL, etc.)
- HuggingFace repository URLs
- Download paths and file mappings

### 3. setup_comfyui.py
**Purpose:** Download models and setup ComfyUI  
**Runs:** After setup.sh completes  
**Actions:**
1. Clone ComfyUI repository
2. Install ComfyUI requirements
3. Download models based on MODEL_TYPE
4. Auto-start ComfyUI server on port 8188

## 🔄 Workflow

When a VastAI instance is created, the onstart script:

```bash
1. wget setup.sh (from GitHub/URL)
2. wget models_config.json (from GitHub/URL)
3. wget setup_comfyui.py (from GitHub/URL)
4. chmod +x setup.sh
5. ./setup.sh  # Install system dependencies
6. python3 setup_comfyui.py --model-type flux  # Setup ComfyUI
```

## 🌐 Hosting Setup

### Option 1: GitHub Repository

1. **Create a public repository:**
   ```bash
   cd /h/local-ai
   cd vast-ai
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/vast-ai.git
   git push -u origin main
   ```

2. **Get raw file URLs:**
   - setup.sh: `https://raw.githubusercontent.com/your-username/vast-ai/main/setup.sh`
   - models_config.json: `https://raw.githubusercontent.com/your-username/vast-ai/main/models_config.json`
   - setup_comfyui.py: `https://raw.githubusercontent.com/your-username/vast-ai/main/setup_comfyui.py`

3. **Update config.json:**
   ```json
   {
     "onstart_files": {
       "setup_script": "https://raw.githubusercontent.com/your-username/vast-ai/main/setup.sh",
       "models_config": "https://raw.githubusercontent.com/your-username/vast-ai/main/models_config.json",
       "setup_comfyui": "https://raw.githubusercontent.com/your-username/vast-ai/main/setup_comfyui.py"
     }
   }
   ```

### Option 2: Self-hosted Server

1. **Upload to web server:**
   ```bash
   scp setup.sh user@your-server.com:/var/www/html/
   scp models_config.json user@your-server.com:/var/www/html/
   scp setup_comfyui.py user@your-server.com:/var/www/html/
   ```

2. **Update config.json:**
   ```json
   {
     "onstart_files": {
       "setup_script": "https://your-server.com/setup.sh",
       "models_config": "https://your-server.com/models_config.json",
       "setup_comfyui": "https://your-server.com/setup_comfyui.py"
     }
   }
   ```

### Option 3: Pastebin/Gist (Quick Testing)

1. Create GitHub Gist with files
2. Get raw URLs
3. Update config.json

## 🛠️ Development

### Testing Locally

```bash
# Test setup.sh
bash setup.sh

# Test setup_comfyui.py
export MODEL_TYPE=flux
python3 setup_comfyui.py --workspace /workspace --model-type flux
```

### Adding New Models

Edit `models_config.json`:

```json
{
  "my-model": {
    "name": "My Custom Model",
    "description": "Model description",
    "checkpoints": [
      {
        "repo_id": "username/repo",
        "filename": "model.safetensors",
        "local_dir": "models/checkpoints"
      }
    ],
    "clip": [],
    "vae": []
  }
}
```

### Modifying Setup

**setup.sh:**
- Add/remove apt packages
- Install additional Python libraries
- Configure system settings

**setup_comfyui.py:**
- Change ComfyUI auto-start behavior
- Modify port (default: 8188)
- Add custom post-setup commands

## 📝 Notes

- Files must be publicly accessible (no authentication)
- Use HTTPS URLs for security
- Keep files under 1MB for fast downloads
- Test changes locally before updating URLs
- Git push changes to update remote files

## 🔒 Security

- **Don't include sensitive data** (API keys, passwords)
- **Use environment variables** for secrets (HF_TOKEN, etc.)
- **Validate downloads** with checksums if needed
- **Use HTTPS** to prevent MITM attacks

## 🚀 Usage

Once setup is complete:

1. **Access ComfyUI:**
   ```bash
   ssh -L 8188:localhost:8188 root@instance-ip -p ssh-port
   ```
   Then open: http://localhost:8188

2. **Check logs:**
   ```bash
   ssh root@instance-ip -p ssh-port
   cat /workspace/log.txt           # Setup logs
   cat /workspace/comfyui.log       # ComfyUI logs
   ```

3. **Verify ComfyUI:**
   ```bash
   ssh root@instance-ip -p ssh-port
   ps aux | grep main.py            # Check if running
   ```

## 🆘 Troubleshooting

### Setup fails
- Check `/workspace/log.txt` for errors
- Verify URLs are accessible: `wget -O- <url>`
- Check internet connection on instance

### ComfyUI doesn't start
- Check `/workspace/comfyui.log`
- Manually start: `cd /workspace/ComfyUI && python main.py --listen --port 8188`
- Verify models downloaded: `ls /workspace/ComfyUI/models/`

### Models not downloading
- Check HuggingFace token in config
- Verify model IDs in models_config.json
- Check disk space: `df -h`

---

**Ready to provision! 🎉**
