#!/usr/bin/env python3
"""
Setup script for ComfyUI and model downloads
Reads model configuration from models_config.json
Model type is determined by environment variable MODEL_TYPE
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# Redirect all output to log file
class LogRedirector:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start(self):
        self.log_file = open(self.log_path, 'a', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        if message.strip():  # Only write non-empty messages
            self.original_stdout.write(message)
            if self.log_file:
                self.log_file.write(message)
                self.log_file.flush()

    def flush(self):
        if self.log_file:
            self.log_file.flush()

    def stop(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.log_file:
            self.log_file.close()


# Global log file handler
LOG_FILE = None
LOG_REDIRECTOR = None


def init_log_file(workspace_dir: Path):
    """Initialize log file for writing"""
    global LOG_FILE, LOG_REDIRECTOR
    log_path = workspace_dir / "log.txt"

    # Start redirector
    LOG_REDIRECTOR = LogRedirector(log_path)
    LOG_REDIRECTOR.start()

    LOG_FILE = LOG_REDIRECTOR.log_file
    log_message(f"\n{'='*60}")
    log_message(f"Setup started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"{'='*60}\n")


def close_log_file():
    """Close log file"""
    global LOG_FILE, LOG_REDIRECTOR
    if LOG_FILE:
        log_message(f"\n{'='*60}")
        log_message(f"Setup finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"{'='*60}\n")

    if LOG_REDIRECTOR:
        LOG_REDIRECTOR.stop()


def log_message(message: str):
    """Write message to both console and log file"""
    print(message)
    if LOG_FILE:
        LOG_FILE.write(message + '\n')
        LOG_FILE.flush()


def load_models_config(config_path: str = "models_config.json") -> Dict:
    """Load models configuration from JSON file"""
    script_dir = Path(__file__).parent
    config_file = script_dir / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_command(command: List[str], description: str = "") -> bool:
    """Execute a shell command and return success status"""
    if description:
        log_message(f"\n{'='*60}")
        log_message(f"⚙️  {description}")
        log_message(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            log_message(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"❌ Error: {e}")
        if e.stderr:
            log_message(f"Error output: {e.stderr}")
        return False


def install_dependencies():
    """Install necessary system dependencies"""
    log_message("📦 Installing dependencies...")
    
    # Install Python packages
    packages = ["huggingface_hub"]
    for package in packages:
        run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"Installing {package}"
        )


def clone_comfyui(workspace_dir: Path) -> bool:
    """Clone ComfyUI repository if not exists"""
    comfy_dir = workspace_dir / "ComfyUI"
    
    if comfy_dir.exists():
        log_message(f"✅ ComfyUI already exists at {comfy_dir}")
        return True
    
    log_message(f"📥 Cloning ComfyUI to {comfy_dir}")
    
    # Clone ComfyUI
    if not run_command(
        ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)],
        "Cloning ComfyUI repository"
    ):
        return False
    
    # Install ComfyUI requirements
    requirements_file = comfy_dir / "requirements.txt"
    if requirements_file.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            "Installing ComfyUI requirements"
        )
    
    # Clone ComfyUI Manager
    custom_nodes_dir = comfy_dir / "custom_nodes"
    custom_nodes_dir.mkdir(exist_ok=True)
    
    manager_dir = custom_nodes_dir / "ComfyUI-Manager"
    if not manager_dir.exists():
        run_command(
            ["git", "clone", "https://github.com/ltdrdata/ComfyUI-Manager.git", str(manager_dir)],
            "Installing ComfyUI Manager"
        )
    
    return True


def download_model_file(repo_id: str, filename: str, local_dir: Path, rename_to: Optional[str] = None) -> bool:
    """Download a single model file from HuggingFace"""
    try:
        from huggingface_hub import hf_hub_download
        
        log_message(f"📥 Downloading {filename} from {repo_id}...")
        
        # Ensure local directory exists
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        # Rename if needed
        if rename_to:
            old_path = Path(downloaded_path)
            new_path = local_dir / rename_to
            if old_path != new_path:
                old_path.rename(new_path)
                log_message(f"✅ Renamed to {rename_to}")
        
        log_message(f"✅ Downloaded successfully")
        return True
        
    except Exception as e:
        log_message(f"❌ Failed to download {filename}: {e}")
        return False


def download_models(workspace_dir: Path, model_type: str, config: Dict) -> bool:
    """Download models based on configuration"""
    comfy_dir = workspace_dir / "ComfyUI"
    
    if model_type not in config:
        log_message(f"❌ Unknown model type: {model_type}")
        log_message(f"Available types: {', '.join(config.keys())}")
        return False
    
    model_config = config[model_type]
    log_message(f"\n{'='*60}")
    log_message(f"📦 Setting up {model_config['name']}")
    log_message(f"   {model_config['description']}")
    log_message(f"{'='*60}\n")
    
    success = True
    
    # Download checkpoints
    if model_config.get('checkpoints'):
        log_message("🎨 Downloading checkpoints...")
        for checkpoint in model_config['checkpoints']:
            local_dir = comfy_dir / checkpoint['local_dir']
            rename_to = checkpoint.get('rename_to')
            if not download_model_file(
                checkpoint['repo_id'],
                checkpoint['filename'],
                local_dir,
                rename_to
            ):
                success = False
    
    # Download CLIP models
    if model_config.get('clip'):
        log_message("\n📝 Downloading CLIP models...")
        for clip in model_config['clip']:
            local_dir = comfy_dir / clip['local_dir']
            rename_to = clip.get('rename_to')
            if not download_model_file(
                clip['repo_id'],
                clip['filename'],
                local_dir,
                rename_to
            ):
                success = False
    
    # Download VAE models
    if model_config.get('vae'):
        log_message("\n🎭 Downloading VAE models...")
        for vae in model_config['vae']:
            local_dir = comfy_dir / vae['local_dir']
            rename_to = vae.get('rename_to')
            if not download_model_file(
                vae['repo_id'],
                vae['filename'],
                local_dir,
                rename_to
            ):
                success = False
    
    return success


def setup_comfyui(workspace_dir: str = "/workspace", model_type: Optional[str] = None) -> bool:
    """
    Main setup function for ComfyUI
    
    Args:
        workspace_dir: Directory where ComfyUI will be installed
        model_type: Type of model(s) to download (flux, stable-diffusion-xl, sdxl)
                   Multiple models can be separated by semicolons (e.g., "flux;sdxl")
                   If None, will read from MODEL_TYPE environment variable
    
    Returns:
        True if setup was successful, False otherwise
    """
    # Get model type from environment if not provided
    if model_type is None:
        model_type = os.environ.get('MODEL_TYPE', 'stable-diffusion-xl')
    
    # Parse multiple model types
    model_types = [m.strip() for m in model_type.split(';') if m.strip()]
    
    # Convert to Path object
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize log file
    init_log_file(workspace_path)
    
    log_message(f"\n{'='*60}")
    log_message(f"🚀 ComfyUI Setup Script")
    log_message(f"{'='*60}")
    log_message(f"Workspace: {workspace_dir}")
    log_message(f"Model Types: {', '.join(model_types)}")
    log_message(f"{'='*60}\n")
    
    # Install dependencies
    install_dependencies()
    
    # Load models configuration
    try:
        config = load_models_config()
    except Exception as e:
        log_message(f"❌ Failed to load models config: {e}")
        close_log_file()
        return False
    
    # Clone ComfyUI
    if not clone_comfyui(workspace_path):
        log_message("❌ Failed to clone ComfyUI")
        close_log_file()
        return False
    
    # Download models for each type
    overall_success = True
    for mt in model_types:
        log_message(f"\n{'='*60}")
        log_message(f"📦 Processing model type: {mt}")
        log_message(f"{'='*60}")
        if not download_models(workspace_path, mt, config):
            log_message(f"⚠️  Failed to download some models for {mt}")
            overall_success = False
    
    if not overall_success:
        log_message("\n⚠️  Some models failed to download")
        close_log_file()
        return False
    
    log_message(f"\n{'='*60}")
    log_message(f"✅ Setup completed successfully!")
    log_message(f"{'='*60}")
    
    # Start ComfyUI in background
    log_message(f"\n🚀 Starting ComfyUI server in background...")
    comfy_dir = workspace_path / 'ComfyUI'
    
    try:
        import subprocess
        
        # Start ComfyUI with nohup in background
        comfy_log = workspace_path / 'comfyui.log'
        comfy_cmd = f"cd {comfy_dir} && nohup python main.py --listen --port 8188 > {comfy_log} 2>&1 &"
        
        subprocess.run(comfy_cmd, shell=True, check=False)
        log_message(f"✅ ComfyUI started on port 8188")
        log_message(f"📝 Logs: {comfy_log}")
        log_message(f"\n🌐 Access ComfyUI at: http://localhost:8188")
        log_message(f"🔗 Or via SSH tunnel: ssh -L 8188:localhost:8188 root@<instance_ip>")
        
    except Exception as e:
        log_message(f"⚠️ Failed to auto-start ComfyUI: {e}")
        log_message(f"\nTo start manually, run:")
        log_message(f"  cd {comfy_dir}")
        log_message(f"  python main.py --listen --port 8188")
    
    close_log_file()
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ComfyUI and download models")
    parser.add_argument(
        "--workspace",
        default="/workspace",
        help="Workspace directory (default: /workspace)"
    )
    parser.add_argument(
        "--model-type",
        help="Model type(s) to download, separated by semicolons (e.g., 'flux;sdxl'). "
             "Available: flux, stable-diffusion-xl, sdxl. "
             "Default: from MODEL_TYPE env var or 'stable-diffusion-xl'"
    )
    
    args = parser.parse_args()
    
    success = setup_comfyui(
        workspace_dir=args.workspace,
        model_type=args.model_type
    )
    
    sys.exit(0 if success else 1)
