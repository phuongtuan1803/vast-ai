#!/usr/bin/env python3
"""
Setup script for ComfyUI and model downloads
Reads model configuration from models_config.json
Model type is determined by environment variable MODEL_TYPE
Supports workflow-based auto-provisioning
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
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
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read and display output line by line in real-time
        for line in process.stdout:
            log_message(line.rstrip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            log_message(f"❌ Command failed with exit code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        log_message(f"❌ Error: {e}")
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
    
    # Download all model types
    model_categories = [
        ('checkpoints', '🎨 Downloading checkpoints...'),
        ('clip', '📝 Downloading CLIP models...'),
        ('vae', '🎭 Downloading VAE models...'),
        ('loras', '🎨 Downloading LoRA models...'),
        ('controlnet', '🎮 Downloading ControlNet models...'),
        ('ipadapter', '🖼️  Downloading IP-Adapter models...'),
        ('clip_vision', '👁️  Downloading CLIP Vision models...'),
        ('upscale_models', '📈 Downloading Upscale models...'),
        ('unet', '🧠 Downloading UNET models...'),
    ]
    
    for category, message in model_categories:
        if model_config.get(category):
            log_message(f"\n{message}")
            for model in model_config[category]:
                local_dir = comfy_dir / model['local_dir']
                rename_to = model.get('rename_to')
                if not download_model_file(
                    model['repo_id'],
                    model['filename'],
                    local_dir,
                    rename_to
                ):
                    success = False
    
    return success


def download_workflow_models(workspace_dir: Path, workflow_path: str, config: Dict) -> bool:
    """
    Download models required by a specific workflow
    
    Args:
        workspace_dir: Directory where ComfyUI is installed
        workflow_path: Path to ComfyUI workflow JSON file
        config: Models configuration dictionary
    
    Returns:
        True if all downloads successful, False otherwise
    """
    try:
        # Import workflow parser
        from workflow_parser import WorkflowParser, ModelDependency
        from download_models import ModelDownloader
        
        log_message(f"\n{'='*60}")
        log_message(f"🔍 Parsing workflow: {workflow_path}")
        log_message(f"{'='*60}\n")
        
        # Parse workflow
        parser = WorkflowParser(workflow_path)
        dependencies = parser.parse()
        
        log_message(f"📦 Found {len(dependencies.models)} model dependencies")
        log_message(f"🔧 Found {len(dependencies.custom_nodes)} custom node dependencies\n")
        
        # Initialize downloader
        comfy_dir = workspace_dir / "ComfyUI"
        downloader = ModelDownloader(comfy_dir, log_callback=log_message)
        
        # Track models to download
        models_to_download = []
        missing_models = []
        
        # Check each model dependency
        for model_dep in dependencies.models:
            model_path = downloader.get_model_path(model_dep.name, model_dep.model_type)
            
            if model_path.exists():
                log_message(f"✅ Already exists: [{model_dep.model_type}] {model_dep.name}")
            else:
                log_message(f"📥 Need to download: [{model_dep.model_type}] {model_dep.name}")
                models_to_download.append(model_dep)
        
        if not models_to_download:
            log_message(f"\n✅ All required models are already present!")
            return True
        
        log_message(f"\n{'='*60}")
        log_message(f"📥 Downloading {len(models_to_download)} missing models")
        log_message(f"{'='*60}\n")
        
        # Try to download each model
        success_count = 0
        for model_dep in models_to_download:
            # Check if model has URL in workflow
            if model_dep.url:
                log_message(f"\n📥 Downloading {model_dep.name} from workflow URL...")
                success = downloader.download_from_url(
                    url=model_dep.url,
                    model_name=model_dep.name,
                    model_type=model_dep.model_type,
                    expected_hash=model_dep.hash,
                    hash_type=model_dep.hash_type or 'sha256'
                )
                if success:
                    success_count += 1
                else:
                    missing_models.append(model_dep)
            else:
                # Try to find in config
                found_in_config = False
                for config_type, config_data in config.items():
                    # Check all model categories in config
                    for category in ['checkpoints', 'vae', 'clip', 'loras', 'controlnet', 
                                   'ipadapter', 'clip_vision', 'upscale_models', 'unet']:
                        if category in config_data:
                            for model_info in config_data[category]:
                                # Match by filename
                                if model_info['filename'].endswith(model_dep.name) or model_dep.name == model_info.get('rename_to'):
                                    log_message(f"\n📥 Found in config: {model_dep.name}")
                                    success = downloader.download_from_huggingface(
                                        repo_id=model_info['repo_id'],
                                        filename=model_info['filename'],
                                        model_type=model_dep.model_type,
                                        rename_to=model_info.get('rename_to')
                                    )
                                    if success:
                                        success_count += 1
                                        found_in_config = True
                                    break
                        if found_in_config:
                            break
                    if found_in_config:
                        break
                
                if not found_in_config:
                    log_message(f"⚠️  Could not find download source for: {model_dep.name}")
                    missing_models.append(model_dep)
        
        # Report results
        log_message(f"\n{'='*60}")
        log_message(f"📊 Download Summary")
        log_message(f"{'='*60}")
        log_message(f"✅ Successfully downloaded: {success_count}/{len(models_to_download)}")
        
        if missing_models:
            log_message(f"\n⚠️  Missing models (manual download required):")
            for model in missing_models:
                log_message(f"  - [{model.model_type}] {model.name}")
            log_message(f"\nPlease download these models manually to the appropriate directories.")
            return False
        
        # Handle custom nodes
        if dependencies.custom_nodes:
            log_message(f"\n{'='*60}")
            log_message(f"🔧 Custom Nodes Detected")
            log_message(f"{'='*60}")
            log_message(f"The workflow requires {len(dependencies.custom_nodes)} custom nodes:")
            for node in dependencies.custom_nodes:
                log_message(f"  - {node.node_type}")
            log_message(f"\n💡 Install via ComfyUI-Manager or manually clone to custom_nodes/")
        
        return True
        
    except Exception as e:
        log_message(f"❌ Error processing workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_comfyui(
    workspace_dir: str = "/workspace",
    model_type: Optional[str] = None,
    workflow_path: Optional[str] = None,
    workflow_paths: Optional[List[str]] = None
) -> bool:
    """
    Main setup function for ComfyUI
    
    Args:
        workspace_dir: Directory where ComfyUI will be installed
        model_type: Type of model(s) to download (flux, stable-diffusion-xl, sdxl)
                   Multiple models can be separated by semicolons (e.g., "flux;sdxl")
                   If None, will read from MODEL_TYPE environment variable
                   Ignored if workflow_path is provided
        workflow_path: Path to single ComfyUI workflow JSON file for auto-provisioning
                      If provided, models will be downloaded based on workflow requirements
        workflow_paths: List of paths to ComfyUI workflow JSON files for multi-workflow provisioning
    
    Returns:
        True if setup was successful, False otherwise
    """
    # Convert to Path object
    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize log file
    init_log_file(workspace_path)
    
    log_message(f"\n{'='*60}")
    log_message(f"🚀 ComfyUI Setup Script")
    log_message(f"{'='*60}")
    log_message(f"Workspace: {workspace_dir}")
    
    # Determine provisioning mode
    all_workflows = []
    if workflow_paths:
        all_workflows = [wp for wp in workflow_paths if wp and Path(wp).exists()]
    if workflow_path and Path(workflow_path).exists():
        all_workflows.append(workflow_path)
    
    if all_workflows:
        log_message(f"Mode: Workflow-based provisioning ({len(all_workflows)} workflows)")
        for wp in all_workflows:
            log_message(f"  - {wp}")
    elif model_type is None:
        # Check environment variable
        model_type = os.environ.get('MODEL_TYPE', '')
    
    if not all_workflows and not model_type:
        log_message(f"Mode: ComfyUI only (no model downloads)")
    elif not all_workflows:
        # Parse multiple model types
        model_types = [m.strip() for m in model_type.split(';') if m.strip()]
        log_message(f"Mode: Model type provisioning")
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
    
    # Download models based on mode
    overall_success = True
    
    if all_workflows:
        # Multi-workflow provisioning
        for wp in all_workflows:
            log_message(f"\n{'='*60}")
            log_message(f"📄 Processing workflow: {Path(wp).name}")
            log_message(f"{'='*60}")
            if not download_workflow_models(workspace_path, wp, config):
                log_message(f"⚠️  Failed to download some workflow models from {wp}")
                overall_success = False
    elif model_type:
        # Model type based provisioning
        model_types = [m.strip() for m in model_type.split(';') if m.strip()]
        for mt in model_types:
            log_message(f"\n{'='*60}")
            log_message(f"📦 Processing model type: {mt}")
            log_message(f"{'='*60}")
            if not download_models(workspace_path, mt, config):
                log_message(f"⚠️  Failed to download some models for {mt}")
                overall_success = False
    else:
        # ComfyUI only mode - no model downloads
        log_message(f"\n{'='*60}")
        log_message(f"✅ ComfyUI only mode - skipping model downloads")
        log_message(f"{'='*60}")
    
    if not overall_success:
        log_message("\n⚠️  Some models failed to download")
        # Don't return False, continue to start ComfyUI
    
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
    
    parser = argparse.ArgumentParser(
        description="Setup ComfyUI and download models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup ComfyUI only (no model downloads)
  python setup_comfyui.py --workspace /workspace
  
  # Setup with model types
  python setup_comfyui.py --workspace /workspace --model-type flux;sdxl
  
  # Setup from single workflow file
  python setup_comfyui.py --workspace /workspace --workflow my_workflow.json
  
  # Setup from multiple workflow files
  python setup_comfyui.py --workspace /workspace --workflows wf1.json wf2.json wf3.json
  
  # Auto-provision all models required by workflows
  python setup_comfyui.py --workflows workflow1.json workflow2.json --workspace ./ComfyUI
        """
    )
    parser.add_argument(
        "--workspace",
        default="/workspace",
        help="Workspace directory (default: /workspace)"
    )
    parser.add_argument(
        "--model-type",
        help="Model type(s) to download, separated by semicolons (e.g., 'flux;sdxl'). "
             "Available: flux, stable-diffusion-xl, sdxl, controlnet, ip_adapter, upscaler. "
             "Default: ComfyUI only if not specified. "
             "Ignored if --workflow or --workflows is provided."
    )
    parser.add_argument(
        "--workflow",
        help="Path to single ComfyUI workflow JSON file. If provided, will auto-download all required models."
    )
    parser.add_argument(
        "--workflows",
        nargs='+',
        help="Paths to multiple ComfyUI workflow JSON files. If provided, will auto-download all required models."
    )
    
    args = parser.parse_args()
    
    success = setup_comfyui(
        workspace_dir=args.workspace,
        model_type=args.model_type,
        workflow_path=args.workflow,
        workflow_paths=args.workflows
    )
    
    sys.exit(0 if success else 1)
