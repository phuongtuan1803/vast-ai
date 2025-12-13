#!/usr/bin/env python3
"""
Model downloader utility for ComfyUI
Handles downloading models from HuggingFace and other sources with validation
"""

import hashlib
import json
import os
import requests
from pathlib import Path
from typing import Optional, Dict, List, Callable
from urllib.parse import urlparse


class ModelDownloader:
    """Handles downloading and validating model files"""
    
    def __init__(self, comfyui_dir: Path, log_callback: Optional[Callable] = None):
        """
        Initialize model downloader
        
        Args:
            comfyui_dir: Path to ComfyUI root directory
            log_callback: Optional callback function for logging (default: print)
        """
        self.comfyui_dir = Path(comfyui_dir)
        self.log = log_callback if log_callback else print
        
        # Model directory mapping
        self.model_dirs = {
            'checkpoints': self.comfyui_dir / 'models' / 'checkpoints',
            'vae': self.comfyui_dir / 'models' / 'vae',
            'clip': self.comfyui_dir / 'models' / 'clip',
            'loras': self.comfyui_dir / 'models' / 'loras',
            'controlnet': self.comfyui_dir / 'models' / 'controlnet',
            'unet': self.comfyui_dir / 'models' / 'unet',
            'upscale_models': self.comfyui_dir / 'models' / 'upscale_models',
            'gligen': self.comfyui_dir / 'models' / 'gligen',
            'hypernetworks': self.comfyui_dir / 'models' / 'hypernetworks',
            'style_models': self.comfyui_dir / 'models' / 'style_models',
            'clip_vision': self.comfyui_dir / 'models' / 'clip_vision',
            'photomaker': self.comfyui_dir / 'models' / 'photomaker',
            'ipadapter': self.comfyui_dir / 'models' / 'ipadapter',
            'embeddings': self.comfyui_dir / 'models' / 'embeddings',
        }
    
    def get_model_path(self, model_name: str, model_type: str) -> Path:
        """Get full path for a model file"""
        if model_type not in self.model_dirs:
            # Default to checkpoints if type unknown
            model_type = 'checkpoints'
        
        return self.model_dirs[model_type] / model_name
    
    def model_exists(self, model_name: str, model_type: str) -> bool:
        """Check if model file already exists"""
        model_path = self.get_model_path(model_name, model_type)
        return model_path.exists()
    
    def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        model_type: str,
        rename_to: Optional[str] = None
    ) -> bool:
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID (e.g., 'stabilityai/stable-diffusion-xl-base-1.0')
            filename: File name in the repository
            model_type: Type of model (checkpoints, vae, clip, etc.)
            rename_to: Optional new name for the downloaded file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import hf_hub_download
            
            final_name = rename_to if rename_to else filename
            
            # Check if already exists
            if self.model_exists(final_name, model_type):
                self.log(f"✅ Model already exists: {final_name}")
                return True
            
            self.log(f"📥 Downloading {filename} from {repo_id}...")
            
            # Ensure directory exists
            target_dir = self.model_dirs[model_type]
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False
            )
            
            # Rename if needed
            if rename_to:
                old_path = Path(downloaded_path)
                new_path = target_dir / rename_to
                if old_path != new_path:
                    old_path.rename(new_path)
                    self.log(f"✅ Renamed to {rename_to}")
            
            self.log(f"✅ Downloaded successfully: {final_name}")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to download {filename}: {e}")
            return False
    
    def download_from_url(
        self,
        url: str,
        model_name: str,
        model_type: str,
        expected_hash: Optional[str] = None,
        hash_type: str = 'sha256'
    ) -> bool:
        """
        Download model from direct URL
        
        Args:
            url: Direct download URL
            model_name: Name to save the file as
            model_type: Type of model (checkpoints, vae, clip, etc.)
            expected_hash: Optional hash for verification
            hash_type: Hash algorithm (sha256, md5, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already exists
            if self.model_exists(model_name, model_type):
                self.log(f"✅ Model already exists: {model_name}")
                return True
            
            self.log(f"📥 Downloading {model_name} from {url}...")
            
            # Ensure directory exists
            target_dir = self.model_dirs[model_type]
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / model_name
            
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            # Create hash object if verification needed
            hash_obj = None
            if expected_hash:
                hash_obj = hashlib.new(hash_type)
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if hash_obj:
                            hash_obj.update(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (block_size * 100) == 0:  # Update every ~800KB
                                self.log(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
            
            # Verify hash if provided
            if expected_hash and hash_obj:
                computed_hash = hash_obj.hexdigest()
                if computed_hash.lower() != expected_hash.lower():
                    self.log(f"❌ Hash mismatch!")
                    self.log(f"  Expected: {expected_hash}")
                    self.log(f"  Got: {computed_hash}")
                    target_path.unlink()  # Delete corrupted file
                    return False
                self.log(f"✅ Hash verified: {hash_type}")
            
            self.log(f"✅ Downloaded successfully: {model_name}")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to download {model_name}: {e}")
            if target_path.exists():
                target_path.unlink()  # Clean up partial download
            return False
    
    def download_from_civitai(
        self,
        model_id: str,
        version_id: Optional[str],
        model_name: str,
        model_type: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Download model from CivitAI
        
        Args:
            model_id: CivitAI model ID
            version_id: Optional version ID (uses latest if not provided)
            model_name: Name to save the file as
            model_type: Type of model
            api_key: Optional CivitAI API key for access to restricted models
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already exists
            if self.model_exists(model_name, model_type):
                self.log(f"✅ Model already exists: {model_name}")
                return True
            
            # Construct CivitAI API URL
            base_url = f"https://civitai.com/api/v1/models/{model_id}"
            
            # Get model info
            response = requests.get(base_url)
            response.raise_for_status()
            model_info = response.json()
            
            # Find version
            if version_id:
                version = next((v for v in model_info['modelVersions'] if str(v['id']) == version_id), None)
            else:
                version = model_info['modelVersions'][0]  # Latest version
            
            if not version:
                self.log(f"❌ Version not found for model {model_id}")
                return False
            
            # Get download URL
            download_url = version['downloadUrl']
            if api_key:
                download_url += f"?token={api_key}"
            
            # Get hash if available
            expected_hash = None
            hash_type = 'sha256'
            if 'files' in version and version['files']:
                file_info = version['files'][0]
                if 'hashes' in file_info and 'SHA256' in file_info['hashes']:
                    expected_hash = file_info['hashes']['SHA256']
            
            self.log(f"📥 Downloading from CivitAI: {model_info['name']} v{version['name']}")
            
            return self.download_from_url(
                url=download_url,
                model_name=model_name,
                model_type=model_type,
                expected_hash=expected_hash,
                hash_type=hash_type
            )
            
        except Exception as e:
            self.log(f"❌ Failed to download from CivitAI: {e}")
            return False
    
    def download_model(
        self,
        model_name: str,
        model_type: str,
        source: Optional[str] = None,
        url: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        expected_hash: Optional[str] = None,
        hash_type: str = 'sha256',
        **kwargs
    ) -> bool:
        """
        Generic download method that routes to appropriate handler
        
        Args:
            model_name: Name to save the file as
            model_type: Type of model
            source: Source type ('huggingface', 'url', 'civitai')
            url: Direct URL (for 'url' source)
            repo_id: HuggingFace repo ID (for 'huggingface' source)
            filename: Filename in repo (for 'huggingface' source)
            expected_hash: Optional hash for verification
            hash_type: Hash algorithm
            **kwargs: Additional source-specific parameters
        
        Returns:
            True if successful, False otherwise
        """
        # Auto-detect source if not specified
        if not source:
            if repo_id:
                source = 'huggingface'
            elif url:
                if 'civitai.com' in url:
                    source = 'civitai'
                else:
                    source = 'url'
            else:
                self.log(f"❌ Cannot determine download source for {model_name}")
                return False
        
        # Route to appropriate handler
        if source == 'huggingface':
            if not repo_id or not filename:
                self.log(f"❌ HuggingFace download requires repo_id and filename")
                return False
            return self.download_from_huggingface(
                repo_id=repo_id,
                filename=filename,
                model_type=model_type,
                rename_to=kwargs.get('rename_to')
            )
        
        elif source == 'url':
            if not url:
                self.log(f"❌ URL download requires url parameter")
                return False
            return self.download_from_url(
                url=url,
                model_name=model_name,
                model_type=model_type,
                expected_hash=expected_hash,
                hash_type=hash_type
            )
        
        elif source == 'civitai':
            model_id = kwargs.get('model_id')
            version_id = kwargs.get('version_id')
            api_key = kwargs.get('api_key')
            
            if not model_id:
                self.log(f"❌ CivitAI download requires model_id")
                return False
            
            return self.download_from_civitai(
                model_id=model_id,
                version_id=version_id,
                model_name=model_name,
                model_type=model_type,
                api_key=api_key
            )
        
        else:
            self.log(f"❌ Unknown source: {source}")
            return False
    
    def validate_models(self, required_models: List[Dict]) -> Dict[str, bool]:
        """
        Validate that required models exist
        
        Args:
            required_models: List of model dicts with 'name' and 'model_type'
        
        Returns:
            Dictionary mapping model names to existence status
        """
        results = {}
        for model in required_models:
            name = model.get('name')
            model_type = model.get('model_type', 'checkpoints')
            if name:
                results[name] = self.model_exists(name, model_type)
        return results


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ComfyUI models")
    parser.add_argument('--comfyui-dir', required=True, help="Path to ComfyUI directory")
    parser.add_argument('--model-name', required=True, help="Model filename")
    parser.add_argument('--model-type', default='checkpoints', help="Model type (default: checkpoints)")
    parser.add_argument('--source', choices=['huggingface', 'url', 'civitai'], help="Download source")
    parser.add_argument('--url', help="Direct download URL")
    parser.add_argument('--repo-id', help="HuggingFace repository ID")
    parser.add_argument('--filename', help="Filename in HuggingFace repo")
    parser.add_argument('--hash', help="Expected file hash for verification")
    parser.add_argument('--hash-type', default='sha256', help="Hash algorithm (default: sha256)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.comfyui_dir)
    
    success = downloader.download_model(
        model_name=args.model_name,
        model_type=args.model_type,
        source=args.source,
        url=args.url,
        repo_id=args.repo_id,
        filename=args.filename,
        expected_hash=args.hash,
        hash_type=args.hash_type
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
