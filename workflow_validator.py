#!/usr/bin/env python3
"""
Workflow validation utilities for ComfyUI
Validates workflow requirements and checks model availability
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from workflow_parser import WorkflowParser, WorkflowDependencies
from download_models import ModelDownloader


@dataclass
class ValidationResult:
    """Result of workflow validation"""
    valid: bool
    missing_models: List[Dict] = field(default_factory=list)
    available_models: List[Dict] = field(default_factory=list)
    missing_custom_nodes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'valid': self.valid,
            'missing_models': self.missing_models,
            'available_models': self.available_models,
            'missing_custom_nodes': self.missing_custom_nodes,
            'warnings': self.warnings,
            'errors': self.errors
        }
    
    def print_summary(self):
        """Print human-readable summary"""
        print(f"\n{'='*60}")
        print(f"Workflow Validation Summary")
        print(f"{'='*60}\n")
        
        if self.valid:
            print("✅ Workflow is ready to run!")
        else:
            print("❌ Workflow has missing dependencies")
        
        print(f"\n📊 Models Status:")
        print(f"  ✅ Available: {len(self.available_models)}")
        print(f"  ❌ Missing: {len(self.missing_models)}")
        
        if self.missing_models:
            print(f"\n❌ Missing Models:")
            for model in self.missing_models:
                print(f"  - [{model['model_type']}] {model['name']}")
        
        if self.missing_custom_nodes:
            print(f"\n🔧 Missing Custom Nodes: {len(self.missing_custom_nodes)}")
            for node in self.missing_custom_nodes:
                print(f"  - {node}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"  - {error}")


class WorkflowValidator:
    """Validates ComfyUI workflows"""
    
    def __init__(self, comfyui_dir: Path):
        """
        Initialize validator
        
        Args:
            comfyui_dir: Path to ComfyUI installation directory
        """
        self.comfyui_dir = Path(comfyui_dir)
        self.downloader = ModelDownloader(self.comfyui_dir)
        
        # Custom nodes directory
        self.custom_nodes_dir = self.comfyui_dir / 'custom_nodes'
    
    def validate_workflow(
        self,
        workflow_path: str,
        check_custom_nodes: bool = True
    ) -> ValidationResult:
        """
        Validate a workflow file
        
        Args:
            workflow_path: Path to workflow JSON file
            check_custom_nodes: Whether to check for custom nodes (default: True)
        
        Returns:
            ValidationResult object
        """
        result = ValidationResult(valid=True)
        
        try:
            # Parse workflow
            parser = WorkflowParser(workflow_path)
            dependencies = parser.parse()
            
            # Validate models
            self._validate_models(dependencies, result)
            
            # Validate custom nodes if requested
            if check_custom_nodes:
                self._validate_custom_nodes(dependencies, result)
            
            # Overall validation status
            result.valid = (
                len(result.missing_models) == 0 and
                len(result.errors) == 0
            )
            
        except Exception as e:
            result.valid = False
            result.errors.append(f"Failed to parse workflow: {e}")
        
        return result
    
    def _validate_models(self, dependencies: WorkflowDependencies, result: ValidationResult):
        """Validate model dependencies"""
        for model in dependencies.models:
            model_path = self.downloader.get_model_path(model.name, model.model_type)
            
            model_info = {
                'name': model.name,
                'model_type': model.model_type,
                'path': str(model_path)
            }
            
            if model_path.exists():
                result.available_models.append(model_info)
            else:
                result.missing_models.append(model_info)
                
                # Add URL info if available
                if model.url:
                    model_info['url'] = model.url
                if model.hash:
                    model_info['hash'] = model.hash
    
    def _validate_custom_nodes(self, dependencies: WorkflowDependencies, result: ValidationResult):
        """Validate custom node dependencies"""
        if not self.custom_nodes_dir.exists():
            result.warnings.append("Custom nodes directory not found")
            return
        
        # Get list of installed custom nodes
        installed_nodes = self._get_installed_custom_nodes()
        
        for custom_node in dependencies.custom_nodes:
            # Simple heuristic: check if any installed node might provide this node type
            found = False
            
            for installed in installed_nodes:
                # Check if node type matches or is contained in installed node name
                node_type_lower = custom_node.node_type.lower()
                installed_lower = installed.lower()
                
                # Direct match or partial match
                if node_type_lower in installed_lower or installed_lower in node_type_lower:
                    found = True
                    break
            
            if not found:
                result.missing_custom_nodes.append(custom_node.node_type)
    
    def _get_installed_custom_nodes(self) -> List[str]:
        """Get list of installed custom node directories"""
        if not self.custom_nodes_dir.exists():
            return []
        
        installed = []
        for item in self.custom_nodes_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                installed.append(item.name)
        
        return installed
    
    def check_model_exists(self, model_name: str, model_type: str) -> bool:
        """
        Check if a specific model exists
        
        Args:
            model_name: Name of the model file
            model_type: Type of model (checkpoints, vae, etc.)
        
        Returns:
            True if model exists, False otherwise
        """
        return self.downloader.model_exists(model_name, model_type)
    
    def get_missing_models_report(self, workflow_path: str) -> Dict:
        """
        Generate a report of missing models for a workflow
        
        Args:
            workflow_path: Path to workflow JSON file
        
        Returns:
            Dictionary with missing models information
        """
        result = self.validate_workflow(workflow_path, check_custom_nodes=False)
        
        return {
            'workflow': workflow_path,
            'total_models': len(result.available_models) + len(result.missing_models),
            'available': len(result.available_models),
            'missing': len(result.missing_models),
            'missing_models': result.missing_models
        }


def validate_multiple_workflows(
    comfyui_dir: Path,
    workflow_paths: List[str]
) -> Dict[str, ValidationResult]:
    """
    Validate multiple workflows
    
    Args:
        comfyui_dir: Path to ComfyUI directory
        workflow_paths: List of workflow file paths
    
    Returns:
        Dictionary mapping workflow paths to validation results
    """
    validator = WorkflowValidator(comfyui_dir)
    results = {}
    
    for workflow_path in workflow_paths:
        try:
            result = validator.validate_workflow(workflow_path)
            results[workflow_path] = result
        except Exception as e:
            result = ValidationResult(valid=False)
            result.errors.append(f"Validation failed: {e}")
            results[workflow_path] = result
    
    return results


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ComfyUI workflow dependencies")
    parser.add_argument('workflow', help="Path to ComfyUI workflow JSON file")
    parser.add_argument(
        '--comfyui-dir',
        required=True,
        help="Path to ComfyUI installation directory"
    )
    parser.add_argument(
        '--no-custom-nodes',
        action='store_true',
        help="Skip custom node validation"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        validator = WorkflowValidator(args.comfyui_dir)
        result = validator.validate_workflow(
            args.workflow,
            check_custom_nodes=not args.no_custom_nodes
        )
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            result.print_summary()
        
        return 0 if result.valid else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
