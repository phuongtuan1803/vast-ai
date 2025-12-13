#!/usr/bin/env python3
"""
ComfyUI Workflow Parser
Parses ComfyUI workflow JSON files to extract model and custom node dependencies
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelDependency:
    """Represents a model dependency from workflow"""
    name: str
    model_type: str  # checkpoints, vae, clip, lora, controlnet, etc.
    url: Optional[str] = None
    hash: Optional[str] = None
    hash_type: Optional[str] = None
    directory: Optional[str] = None
    
    def __hash__(self):
        return hash((self.name, self.model_type))
    
    def __eq__(self, other):
        if not isinstance(other, ModelDependency):
            return False
        return self.name == other.name and self.model_type == other.model_type


@dataclass
class CustomNodeDependency:
    """Represents a custom node dependency"""
    node_type: str
    reference: Optional[str] = None  # git URL or custom node name
    
    def __hash__(self):
        return hash(self.node_type)
    
    def __eq__(self, other):
        if not isinstance(other, CustomNodeDependency):
            return False
        return self.node_type == other.node_type


@dataclass
class WorkflowDependencies:
    """Container for all workflow dependencies"""
    models: Set[ModelDependency] = field(default_factory=set)
    custom_nodes: Set[CustomNodeDependency] = field(default_factory=set)
    
    def add_model(self, dependency: ModelDependency):
        """Add model dependency"""
        self.models.add(dependency)
    
    def add_custom_node(self, dependency: CustomNodeDependency):
        """Add custom node dependency"""
        self.custom_nodes.add(dependency)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'models': [
                {
                    'name': m.name,
                    'model_type': m.model_type,
                    'url': m.url,
                    'hash': m.hash,
                    'hash_type': m.hash_type,
                    'directory': m.directory
                }
                for m in self.models
            ],
            'custom_nodes': [
                {
                    'node_type': n.node_type,
                    'reference': n.reference
                }
                for n in self.custom_nodes
            ]
        }


class WorkflowParser:
    """Parser for ComfyUI workflow JSON files"""
    
    # Mapping of node types to model types and widget indices
    NODE_TYPE_MAPPING = {
        'CheckpointLoaderSimple': ('checkpoints', 0),
        'CheckpointLoader': ('checkpoints', 0),
        'CheckpointLoaderSimpleWithNoiseSelect': ('checkpoints', 0),
        'unCLIPCheckpointLoader': ('checkpoints', 0),
        'VAELoader': ('vae', 0),
        'VAELoaderNF4': ('vae', 0),
        'CLIPLoader': ('clip', 0),
        'DualCLIPLoader': ('clip', [0, 1]),  # Multiple CLIP models
        'TripleCLIPLoader': ('clip', [0, 1, 2]),  # Multiple CLIP models
        'ControlNetLoader': ('controlnet', 0),
        'DiffControlNetLoader': ('controlnet', 0),
        'LoraLoader': ('loras', 0),
        'LoraLoaderModelOnly': ('loras', 0),
        'UNETLoader': ('unet', 0),
        'StyleModelLoader': ('style_models', 0),
        'CLIPVisionLoader': ('clip_vision', 0),
        'UpscaleModelLoader': ('upscale_models', 0),
        'GLIGEN Loader': ('gligen', 0),
        'hypernetwork_loader': ('hypernetworks', 0),
        'PhotoMakerLoader': ('photomaker', 0),
        'IPAdapterModelLoader': ('ipadapter', 0),
    }
    
    # Known custom node prefixes (nodes not in base ComfyUI)
    KNOWN_CUSTOM_NODE_PREFIXES = [
        'CR ',  # ComfyRoll
        'WAS ',  # WAS Node Suite
        'ImpactPack',
        'Inspire',
        'Efficiency',
        'Anything Everywhere',
        'AnimateDiff',
        'ControlNet',
        'IPAdapter',
        'PhotoMaker',
        'InstantID',
        'FaceRestore',
        'Roop',
        'ReActor',
        'Ultimate SD Upscale',
    ]
    
    def __init__(self, workflow_path: str):
        """
        Initialize parser with workflow file path
        
        Args:
            workflow_path: Path to ComfyUI workflow JSON file
        """
        self.workflow_path = Path(workflow_path)
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
        with open(self.workflow_path, 'r', encoding='utf-8') as f:
            self.workflow_data = json.load(f)
    
    def parse(self) -> WorkflowDependencies:
        """
        Parse workflow and extract all dependencies
        
        Returns:
            WorkflowDependencies object containing models and custom nodes
        """
        dependencies = WorkflowDependencies()
        
        # Try parsing models array (v1.0 workflow spec)
        if 'models' in self.workflow_data and isinstance(self.workflow_data['models'], list):
            self._parse_models_array(self.workflow_data['models'], dependencies)
        
        # Parse nodes for model dependencies
        if 'nodes' in self.workflow_data:
            self._parse_nodes(self.workflow_data['nodes'], dependencies)
        
        # Also check 'extra' field which some workflows use
        if 'extra' in self.workflow_data and 'models' in self.workflow_data['extra']:
            self._parse_models_array(self.workflow_data['extra']['models'], dependencies)
        
        return dependencies
    
    def _parse_models_array(self, models: List[Dict], dependencies: WorkflowDependencies):
        """Parse models array from workflow (v1.0 spec)"""
        for model in models:
            if not isinstance(model, dict):
                continue
            
            name = model.get('name')
            if not name:
                continue
            
            # Determine model type from directory or name
            directory = model.get('directory', '')
            model_type = self._infer_model_type_from_directory(directory)
            
            dep = ModelDependency(
                name=name,
                model_type=model_type,
                url=model.get('url'),
                hash=model.get('hash'),
                hash_type=model.get('hash_type'),
                directory=directory
            )
            dependencies.add_model(dep)
    
    def _parse_nodes(self, nodes: List[Dict], dependencies: WorkflowDependencies):
        """Parse nodes array to extract model and custom node dependencies"""
        for node in nodes:
            if not isinstance(node, dict):
                continue
            
            node_type = node.get('type')
            if not node_type:
                continue
            
            # Check if this is a known loader node
            if node_type in self.NODE_TYPE_MAPPING:
                self._extract_model_from_node(node, dependencies)
            
            # Check if this might be a custom node
            if self._is_custom_node(node_type):
                custom_node = CustomNodeDependency(node_type=node_type)
                dependencies.add_custom_node(custom_node)
    
    def _extract_model_from_node(self, node: Dict, dependencies: WorkflowDependencies):
        """Extract model dependency from a loader node"""
        node_type = node.get('type')
        if node_type not in self.NODE_TYPE_MAPPING:
            return
        
        model_type, widget_index = self.NODE_TYPE_MAPPING[node_type]
        widgets_values = node.get('widgets_values', [])
        
        if not widgets_values:
            return
        
        # Handle single or multiple indices
        if isinstance(widget_index, list):
            # Multiple models (e.g., DualCLIPLoader)
            for idx in widget_index:
                if idx < len(widgets_values):
                    model_name = widgets_values[idx]
                    if model_name and isinstance(model_name, str):
                        dep = ModelDependency(
                            name=model_name,
                            model_type=model_type
                        )
                        dependencies.add_model(dep)
        else:
            # Single model
            if widget_index < len(widgets_values):
                model_name = widgets_values[widget_index]
                if model_name and isinstance(model_name, str):
                    dep = ModelDependency(
                        name=model_name,
                        model_type=model_type
                    )
                    dependencies.add_model(dep)
    
    def _is_custom_node(self, node_type: str) -> bool:
        """Check if node type is likely a custom node"""
        # Check known prefixes
        for prefix in self.KNOWN_CUSTOM_NODE_PREFIXES:
            if node_type.startswith(prefix):
                return True
        
        # Check if it contains certain patterns
        custom_indicators = [
            '++',  # ComfyUI++
            'Advanced',
            'Ultimate',
            'Pack',
            'Suite',
        ]
        
        for indicator in custom_indicators:
            if indicator in node_type:
                return True
        
        # If not in standard node mapping and has unusual capitalization, likely custom
        if node_type not in self.NODE_TYPE_MAPPING:
            # Check for mixed case or special patterns
            if any(c.isupper() for c in node_type[1:]) or '_' in node_type:
                return True
        
        return False
    
    def _infer_model_type_from_directory(self, directory: str) -> str:
        """Infer model type from directory path"""
        directory = directory.lower()
        
        type_mappings = {
            'checkpoint': 'checkpoints',
            'vae': 'vae',
            'clip': 'clip',
            'lora': 'loras',
            'controlnet': 'controlnet',
            'unet': 'unet',
            'upscale': 'upscale_models',
            'gligen': 'gligen',
            'hypernetwork': 'hypernetworks',
            'style': 'style_models',
            'clip_vision': 'clip_vision',
            'photomaker': 'photomaker',
            'ipadapter': 'ipadapter',
        }
        
        for key, value in type_mappings.items():
            if key in directory:
                return value
        
        return 'checkpoints'  # Default to checkpoints


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse ComfyUI workflow and extract dependencies")
    parser.add_argument('workflow', help="Path to ComfyUI workflow JSON file")
    parser.add_argument('--output', '-o', help="Output JSON file for dependencies")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Parse workflow
        workflow_parser = WorkflowParser(args.workflow)
        dependencies = workflow_parser.parse()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Workflow: {args.workflow}")
        print(f"{'='*60}\n")
        
        print(f"📦 Models Found: {len(dependencies.models)}")
        for model in sorted(dependencies.models, key=lambda m: (m.model_type, m.name)):
            print(f"  - [{model.model_type}] {model.name}")
            if args.verbose and model.url:
                print(f"    URL: {model.url}")
            if args.verbose and model.directory:
                print(f"    Dir: {model.directory}")
        
        print(f"\n🔧 Custom Nodes Found: {len(dependencies.custom_nodes)}")
        for node in sorted(dependencies.custom_nodes, key=lambda n: n.node_type):
            print(f"  - {node.node_type}")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dependencies.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\n✅ Dependencies saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
