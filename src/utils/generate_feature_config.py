"""Generate feature configuration from code.

This script extracts configuration parameters from the feature extraction modules
and generates a YAML configuration file, using the existing config as a base template.
"""

import yaml
from pathlib import Path
import importlib
import inspect
from typing import Dict, Any

def load_base_config() -> Dict[str, Any]:
    """Load the base configuration template.
    
    Returns:
        Dictionary containing the base configuration
    """
    config_path = Path('configs/feature_config.yaml')
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def extract_config_from_module(module_name: str) -> Dict[str, Any]:
    """Extract configuration from a module.
    
    Args:
        module_name: Name of the module to extract config from
        
    Returns:
        Dictionary of configuration parameters
    """
    # Import module
    module = importlib.import_module(module_name)
    
    # Find configuration dictionaries
    config = {}
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, dict) and name.endswith('_config'):
            config[name] = obj
    
    return config

def merge_configs(base_config: Dict[str, Any], code_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with code-generated configs.
    
    Args:
        base_config: Base configuration template
        code_configs: Configuration extracted from code
        
    Returns:
        Merged configuration dictionary
    """
    # Start with base config
    merged = base_config.copy()
    
    # Update with code configs
    for module_name, module_config in code_configs.items():
        if module_name not in merged:
            merged[module_name] = {}
        
        # Deep merge dictionaries
        for key, value in module_config.items():
            if isinstance(value, dict) and key in merged[module_name]:
                merged[module_name][key].update(value)
            else:
                merged[module_name][key] = value
    
    return merged

def generate_feature_config():
    """Generate feature configuration file."""
    # Load base config
    base_config = load_base_config()
    
    # Extract configs from modules
    code_configs = {
        'symptom_matcher': extract_config_from_module('src.features.symptom_matcher'),
        'temporal_parser': extract_config_from_module('src.features.temporal_parser'),
        'emoji_processor': extract_config_from_module('src.features.emoji_processor'),
        'response_attribution': extract_config_from_module('src.features.response_attribution')
    }
    
    # Merge configs
    merged_config = merge_configs(base_config, code_configs)
    
    # Create output directory if it doesn't exist
    output_dir = Path('configs')
    output_dir.mkdir(exist_ok=True)
    
    # Write config file
    with open(output_dir / 'feature_config.yaml', 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

if __name__ == '__main__':
    generate_feature_config() 