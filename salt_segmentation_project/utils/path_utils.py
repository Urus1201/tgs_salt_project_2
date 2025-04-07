import os
import re
from pathlib import Path
from typing import Union, Dict, Any

def resolve_path(path_str: str, config: Dict[str, Any] = None) -> str:
    """
    Resolve a path string that may contain environment variable references.
    Format: ${ENV_VAR:default_value}
    
    Args:
        path_str: Path string that may contain environment variable references
        config: Optional configuration dictionary for additional substitutions
        
    Returns:
        Resolved path string
    """
    # Handle environment variable with default value format: ${ENV_VAR:default}
    def replace_env_var(match):
        env_var, default = match.group(1).split(':', 1) if ':' in match.group(1) else (match.group(1), '')
        return os.environ.get(env_var, default)
    
    if isinstance(path_str, str):
        # Replace environment variables
        path_str = re.sub(r'\${([^}]+)}', replace_env_var, path_str)
    
    return path_str

def get_absolute_path(relative_path: str, base_dir: str) -> str:
    """
    Combine base directory with relative path to get absolute path.
    
    Args:
        relative_path: Relative path
        base_dir: Base directory
        
    Returns:
        Absolute path
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(base_dir, relative_path)

def prepare_data_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with resolved absolute paths for data files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Resolve base directory
    base_dir = resolve_path(config['data']['base_dir'])
    
    # Update paths in config
    for key in ['train_csv', 'test_csv', 'depths_csv', 'train_images', 'train_masks', 'test_images']:
        if key in config['data']:
            config['data'][key] = get_absolute_path(config['data'][key], base_dir)
    
    return config
