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

def prepare_data_paths(config: Dict) -> Dict:
    """Resolve path templates in config, especially with environment variables.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Config with resolved paths
    """
    # Resolve DATA_DIR environment variable in data paths
    if 'data' in config:
        data_dir = config['data']['base_dir']
        if '${' in data_dir:
            # Extract environment variable name and default
            var_name, default = data_dir[2:-1].split(':', 1)
            data_dir = os.environ.get(var_name, default)
            config['data']['base_dir'] = data_dir
        
        # Update paths to be absolute
        for key in ['train_csv', 'test_csv', 'depths_csv']:
            if key in config['data']:
                config['data'][key] = os.path.join(data_dir, config['data'][key])
                
        for key in ['train_images', 'train_masks', 'test_images']:
            if key in config['data']:
                config['data'][key] = os.path.join(data_dir, config['data'][key])
    
    # Resolve checkpoint paths for refinement 
    if 'refinement' in config and 'main_model_checkpoint' in config['refinement']:
        main_checkpoint = config['refinement']['main_model_checkpoint']
        if '${' in main_checkpoint:
            # Extract environment variable name and default
            var_name, default = main_checkpoint[2:-1].split(':', 1)
            main_checkpoint = os.environ.get(var_name, default)
            config['refinement']['main_model_checkpoint'] = main_checkpoint
    
    return config
