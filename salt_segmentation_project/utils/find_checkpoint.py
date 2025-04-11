import os

def find_checkpoint(config):
    """
    Find checkpoint based on a simple identifier.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Full path to the checkpoint file
    """
    save_dir = config['training']['save_dir']
    subdirs = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    latest_dir = max(subdirs, key=os.path.getctime)
    best_path = os.path.join(latest_dir, "model_best.pth")
    if os.path.exists(best_path):
        return best_path
    else:
        raise FileNotFoundError(f"No model found.")
    
    