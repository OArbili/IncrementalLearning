import numpy as np
import random
import os
import optuna

# Global seed value
SEED = 42
# Add at the beginning of your script
os.environ['PYTHONHASHSEED'] = str(SEED)

def set_all_seeds():
    """Set all seeds for reproducibility"""
    # Python's built-in random
    random.seed(SEED)
    
    # NumPy
    np.random.seed(SEED)
    
    # Optuna (for hyperparameter optimization)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Set environment variables related to Python hash seed
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # TensorFlow (if you're using it)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
        tf.compat.v1.set_random_seed(SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except:
        pass
    
    print("All random seeds have been set to:", SEED)

def detect_device():
    """Detect if CUDA is available, fallback to CPU."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'cuda'
    except FileNotFoundError:
        pass
    return 'cpu'

DEVICE = detect_device()