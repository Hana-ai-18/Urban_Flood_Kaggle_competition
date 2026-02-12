"""
Helper utilities
"""

import torch
import random
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any
import json


def set_seed(seed: int):
    """Set random seed cho reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration từ YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   save_path: str):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Metrics dictionary
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str,
                   device: str = 'cuda') -> Dict:
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    Get torch device
    
    Args:
        device_name: 'cuda' or 'cpu'
        
    Returns:
        torch.device
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop
        
        Args:
            score: Current score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """Compute and store average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update with new value
        
        Args:
            val: New value
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_directory(path: str):
    """Create directory if not exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_predictions(predictions: Dict,
                    save_path: str,
                    format: str = 'csv'):
    """
    Save predictions to file
    
    Args:
        predictions: Dictionary of predictions
        save_path: Path to save
        format: 'csv' or 'parquet'
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    if format == 'csv':
        df.to_csv(save_path, index=False)
    elif format == 'parquet':
        df.to_parquet(save_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Predictions saved to {save_path}")


def format_time(seconds: float) -> str:
    """Format seconds to readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Logger:
    """Simple logger"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        
        if log_file:
            # Create log directory
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str):
        """Log message"""
        print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics dictionary"""
        message = prefix
        for key, value in metrics.items():
            if isinstance(value, float):
                message += f"{key}: {value:.6f} | "
            else:
                message += f"{key}: {value} | "
        
        self.log(message)