"""
GPU utilities for efficient training and inference
"""

import torch
import torch.nn as nn
from typing import Optional, List
import subprocess
import os


def get_gpu_info():
    """
    Get detailed GPU information
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'devices': []
        }
    
    gpu_info = {
        'available': True,
        'count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'capability': torch.cuda.get_device_capability(i),
            'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
        }
        
        # Current memory usage
        if torch.cuda.is_initialized():
            device_info['allocated_memory'] = torch.cuda.memory_allocated(i) / (1024**3)
            device_info['cached_memory'] = torch.cuda.memory_reserved(i) / (1024**3)
        
        gpu_info['devices'].append(device_info)
    
    return gpu_info


def print_gpu_info():
    """Print GPU information in readable format"""
    info = get_gpu_info()
    
    if not info['available']:
        print("=" * 60)
        print("GPU: Not available")
        print("=" * 60)
        return
    
    print("=" * 60)
    print(f"GPU Information")
    print("=" * 60)
    print(f"Number of GPUs: {info['count']}")
    
    for device in info['devices']:
        print(f"\nGPU {device['id']}: {device['name']}")
        print(f"  Compute Capability: {device['capability']}")
        print(f"  Total Memory: {device['total_memory']:.2f} GB")
        
        if 'allocated_memory' in device:
            print(f"  Allocated Memory: {device['allocated_memory']:.2f} GB")
            print(f"  Cached Memory: {device['cached_memory']:.2f} GB")
            free_memory = device['total_memory'] - device['allocated_memory']
            print(f"  Free Memory: {free_memory:.2f} GB")
    
    print("=" * 60)


def get_optimal_device(device_name: Optional[str] = None, 
                       gpu_id: Optional[int] = None) -> torch.device:
    """
    Get optimal device for computation
    
    Args:
        device_name: 'cuda', 'cpu', or None for auto
        gpu_id: Specific GPU ID to use
        
    Returns:
        torch.device
    """
    if device_name == 'cpu':
        device = torch.device('cpu')
        print("Using CPU")
        return device
    
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU")
        return torch.device('cpu')
    
    if gpu_id is not None:
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"GPU {gpu_id} not available. Using default GPU")
            device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    return device


def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


def get_memory_summary(device: torch.device) -> dict:
    """
    Get memory summary for device
    
    Args:
        device: torch.device
        
    Returns:
        Dictionary with memory info
    """
    if device.type != 'cuda':
        return {'device': 'cpu', 'available': False}
    
    gpu_id = device.index if device.index is not None else 0
    
    return {
        'device': f'cuda:{gpu_id}',
        'allocated': torch.cuda.memory_allocated(gpu_id) / (1024**3),
        'reserved': torch.cuda.memory_reserved(gpu_id) / (1024**3),
        'max_allocated': torch.cuda.max_memory_allocated(gpu_id) / (1024**3),
        'max_reserved': torch.cuda.max_memory_reserved(gpu_id) / (1024**3)
    }


def print_memory_summary(device: torch.device):
    """Print memory summary in readable format"""
    summary = get_memory_summary(device)
    
    if not summary.get('available', True):
        return
    
    print(f"\nGPU Memory Summary ({summary['device']}):")
    print(f"  Allocated: {summary['allocated']:.2f} GB")
    print(f"  Reserved:  {summary['reserved']:.2f} GB")
    print(f"  Max Allocated: {summary['max_allocated']:.2f} GB")
    print(f"  Max Reserved:  {summary['max_reserved']:.2f} GB")


def setup_cudnn_benchmark(enable: bool = True):
    """
    Setup cuDNN benchmark mode for optimal performance
    
    Args:
        enable: Enable benchmark mode
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        torch.backends.cudnn.enabled = True
        
        if enable:
            print("cuDNN benchmark mode enabled (faster but non-deterministic)")
        else:
            print("cuDNN benchmark mode disabled (slower but deterministic)")


def setup_mixed_precision() -> bool:
    """
    Check if mixed precision training is available
    
    Returns:
        True if available
    """
    if not torch.cuda.is_available():
        return False
    
    # Check if GPU supports mixed precision
    capability = torch.cuda.get_device_capability()
    
    # Mixed precision requires compute capability >= 7.0 (Volta and newer)
    if capability[0] >= 7:
        print(f"Mixed precision training available (Compute {capability[0]}.{capability[1]})")
        return True
    else:
        print(f"Mixed precision not recommended (Compute {capability[0]}.{capability[1]} < 7.0)")
        return False


def get_gpu_utilization():
    """
    Get GPU utilization using nvidia-smi
    
    Returns:
        List of dictionaries with GPU utilization info
    """
    if not torch.cuda.is_available():
        return []
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        utilization = []
        for line in result.stdout.strip().split('\n'):
            values = line.split(', ')
            if len(values) == 4:
                utilization.append({
                    'id': int(values[0]),
                    'gpu_util': float(values[1]),
                    'memory_util': float(values[2]),
                    'temperature': float(values[3])
                })
        
        return utilization
    
    except Exception as e:
        print(f"Could not get GPU utilization: {e}")
        return []


def print_gpu_utilization():
    """Print GPU utilization"""
    utilization = get_gpu_utilization()
    
    if not utilization:
        return
    
    print("\nGPU Utilization:")
    for gpu in utilization:
        print(f"  GPU {gpu['id']}: {gpu['gpu_util']:.1f}% | "
              f"Memory: {gpu['memory_util']:.1f}% | "
              f"Temp: {gpu['temperature']:.0f}°C")


def select_best_gpu() -> int:
    """
    Select GPU with most free memory
    
    Returns:
        GPU ID with most free memory
    """
    if not torch.cuda.is_available():
        return -1
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        return 0
    
    # Get free memory for each GPU
    free_memory = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(i)
        free = total - allocated
        free_memory.append((i, free))
    
    # Select GPU with most free memory
    best_gpu = max(free_memory, key=lambda x: x[1])
    
    print(f"Selected GPU {best_gpu[0]} with {best_gpu[1]/(1024**3):.2f} GB free memory")
    
    return best_gpu[0]


def setup_multi_gpu(model: nn.Module, gpu_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Setup model for multi-GPU training using DataParallel
    
    Args:
        model: Model to wrap
        gpu_ids: List of GPU IDs to use (None for all)
        
    Returns:
        Wrapped model
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot use multi-GPU")
        return model
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        print(f"Only {num_gpus} GPU available. Multi-GPU not enabled")
        return model
    
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    
    print(f"Using DataParallel with GPUs: {gpu_ids}")
    
    model = nn.DataParallel(model, device_ids=gpu_ids)
    
    return model


def enable_tf32():
    """
    Enable TF32 for faster training on Ampere GPUs
    TF32 provides better performance with minimal accuracy impact
    """
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        
        # TF32 is available on Ampere (8.x) and newer
        if capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for faster training (Ampere or newer GPU)")
        else:
            print(f"TF32 not available (requires Ampere GPU, got Compute {capability[0]}.{capability[1]})")


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference
    
    Args:
        model: Model to optimize
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Use torch.jit if possible for faster inference
    if torch.cuda.is_available():
        try:
            print("Attempting to optimize model with torch.jit...")
            # This may not work for all models
            # model = torch.jit.script(model)
            # print("Model optimized with torch.jit")
        except Exception as e:
            print(f"Could not optimize with torch.jit: {e}")
    
    return model


class GPUMonitor:
    """Monitor GPU usage during training"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == 'cuda'
        
        if self.enabled:
            self.gpu_id = device.index if device.index is not None else 0
            torch.cuda.reset_peak_memory_stats(self.gpu_id)
    
    def get_stats(self) -> dict:
        """Get current GPU stats"""
        if not self.enabled:
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated(self.gpu_id) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.gpu_id) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.gpu_id) / (1024**3),
        }
    
    def print_stats(self):
        """Print current stats"""
        if not self.enabled:
            return
        
        stats = self.get_stats()
        print(f"GPU Memory: {stats['allocated_gb']:.2f} GB / "
              f"Peak: {stats['max_allocated_gb']:.2f} GB")
    
    def reset_peak_stats(self):
        """Reset peak memory stats"""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.gpu_id)


def set_gpu_growth(enable: bool = True):
    """
    Set GPU memory growth (PyTorch doesn't have direct equivalent to TF's GPU growth)
    This is a placeholder for potential future functionality
    
    Args:
        enable: Enable memory growth
    """
    if not torch.cuda.is_available():
        return
    
    if enable:
        # PyTorch's caching allocator already does something similar
        print("PyTorch's caching allocator handles memory efficiently by default")
    else:
        # Can disable caching allocator if needed
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        print("CUDA memory caching disabled")