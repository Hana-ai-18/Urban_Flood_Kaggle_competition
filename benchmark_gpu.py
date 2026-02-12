"""
Benchmark GPU performance for the model
"""

import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path

from models import SpatioTemporalGNN
from utils.gpu_utils import (
    print_gpu_info, get_optimal_device, setup_cudnn_benchmark,
    setup_mixed_precision, enable_tf32, GPUMonitor, clear_gpu_cache
)
from utils import load_config


def create_dummy_batch(batch_size, seq_len, num_1d_nodes, num_2d_nodes,
                       static_1d_dim, static_2d_dim,
                       dynamic_1d_dim, dynamic_2d_dim,
                       device):
    """Create dummy batch for benchmarking"""
    
    class DummyBatch:
        pass
    
    batch = DummyBatch()
    
    # Static features (shared across batch)
    batch.x_1d_static = torch.randn(num_1d_nodes, static_1d_dim).to(device)
    batch.x_2d_static = torch.randn(num_2d_nodes, static_2d_dim).to(device)
    
    # Dynamic features
    batch.x_1d_dynamic = torch.randn(batch_size, seq_len, num_1d_nodes, dynamic_1d_dim).to(device)
    batch.x_2d_dynamic = torch.randn(batch_size, seq_len, num_2d_nodes, dynamic_2d_dim).to(device)
    
    # Edge indices
    # Create random edges
    num_1d_edges = num_1d_nodes * 2
    num_2d_edges = num_2d_nodes * 3
    
    batch.edge_index_1d = torch.randint(0, num_1d_nodes, (2, num_1d_edges)).to(device)
    batch.edge_index_2d = torch.randint(0, num_2d_nodes, (2, num_2d_edges)).to(device)
    batch.edge_index_1d2d = torch.randint(0, min(num_1d_nodes, num_2d_nodes), (2, 10)).to(device)
    
    # Targets
    batch.y_1d = torch.randn(batch_size, num_1d_nodes).to(device)
    batch.y_2d = torch.randn(batch_size, num_2d_nodes).to(device)
    
    batch.model_id = torch.tensor([1] * batch_size).to(device)
    
    return batch


def benchmark_forward_pass(model, batch, num_iterations=100, warmup=10):
    """
    Benchmark forward pass
    
    Args:
        model: Model to benchmark
        batch: Input batch
        num_iterations: Number of iterations
        warmup: Number of warmup iterations
        
    Returns:
        Average time per iteration (ms)
    """
    model.eval()
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_time, times


def benchmark_backward_pass(model, batch, criterion, num_iterations=50, warmup=5):
    """
    Benchmark forward + backward pass
    
    Args:
        model: Model to benchmark
        batch: Input batch
        criterion: Loss criterion
        num_iterations: Number of iterations
        warmup: Warmup iterations
        
    Returns:
        Average time per iteration (ms)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        optimizer.zero_grad()
        pred_1d, pred_2d = model(batch)
        loss = criterion(pred_1d, pred_2d, batch.y_1d, batch.y_2d, batch.model_id[0].item())
        loss.backward()
        optimizer.step()
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        
        optimizer.zero_grad()
        pred_1d, pred_2d = model(batch)
        loss = criterion(pred_1d, pred_2d, batch.y_1d, batch.y_2d, batch.model_id[0].item())
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append((end - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_time, times


def run_benchmark(config_path: str, gpu_id: int = None):
    """
    Run complete benchmark
    
    Args:
        config_path: Path to config file
        gpu_id: GPU ID to use
    """
    print("=" * 80)
    print("GPU PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Load config
    config = load_config(config_path)
    
    # GPU info
    print_gpu_info()
    
    # Setup device
    device = get_optimal_device('cuda', gpu_id)
    
    if device.type == 'cpu':
        print("\nWarning: Running on CPU. Benchmark may be slow.")
        return
    
    # GPU optimizations
    print("\nSetting up GPU optimizations...")
    setup_cudnn_benchmark(enable=True)
    enable_tf32()
    use_amp = setup_mixed_precision()
    
    # Clear cache
    clear_gpu_cache()
    
    # Model configuration
    model_config = {
        'static_1d_dim': config['model']['node_1d_static_dim'],
        'static_2d_dim': config['model']['node_2d_static_dim'],
        'dynamic_1d_dim': config['model']['node_1d_dynamic_dim'],
        'dynamic_2d_dim': config['model']['node_2d_dynamic_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'num_gnn_layers': config['model']['num_gnn_layers'],
        'num_temporal_layers': config['model']['num_temporal_layers'],
        'gnn_type': config['model']['gnn_type'],
        'dropout': config['model']['dropout']
    }
    
    # Create model
    print("\nCreating model...")
    model = SpatioTemporalGNN(**model_config).to(device)
    
    from utils.helpers import count_parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # GPU monitor
    gpu_monitor = GPUMonitor(device)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    seq_len = config['model']['sequence_length']
    num_1d_nodes = 50  # Example
    num_2d_nodes = 100  # Example
    
    results = []
    
    for batch_size in batch_sizes:
        print("\n" + "=" * 80)
        print(f"BATCH SIZE: {batch_size}")
        print("=" * 80)
        
        # Create dummy batch
        print("\nCreating dummy batch...")
        batch = create_dummy_batch(
            batch_size, seq_len, num_1d_nodes, num_2d_nodes,
            model_config['static_1d_dim'], model_config['static_2d_dim'],
            model_config['dynamic_1d_dim'], model_config['dynamic_2d_dim'],
            device
        )
        
        print(f"Batch shape:")
        print(f"  1D dynamic: {batch.x_1d_dynamic.shape}")
        print(f"  2D dynamic: {batch.x_2d_dynamic.shape}")
        
        # Memory before
        print("\nGPU Memory before benchmark:")
        gpu_monitor.print_stats()
        
        # Benchmark forward pass
        print("\n--- Forward Pass Benchmark ---")
        avg_forward, std_forward, _ = benchmark_forward_pass(model, batch, num_iterations=100)
        print(f"Average time: {avg_forward:.2f} ± {std_forward:.2f} ms")
        print(f"Throughput: {1000.0 / avg_forward:.2f} samples/sec")
        
        # Memory after forward
        print("\nGPU Memory after forward:")
        gpu_monitor.print_stats()
        
        # Benchmark backward pass
        print("\n--- Forward + Backward Pass Benchmark ---")
        from utils import StandardizedRMSE
        std_devs = {
            (1, 1): 16.877747,
            (1, 2): 14.378797
        }
        criterion = StandardizedRMSE(std_devs)
        
        avg_backward, std_backward, _ = benchmark_backward_pass(
            model, batch, criterion, num_iterations=50
        )
        print(f"Average time: {avg_backward:.2f} ± {std_backward:.2f} ms")
        print(f"Throughput: {1000.0 / avg_backward:.2f} samples/sec")
        
        # Memory after backward
        print("\nGPU Memory after backward:")
        gpu_monitor.print_stats()
        
        # Store results
        results.append({
            'batch_size': batch_size,
            'forward_time_ms': avg_forward,
            'forward_std_ms': std_forward,
            'backward_time_ms': avg_backward,
            'backward_std_ms': std_backward,
            'forward_throughput': 1000.0 / avg_forward,
            'backward_throughput': 1000.0 / avg_backward
        })
        
        # Clear cache between tests
        clear_gpu_cache()
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Batch Size':<12} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Forward FPS':<15} {'Backward FPS':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['batch_size']:<12} "
              f"{result['forward_time_ms']:>6.2f} ± {result['forward_std_ms']:<5.2f} "
              f"{result['backward_time_ms']:>6.2f} ± {result['backward_std_ms']:<5.2f} "
              f"{result['forward_throughput']:>12.2f} "
              f"{result['backward_throughput']:>14.2f}")
    
    print("\n" + "=" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    # Find best batch size for throughput
    best_batch = max(results, key=lambda x: x['backward_throughput'] * x['batch_size'])
    print(f"  Best batch size for training: {best_batch['batch_size']}")
    print(f"    (Throughput: {best_batch['backward_throughput'] * best_batch['batch_size']:.2f} samples/sec)")
    
    if use_amp:
        print(f"  Mixed precision training: ENABLED (faster training)")
    else:
        print(f"  Mixed precision training: NOT AVAILABLE (GPU too old)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark GPU performance')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    run_benchmark(args.config, args.gpu)