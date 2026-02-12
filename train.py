"""
Training script for UrbanFloodBench
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path

from data import create_dataloaders
from models import SpatioTemporalGNN
from utils import (
    set_seed, load_config, save_checkpoint, get_device,
    StandardizedRMSE, compute_metrics, EarlyStopping,
    AverageMeter, Logger, create_directory,
    print_gpu_info, setup_cudnn_benchmark, setup_mixed_precision,
    enable_tf32, GPUMonitor, clear_gpu_cache
)


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    std_rmse_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        # Move to device
        batch = batch.to(device)
        
        # Forward pass with mixed precision
        if config['training'].get('use_amp', True):
            with autocast():
                pred_1d, pred_2d = model(batch)
                
                # Compute loss
                loss = criterion(
                    pred_1d, pred_2d,
                    batch.y_1d, batch.y_2d,
                    batch.model_id[0].item()
                )
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training'].get('gradient_clip'):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_1d, pred_2d = model(batch)
            loss = criterion(
                pred_1d, pred_2d,
                batch.y_1d, batch.y_2d,
                batch.model_id[0].item()
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            if config['training'].get('gradient_clip'):
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), batch.y_1d.size(0))
        std_rmse_meter.update(loss.item(), batch.y_1d.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'std_rmse': f'{std_rmse_meter.avg:.4f}'
        })
    
    return {
        'train_loss': losses.avg,
        'train_std_rmse': std_rmse_meter.avg
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    std_rmse_meter = AverageMeter()
    
    all_metrics = []
    
    for batch in tqdm(val_loader, desc='Validation'):
        batch = batch.to(device)
        
        # Forward pass
        pred_1d, pred_2d = model(batch)
        
        # Compute loss
        loss = criterion(
            pred_1d, pred_2d,
            batch.y_1d, batch.y_2d,
            batch.model_id[0].item()
        )
        
        # Compute metrics
        metrics = compute_metrics(
            pred_1d, pred_2d,
            batch.y_1d, batch.y_2d,
            batch.model_id[0].item(),
            config['loss']['std_devs']
        )
        
        losses.update(loss.item(), batch.y_1d.size(0))
        std_rmse_meter.update(metrics['standardized_rmse'], batch.y_1d.size(0))
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        'val_loss': losses.avg,
        'val_std_rmse': std_rmse_meter.avg
    }
    
    return avg_metrics


def train(config_path: str, model_id: int):
    """
    Main training function
    
    Args:
        config_path: Path to config file
        model_id: Model ID (1 or 2)
    """
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Print GPU info
    print_gpu_info()
    
    # Get device
    device = get_device(config.get('device', 'cuda'))
    
    # GPU optimizations
    if device.type == 'cuda':
        # Enable cuDNN benchmark for faster training
        setup_cudnn_benchmark(enable=True)
        
        # Enable TF32 on Ampere GPUs for faster training
        enable_tf32()
        
        # Check mixed precision support
        use_amp = setup_mixed_precision() and config['training'].get('use_amp', True)
        config['training']['use_amp'] = use_amp
        
        # Clear GPU cache
        clear_gpu_cache()
    else:
        config['training']['use_amp'] = False
    
    # Create output directories
    checkpoint_dir = Path(config['data']['checkpoint_dir']) / f"model_{model_id}"
    create_directory(checkpoint_dir)
    
    log_dir = Path(config['data']['output_dir']) / 'logs'
    create_directory(log_dir)
    
    # Logger
    logger = Logger(log_dir / f'train_model_{model_id}.log')
    logger.log(f"Training Model {model_id}")
    logger.log(f"Device: {device}")
    logger.log(f"Mixed Precision: {config['training']['use_amp']}")
    
    # Create dataloaders
    logger.log("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config['data']['data_root'],
        model_id,
        config,
        use_graph=True
    )
    
    logger.log(f"Train samples: {len(train_loader.dataset)}")
    logger.log(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.log("Creating model...")
    
    # Get std_devs for this model
    std_devs = {
        (model_id, 1): config['loss']['std_devs'][f'model_{model_id}']['node_type_1'],
        (model_id, 2): config['loss']['std_devs'][f'model_{model_id}']['node_type_2']
    }
    
    model = SpatioTemporalGNN(
        static_1d_dim=config['model']['node_1d_static_dim'],
        static_2d_dim=config['model']['node_2d_static_dim'],
        dynamic_1d_dim=config['model']['node_1d_dynamic_dim'],
        dynamic_2d_dim=config['model']['node_2d_dynamic_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        num_temporal_layers=config['model']['num_temporal_layers'],
        gnn_type=config['model']['gnn_type'],
        temporal_type='LSTM',
        dropout=config['model']['dropout']
    ).to(device)
    
    from utils.helpers import count_parameters
    num_params = count_parameters(model)
    logger.log(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = StandardizedRMSE(std_devs)
    
    # Optimizer
    if config['training']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Scheduler
    if config['training'].get('scheduler') == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('factor', 0.5),
            patience=config['training'].get('patience', 10),
            verbose=True
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training'].get('use_amp', True) and device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 20),
        mode='min'
    )
    
    # GPU Monitor
    gpu_monitor = GPUMonitor(device)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        logger.log(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        logger.log(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Print GPU stats at start of epoch
        if device.type == 'cuda':
            gpu_monitor.print_stats()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device, config
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, config
        )
        
        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}
        logger.log_metrics(all_metrics, prefix=f"Epoch {epoch+1}: ")
        
        # Print GPU stats after epoch
        if device.type == 'cuda':
            gpu_monitor.print_stats()
        
        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                model, optimizer, epoch,
                all_metrics,
                checkpoint_dir / 'best_model.pt'
            )
            logger.log(f"✓ Best model saved! Val loss: {best_val_loss:.6f}")
        
        # Save latest checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch,
                all_metrics,
                checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and (epoch + 1) % 5 == 0:
            clear_gpu_cache()
        
        # Early stopping
        if early_stopping(val_metrics['val_loss']):
            logger.log(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.log("Training completed!")
    
    # Final GPU stats
    if device.type == 'cuda':
        logger.log("\nFinal GPU Memory Usage:")
        gpu_monitor.print_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train flood prediction model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_id', type=int, required=True,
                       help='Model ID (1 or 2)')
    
    args = parser.parse_args()
    
    train(args.config, args.model_id)