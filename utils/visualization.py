"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         save_path: str = None):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Standardized RMSE)', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_predictions(predictions: np.ndarray,
                    targets: np.ndarray,
                    node_ids: np.ndarray = None,
                    title: str = 'Predictions vs Targets',
                    save_path: str = None):
    """
    Plot predictions vs targets scatter plot
    
    Args:
        predictions: Predicted values
        targets: Target values
        node_ids: Node IDs (optional)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    # Add text box with metrics
    textstr = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Target Water Level', fontsize=12)
    ax.set_ylabel('Predicted Water Level', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()


def plot_time_series_predictions(predictions: np.ndarray,
                                 targets: np.ndarray,
                                 timestamps: np.ndarray = None,
                                 node_id: int = None,
                                 title: str = 'Time Series Prediction',
                                 save_path: str = None):
    """
    Plot time series predictions for a single node
    
    Args:
        predictions: Predicted time series
        targets: Target time series
        timestamps: Timestamps (optional)
        node_id: Node ID for title
        title: Plot title
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if timestamps is None:
        timestamps = np.arange(len(predictions))
    
    ax.plot(timestamps, targets, 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(timestamps, predictions, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    
    # Calculate error
    error = predictions - targets
    rmse = np.sqrt(np.mean(error ** 2))
    
    if node_id is not None:
        title = f'{title} - Node {node_id} (RMSE: {rmse:.4f})'
    else:
        title = f'{title} (RMSE: {rmse:.4f})'
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Water Level', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_distribution(errors: np.ndarray,
                           title: str = 'Prediction Error Distribution',
                           save_path: str = None):
    """
    Plot distribution of prediction errors
    
    Args:
        errors: Prediction errors
        title: Plot title
        save_path: Path to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Histogram', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(errors, vert=True)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Error Box Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    textstr = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.5, 0.95, textstr, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            horizontalalignment='center', bbox=props)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_node_level_errors(errors_1d: np.ndarray,
                           errors_2d: np.ndarray,
                           save_path: str = None):
    """
    Plot node-level errors for 1D and 2D nodes
    
    Args:
        errors_1d: Errors for 1D nodes
        errors_2d: Errors for 2D nodes
        save_path: Path to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1D nodes
    node_ids_1d = np.arange(len(errors_1d))
    ax1.bar(node_ids_1d, errors_1d, alpha=0.7, color='blue')
    ax1.set_xlabel('1D Node Index', fontsize=12)
    ax1.set_ylabel('Standardized RMSE', fontsize=12)
    ax1.set_title('1D Node Errors', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2D nodes
    node_ids_2d = np.arange(len(errors_2d))
    ax2.bar(node_ids_2d, errors_2d, alpha=0.7, color='red')
    ax2.set_xlabel('2D Node Index', fontsize=12)
    ax2.set_ylabel('Standardized RMSE', fontsize=12)
    ax2.set_title('2D Node Errors', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_spatial_errors(errors: np.ndarray,
                       positions_x: np.ndarray,
                       positions_y: np.ndarray,
                       title: str = 'Spatial Error Distribution',
                       save_path: str = None):
    """
    Plot errors on spatial map
    
    Args:
        errors: Node errors
        positions_x: X coordinates
        positions_y: Y coordinates
        title: Plot title
        save_path: Path to save
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot với color mapping theo error
    scatter = ax.scatter(positions_x, positions_y, 
                        c=errors, cmap='RdYlGn_r',
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Standardized RMSE', fontsize=12)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_visualization_report(predictions_1d: np.ndarray,
                                predictions_2d: np.ndarray,
                                targets_1d: np.ndarray,
                                targets_2d: np.ndarray,
                                output_dir: str):
    """
    Create comprehensive visualization report
    
    Args:
        predictions_1d: 1D predictions
        predictions_2d: 2D predictions
        targets_1d: 1D targets
        targets_2d: 2D targets
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualization report...")
    
    # 1. Prediction scatter plots
    plot_predictions(
        predictions_1d.flatten(), targets_1d.flatten(),
        title='1D Nodes: Predictions vs Targets',
        save_path=output_dir / '1d_predictions.png'
    )
    
    plot_predictions(
        predictions_2d.flatten(), targets_2d.flatten(),
        title='2D Nodes: Predictions vs Targets',
        save_path=output_dir / '2d_predictions.png'
    )
    
    # 2. Error distributions
    errors_1d = predictions_1d.flatten() - targets_1d.flatten()
    errors_2d = predictions_2d.flatten() - targets_2d.flatten()
    
    plot_error_distribution(
        errors_1d,
        title='1D Node Error Distribution',
        save_path=output_dir / '1d_error_dist.png'
    )
    
    plot_error_distribution(
        errors_2d,
        title='2D Node Error Distribution',
        save_path=output_dir / '2d_error_dist.png'
    )
    
    print(f"Visualization report saved to {output_dir}")