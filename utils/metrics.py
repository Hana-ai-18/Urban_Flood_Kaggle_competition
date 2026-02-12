"""
Evaluation metrics for flood prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class StandardizedRMSE(nn.Module):
    """
    Standardized RMSE metric theo specification của competition
    """
    
    def __init__(self, std_devs: Dict[Tuple[int, int], float]):
        """
        Args:
            std_devs: Dictionary mapping (model_id, node_type) -> std_dev
                Example: {(1, 1): 16.877747, (1, 2): 14.378797, ...}
        """
        super().__init__()
        self.std_devs = std_devs
        
    def forward(self,
                pred_1d: torch.Tensor,
                pred_2d: torch.Tensor,
                target_1d: torch.Tensor,
                target_2d: torch.Tensor,
                model_id: int) -> torch.Tensor:
        """
        Compute Standardized RMSE
        
        Args:
            pred_1d: Predictions for 1D nodes [batch, num_1d_nodes]
            pred_2d: Predictions for 2D nodes [batch, num_2d_nodes]
            target_1d: Targets for 1D nodes [batch, num_1d_nodes]
            target_2d: Targets for 2D nodes [batch, num_2d_nodes]
            model_id: Model ID (1 or 2)
            
        Returns:
            Standardized RMSE value
        """
        # RMSE for 1D
        rmse_1d = torch.sqrt(torch.mean((pred_1d - target_1d) ** 2))
        std_1d = self.std_devs[(model_id, 1)]
        standardized_rmse_1d = rmse_1d / std_1d
        
        # RMSE for 2D
        rmse_2d = torch.sqrt(torch.mean((pred_2d - target_2d) ** 2))
        std_2d = self.std_devs[(model_id, 2)]
        standardized_rmse_2d = rmse_2d / std_2d
        
        # Average
        standardized_rmse = (standardized_rmse_1d + standardized_rmse_2d) / 2.0
        
        return standardized_rmse
    
    def compute_node_level(self,
                          pred: torch.Tensor,
                          target: torch.Tensor,
                          std_dev: float) -> torch.Tensor:
        """
        Compute node-level standardized RMSE
        
        Args:
            pred: Predictions [batch, num_nodes] or [num_timesteps, num_nodes]
            target: Targets [batch, num_nodes] or [num_timesteps, num_nodes]
            std_dev: Standard deviation for normalization
            
        Returns:
            Node-level standardized RMSE [num_nodes]
        """
        # RMSE per node
        rmse_per_node = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
        
        # Standardize
        standardized_rmse_per_node = rmse_per_node / std_dev
        
        return standardized_rmse_per_node


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute regular RMSE
    
    Args:
        pred: Predictions
        target: Targets
        
    Returns:
        RMSE value
    """
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error
    
    Args:
        pred: Predictions
        target: Targets
        
    Returns:
        MAE value
    """
    mae = torch.mean(torch.abs(pred - target))
    return mae.item()


def compute_metrics(pred_1d: torch.Tensor,
                   pred_2d: torch.Tensor,
                   target_1d: torch.Tensor,
                   target_2d: torch.Tensor,
                   model_id: int,
                   std_devs: Dict[Tuple[int, int], float]) -> Dict[str, float]:
    """
    Compute all metrics
    
    Args:
        pred_1d: 1D predictions
        pred_2d: 2D predictions
        target_1d: 1D targets
        target_2d: 2D targets
        model_id: Model ID
        std_devs: Standard deviations dictionary
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Standardized RMSE
    std_rmse_metric = StandardizedRMSE(std_devs)
    metrics['standardized_rmse'] = std_rmse_metric(
        pred_1d, pred_2d, target_1d, target_2d, model_id
    ).item()
    
    # Regular RMSE
    metrics['rmse_1d'] = compute_rmse(pred_1d, target_1d)
    metrics['rmse_2d'] = compute_rmse(pred_2d, target_2d)
    metrics['rmse_overall'] = (metrics['rmse_1d'] + metrics['rmse_2d']) / 2.0
    
    # MAE
    metrics['mae_1d'] = compute_mae(pred_1d, target_1d)
    metrics['mae_2d'] = compute_mae(pred_2d, target_2d)
    metrics['mae_overall'] = (metrics['mae_1d'] + metrics['mae_2d']) / 2.0
    
    return metrics


class NodeLevelMetrics:
    """Compute và track node-level metrics"""
    
    def __init__(self, num_1d_nodes: int, num_2d_nodes: int):
        self.num_1d_nodes = num_1d_nodes
        self.num_2d_nodes = num_2d_nodes
        
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.squared_errors_1d = np.zeros(self.num_1d_nodes)
        self.squared_errors_2d = np.zeros(self.num_2d_nodes)
        self.count = 0
    
    def update(self, pred_1d: np.ndarray, pred_2d: np.ndarray,
               target_1d: np.ndarray, target_2d: np.ndarray):
        """
        Update với batch mới
        
        Args:
            pred_1d: [batch, num_1d_nodes]
            pred_2d: [batch, num_2d_nodes]
            target_1d: [batch, num_1d_nodes]
            target_2d: [batch, num_2d_nodes]
        """
        # Accumulate squared errors
        self.squared_errors_1d += np.sum((pred_1d - target_1d) ** 2, axis=0)
        self.squared_errors_2d += np.sum((pred_2d - target_2d) ** 2, axis=0)
        
        self.count += pred_1d.shape[0]
    
    def compute(self, std_1d: float, std_2d: float) -> Dict[str, np.ndarray]:
        """
        Compute node-level standardized RMSE
        
        Args:
            std_1d: Standard deviation for 1D nodes
            std_2d: Standard deviation for 2D nodes
            
        Returns:
            Dictionary with node-level metrics
        """
        if self.count == 0:
            return {
                'rmse_1d_per_node': np.zeros(self.num_1d_nodes),
                'rmse_2d_per_node': np.zeros(self.num_2d_nodes),
                'standardized_rmse_1d_per_node': np.zeros(self.num_1d_nodes),
                'standardized_rmse_2d_per_node': np.zeros(self.num_2d_nodes)
            }
        
        # RMSE per node
        rmse_1d_per_node = np.sqrt(self.squared_errors_1d / self.count)
        rmse_2d_per_node = np.sqrt(self.squared_errors_2d / self.count)
        
        # Standardized RMSE per node
        std_rmse_1d_per_node = rmse_1d_per_node / std_1d
        std_rmse_2d_per_node = rmse_2d_per_node / std_2d
        
        return {
            'rmse_1d_per_node': rmse_1d_per_node,
            'rmse_2d_per_node': rmse_2d_per_node,
            'standardized_rmse_1d_per_node': std_rmse_1d_per_node,
            'standardized_rmse_2d_per_node': std_rmse_2d_per_node,
            'avg_standardized_rmse': (std_rmse_1d_per_node.mean() + 
                                     std_rmse_2d_per_node.mean()) / 2.0
        }