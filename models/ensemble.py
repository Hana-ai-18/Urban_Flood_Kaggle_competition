"""
Ensemble models
"""

import torch
import torch.nn as nn
from typing import List, Dict


class EnsembleModel(nn.Module):
    """Ensemble của nhiều models"""
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Args:
            models: List of models
            weights: Weights cho từng model (default: equal weights)
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, batch):
        """
        Forward pass through all models và average predictions
        
        Args:
            batch: Input batch
            
        Returns:
            pred_1d: Averaged predictions for 1D nodes
            pred_2d: Averaged predictions for 2D nodes
        """
        predictions_1d = []
        predictions_2d = []
        
        for model in self.models:
            pred_1d, pred_2d = model(batch)
            predictions_1d.append(pred_1d)
            predictions_2d.append(pred_2d)
        
        # Stack predictions
        predictions_1d = torch.stack(predictions_1d, dim=0)  # [num_models, batch, nodes]
        predictions_2d = torch.stack(predictions_2d, dim=0)
        
        # Weighted average
        weights = self.weights.to(predictions_1d.device).view(-1, 1, 1)
        
        pred_1d = (predictions_1d * weights).sum(dim=0)
        pred_2d = (predictions_2d * weights).sum(dim=0)
        
        return pred_1d, pred_2d