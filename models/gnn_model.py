"""
Graph Neural Network models for flood prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from typing import Optional, Tuple


class GNNLayer(nn.Module):
    """Single GNN layer với nhiều options"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 gnn_type: str = 'GAT',
                 heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            gnn_type: Type of GNN (GCN, GAT, GraphSAGE)
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gnn_type = gnn_type
        
        if gnn_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn_type == 'GAT':
            self.conv = GATConv(
                in_channels, 
                out_channels // heads,
                heads=heads,
                dropout=dropout
            )
        elif gnn_type == 'GraphSAGE':
            self.conv = SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # GNN convolution
        x_new = self.conv(x, edge_index)
        
        # Residual connection if dimensions match
        if x.size(-1) == x_new.size(-1):
            x_new = x_new + x
        
        # Normalization and dropout
        x_new = self.norm(x_new)
        x_new = F.relu(x_new)
        x_new = self.dropout(x_new)
        
        return x_new


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal GNN for flood prediction
    Xử lý cả 1D và 2D graphs đồng thời
    """
    
    def __init__(self,
                 # Feature dimensions
                 static_1d_dim: int,
                 static_2d_dim: int,
                 dynamic_1d_dim: int,
                 dynamic_2d_dim: int,
                 
                 # Model architecture
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 num_temporal_layers: int = 2,
                 gnn_type: str = 'GAT',
                 temporal_type: str = 'LSTM',
                 
                 # Other params
                 dropout: float = 0.1,
                 heads: int = 4):
        """
        Args:
            static_1d_dim: Dimension of 1D static features
            static_2d_dim: Dimension of 2D static features
            dynamic_1d_dim: Dimension of 1D dynamic features
            dynamic_2d_dim: Dimension of 2D dynamic features
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            num_temporal_layers: Number of temporal layers (LSTM/GRU)
            gnn_type: Type of GNN
            temporal_type: LSTM or GRU
            dropout: Dropout rate
            heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.temporal_type = temporal_type
        
        # Feature projection layers
        # Combine static and dynamic features
        self.proj_1d = nn.Linear(static_1d_dim + dynamic_1d_dim, hidden_dim)
        self.proj_2d = nn.Linear(static_2d_dim + dynamic_2d_dim, hidden_dim)
        
        # GNN layers for 1D graph
        self.gnn_1d_layers = nn.ModuleList([
            GNNLayer(
                hidden_dim, 
                hidden_dim, 
                gnn_type=gnn_type,
                heads=heads,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])
        
        # GNN layers for 2D graph
        self.gnn_2d_layers = nn.ModuleList([
            GNNLayer(
                hidden_dim,
                hidden_dim,
                gnn_type=gnn_type,
                heads=heads,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])
        
        # Cross-graph interaction (1D <-> 2D)
        self.cross_attn_1d = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_2d = nn.MultiheadAttention(
            hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal encoder
        if temporal_type == 'LSTM':
            self.temporal_1d = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout if num_temporal_layers > 1 else 0,
                batch_first=True
            )
            self.temporal_2d = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout if num_temporal_layers > 1 else 0,
                batch_first=True
            )
        elif temporal_type == 'GRU':
            self.temporal_1d = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout if num_temporal_layers > 1 else 0,
                batch_first=True
            )
            self.temporal_2d = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_temporal_layers,
                dropout=dropout if num_temporal_layers > 1 else 0,
                batch_first=True
            )
        
        # Output heads (predict water level)
        self.output_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, batch):
        """
        Forward pass
        
        Args:
            batch: Batch from FloodGraphDataset containing:
                - x_1d_static: [batch_size, num_1d_nodes, static_dim]
                - x_2d_static: [batch_size, num_2d_nodes, static_dim]
                - x_1d_dynamic: [batch_size, seq_len, num_1d_nodes, dynamic_dim]
                - x_2d_dynamic: [batch_size, seq_len, num_2d_nodes, dynamic_dim]
                - edge_index_1d: [2, num_1d_edges]
                - edge_index_2d: [2, num_2d_edges]
                - edge_index_1d2d: [2, num_connections]
                
        Returns:
            pred_1d: [batch_size, num_1d_nodes] - Predicted water levels
            pred_2d: [batch_size, num_2d_nodes] - Predicted water levels
        """
        # Extract features
        x_1d_static = batch.x_1d_static
        x_2d_static = batch.x_2d_static
        x_1d_dynamic = batch.x_1d_dynamic
        x_2d_dynamic = batch.x_2d_dynamic
        
        edge_index_1d = batch.edge_index_1d
        edge_index_2d = batch.edge_index_2d
        
        batch_size, seq_len, num_1d_nodes, _ = x_1d_dynamic.shape
        _, _, num_2d_nodes, _ = x_2d_dynamic.shape
        
        # Process each timestep
        h_1d_seq = []
        h_2d_seq = []
        
        for t in range(seq_len):
            # Get dynamic features at time t
            x_1d_t = x_1d_dynamic[:, t, :, :]  # [batch, num_1d_nodes, dynamic_dim]
            x_2d_t = x_2d_dynamic[:, t, :, :]  # [batch, num_2d_nodes, dynamic_dim]
            
            # Combine static and dynamic
            # Expand static to match batch
            x_1d_static_expanded = x_1d_static.unsqueeze(0).expand(batch_size, -1, -1)
            x_2d_static_expanded = x_2d_static.unsqueeze(0).expand(batch_size, -1, -1)
            
            x_1d_combined = torch.cat([x_1d_static_expanded, x_1d_t], dim=-1)
            x_2d_combined = torch.cat([x_2d_static_expanded, x_2d_t], dim=-1)
            
            # Project to hidden dimension
            h_1d = self.proj_1d(x_1d_combined)  # [batch, num_1d_nodes, hidden]
            h_2d = self.proj_2d(x_2d_combined)  # [batch, num_2d_nodes, hidden]
            
            # Apply GNN layers (process batch together)
            # Flatten batch dimension for GNN
            h_1d_flat = h_1d.view(-1, self.hidden_dim)  # [batch*num_1d_nodes, hidden]
            h_2d_flat = h_2d.view(-1, self.hidden_dim)  # [batch*num_2d_nodes, hidden]
            
            # Apply GNN layers
            for gnn_layer in self.gnn_1d_layers:
                h_1d_flat = gnn_layer(h_1d_flat, edge_index_1d)
            
            for gnn_layer in self.gnn_2d_layers:
                h_2d_flat = gnn_layer(h_2d_flat, edge_index_2d)
            
            # Reshape back
            h_1d = h_1d_flat.view(batch_size, num_1d_nodes, self.hidden_dim)
            h_2d = h_2d_flat.view(batch_size, num_2d_nodes, self.hidden_dim)
            
            # Cross-graph interaction
            h_1d_cross, _ = self.cross_attn_1d(h_1d, h_2d, h_2d)
            h_2d_cross, _ = self.cross_attn_2d(h_2d, h_1d, h_1d)
            
            h_1d = h_1d + h_1d_cross
            h_2d = h_2d + h_2d_cross
            
            h_1d_seq.append(h_1d)
            h_2d_seq.append(h_2d)
        
        # Stack sequences
        h_1d_seq = torch.stack(h_1d_seq, dim=1)  # [batch, seq_len, num_1d_nodes, hidden]
        h_2d_seq = torch.stack(h_2d_seq, dim=1)  # [batch, seq_len, num_2d_nodes, hidden]
        
        # Reshape for temporal processing
        # Process each node separately through time
        h_1d_seq = h_1d_seq.transpose(1, 2).contiguous()  # [batch, num_1d_nodes, seq_len, hidden]
        h_2d_seq = h_2d_seq.transpose(1, 2).contiguous()  # [batch, num_2d_nodes, seq_len, hidden]
        
        h_1d_seq = h_1d_seq.view(batch_size * num_1d_nodes, seq_len, self.hidden_dim)
        h_2d_seq = h_2d_seq.view(batch_size * num_2d_nodes, seq_len, self.hidden_dim)
        
        # Apply temporal model
        _, (h_1d_final, _) = self.temporal_1d(h_1d_seq)  # h_final: [num_layers, batch*nodes, hidden]
        _, (h_2d_final, _) = self.temporal_2d(h_2d_seq)
        
        # Take last layer output
        h_1d_final = h_1d_final[-1]  # [batch*num_1d_nodes, hidden]
        h_2d_final = h_2d_final[-1]  # [batch*num_2d_nodes, hidden]
        
        # Reshape back
        h_1d_final = h_1d_final.view(batch_size, num_1d_nodes, self.hidden_dim)
        h_2d_final = h_2d_final.view(batch_size, num_2d_nodes, self.hidden_dim)
        
        # Generate predictions
        pred_1d = self.output_1d(h_1d_final).squeeze(-1)  # [batch, num_1d_nodes]
        pred_2d = self.output_2d(h_2d_final).squeeze(-1)  # [batch, num_2d_nodes]
        
        return pred_1d, pred_2d


class TemporalGNN(nn.Module):
    """
    Simplified Temporal GNN (separate processing for 1D and 2D)
    """
    
    def __init__(self,
                 static_dim: int,
                 dynamic_dim: int,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 3,
                 gnn_type: str = 'GAT',
                 dropout: float = 0.1):
        """
        Args:
            static_dim: Static feature dimension
            dynamic_dim: Dynamic feature dimension
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN type
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature projection
        self.proj = nn.Linear(static_dim + dynamic_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, gnn_type=gnn_type, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # Temporal aggregation
        self.temporal_pool = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x_static, x_dynamic, edge_index):
        """
        Args:
            x_static: [num_nodes, static_dim]
            x_dynamic: [seq_len, num_nodes, dynamic_dim]
            edge_index: [2, num_edges]
            
        Returns:
            predictions: [num_nodes]
        """
        seq_len, num_nodes, _ = x_dynamic.shape
        
        h_seq = []
        for t in range(seq_len):
            # Combine static and dynamic
            x_t = torch.cat([
                x_static.unsqueeze(0).expand(1, -1, -1).squeeze(0),
                x_dynamic[t]
            ], dim=-1)
            
            # Project
            h = self.proj(x_t)
            
            # Apply GNN
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index)
            
            h_seq.append(h)
        
        # Stack and apply temporal
        h_seq = torch.stack(h_seq, dim=0)  # [seq_len, num_nodes, hidden]
        h_seq = h_seq.transpose(0, 1)  # [num_nodes, seq_len, hidden]
        
        _, (h_final, _) = self.temporal_pool(h_seq)
        h_final = h_final[-1]  # [num_nodes, hidden]
        
        # Predict
        pred = self.output(h_final).squeeze(-1)
        
        return pred