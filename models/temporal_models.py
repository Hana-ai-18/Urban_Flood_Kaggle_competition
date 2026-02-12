"""
Temporal encoding models
"""

import torch
import torch.nn as nn
import math


class LSTMEncoder(nn.Module):
    """LSTM Encoder for temporal sequences"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            output: [batch, hidden_dim]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_n = h_n[-1]
        
        # Project
        output = self.fc(h_n)
        
        return output


class GRUEncoder(nn.Module):
    """GRU Encoder"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            output: [batch, hidden_dim]
        """
        gru_out, h_n = self.gru(x)
        
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_n = h_n[-1]
        
        output = self.fc(h_n)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Transformer Encoder for temporal sequences"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            output: [batch, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, hidden]
        
        # Pool over sequence
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, hidden]
        
        return x