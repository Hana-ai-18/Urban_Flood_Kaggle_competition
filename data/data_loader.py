"""
DataLoader utilities
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Tuple, Dict, List
import numpy as np

from .dataset import FloodDataset, FloodGraphDataset


def create_dataloaders(
    data_root: str,
    model_id: int,
    config: Dict,
    use_graph: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo train và validation dataloaders
    
    Args:
        data_root: Thư mục gốc data
        model_id: Model ID
        config: Configuration dictionary
        use_graph: Dùng Graph dataset hay không
        
    Returns:
        train_loader, val_loader
    """
    from .preprocessing import DataPreprocessor
    
    # Get list of training events
    preprocessor = DataPreprocessor(data_root, model_id)
    train_event_ids = preprocessor.get_train_events()
    
    # Split train/val
    train_val_split = config['data'].get('train_val_split', 0.8)
    num_train = int(len(train_event_ids) * train_val_split)
    
    np.random.seed(config.get('seed', 42))
    np.random.shuffle(train_event_ids)
    
    train_events = train_event_ids[:num_train]
    val_events = train_event_ids[num_train:]
    
    # Create datasets
    if use_graph:
        train_dataset = FloodGraphDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=train_events,
            split='train',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        val_dataset = FloodGraphDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=val_events,
            split='train',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        # Use PyG DataLoader
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
        
    else:
        train_dataset = FloodDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=train_events,
            split='train',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        val_dataset = FloodDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=val_events,
            split='train',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
    
    return train_loader, val_loader


def create_test_loader(
    data_root: str,
    model_id: int,
    config: Dict,
    use_graph: bool = True
) -> DataLoader:
    """
    Tạo test dataloader
    
    Args:
        data_root: Thư mục gốc
        model_id: Model ID
        config: Configuration
        use_graph: Use graph dataset
        
    Returns:
        test_loader
    """
    from .preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(data_root, model_id)
    test_event_ids = preprocessor.get_test_events()
    
    if use_graph:
        test_dataset = FloodGraphDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=test_event_ids,
            split='test',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=1,  # Process one at a time for autoregressive
            shuffle=False,
            num_workers=0
        )
    else:
        test_dataset = FloodDataset(
            data_root=data_root,
            model_id=model_id,
            event_ids=test_event_ids,
            split='test',
            sequence_length=config['model']['sequence_length'],
            prediction_horizon=config['model']['prediction_horizon'],
            normalize=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    
    return test_loader


class AutoregressiveDataLoader:
    """
    Custom DataLoader cho autoregressive prediction
    """
    
    def __init__(self,
                 data_root: str,
                 model_id: int,
                 event_ids: List[int],
                 config: Dict,
                 split: str = 'test'):
        """
        Args:
            data_root: Thư mục data
            model_id: Model ID
            event_ids: Event IDs
            config: Configuration
            split: train/test
        """
        self.data_root = data_root
        self.model_id = model_id
        self.event_ids = event_ids
        self.config = config
        self.split = split
        
        from .preprocessing import DataPreprocessor
        self.preprocessor = DataPreprocessor(data_root, model_id)
        
        # Load static features
        self.static_features = self.preprocessor.load_static_features(split)
        
    def get_event_iterator(self):
        """
        Generator cho từng event
        
        Yields:
            event_id, initial_states, rainfall_sequence, metadata
        """
        for event_id in self.event_ids:
            # Load event data
            event_data = self.preprocessor.load_event_data(event_id, self.split)
            
            # Get spinup data (first 10 timesteps)
            spinup_steps = self.config['data'].get('spinup_steps', 10)
            
            # Initial states
            initial_1d = event_data['1d_nodes_dynamic'][
                event_data['1d_nodes_dynamic']['timestep'] < spinup_steps
            ]
            initial_2d = event_data['2d_nodes_dynamic'][
                event_data['2d_nodes_dynamic']['timestep'] < spinup_steps
            ]
            
            # Full rainfall sequence
            rainfall_2d = event_data['2d_nodes_dynamic'][
                ['timestep', 'node_id', 'rainfall_depth']
            ]
            
            # Metadata
            num_timesteps = len(event_data['timesteps'])
            
            metadata = {
                'event_id': event_id,
                'model_id': self.model_id,
                'num_timesteps': num_timesteps,
                'spinup_steps': spinup_steps
            }
            
            yield event_id, initial_1d, initial_2d, rainfall_2d, metadata


def collate_graph_batch(batch):
    """
    Custom collate function for graph batches
    
    Args:
        batch: List of Data objects
        
    Returns:
        Batched Data object
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)