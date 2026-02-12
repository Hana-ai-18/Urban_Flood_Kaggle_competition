"""
Dataset classes for flood prediction
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .preprocessing import DataPreprocessor, FeatureExtractor


class FloodDataset(Dataset):
    """Dataset cơ bản cho flood prediction"""
    
    def __init__(self,
                 data_root: str,
                 model_id: int,
                 event_ids: List[int],
                 split: str = 'train',
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 normalize: bool = True):
        """
        Args:
            data_root: Thư mục gốc chứa dữ liệu
            model_id: ID của model (1 hoặc 2)
            event_ids: Danh sách event IDs cần load
            split: 'train' hoặc 'test'
            sequence_length: Độ dài sequence input
            prediction_horizon: Số bước dự đoán
            normalize: Có chuẩn hóa features không
        """
        self.data_root = Path(data_root)
        self.model_id = model_id
        self.event_ids = event_ids
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        
        self.preprocessor = DataPreprocessor(data_root, model_id)
        
        # Load static features
        self.static_features = self.preprocessor.load_static_features(split)
        
        # Load tất cả events
        self.events_data = {}
        for event_id in event_ids:
            self.events_data[event_id] = self.preprocessor.load_event_data(event_id, split)
        
        # Create samples
        self.samples = self._create_samples()
        
        # Statistics cho normalization
        self.stats = self._compute_statistics() if normalize else None
        
    def _create_samples(self) -> List[Dict]:
        """Tạo danh sách samples từ tất cả events"""
        samples = []
        
        for event_id in self.event_ids:
            event_data = self.events_data[event_id]
            timesteps_df = event_data['timesteps']
            num_timesteps = len(timesteps_df)
            
            # Tạo sequences
            for t in range(self.sequence_length, num_timesteps - self.prediction_horizon + 1):
                sample = {
                    'event_id': event_id,
                    'start_timestep': t - self.sequence_length,
                    'end_timestep': t,
                    'target_timestep': t + self.prediction_horizon - 1
                }
                samples.append(sample)
        
        return samples
    
    def _compute_statistics(self) -> Dict:
        """Tính statistics cho normalization"""
        # Collect tất cả data để tính mean và std
        all_1d_dynamic = []
        all_2d_dynamic = []
        
        for event_data in self.events_data.values():
            all_1d_dynamic.append(event_data['1d_nodes_dynamic'])
            all_2d_dynamic.append(event_data['2d_nodes_dynamic'])
        
        # Concatenate
        df_1d = pd.concat(all_1d_dynamic, ignore_index=True)
        df_2d = pd.concat(all_2d_dynamic, ignore_index=True)
        
        stats = {
            '1d_mean': df_1d.drop(columns=['node_id', 'timestep']).mean().values,
            '1d_std': df_1d.drop(columns=['node_id', 'timestep']).std().values,
            '2d_mean': df_2d.drop(columns=['node_id', 'timestep']).mean().values,
            '2d_std': df_2d.drop(columns=['node_id', 'timestep']).std().values,
        }
        
        # Replace 0 std with 1
        stats['1d_std'][stats['1d_std'] == 0] = 1.0
        stats['2d_std'][stats['2d_std'] == 0] = 1.0
        
        return stats
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get một sample
        
        Returns:
            Dictionary chứa:
            - x_1d: 1D node features [sequence_length, num_1d_nodes, features]
            - x_2d: 2D node features [sequence_length, num_2d_nodes, features]
            - y_1d: 1D target [num_1d_nodes]
            - y_2d: 2D target [num_2d_nodes]
        """
        sample = self.samples[idx]
        event_id = sample['event_id']
        event_data = self.events_data[event_id]
        
        # Extract sequences
        x_1d = []
        x_2d = []
        
        for t in range(sample['start_timestep'], sample['end_timestep']):
            # 1D nodes
            nodes_1d_t = event_data['1d_nodes_dynamic'][
                event_data['1d_nodes_dynamic']['timestep'] == t
            ].sort_values('node_id')
            
            # 2D nodes
            nodes_2d_t = event_data['2d_nodes_dynamic'][
                event_data['2d_nodes_dynamic']['timestep'] == t
            ].sort_values('node_id')
            
            # Extract features (excluding node_id and timestep)
            x_1d.append(nodes_1d_t.drop(columns=['node_id', 'timestep']).values)
            x_2d.append(nodes_2d_t.drop(columns=['node_id', 'timestep']).values)
        
        x_1d = np.array(x_1d)  # [seq_len, num_nodes, features]
        x_2d = np.array(x_2d)
        
        # Extract targets
        target_t = sample['target_timestep']
        
        y_1d = event_data['1d_nodes_dynamic'][
            event_data['1d_nodes_dynamic']['timestep'] == target_t
        ].sort_values('node_id')['water_level'].values
        
        y_2d = event_data['2d_nodes_dynamic'][
            event_data['2d_nodes_dynamic']['timestep'] == target_t
        ].sort_values('node_id')['water_level'].values
        
        # Normalize if needed
        if self.normalize and self.stats is not None:
            x_1d = (x_1d - self.stats['1d_mean']) / self.stats['1d_std']
            x_2d = (x_2d - self.stats['2d_mean']) / self.stats['2d_std']
        
        return {
            'x_1d': torch.FloatTensor(x_1d),
            'x_2d': torch.FloatTensor(x_2d),
            'y_1d': torch.FloatTensor(y_1d),
            'y_2d': torch.FloatTensor(y_2d),
            'event_id': event_id,
            'timestep': target_t
        }


class FloodGraphDataset(Dataset):
    """Dataset cho Graph Neural Networks"""
    
    def __init__(self,
                 data_root: str,
                 model_id: int,
                 event_ids: List[int],
                 split: str = 'train',
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 normalize: bool = True,
                 add_spatial_features: bool = True):
        """
        Args:
            data_root: Thư mục gốc
            model_id: Model ID
            event_ids: Event IDs
            split: train/test
            sequence_length: Độ dài sequence
            prediction_horizon: Prediction horizon
            normalize: Normalize features
            add_spatial_features: Thêm spatial features
        """
        self.data_root = Path(data_root)
        self.model_id = model_id
        self.event_ids = event_ids
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.add_spatial_features = add_spatial_features
        
        self.preprocessor = DataPreprocessor(data_root, model_id)
        self.feature_extractor = FeatureExtractor()
        
        # Load static features và graph structure
        self.static_features = self.preprocessor.load_static_features(split)
        self.node_mapping = self.preprocessor.create_node_mapping(self.static_features)
        
        # Build graph structure
        self.edge_index_1d = self._build_edge_index('1d')
        self.edge_index_2d = self._build_edge_index('2d')
        self.edge_index_1d2d = self._build_1d2d_connections()
        
        # Static node features
        self.static_node_features_1d = self._build_static_node_features('1d')
        self.static_node_features_2d = self._build_static_node_features('2d')
        
        # Load events
        self.events_data = {}
        for event_id in event_ids:
            self.events_data[event_id] = self.preprocessor.load_event_data(event_id, split)
        
        # Create samples
        self.samples = self._create_samples()
        
        # Compute statistics
        self.stats = self._compute_statistics() if normalize else None
        
    def _build_edge_index(self, node_type: str) -> torch.LongTensor:
        """Build edge index cho 1D hoặc 2D graph"""
        edge_index_df = self.static_features[f'{node_type}_edge_index']
        
        # Map node IDs to indices
        mapping = self.node_mapping[f'node_{node_type}_to_idx']
        
        from_nodes = edge_index_df['from_node'].map(mapping).values
        to_nodes = edge_index_df['to_node'].map(mapping).values
        
        edge_index = np.stack([from_nodes, to_nodes], axis=0)
        
        return torch.LongTensor(edge_index)
    
    def _build_1d2d_connections(self) -> torch.LongTensor:
        """Build connections giữa 1D và 2D nodes"""
        connections_df = self.static_features['1d2d_connections']
        
        # Map IDs to indices
        mapping_1d = self.node_mapping['node_1d_to_idx']
        mapping_2d = self.node_mapping['node_2d_to_idx']
        
        nodes_1d = connections_df['node_1d_id'].map(mapping_1d).values
        nodes_2d = connections_df['node_2d_id'].map(mapping_2d).values
        
        # Bi-directional connections
        edge_index = np.stack([nodes_1d, nodes_2d], axis=0)
        
        return torch.LongTensor(edge_index)
    
    def _build_static_node_features(self, node_type: str) -> torch.FloatTensor:
        """Build static node features"""
        static_df = self.static_features[f'{node_type}_nodes_static']
        
        # Sort by node_id to ensure consistent ordering
        static_df = static_df.sort_values('node_id')
        
        # Extract features
        feature_cols = [col for col in static_df.columns if col != 'node_id']
        features = static_df[feature_cols].values
        
        # Add spatial features if requested
        if self.add_spatial_features:
            pos_x = static_df[f'{node_type}_position_x'].values
            pos_y = static_df[f'{node_type}_position_y'].values
            
            spatial_features = self.feature_extractor.compute_distance_features(pos_x, pos_y)
            
            # Concatenate
            spatial_array = np.stack([
                spatial_features['distance_to_centroid'],
                spatial_features['normalized_x'],
                spatial_features['normalized_y']
            ], axis=1)
            
            features = np.concatenate([features, spatial_array], axis=1)
        
        return torch.FloatTensor(features)
    
    def _create_samples(self) -> List[Dict]:
        """Create samples"""
        samples = []
        
        for event_id in self.event_ids:
            event_data = self.events_data[event_id]
            num_timesteps = len(event_data['timesteps'])
            
            # For test set, start from spinup_steps (usually 10)
            if self.split == 'test':
                start_idx = 10  # After spinup
            else:
                start_idx = self.sequence_length
            
            for t in range(start_idx, num_timesteps - self.prediction_horizon + 1):
                sample = {
                    'event_id': event_id,
                    'start_timestep': t - self.sequence_length,
                    'end_timestep': t,
                    'target_timestep': t + self.prediction_horizon - 1
                }
                samples.append(sample)
        
        return samples
    
    def _compute_statistics(self) -> Dict:
        """Compute normalization statistics"""
        all_1d_dynamic = []
        all_2d_dynamic = []
        
        for event_data in self.events_data.values():
            all_1d_dynamic.append(event_data['1d_nodes_dynamic'])
            all_2d_dynamic.append(event_data['2d_nodes_dynamic'])
        
        df_1d = pd.concat(all_1d_dynamic, ignore_index=True)
        df_2d = pd.concat(all_2d_dynamic, ignore_index=True)
        
        # Exclude node_id and timestep
        feature_cols_1d = [col for col in df_1d.columns if col not in ['node_id', 'timestep']]
        feature_cols_2d = [col for col in df_2d.columns if col not in ['node_id', 'timestep']]
        
        stats = {
            '1d_dynamic_mean': df_1d[feature_cols_1d].mean().values,
            '1d_dynamic_std': df_1d[feature_cols_1d].std().values,
            '2d_dynamic_mean': df_2d[feature_cols_2d].mean().values,
            '2d_dynamic_std': df_2d[feature_cols_2d].std().values,
        }
        
        # Prevent division by zero
        stats['1d_dynamic_std'][stats['1d_dynamic_std'] == 0] = 1.0
        stats['2d_dynamic_std'][stats['2d_dynamic_std'] == 0] = 1.0
        
        return stats
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get graph data sample
        
        Returns:
            PyG Data object
        """
        sample = self.samples[idx]
        event_id = sample['event_id']
        event_data = self.events_data[event_id]
        
        # Extract dynamic features for sequence
        dynamic_1d_seq = []
        dynamic_2d_seq = []
        
        for t in range(sample['start_timestep'], sample['end_timestep']):
            # 1D
            df_1d_t = event_data['1d_nodes_dynamic'][
                event_data['1d_nodes_dynamic']['timestep'] == t
            ].sort_values('node_id')
            
            feature_cols_1d = [col for col in df_1d_t.columns 
                              if col not in ['node_id', 'timestep']]
            dynamic_1d_seq.append(df_1d_t[feature_cols_1d].values)
            
            # 2D
            df_2d_t = event_data['2d_nodes_dynamic'][
                event_data['2d_nodes_dynamic']['timestep'] == t
            ].sort_values('node_id')
            
            feature_cols_2d = [col for col in df_2d_t.columns 
                              if col not in ['node_id', 'timestep']]
            dynamic_2d_seq.append(df_2d_t[feature_cols_2d].values)
        
        dynamic_1d_seq = np.array(dynamic_1d_seq)  # [seq_len, num_nodes, features]
        dynamic_2d_seq = np.array(dynamic_2d_seq)
        
        # Normalize dynamic features
        if self.normalize and self.stats is not None:
            dynamic_1d_seq = (dynamic_1d_seq - self.stats['1d_dynamic_mean']) / self.stats['1d_dynamic_std']
            dynamic_2d_seq = (dynamic_2d_seq - self.stats['2d_dynamic_mean']) / self.stats['2d_dynamic_std']
        
        # Target water levels
        target_t = sample['target_timestep']
        
        y_1d = event_data['1d_nodes_dynamic'][
            event_data['1d_nodes_dynamic']['timestep'] == target_t
        ].sort_values('node_id')['water_level'].values
        
        y_2d = event_data['2d_nodes_dynamic'][
            event_data['2d_nodes_dynamic']['timestep'] == target_t
        ].sort_values('node_id')['water_level'].values
        
        # Create PyG Data object
        data = Data(
            # Static features
            x_1d_static=self.static_node_features_1d,
            x_2d_static=self.static_node_features_2d,
            
            # Dynamic features
            x_1d_dynamic=torch.FloatTensor(dynamic_1d_seq),
            x_2d_dynamic=torch.FloatTensor(dynamic_2d_seq),
            
            # Edge indices
            edge_index_1d=self.edge_index_1d,
            edge_index_2d=self.edge_index_2d,
            edge_index_1d2d=self.edge_index_1d2d,
            
            # Targets
            y_1d=torch.FloatTensor(y_1d),
            y_2d=torch.FloatTensor(y_2d),
            
            # Metadata
            event_id=event_id,
            timestep=target_t,
            model_id=self.model_id
        )
        
        return data