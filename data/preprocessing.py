"""
Preprocessing utilities for flood data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import geopandas as gpd


class DataPreprocessor:
    """Tiền xử lý dữ liệu lũ đô thị"""
    
    def __init__(self, data_root: str, model_id: int):
        """
        Args:
            data_root: Thư mục gốc chứa dữ liệu
            model_id: ID của mô hình (1 hoặc 2)
        """
        self.data_root = Path(data_root)
        self.model_id = model_id
        self.model_path = self.data_root / f"Model_{model_id}"
        
    def load_static_features(self, split: str = 'train') -> Dict[str, pd.DataFrame]:
        """
        Load static features
        
        Args:
            split: 'train' hoặc 'test'
            
        Returns:
            Dictionary chứa các static features
        """
        split_path = self.model_path / split
        
        static_features = {
            '1d_nodes_static': pd.read_csv(split_path / '1d_nodes_static.csv'),
            '2d_nodes_static': pd.read_csv(split_path / '2d_nodes_static.csv'),
            '1d_edges_static': pd.read_csv(split_path / '1d_edges_static.csv'),
            '2d_edges_static': pd.read_csv(split_path / '2d_edges_static.csv'),
            '1d_edge_index': pd.read_csv(split_path / '1d_edge_index.csv'),
            '2d_edge_index': pd.read_csv(split_path / '2d_edge_index.csv'),
            '1d2d_connections': pd.read_csv(split_path / '1d2d_connections.csv')
        }
        
        return static_features
    
    def load_event_data(self, event_id: int, split: str = 'train') -> Dict[str, pd.DataFrame]:
        """
        Load dynamic features cho một event
        
        Args:
            event_id: ID của event
            split: 'train' hoặc 'test'
            
        Returns:
            Dictionary chứa dynamic features
        """
        event_path = self.model_path / split / f"event_{event_id}"
        
        if split == 'test':
            prefix = 'test_'
        else:
            prefix = ''
        
        dynamic_features = {
            '1d_nodes_dynamic': pd.read_csv(event_path / f'{prefix}1d_nodes_dynamic_all.csv'),
            '2d_nodes_dynamic': pd.read_csv(event_path / f'{prefix}2d_nodes_dynamic_all.csv'),
            '1d_edges_dynamic': pd.read_csv(event_path / f'{prefix}1d_edges_dynamic_all.csv'),
            '2d_edges_dynamic': pd.read_csv(event_path / f'{prefix}2d_edges_dynamic_all.csv'),
            'timesteps': pd.read_csv(event_path / f'{prefix}timesteps.csv')
        }
        
        return dynamic_features
    
    def normalize_features(self, 
                          features: np.ndarray, 
                          mean: np.ndarray = None, 
                          std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn hóa features
        
        Args:
            features: Array features cần chuẩn hóa
            mean: Mean để chuẩn hóa (None để tính từ data)
            std: Std để chuẩn hóa (None để tính từ data)
            
        Returns:
            normalized_features, mean, std
        """
        if mean is None:
            mean = np.nanmean(features, axis=0)
        if std is None:
            std = np.nanstd(features, axis=0)
            std[std == 0] = 1.0  # Tránh chia cho 0
        
        normalized = (features - mean) / std
        
        return normalized, mean, std
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        sequence_length: int,
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo sequences cho time series prediction
        
        Args:
            data: Time series data [timesteps, features]
            sequence_length: Độ dài sequence input
            prediction_horizon: Số bước dự đoán
            
        Returns:
            X: Input sequences [num_sequences, sequence_length, features]
            y: Target sequences [num_sequences, prediction_horizon, features]
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def extract_node_features(self, 
                            static_df: pd.DataFrame,
                            dynamic_df: pd.DataFrame,
                            timestep: int) -> np.ndarray:
        """
        Trích xuất node features tại một timestep
        
        Args:
            static_df: DataFrame chứa static features
            dynamic_df: DataFrame chứa dynamic features
            timestep: Timestep cần extract
            
        Returns:
            Node features array
        """
        # Get dynamic features at timestep
        dynamic_at_t = dynamic_df[dynamic_df['timestep'] == timestep]
        
        # Combine static and dynamic
        # Cần đảm bảo order của nodes
        node_ids = static_df['node_id'].values
        
        static_features = static_df.drop(columns=['node_id']).values
        
        # Match dynamic features to node order
        dynamic_features = []
        for node_id in node_ids:
            node_dynamic = dynamic_at_t[dynamic_at_t['node_id'] == node_id]
            if len(node_dynamic) > 0:
                dynamic_features.append(node_dynamic.drop(columns=['node_id', 'timestep']).values[0])
            else:
                # Nếu không có data, dùng zero
                num_dynamic_features = len(dynamic_at_t.columns) - 2  # Trừ node_id và timestep
                dynamic_features.append(np.zeros(num_dynamic_features))
        
        dynamic_features = np.array(dynamic_features)
        
        # Concatenate
        combined_features = np.concatenate([static_features, dynamic_features], axis=1)
        
        return combined_features
    
    def get_edge_index(self, edge_index_df: pd.DataFrame) -> np.ndarray:
        """
        Chuyển edge index DataFrame thành tensor format
        
        Args:
            edge_index_df: DataFrame with 'from_node' and 'to_node'
            
        Returns:
            Edge index array [2, num_edges]
        """
        edge_index = edge_index_df[['from_node', 'to_node']].values.T
        return edge_index
    
    def load_shapefiles(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Load shapefiles cho visualization
        
        Returns:
            Dictionary of GeoDataFrames
        """
        shapefile_path = self.model_path / 'shapefiles'
        
        shapefiles = {}
        
        # Load các shapefile có sẵn
        shapefile_types = ['Nodes_1D', 'Nodes_2D', 'Links_1D', 'Links_2D', 'Mesh_2D']
        
        for sf_type in shapefile_types:
            try:
                sf_file = list(shapefile_path.glob(f"{sf_type}*.shp"))
                if sf_file:
                    shapefiles[sf_type] = gpd.read_file(sf_file[0])
            except Exception as e:
                print(f"Warning: Could not load {sf_type}: {e}")
        
        return shapefiles
    
    def get_train_events(self) -> List[int]:
        """Lấy danh sách event IDs cho training"""
        train_path = self.model_path / 'train'
        event_dirs = [d for d in train_path.iterdir() if d.is_dir() and d.name.startswith('event_')]
        event_ids = [int(d.name.split('_')[1]) for d in event_dirs]
        return sorted(event_ids)
    
    def get_test_events(self) -> List[int]:
        """Lấy danh sách event IDs cho testing"""
        test_path = self.model_path / 'test'
        event_dirs = [d for d in test_path.iterdir() if d.is_dir() and d.name.startswith('event_')]
        event_ids = [int(d.name.split('_')[1]) for d in event_dirs]
        return sorted(event_ids)
    
    def create_node_mapping(self, static_features: Dict[str, pd.DataFrame]) -> Dict:
        """
        Tạo mapping giữa node IDs và indices
        
        Args:
            static_features: Dictionary of static feature DataFrames
            
        Returns:
            Dictionary với mappings
        """
        node_1d_ids = static_features['1d_nodes_static']['node_id'].values
        node_2d_ids = static_features['2d_nodes_static']['node_id'].values
        
        mapping = {
            'node_1d_to_idx': {nid: idx for idx, nid in enumerate(node_1d_ids)},
            'node_2d_to_idx': {nid: idx for idx, nid in enumerate(node_2d_ids)},
            'idx_to_node_1d': {idx: nid for idx, nid in enumerate(node_1d_ids)},
            'idx_to_node_2d': {idx: nid for idx, nid in enumerate(node_2d_ids)},
            'num_1d_nodes': len(node_1d_ids),
            'num_2d_nodes': len(node_2d_ids)
        }
        
        return mapping


class FeatureExtractor:
    """Trích xuất thêm các features từ spatial data"""
    
    @staticmethod
    def compute_distance_features(pos_x: np.ndarray, pos_y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Tính toán distance-based features
        
        Args:
            pos_x: X coordinates
            pos_y: Y coordinates
            
        Returns:
            Dictionary of distance features
        """
        features = {}
        
        # Distance to centroid
        centroid_x = np.mean(pos_x)
        centroid_y = np.mean(pos_y)
        features['distance_to_centroid'] = np.sqrt(
            (pos_x - centroid_x)**2 + (pos_y - centroid_y)**2
        )
        
        # Relative position (normalized)
        x_range = pos_x.max() - pos_x.min()
        y_range = pos_y.max() - pos_y.min()
        
        if x_range > 0:
            features['normalized_x'] = (pos_x - pos_x.min()) / x_range
        else:
            features['normalized_x'] = np.zeros_like(pos_x)
            
        if y_range > 0:
            features['normalized_y'] = (pos_y - pos_y.min()) / y_range
        else:
            features['normalized_y'] = np.zeros_like(pos_y)
        
        return features
    
    @staticmethod
    def compute_temporal_features(rainfall: np.ndarray, 
                                  window: int = 5) -> Dict[str, np.ndarray]:
        """
        Tính toán temporal features từ rainfall
        
        Args:
            rainfall: Rainfall time series [timesteps, num_nodes]
            window: Window size cho rolling statistics
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Cumulative rainfall
        features['cumulative_rainfall'] = np.cumsum(rainfall, axis=0)
        
        # Rolling mean
        df = pd.DataFrame(rainfall)
        features['rolling_mean'] = df.rolling(window=window, min_periods=1).mean().values
        
        # Rolling max
        features['rolling_max'] = df.rolling(window=window, min_periods=1).max().values
        
        # Rate of change
        features['rainfall_rate'] = np.diff(rainfall, axis=0, prepend=rainfall[0:1])
        
        return features