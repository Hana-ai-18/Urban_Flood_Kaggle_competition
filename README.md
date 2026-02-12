Tôi sẽ giải thích chi tiết về thuật toán và phương pháp xử lý data trong bài toán này.

## 🧠 THUẬT TOÁN SỬ DỤNG

### 1. **Spatio-Temporal Graph Neural Network (ST-GNN)**

Đây là thuật toán chính được implement trong `models/gnn_model.py`

#### **Kiến Trúc Tổng Thể:**

```
Input (Rainfall + Water States)
    ↓
Feature Projection (Static + Dynamic)
    ↓
┌─────────────────────────────────────┐
│  SPATIAL PROCESSING (GNN Layers)    │
│  ├── 1D Graph (Drainage Network)    │
│  │   └── GAT/GCN/GraphSAGE         │
│  └── 2D Graph (Surface Mesh)        │
│      └── GAT/GCN/GraphSAGE          │
└─────────────────────────────────────┘
    ↓
Cross-Graph Attention (1D ↔ 2D)
    ↓
┌─────────────────────────────────────┐
│  TEMPORAL PROCESSING                 │
│  └── LSTM/GRU/Transformer            │
└─────────────────────────────────────┘
    ↓
Output Heads (Water Level Prediction)
    ├── 1D Nodes → Water levels
    └── 2D Nodes → Water levels
```

### 2. **Chi Tiết Các Thành Phần**

#### **A. Graph Neural Networks (GNN)**

**Mục đích**: Học spatial dependencies giữa các nodes

**3 Variants được support:**

1. **GAT (Graph Attention Networks)** - Default
   ```python
   # Attention mechanism
   α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
   h_i' = σ(Σ α_ij W h_j)
   ```
   - **Ưu điểm**: Học được importance của neighbors
   - **Use case**: Khi một số connections quan trọng hơn

2. **GCN (Graph Convolutional Networks)**
   ```python
   # Spectral convolution
   h_i' = σ(Σ (1/√(d_i·d_j)) W h_j)
   ```
   - **Ưu điểm**: Simple, efficient
   - **Use case**: Đồ thị đồng nhất

3. **GraphSAGE**
   ```python
   # Neighborhood sampling
   h_i' = σ(W · CONCAT(h_i, AGGREGATE({h_j})))
   ```
   - **Ưu điểm**: Scalable cho large graphs
   - **Use case**: Khi có nhiều nodes

#### **B. Temporal Models**

**Mục đích**: Học temporal dependencies qua thời gian

**3 Options:**

1. **LSTM (Long Short-Term Memory)** - Default
   ```python
   # Gates
   f_t = σ(W_f · [h_{t-1}, x_t])  # Forget gate
   i_t = σ(W_i · [h_{t-1}, x_t])  # Input gate
   o_t = σ(W_o · [h_{t-1}, x_t])  # Output gate
   
   # Cell state update
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t])
   h_t = o_t ⊙ tanh(c_t)
   ```
   - **Ưu điểm**: Capture long-term dependencies
   - **Use case**: Time series with long memory

2. **GRU (Gated Recurrent Unit)**
   ```python
   # Simpler than LSTM
   z_t = σ(W_z · [h_{t-1}, x_t])  # Update gate
   r_t = σ(W_r · [h_{t-1}, x_t])  # Reset gate
   h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ tanh(W · [r_t ⊙ h_{t-1}, x_t])
   ```
   - **Ưu điểm**: Faster, fewer parameters
   - **Use case**: Khi cần speed

3. **Transformer**
   ```python
   # Self-attention
   Attention(Q,K,V) = softmax(QK^T/√d_k)V
   ```
   - **Ưu điểm**: Parallel processing
   - **Use case**: Long sequences

#### **C. Cross-Graph Attention**

**Mục đích**: Trao đổi information giữa 1D và 2D graphs

```python
# 1D nodes attend to 2D nodes
h_1d_enhanced = MultiHeadAttention(
    query=h_1d,
    key=h_2d,
    value=h_2d
)

# 2D nodes attend to 1D nodes  
h_2d_enhanced = MultiHeadAttention(
    query=h_2d,
    key=h_1d,
    value=h_1d
)
```

**Physical meaning**: 
- Surface water ảnh hưởng đến drainage system
- Drainage system ảnh hưởng đến surface flooding

---

## 📊 PHƯƠNG PHÁP XỬ LÝ DATA

### 1. **Data Loading & Preprocessing**

#### **A. Static Features** (`data/preprocessing.py`)

```python
class DataPreprocessor:
    def load_static_features(self, split='train'):
        # Load CSV files
        static_features = {
            '1d_nodes_static': pd.read_csv(...),
            '2d_nodes_static': pd.read_csv(...),
            '1d_edges_static': pd.read_csv(...),
            '2d_edges_static': pd.read_csv(...),
            '1d_edge_index': pd.read_csv(...),
            '2d_edge_index': pd.read_csv(...),
            '1d2d_connections': pd.read_csv(...)
        }
        return static_features
```

**Static features được process:**

**1D Nodes** (6 features):
- Position (x, y)
- Depth, Invert elevation, Surface elevation
- Base area

**2D Nodes** (9 features):
- Position (x, y)
- Area, Roughness
- Elevation (min, centroid)
- Aspect, Curvature, Flow accumulation

**Edges** (1D: 7 features, 2D: 5 features):
- Length, Diameter, Shape, Roughness, Slope
- Relative positions

#### **B. Dynamic Features**

```python
def load_event_data(self, event_id, split='train'):
    # Load time-varying data
    dynamic_features = {
        '1d_nodes_dynamic': pd.read_csv(...),  # Water level, inlet flow
        '2d_nodes_dynamic': pd.read_csv(...),  # Rainfall, water level, volume
        '1d_edges_dynamic': pd.read_csv(...),  # Flow, velocity
        '2d_edges_dynamic': pd.read_csv(...),  # Flow, velocity
        'timesteps': pd.read_csv(...)
    }
    return dynamic_features
```

**Dynamic features:**
- **Input**: Rainfall intensity (known)
- **State**: Water level, volume (predicted)
- **Flow**: Discharge, velocity (auxiliary)

#### **C. Normalization**

```python
def normalize_features(self, features, mean=None, std=None):
    # Z-score normalization
    if mean is None:
        mean = np.nanmean(features, axis=0)
    if std is None:
        std = np.nanstd(features, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std
```

**Normalization strategy:**
- Compute statistics from **training data only**
- Apply same statistics to validation/test
- Handle missing values (NaN)
- Prevent division by zero

### 2. **Graph Construction** (`data/dataset.py`)

#### **A. Build Graph Structure**

```python
def _build_edge_index(self, node_type='1d'):
    # Read connectivity
    edge_index_df = self.static_features[f'{node_type}_edge_index']
    
    # Map node IDs to indices
    mapping = self.node_mapping[f'node_{node_type}_to_idx']
    
    from_nodes = edge_index_df['from_node'].map(mapping).values
    to_nodes = edge_index_df['to_node'].map(mapping).values
    
    # Create edge index [2, num_edges]
    edge_index = np.stack([from_nodes, to_nodes], axis=0)
    
    return torch.LongTensor(edge_index)
```

**Graph structure:**
```
1D Graph:
  Nodes: Drainage junctions
  Edges: Pipes connecting junctions
  
2D Graph:
  Nodes: Surface grid cells
  Edges: Surface flow connections
  
1D-2D Coupling:
  Edges: Drainage inlets/outlets
```

#### **B. Create Sequences**

```python
def create_sequences(self, data, sequence_length, prediction_horizon=1):
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Input: past sequence_length timesteps
        X.append(data[i:i + sequence_length])
        
        # Target: next prediction_horizon timesteps
        y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
    
    return np.array(X), np.array(y)
```

**Sequence structure:**
```
Timesteps:  0  1  2  3  4  5  6  7  8  9  10 11 12 ...
            |--------X---------|  y              (sample 1)
               |--------X---------|  y           (sample 2)
                  |--------X---------|  y        (sample 3)
                  
X: [seq_len=10, num_nodes, features]
y: [num_nodes, 1]  (next water level)
```

### 3. **Feature Engineering**

#### **A. Spatial Features**

```python
class FeatureExtractor:
    @staticmethod
    def compute_distance_features(pos_x, pos_y):
        # Distance to centroid
        centroid_x = np.mean(pos_x)
        centroid_y = np.mean(pos_y)
        
        distance = np.sqrt(
            (pos_x - centroid_x)**2 + 
            (pos_y - centroid_y)**2
        )
        
        # Normalized positions
        x_norm = (pos_x - pos_x.min()) / (pos_x.max() - pos_x.min())
        y_norm = (pos_y - pos_y.min()) / (pos_y.max() - pos_y.min())
        
        return {
            'distance_to_centroid': distance,
            'normalized_x': x_norm,
            'normalized_y': y_norm
        }
```

**Engineered features:**
- Distance to centroid (relative importance)
- Normalized coordinates (position encoding)
- Relative elevations (hydraulic gradient)

#### **B. Temporal Features**

```python
@staticmethod
def compute_temporal_features(rainfall, window=5):
    # Cumulative rainfall
    cumulative = np.cumsum(rainfall, axis=0)
    
    # Rolling statistics
    rolling_mean = pd.DataFrame(rainfall).rolling(window).mean()
    rolling_max = pd.DataFrame(rainfall).rolling(window).max()
    
    # Rate of change
    rate = np.diff(rainfall, axis=0, prepend=rainfall[0:1])
    
    return {
        'cumulative_rainfall': cumulative,
        'rolling_mean': rolling_mean,
        'rolling_max': rolling_max,
        'rainfall_rate': rate
    }
```

**Temporal aggregations:**
- Cumulative rainfall (total water input)
- Rolling statistics (recent trends)
- Rate of change (intensity changes)

### 4. **Data Augmentation** (Optional)

```python
# Noise injection
def add_noise(features, noise_level=0.01):
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise

# Temporal shifting
def time_shift(sequence, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift+1)
    return np.roll(sequence, shift, axis=0)
```

### 5. **Batch Construction** (`data/data_loader.py`)

```python
class FloodGraphDataset(Dataset):
    def __getitem__(self, idx):
        # Get sample
        sample = self.samples[idx]
        
        # Extract sequence
        seq_1d = []  # [seq_len, num_1d_nodes, features]
        seq_2d = []  # [seq_len, num_2d_nodes, features]
        
        for t in range(start_t, end_t):
            # Combine static + dynamic at time t
            features_1d_t = concat([static_1d, dynamic_1d[t]])
            features_2d_t = concat([static_2d, dynamic_2d[t]])
            
            seq_1d.append(features_1d_t)
            seq_2d.append(features_2d_t)
        
        # Create PyG Data object
        return Data(
            x_1d_static=static_1d,
            x_2d_static=static_2d,
            x_1d_dynamic=seq_1d,
            x_2d_dynamic=seq_2d,
            edge_index_1d=edge_index_1d,
            edge_index_2d=edge_index_2d,
            y_1d=target_1d,
            y_2d=target_2d
        )
```

---

## 🔄 WORKFLOW HOÀN CHỈNH

```
1. DATA LOADING
   ├── Load static CSVs (geometry, topology)
   ├── Load dynamic CSVs (time series)
   └── Load shapefiles (visualization)

2. PREPROCESSING
   ├── Parse CSVs → DataFrames
   ├── Create node/edge mappings
   ├── Build graph structure
   └── Compute statistics (mean, std)

3. FEATURE ENGINEERING
   ├── Normalize features (Z-score)
   ├── Add spatial features (distances)
   ├── Add temporal features (rolling stats)
   └── Combine static + dynamic

4. SEQUENCE CREATION
   ├── Sliding window (seq_len=10)
   ├── Create input sequences X
   └── Create target sequences y

5. GRAPH CONSTRUCTION
   ├── Node features tensor
   ├── Edge index tensor
   └── PyG Data object

6. BATCHING
   ├── Collate multiple graphs
   ├── Batch processing
   └── GPU transfer

7. MODEL FORWARD
   ├── Feature projection
   ├── GNN layers (spatial)
   ├── Cross-attention (1D↔2D)
   ├── LSTM layers (temporal)
   └── Output heads (prediction)

8. TRAINING
   ├── Forward pass
   ├── Compute loss (Standardized RMSE)
   ├── Backward pass
   ├── Update weights
   └── Validation

9. INFERENCE (Autoregressive)
   ├── Use spinup (10 steps)
   ├── Predict t+1
   ├── Use prediction as input for t+2
   └── Repeat for all timesteps

10. SUBMISSION
    ├── Format predictions
    ├── Create CSV
    └── Validate & submit
```

---

## 📈 METRICS & LOSS

### **Standardized RMSE**

```python
def standardized_rmse(pred, target, std_dev):
    # Regular RMSE
    rmse = sqrt(mean((pred - target)^2))
    
    # Standardize by pre-computed std
    standardized = rmse / std_dev
    
    return standardized
```

**Hierarchical averaging:**
```
1. Node-level RMSE (per node)
2. Node-type average (1D vs 2D)
3. Event-level average (across timesteps)
4. Model-level average (across events)
5. Final score (across models)
```

---

## 💡 KEY INSIGHTS

1. **Graph Structure**: Captures physical connectivity (pipes, surface mesh)
2. **Temporal Modeling**: Learns how water propagates over time
3. **Multi-scale**: Handles both 1D (pipes) and 2D (surface) simultaneously
4. **Autoregressive**: Uses own predictions for future timesteps
5. **Physics-informed**: Graph structure reflects actual hydraulic network

Đây là một **state-of-the-art** approach kết hợp **Graph ML** và **Time Series** để giải quyết bài toán flood prediction phức tạp! 🚀