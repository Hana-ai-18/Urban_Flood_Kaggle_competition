"""
Inference script with autoregressive prediction
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

from data import AutoregressiveDataLoader
from models import SpatioTemporalGNN
from utils import load_config, get_device, create_directory, save_predictions
from data.preprocessing import DataPreprocessor


@torch.no_grad()
def autoregressive_predict(model, event_data, config, device, model_id):
    """
    Autoregressive prediction cho một event
    
    Args:
        model: Trained model
        event_data: Tuple of (initial_1d, initial_2d, rainfall_2d, metadata)
        config: Configuration
        device: Device
        model_id: Model ID
        
    Returns:
        predictions: Dictionary of predictions
    """
    model.eval()
    
    initial_1d, initial_2d, rainfall_2d, metadata = event_data
    event_id = metadata['event_id']
    num_timesteps = metadata['num_timesteps']
    spinup_steps = metadata['spinup_steps']
    
    # Get sequence length
    seq_len = config['model']['sequence_length']
    
    # Initialize predictions storage
    all_predictions_1d = []
    all_predictions_2d = []
    
    # Get node IDs
    node_1d_ids = initial_1d['node_id'].unique()
    node_2d_ids = initial_2d['node_id'].unique()
    num_1d_nodes = len(node_1d_ids)
    num_2d_nodes = len(node_2d_ids)
    
    # Current state (water levels from spinup)
    current_state_1d = initial_1d.pivot(
        index='timestep', columns='node_id', values='water_level'
    ).values  # [spinup_steps, num_1d_nodes]
    
    current_state_2d = initial_2d.pivot(
        index='timestep', columns='node_id', values='water_level'
    ).values  # [spinup_steps, num_2d_nodes]
    
    # Predict từng timestep sau spinup
    for t in range(spinup_steps, num_timesteps):
        # Get sequence window
        if t < seq_len:
            # Use all available history
            seq_1d = current_state_1d[:t]
            seq_2d = current_state_2d[:t]
            
            # Pad if necessary
            if len(seq_1d) < seq_len:
                pad_len = seq_len - len(seq_1d)
                seq_1d = np.vstack([
                    np.zeros((pad_len, num_1d_nodes)),
                    seq_1d
                ])
                seq_2d = np.vstack([
                    np.zeros((pad_len, num_2d_nodes)),
                    seq_2d
                ])
        else:
            # Use last seq_len timesteps
            seq_1d = current_state_1d[-seq_len:]
            seq_2d = current_state_2d[-seq_len:]
        
        # Get rainfall at current timestep
        rainfall_t = rainfall_2d[rainfall_2d['timestep'] == t].pivot(
            index='timestep', columns='node_id', values='rainfall_depth'
        ).values[0]  # [num_2d_nodes]
        
        # Prepare batch (simplified - needs proper feature construction)
        # This is a placeholder - you need to properly construct graph batch
        # with static features, dynamic features, edge indices, etc.
        
        # For now, assuming we have a helper function
        batch = prepare_batch_for_inference(
            seq_1d, seq_2d, rainfall_t,
            model_id, config, device
        )
        
        # Predict
        pred_1d, pred_2d = model(batch)
        
        # Get predictions
        pred_1d_np = pred_1d.cpu().numpy()[0]  # [num_1d_nodes]
        pred_2d_np = pred_2d.cpu().numpy()[0]  # [num_2d_nodes]
        
        # Store predictions
        all_predictions_1d.append(pred_1d_np)
        all_predictions_2d.append(pred_2d_np)
        
        # Update current state (append prediction)
        current_state_1d = np.vstack([current_state_1d, pred_1d_np.reshape(1, -1)])
        current_state_2d = np.vstack([current_state_2d, pred_2d_np.reshape(1, -1)])
    
    # Format predictions for submission
    predictions = format_predictions(
        all_predictions_1d, all_predictions_2d,
        node_1d_ids, node_2d_ids,
        event_id, model_id,
        start_timestep=spinup_steps
    )
    
    return predictions


def prepare_batch_for_inference(seq_1d, seq_2d, rainfall, model_id, config, device):
    """
    Prepare batch for inference
    
    This is a simplified version - you need to implement proper graph construction
    """
    # TODO: Implement proper batch construction
    # Should include:
    # - Static node features
    # - Dynamic node features (including rainfall)
    # - Edge indices
    # - Proper PyG Data format
    
    raise NotImplementedError(
        "Need to implement proper batch construction for inference. "
        "This requires loading static features and constructing PyG Data object."
    )


def format_predictions(pred_1d_list, pred_2d_list, 
                      node_1d_ids, node_2d_ids,
                      event_id, model_id, start_timestep):
    """
    Format predictions theo submission format
    
    IMPORTANT: Submission format yêu cầu mỗi row là prediction cho 1 node tại 1 timestep
    Không phải chỉ timestep cuối cùng mà TẤT CẢ timesteps sau spinup
    
    Args:
        pred_1d_list: List of 1D predictions [num_timesteps, num_1d_nodes]
        pred_2d_list: List of 2D predictions [num_timesteps, num_2d_nodes]
        node_1d_ids: 1D node IDs
        node_2d_ids: 2D node IDs
        event_id: Event ID
        model_id: Model ID
        start_timestep: Starting timestep (after spinup, usually 10)
        
    Returns:
        DataFrame with predictions in submission format
    """
    rows = []
    
    # For each timestep sau spinup
    for t_idx, (pred_1d, pred_2d) in enumerate(zip(pred_1d_list, pred_2d_list)):
        timestep = start_timestep + t_idx
        
        # 1D nodes - mỗi node 1 row
        for node_idx, node_id in enumerate(node_1d_ids):
            rows.append({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': 1,
                'node_id': int(node_id),
                'timestep': timestep,  # Để track, sẽ remove sau
                'water_level': float(pred_1d[node_idx])
            })
        
        # 2D nodes - mỗi node 1 row
        for node_idx, node_id in enumerate(node_2d_ids):
            rows.append({
                'model_id': model_id,
                'event_id': event_id,
                'node_type': 2,
                'node_id': int(node_id),
                'timestep': timestep,  # Để track, sẽ remove sau
                'water_level': float(pred_2d[node_idx])
            })
    
    df = pd.DataFrame(rows)
    
    # Sort by timestep, node_type, node_id để đảm bảo consistency
    df = df.sort_values(['timestep', 'node_type', 'node_id']).reset_index(drop=True)
    
    # Remove timestep column (chỉ dùng để sort)
    df = df.drop(columns=['timestep'])
    
    return df


def run_inference(config_path: str, 
                 model_1_checkpoint: str,
                 model_2_checkpoint: str,
                 output_path: str):
    """
    Run inference on test set
    
    Args:
        config_path: Path to config
        model_1_checkpoint: Path to Model 1 checkpoint
        model_2_checkpoint: Path to Model 2 checkpoint
        output_path: Path to save predictions
    """
    # Load config
    config = load_config(config_path)
    device = get_device(config.get('device', 'cuda'))
    
    print("Running inference...")
    print("=" * 60)
    
    all_predictions = []
    
    # Process both models
    for model_id, checkpoint_path in [(1, model_1_checkpoint), (2, model_2_checkpoint)]:
        print(f"\n{'='*60}")
        print(f"Processing Model {model_id}")
        print(f"{'='*60}")
        
        # Load model
        model = SpatioTemporalGNN(
            static_1d_dim=config['model']['node_1d_static_dim'],
            static_2d_dim=config['model']['node_2d_static_dim'],
            dynamic_1d_dim=config['model']['node_1d_dynamic_dim'],
            dynamic_2d_dim=config['model']['node_2d_dynamic_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_gnn_layers=config['model']['num_gnn_layers'],
            num_temporal_layers=config['model']['num_temporal_layers'],
            gnn_type=config['model']['gnn_type'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', 'N/A')}")
        
        # Get test events
        preprocessor = DataPreprocessor(config['data']['data_root'], model_id)
        test_event_ids = preprocessor.get_test_events()
        
        print(f"✓ Test events: {test_event_ids}")
        
        # Create autoregressive data loader
        ar_loader = AutoregressiveDataLoader(
            config['data']['data_root'],
            model_id,
            test_event_ids,
            config,
            split='test'
        )
        
        # Predict for each event
        for event_data in tqdm(ar_loader.get_event_iterator(), 
                              total=len(test_event_ids),
                              desc=f'Model {model_id} Events'):
            
            predictions = autoregressive_predict(
                model, event_data, config, device, model_id
            )
            
            all_predictions.append(predictions)
    
    print(f"\n{'='*60}")
    print("Combining and formatting predictions...")
    print(f"{'='*60}")
    
    # Combine all predictions
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Verify required columns
    required_cols = ['model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    assert all(col in final_predictions.columns for col in required_cols), \
        f"Missing columns. Required: {required_cols}, Got: {final_predictions.columns.tolist()}"
    
    # Sort để đảm bảo consistency
    # Sort order: model_id, event_id, node_type, node_id
    final_predictions = final_predictions.sort_values(
        ['model_id', 'event_id', 'node_type', 'node_id']
    ).reset_index(drop=True)
    
    # Add row_id (0-indexed)
    final_predictions.insert(0, 'row_id', range(len(final_predictions)))
    
    # Reorder columns theo đúng format yêu cầu
    final_predictions = final_predictions[
        ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    ]
    
    # Validate data types
    final_predictions['row_id'] = final_predictions['row_id'].astype(int)
    final_predictions['model_id'] = final_predictions['model_id'].astype(int)
    final_predictions['event_id'] = final_predictions['event_id'].astype(int)
    final_predictions['node_type'] = final_predictions['node_type'].astype(int)
    final_predictions['node_id'] = final_predictions['node_id'].astype(int)
    final_predictions['water_level'] = final_predictions['water_level'].astype(float)
    
    # Check for NaN or inf values
    if final_predictions['water_level'].isna().any():
        print("WARNING: Found NaN values in predictions!")
        num_nan = final_predictions['water_level'].isna().sum()
        print(f"  Number of NaN values: {num_nan}")
        # Fill with 0 or median
        final_predictions['water_level'].fillna(0, inplace=True)
    
    if not np.isfinite(final_predictions['water_level']).all():
        print("WARNING: Found infinite values in predictions!")
        # Clip extreme values
        final_predictions['water_level'] = final_predictions['water_level'].clip(-1e6, 1e6)
    
    # Save predictions
    output_format = Path(output_path).suffix[1:]  # csv or parquet
    save_predictions(final_predictions, output_path, format=output_format)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Submission Summary")
    print(f"{'='*60}")
    print(f"Total predictions: {len(final_predictions):,}")
    print(f"Models: {final_predictions['model_id'].unique().tolist()}")
    print(f"Events per model:")
    for model_id in sorted(final_predictions['model_id'].unique()):
        events = final_predictions[final_predictions['model_id'] == model_id]['event_id'].unique()
        print(f"  Model {model_id}: {len(events)} events - {events.tolist()}")
    
    print(f"\nNode types:")
    print(f"  1D nodes: {(final_predictions['node_type'] == 1).sum():,} predictions")
    print(f"  2D nodes: {(final_predictions['node_type'] == 2).sum():,} predictions")
    
    print(f"\nWater level statistics:")
    print(f"  Min: {final_predictions['water_level'].min():.4f}")
    print(f"  Max: {final_predictions['water_level'].max():.4f}")
    print(f"  Mean: {final_predictions['water_level'].mean():.4f}")
    print(f"  Std: {final_predictions['water_level'].std():.4f}")
    
    print(f"\n{'='*60}")
    print(f"✓ Submission saved to: {output_path}")
    print(f"{'='*60}")
    
    # Save a copy of first 10 rows for inspection
    sample_path = Path(output_path).parent / 'submission_sample.csv'
    final_predictions.head(20).to_csv(sample_path, index=False)
    print(f"✓ Sample (first 20 rows) saved to: {sample_path}")
    
    return final_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to Model 1 checkpoint')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to Model 2 checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    run_inference(args.config, args.model1, args.model2, args.output)