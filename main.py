"""
Main pipeline for UrbanFloodBench
"""

import argparse
import subprocess
from pathlib import Path
from utils import load_config, create_directory


def run_command(cmd):
    """Run shell command"""
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main(config_path: str, mode: str = 'all'):
    """
    Main pipeline
    
    Args:
        config_path: Path to config file
        mode: 'train', 'inference', or 'all'
    """
    config = load_config(config_path)
    
    # Create directories
    create_directory(config['data']['checkpoint_dir'])
    create_directory(config['data']['output_dir'])
    
    if mode in ['train', 'all']:
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        # Train Model 1
        print("\nTraining Model 1...")
        run_command([
            'python', 'train.py',
            '--config', config_path,
            '--model_id', '1'
        ])
        
        # Train Model 2
        print("\nTraining Model 2...")
        run_command([
            'python', 'train.py',
            '--config', config_path,
            '--model_id', '2'
        ])
    
    if mode in ['inference', 'all']:
        print("\n" + "="*50)
        print("INFERENCE PHASE")
        print("="*50)
        
        # Paths to best models
        checkpoint_dir = Path(config['data']['checkpoint_dir'])
        model1_path = checkpoint_dir / 'model_1' / 'best_model.pt'
        model2_path = checkpoint_dir / 'model_2' / 'best_model.pt'
        
        # Check if checkpoints exist
        if not model1_path.exists():
            print(f"Warning: Model 1 checkpoint not found at {model1_path}")
            print("Please train Model 1 first or specify checkpoint path")
            return
        
        if not model2_path.exists():
            print(f"Warning: Model 2 checkpoint not found at {model2_path}")
            print("Please train Model 2 first or specify checkpoint path")
            return
        
        # Run inference
        output_path = Path(config['data']['output_dir']) / 'submission.csv'
        
        print(f"\nRunning inference...")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Output: {output_path}")
        
        run_command([
            'python', 'inference.py',
            '--config', config_path,
            '--model1', str(model1_path),
            '--model2', str(model2_path),
            '--output', str(output_path)
        ])
        
        print(f"\nSubmission file created: {output_path}")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UrbanFloodBench Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'inference', 'all'],
                       help='Pipeline mode')
    
    args = parser.parse_args()
    
    main(args.config, args.mode)