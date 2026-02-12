"""
Validate submission file format
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def validate_submission(submission_path: str, verbose: bool = True):
    """
    Validate submission file theo yêu cầu của competition
    
    Args:
        submission_path: Path to submission file
        verbose: Print detailed validation results
        
    Returns:
        is_valid: Boolean indicating if submission is valid
        errors: List of error messages
    """
    errors = []
    warnings = []
    
    if verbose:
        print("=" * 60)
        print("VALIDATING SUBMISSION FILE")
        print("=" * 60)
        print(f"File: {submission_path}\n")
    
    # 1. Check file exists
    if not Path(submission_path).exists():
        errors.append(f"File not found: {submission_path}")
        return False, errors
    
    # 2. Load file
    try:
        if submission_path.endswith('.csv'):
            df = pd.read_csv(submission_path)
        elif submission_path.endswith('.parquet'):
            df = pd.read_parquet(submission_path)
        else:
            errors.append(f"Invalid file format. Must be .csv or .parquet")
            return False, errors
        
        if verbose:
            print(f"✓ File loaded successfully")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {df.columns.tolist()}\n")
    
    except Exception as e:
        errors.append(f"Failed to load file: {str(e)}")
        return False, errors
    
    # 3. Check required columns
    required_columns = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    else:
        if verbose:
            print(f"✓ All required columns present")
    
    # 4. Check column order
    if df.columns.tolist()[:6] != required_columns:
        warnings.append(
            f"Column order mismatch. Expected: {required_columns}, "
            f"Got: {df.columns.tolist()[:6]}"
        )
    else:
        if verbose:
            print(f"✓ Column order correct")
    
    # 5. Check for extra columns
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns:
        warnings.append(f"Extra columns found (will be ignored): {extra_columns}")
    
    # 6. Check data types
    if verbose:
        print(f"\nData Types:")
    
    for col in required_columns[:5]:  # All except water_level should be int
        if not pd.api.types.is_integer_dtype(df[col]):
            errors.append(f"Column '{col}' must be integer type, got {df[col].dtype}")
        elif verbose:
            print(f"  {col}: {df[col].dtype} ✓")
    
    if not pd.api.types.is_numeric_dtype(df['water_level']):
        errors.append(f"Column 'water_level' must be numeric, got {df['water_level'].dtype}")
    elif verbose:
        print(f"  water_level: {df['water_level'].dtype} ✓")
    
    # 7. Check for missing values
    if verbose:
        print(f"\nMissing Values Check:")
    
    for col in required_columns:
        num_missing = df[col].isna().sum()
        if num_missing > 0:
            errors.append(f"Column '{col}' has {num_missing} missing values")
        elif verbose:
            print(f"  {col}: No missing values ✓")
    
    # 8. Check for infinite values in water_level
    num_inf = np.isinf(df['water_level']).sum()
    if num_inf > 0:
        errors.append(f"'water_level' has {num_inf} infinite values")
    elif verbose:
        print(f"\n✓ No infinite values in water_level")
    
    # 9. Check model_id values
    valid_model_ids = {1, 2}
    actual_model_ids = set(df['model_id'].unique())
    
    if not actual_model_ids.issubset(valid_model_ids):
        errors.append(
            f"Invalid model_id values. Expected: {valid_model_ids}, "
            f"Got: {actual_model_ids}"
        )
    elif len(actual_model_ids) != 2:
        errors.append(
            f"Must have exactly 2 model_ids (1 and 2). "
            f"Found: {len(actual_model_ids)} - {actual_model_ids}"
        )
    elif verbose:
        print(f"✓ model_id values valid: {sorted(actual_model_ids)}")
    
    # 10. Check node_type values
    valid_node_types = {1, 2}
    actual_node_types = set(df['node_type'].unique())
    
    if not actual_node_types.issubset(valid_node_types):
        errors.append(
            f"Invalid node_type values. Expected: {valid_node_types}, "
            f"Got: {actual_node_types}"
        )
    elif verbose:
        print(f"✓ node_type values valid: {sorted(actual_node_types)}")
    
    # 11. Check row_id is sequential
    if not (df['row_id'] == range(len(df))).all():
        errors.append("row_id must be sequential starting from 0")
    elif verbose:
        print(f"✓ row_id is sequential (0 to {len(df)-1})")
    
    # 12. Statistics
    if verbose:
        print(f"\n" + "=" * 60)
        print("SUBMISSION STATISTICS")
        print("=" * 60)
        
        print(f"\nTotal predictions: {len(df):,}")
        
        print(f"\nPredictions by model:")
        for model_id in sorted(df['model_id'].unique()):
            count = (df['model_id'] == model_id).sum()
            events = df[df['model_id'] == model_id]['event_id'].unique()
            print(f"  Model {model_id}: {count:,} predictions across {len(events)} events")
        
        print(f"\nPredictions by node type:")
        for node_type in sorted(df['node_type'].unique()):
            count = (df['node_type'] == node_type).sum()
            label = "1D (drainage)" if node_type == 1 else "2D (surface)"
            print(f"  Node type {node_type} ({label}): {count:,} predictions")
        
        print(f"\nWater level statistics:")
        print(f"  Min:    {df['water_level'].min():.6f}")
        print(f"  Max:    {df['water_level'].max():.6f}")
        print(f"  Mean:   {df['water_level'].mean():.6f}")
        print(f"  Median: {df['water_level'].median():.6f}")
        print(f"  Std:    {df['water_level'].std():.6f}")
        
        # Check for duplicate predictions
        duplicate_keys = ['model_id', 'event_id', 'node_type', 'node_id']
        duplicates = df.duplicated(subset=duplicate_keys, keep=False)
        num_duplicates = duplicates.sum()
        
        if num_duplicates > 0:
            errors.append(f"Found {num_duplicates} duplicate predictions (same model, event, node_type, node_id)")
        else:
            print(f"\n✓ No duplicate predictions")
    
    # 13. Print validation result
    if verbose:
        print(f"\n" + "=" * 60)
        print("VALIDATION RESULT")
        print("=" * 60)
    
    is_valid = len(errors) == 0
    
    if is_valid:
        if verbose:
            print("✓ SUBMISSION IS VALID")
            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for i, warning in enumerate(warnings, 1):
                    print(f"  {i}. {warning}")
    else:
        if verbose:
            print(f"✗ SUBMISSION IS INVALID")
            print(f"\nErrors ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            
            if warnings:
                print(f"\nWarnings ({len(warnings)}):")
                for i, warning in enumerate(warnings, 1):
                    print(f"  {i}. {warning}")
    
    if verbose:
        print("=" * 60)
    
    return is_valid, errors


def print_submission_sample(submission_path: str, n: int = 10):
    """
    Print first n rows of submission file
    
    Args:
        submission_path: Path to submission
        n: Number of rows to print
    """
    if submission_path.endswith('.csv'):
        df = pd.read_csv(submission_path)
    else:
        df = pd.read_parquet(submission_path)
    
    print(f"\nFirst {n} rows of submission:")
    print("=" * 100)
    print(df.head(n).to_string(index=False))
    print("=" * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate submission file')
    parser.add_argument('submission', type=str, help='Path to submission file')
    parser.add_argument('--sample', action='store_true', help='Print sample rows')
    parser.add_argument('--quiet', action='store_true', help='Only show errors')
    
    args = parser.parse_args()
    
    is_valid, errors = validate_submission(args.submission, verbose=not args.quiet)
    
    if args.sample:
        print_submission_sample(args.submission)
    
    # Exit code
    exit(0 if is_valid else 1)