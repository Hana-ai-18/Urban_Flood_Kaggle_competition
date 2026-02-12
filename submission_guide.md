# Submission Format Guide - UrbanFloodBench

## 📋 Quick Reference

### Required File Format

```csv
row_id,model_id,event_id,node_type,node_id,water_level
0,1,3,1,50,233.3301
1,1,3,1,51,234.5672
2,1,3,1,52,235.1234
...
```

### Column Specifications

| Column | Type | Valid Values | Description |
|--------|------|--------------|-------------|
| `row_id` | int | 0, 1, 2, ... | Sequential identifier starting from 0 |
| `model_id` | int | 1, 2 | Urban model ID |
| `event_id` | int | varies | Test event identifier |
| `node_type` | int | 1, 2 | 1=1D drainage, 2=2D surface |
| `node_id` | int | varies | Unique node identifier |
| `water_level` | float | any | Predicted water level |

## ⚠️ Critical Understanding

### What Each Row Represents

**IMPORTANT**: Each row = ONE prediction for ONE node at ONE specific timestep

This is **NOT**:
- ❌ One row per node (averaging all timesteps)
- ❌ Only the final timestep prediction
- ❌ Summary statistics

This **IS**:
- ✅ One row for each (model, event, timestep, node_type, node) combination
- ✅ ALL timesteps after spinup (typically 10+ timesteps per node)
- ✅ Complete time series for each node

### Example Breakdown

If you have:
- Model 1 with 2 test events
  - Event 1: 30 timesteps (10 spinup + 20 predict), 50 nodes (30 1D + 20 2D)
  - Event 2: 25 timesteps (10 spinup + 15 predict), 50 nodes (30 1D + 20 2D)

Then your submission needs:
```
Event 1: 20 timesteps × 50 nodes = 1,000 rows
Event 2: 15 timesteps × 50 nodes =   750 rows
Total for Model 1:               1,750 rows
```

Plus similar for Model 2 = **THOUSANDS of rows total**

## 📊 Submission Structure

### Hierarchy

```
Submission File
├── Model 1
│   ├── Event 1
│   │   ├── Timestep 10 (after spinup)
│   │   │   ├── 1D Node 0: water_level
│   │   │   ├── 1D Node 1: water_level
│   │   │   ├── ...
│   │   │   ├── 2D Node 0: water_level
│   │   │   └── ...
│   │   ├── Timestep 11
│   │   │   └── (all nodes again)
│   │   └── ...
│   └── Event 2
│       └── ...
└── Model 2
    └── ...
```

### Sorting Recommendation

Sort by: `model_id → event_id → timestep → node_type → node_id`

This ensures:
- Consistent ordering
- Easy validation
- Logical grouping

## ✅ Validation Checklist

Before submitting, verify:

- [ ] File is .csv or .parquet
- [ ] Exactly 6 columns in correct order
- [ ] All columns have correct data types
- [ ] row_id is sequential (0, 1, 2, ...)
- [ ] Both model_id values present (1 and 2)
- [ ] All test events included
- [ ] Both node_types present (1 and 2)
- [ ] No missing values (NaN)
- [ ] No infinite values
- [ ] No duplicate (model, event, node_type, node) within same timestep
- [ ] All timesteps after spinup included

## 🔧 Using Validation Script

```bash
# Basic validation
python validate_submission.py submission.csv

# With sample preview
python validate_submission.py submission.csv --sample

# Quiet mode (errors only)
python validate_submission.py submission.csv --quiet
```

### Validation Output Example

```
============================================================
VALIDATING SUBMISSION FILE
============================================================
File: submission.csv

✓ File loaded successfully
  Rows: 15,432
  Columns: ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']

✓ All required columns present
✓ Column order correct

Data Types:
  row_id: int64 ✓
  model_id: int64 ✓
  event_id: int64 ✓
  node_type: int64 ✓
  node_id: int64 ✓
  water_level: float64 ✓

✓ No missing values
✓ No infinite values
✓ model_id values valid: [1, 2]
✓ node_type values valid: [1, 2]
✓ row_id is sequential (0 to 15431)

============================================================
SUBMISSION STATISTICS
============================================================

Total predictions: 15,432

Predictions by model:
  Model 1: 8,500 predictions across 3 events
  Model 2: 6,932 predictions across 2 events

Predictions by node type:
  Node type 1 (1D drainage): 6,173 predictions
  Node type 2 (2D surface): 9,259 predictions

Water level statistics:
  Min:    198.234567
  Max:    267.891234
  Mean:   232.456789
  Median: 231.234567
  Std:    15.678901

✓ No duplicate predictions

============================================================
VALIDATION RESULT
============================================================
✓ SUBMISSION IS VALID
============================================================
```

## 🚨 Common Errors

### Error 1: Wrong Number of Rows

**Problem**: Submission has too few rows
```
Expected: ~15,000 rows
Got: 150 rows
```

**Cause**: Only predicting final timestep or only one timestep per node

**Fix**: Predict ALL timesteps after spinup for EACH node

### Error 2: Missing Timesteps

**Problem**: Only have predictions for timestep 10, missing 11-30
```
Timestep 10: 100 predictions ✓
Timestep 11: MISSING ✗
Timestep 12: MISSING ✗
```

**Fix**: Implement autoregressive prediction for all timesteps

### Error 3: Duplicate Entries

**Problem**: Same (model, event, node_type, node_id) appears multiple times at different timesteps
```
This is CORRECT (different timesteps):
model_id=1, event_id=1, node_type=1, node_id=5, timestep=10
model_id=1, event_id=1, node_type=1, node_id=5, timestep=11

This is WRONG (duplicate within timestep):
model_id=1, event_id=1, node_type=1, node_id=5, water_level=230.1
model_id=1, event_id=1, node_type=1, node_id=5, water_level=230.2
```

**Fix**: Ensure one prediction per node per timestep

### Error 4: Missing Nodes

**Problem**: Not all nodes are included
```
Expected 1D nodes: 0-49 (50 nodes)
Got: 0-45 (46 nodes)
Missing: 46, 47, 48, 49
```

**Fix**: Include ALL nodes from test data

### Error 5: Wrong Column Order

**Problem**:
```
Got: model_id, row_id, event_id, ...
Expected: row_id, model_id, event_id, ...
```

**Fix**: Reorder columns to match specification exactly

## 💡 Best Practices

### 1. Test with Sample Data First

```bash
# Create sample submission
python create_sample_submission.py --output test.csv

# Validate it
python validate_submission.py test.csv
```

### 2. Build Submission Incrementally

```python
all_predictions = []

for model_id in [1, 2]:
    for event_id in test_events:
        event_predictions = predict_event(model, event_id)
        all_predictions.append(event_predictions)

final_df = pd.concat(all_predictions)
```

### 3. Always Validate Before Submission

```bash
python validate_submission.py submission.csv
# Only submit if validation passes!
```

### 4. Keep Intermediate Files

Save predictions per event for debugging:
```python
# Save intermediate
event_pred.to_csv(f'debug/model_{model_id}_event_{event_id}.csv')

# Final combined
final_pred.to_csv('submission.csv')
```

### 5. Double-Check Row Counts

```python
# Expected rows calculation
def calculate_expected_rows():
    total = 0
    for model_id, events_info in test_structure.items():
        for event_id, info in events_info.items():
            num_nodes = info['num_1d'] + info['num_2d']
            num_timesteps = info['total_timesteps'] - 10  # After spinup
            total += num_nodes * num_timesteps
    return total

expected = calculate_expected_rows()
actual = len(submission_df)

assert actual == expected, f"Row count mismatch: {actual} vs {expected}"
```

## 📝 Submission Workflow

### Step-by-Step

1. **Train models**
   ```bash
   python train.py --model_id 1
   python train.py --model_id 2
   ```

2. **Run inference**
   ```bash
   python inference.py \
       --model1 checkpoints/model_1/best_model.pt \
       --model2 checkpoints/model_2/best_model.pt \
       --output submission.csv
   ```

3. **Validate**
   ```bash
   python validate_submission.py submission.csv
   ```

4. **Review sample**
   ```bash
   python validate_submission.py submission.csv --sample
   ```

5. **Submit to Kaggle**
   ```bash
   kaggle competitions submit -c urban-flood-modelling \
       -f submission.csv \
       -m "SpatioTemporalGNN v1.0"
   ```

## 🎯 Quick Checks

Before submission, run these quick checks:

```python
import pandas as pd

df = pd.read_csv('submission.csv')

# 1. Column check
assert list(df.columns) == ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']

# 2. Row ID check
assert (df['row_id'] == range(len(df))).all()

# 3. Model ID check
assert set(df['model_id'].unique()) == {1, 2}

# 4. Node type check
assert set(df['node_type'].unique()) == {1, 2}

# 5. No NaN check
assert not df['water_level'].isna().any()

# 6. No duplicates check (within same timestep)
# This check depends on your data structure

print("✓ All quick checks passed!")
```

## 📚 Additional Resources

- **Competition Page**: https://kaggle.com/competitions/urban-flood-modelling
- **Evaluation Details**: See competition page
- **Sample Submission**: Use `create_sample_submission.py`
- **Validation Script**: `validate_submission.py`

---

**Remember**: When in doubt, validate! The validation script will catch most errors before you submit.