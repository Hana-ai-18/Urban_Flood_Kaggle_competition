# Update Notes - Submission Format Fix

## 📋 Những Gì Đã Được Cập Nhật

Dựa trên yêu cầu submission format của competition, tôi đã cập nhật và bổ sung các file sau:

### 1. ✅ File Đã Được Sửa

#### `inference.py`
**Thay đổi chính:**
- ✅ Sửa function `format_predictions()` để đảm bảo **mỗi row = 1 prediction cho 1 node tại 1 timestep**
- ✅ Không chỉ predict timestep cuối cùng mà predict **TẤT CẢ timesteps** sau spinup
- ✅ Add proper sorting: `model_id → event_id → timestep → node_type → node_id`
- ✅ Add validation checks trong `run_inference()`:
  - Check for NaN values
  - Check for infinite values
  - Proper data type conversion
  - Detailed summary statistics
- ✅ Save sample file (first 20 rows) để inspect

**Before:**
```python
# Chỉ predict 1 timestep hoặc format không rõ ràng
```

**After:**
```python
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
            'water_level': float(pred_1d[node_idx])
        })
    # ... tương tự cho 2D nodes
```

#### `README.md`
**Thêm:**
- ✅ Section "Submission Format" với chi tiết về format requirements
- ✅ Validation instructions
- ✅ Sample creation instructions
- ✅ Updated usage examples

### 2. ✅ File Mới Được Tạo

#### `validate_submission.py` (242 lines)
**Chức năng:**
- ✅ Validate submission file format
- ✅ Check all required columns
- ✅ Verify column order
- ✅ Check data types
- ✅ Check for missing/infinite values
- ✅ Verify model_id and node_type values
- ✅ Check for duplicates
- ✅ Print detailed statistics
- ✅ Exit with error code if validation fails

**Usage:**
```bash
# Validate
python validate_submission.py submission.csv

# With sample preview
python validate_submission.py submission.csv --sample

# Quiet mode
python validate_submission.py submission.csv --quiet
```

#### `create_sample_submission.py` (193 lines)
**Chức năng:**
- ✅ Tạo dummy sample submission với random data
- ✅ Tạo template từ actual test data structure
- ✅ Đúng format requirements
- ✅ Có thể dùng để test pipeline

**Usage:**
```bash
# Dummy sample
python create_sample_submission.py --output sample.csv

# From actual data
python create_sample_submission.py --from-data data/raw --output template.csv
```

#### `SUBMISSION_GUIDE.md` (500+ lines)
**Nội dung:**
- ✅ Detailed submission format explanation
- ✅ **Critical understanding**: Each row = 1 node at 1 timestep
- ✅ Example breakdown with calculations
- ✅ Validation checklist
- ✅ Common errors và cách fix
- ✅ Best practices
- ✅ Step-by-step workflow
- ✅ Quick checks code examples

## 🎯 Điểm Quan Trọng Cần Nhớ

### Submission Format Requirements

**MỖI ROW = 1 PREDICTION CHO 1 NODE TẠI 1 TIMESTEP**

Không phải:
- ❌ 1 row per node (average tất cả timesteps)
- ❌ Chỉ predict timestep cuối cùng
- ❌ Summary statistics

Mà là:
- ✅ 1 row cho mỗi (model, event, timestep, node_type, node) combination
- ✅ TẤT CẢ timesteps sau spinup (thường 10+ timesteps per node)
- ✅ Complete time series cho mỗi node

###Ví Dụ Tính Toán

```
Nếu có:
- Event 1: 30 timesteps total (10 spinup + 20 predict)
- 50 nodes (30 1D + 20 2D)

Thì cần:
20 timesteps × 50 nodes = 1,000 rows cho event này
```

### Column Order (QUAN TRỌNG)

```csv
row_id,model_id,event_id,node_type,node_id,water_level
0,1,3,1,50,233.3301
1,1,3,1,51,234.5672
...
```

## 📊 Cấu Trúc File Hiện Tại

```
urban_flood_bench/
├── config/
│   └── config.yaml
├── data/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── gnn_model.py
│   ├── temporal_models.py
│   └── ensemble.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── helpers.py
│   └── visualization.py
│
├── train.py
├── inference.py                    ⭐ UPDATED
├── main.py
│
├── validate_submission.py          ⭐ NEW
├── create_sample_submission.py     ⭐ NEW
│
├── requirements.txt
├── README.md                       ⭐ UPDATED
├── USAGE_GUIDE.md
├── PROJECT_SUMMARY.md
└── SUBMISSION_GUIDE.md             ⭐ NEW
```

## 🚀 Workflow Được Đề Xuất

### 1. Test Với Sample Data

```bash
# Tạo sample submission
python create_sample_submission.py --output test.csv

# Validate
python validate_submission.py test.csv
```

### 2. Development Workflow

```bash
# Train models
python train.py --model_id 1
python train.py --model_id 2

# Run inference
python inference.py \
    --config config/config.yaml \
    --model1 checkpoints/model_1/best_model.pt \
    --model2 checkpoints/model_2/best_model.pt \
    --output submission.csv

# Validate BEFORE submitting
python validate_submission.py submission.csv

# If valid, submit to Kaggle
kaggle competitions submit -c urban-flood-modelling \
    -f submission.csv -m "Description"
```

### 3. Debugging

```bash
# Check sample output
python validate_submission.py submission.csv --sample

# Check intermediate files
ls -lh outputs/submission_sample.csv
head -20 outputs/submission_sample.csv
```

## ✅ Validation Checklist

Trước khi submit, đảm bảo:

- [ ] File có đúng 6 columns: `row_id, model_id, event_id, node_type, node_id, water_level`
- [ ] Columns theo đúng thứ tự
- [ ] row_id sequential từ 0
- [ ] Có cả 2 model_id (1 và 2)
- [ ] Có cả 2 node_type (1 và 2)
- [ ] Không có missing values
- [ ] Không có infinite values
- [ ] Không có duplicate entries
- [ ] Tất cả test events đều có
- [ ] Tất cả timesteps sau spinup đều có

## 🐛 Common Issues & Solutions

### Issue 1: Too Few Rows

**Symptom:**
```
Expected: ~15,000 rows
Got: 150 rows
```

**Solution:** Đảm bảo predict ALL timesteps, không chỉ timestep cuối

### Issue 2: Wrong Format

**Symptom:**
```
Validation fails on column check
```

**Solution:** Check column names và order chính xác

### Issue 3: Missing Data

**Symptom:**
```
Missing some events or nodes
```

**Solution:** Verify autoregressive loop covers all events và nodes

## 📚 Tài Liệu Liên Quan

- `SUBMISSION_GUIDE.md` - Chi tiết về submission format
- `USAGE_GUIDE.md` - Hướng dẫn sử dụng tổng thể
- `PROJECT_SUMMARY.md` - Technical overview
- `README.md` - Quick start guide

## 🎯 Next Steps

1. **Test validation script** với sample data
2. **Implement full autoregressive prediction** trong inference.py
3. **Test với actual data** (khi có)
4. **Verify row counts** match expected
5. **Submit và monitor** leaderboard

## ⚠️ Important Notes

1. **Autoregressive Prediction**: 
   - Must predict ALL timesteps after spinup
   - Each prediction uses previous predictions as input
   - Maintain proper sequence window

2. **Data Consistency**:
   - Node IDs must match test data exactly
   - Event IDs must match test data exactly
   - All nodes must be included

3. **Validation is Mandatory**:
   - Always run validation before submission
   - Fix ALL errors (not just warnings)
   - Double-check row counts

---

**Status**: ✅ Code updated and ready for testing with actual data

**Last Updated**: 2024