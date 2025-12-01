# Update Scripts for New Data Split

## Changes Required

### 1. Update `scripts/data_process/split_data.py`

The old script randomly selects 10% test genes. Update to:

- Read fixed 30 test genes from `data/raw/test_set.csv`
- Use remaining 120 genes for training (train.h5ad includes control + 120 perturb genes)
- Test set contains only the 30 test genes (no control)

### 2. Update `scripts/finetune.sh`

- Uncomment Step 3 (evaluation) and update to use new API
- Add `PYTHONPATH` export for hpdex (same as zeroshot.sh)

### 3. `scripts/convert_to_gears.py` - No changes needed

The GEARS conversion only processes train.h5ad which now contains 120 genes (excluding 30 test genes).

### 4. `scripts/zeroshot.sh` - No changes needed

Already uses correct API.

---

## Files to Modify

| File | Change |
|------|--------|
| `scripts/data_process/split_data.py` | Use fixed test genes from CSV instead of random 10% |
| `scripts/finetune.sh` | Uncomment eval step, add PYTHONPATH for hpdex |