## Final Plan

### Step 1: Refactor `main.py` for cell-level predictions

**Changes to `loader.py`:**
- Modify `prepare_perturbation_batch()` to accept dynamic `n_cells` instead of fixed `batch_size`
- Add method to return the **exact same** control cells for both DE computations

**Changes to `main.py`:**
- For each target gene:
  - Get `n_cells = len(ground_truth_perturbed_cells)`
  - Sample exactly `n_cells` control cells (with fixed seed for reproducibility)
  - Run model prediction → `(n_cells, n_genes)` tensor
  - **Remove** the mean averaging step
  - Store cell-level data for DE analysis

### Step 2: Add hpdex integration and new metrics

**New file `src/utils/de_metrics.py`:**

Implements:
1. **`compute_de_metrics(control, pred, truth, gene_names)`** — per-target metrics:
   - Build `adata_truth` = control + ground_truth (labeled)
   - Build `adata_pred` = control + predictions (labeled)
   - Run `hpdex.pde()` on both → get DE tables
   - Compute:
     - **Pearson**: `pearsonr(truth_log2fc, pred_log2fc)`
     - **MSE**: `mean((truth_log2fc - pred_log2fc)²)`
     - **DES**: overlap of significant DE genes (FDR < 0.05), with the adjustment when `n_pred > n_true`

2. **`compute_pds(all_pred_deltas, all_truth_deltas, control_mean)`** — global metric:
   - For each perturbation `p`:
     - Compute L1 distance to all true deltas (excluding target gene)
     - Find rank of correct perturbation
     - `PDS_p = 1 - (rank - 1) / N`
   - Return mean PDS

**Changes to `main.py`:**
- After processing each target:
  - Call `compute_de_metrics()` → store per-target {pearson, mse, des}
  - Accumulate pseudobulk deltas for PDS
- After all targets:
  - Call `compute_pds()` → get overall PDS
- Save results:
  - `perturbation_metrics.csv`: per-target metrics
  - Log overall means + PDS

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/data/loader.py` | Add `n_cells` param, return consistent control cells |
| `src/main.py` | Remove mean comparison, integrate DE metrics, add PDS |
| `src/utils/de_metrics.py` | **New** — DES, Pearson, MSE on DE, PDS computation |
| `src/utils/metrics.py` | Keep or deprecate old mean-based metrics |

---

## Output Schema

**Per-target CSV columns:**
- `target_gene`
- `n_cells` (number of cells used)
- `pearson_log2fc`
- `mse_log2fc`
- `des`
- `n_de_truth` (number of significant DE genes in truth)
- `n_de_pred` (number of significant DE genes in prediction)

**Summary (logged):**
- Mean Pearson, Mean MSE, Mean DES
- **PDS** (global ranking metric)

---

## Files Modified/Created

### 1. `src/utils/de_metrics.py` (new)
- `build_de_adata()` — constructs AnnData for hpdex DE analysis
- `compute_de_comparison_metrics()` — runs hpdex on truth & pred, returns Pearson, MSE, DES
- `compute_des()` — implements VCC DES formula with |log2FC| fallback
- `compute_pds()` — implements VCC PDS ranking metric
- `compute_pseudobulk_delta()` — computes perturbation delta for PDS

### 2. `src/data/loader.py`
- Added `n_cells` param to `prepare_perturbation_batch()` (dynamic sample size)
- Added `seed` param for reproducibility
- Added `return_control_expr=True` option to return control expression
- Added `get_control_mean()` for PDS delta calculation
- Added `get_gene_names()` helper

### 3. `src/main.py`
- Removed mean-based comparison
- Samples `n_cells = len(ground_truth)` per target
- Calls `compute_de_comparison_metrics()` for per-target metrics
- Accumulates deltas and calls `compute_pds()` after all targets
- Reports: `pearson_log2fc`, `mse_log2fc`, `des`, `pds`, `n_de_truth`, `n_de_pred`

---

## Output CSV Schema

| Column | Description |
|--------|-------------|
| `target_gene` | Perturbation target |
| `n_cells` | Number of cells used |
| `pearson_log2fc` | Pearson correlation of log2FC |
| `mse_log2fc` | MSE of log2FC |
| `des` | Differential Expression Score |
| `pds` | Perturbation Discrimination Score |
| `n_de_truth` | # significant DE genes (truth) |
| `n_de_pred` | # significant DE genes (pred) |

---

To run:
```bash
python src/main.py --model_type baseline --threads 8
python src/main.py --model_type scgpt --threads 8
```