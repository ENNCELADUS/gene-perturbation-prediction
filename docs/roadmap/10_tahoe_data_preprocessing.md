# Tahoe Drug-Condition Preprocessing Plan

Align Tahoe preprocessing and splitting with the Norman pipeline, using drug-based
condition splits for unseen-drug generalization.

## Goals
- Normalize Tahoe counts to match Norman (log1p + library-size normalized).
- Keep condition-level splits (no cell-level leakage).
- Use **drug visibility** as the split rule (unseen drugs test-only).
- Preserve reproducibility with fixed seeds and saved split artifacts.

## Inputs & Assumptions
- Input H5AD: `data/tahoe/tahoe.h5ad` (raw counts in `X`, control counts in `layers["ctrl"]`).
- `obs` has: `condition`, `target_gene`, `drug`, `control`, `cell_line_id`, `plate`.
- Condition format: `target_gene+drug` (single target, single drug).
- Tahoe has no double perturbations; split strata are drug-based.

## Preprocessing (Norman-Aligned)
1. **Load (backed mode)** for memory safety.
2. **Preserve raw counts**:
   - If missing, set `layers["counts"] = X.copy()` to keep raw UMI counts.
   - Keep raw control counts in `layers["ctrl"]`.
3. **Normalize + log1p** (match Norman plan A):
   - `sc.pp.normalize_total(adata, target_sum=1e4)`
   - `sc.pp.log1p(adata)`
4. **Normalize control layer** (paired control expression):
   - Apply the same normalize/log1p on `layers["ctrl"]`.
   - Store normalized control in `layers["ctrl_norm"]` to keep raw control counts.
5. **Optional HVG parity with Norman**:
   - Select top 5k HVGs on normalized `X`.
   - Subset `X`, `layers["counts"]`, and `layers["ctrl_norm"]` to the same genes.
6. **Write output**:
   - `data/processed/tahoe/tahoe_log1p.h5ad` (document in README).

## Drug-Based Condition Split (Condition-Level Rule)
**Split unit = condition** (all cells of a condition stay together).

1. **Filter low-coverage conditions**:
   - Drop conditions with `< min_cells_per_condition` (default 5).
2. **Compute drug sizes**:
   - Count cells per drug (post-filter).
3. **Select unseen drugs** (test-only):
   - Use `unseen_drug_fraction` (default 0.2 ≈ 10 drugs).
   - Stratify by drug size quartiles to avoid skew (sample proportionally per quartile).
4. **Split remaining (seen) drugs**:
   - Split drugs into train/val/test by ratios (e.g., 0.8/0.1/0.1).
   - Assign all conditions for each drug to the same split.
5. **Define test strata**:
   - `drug_unseen`: conditions with unseen drugs.
   - `drug_seen_holdout`: conditions from seen drugs allocated to test.

## Artifacts
- `data/processed/tahoe/splits/tahoe_drug_split_seed{seed}.json`
  - `train_conditions`, `val_conditions`, `test_conditions`
  - `test_strata` (`drug_unseen`, `drug_seen_holdout`)
  - `unseen_drugs`, `seed`
- `data/processed/tahoe/README.md` documenting preprocessing + split stats.

## Implementation Tasks (Tahoe-Only)
1. `src/data/tahoe/preprocess_tahoe.py`
   - Normalize/log1p, optional HVG selection, write processed H5AD.
2. `src/data/tahoe/drug_condition_splits.py`
   - `TahoeConditionSplit` dataclass + `TahoeDrugSplitter`.
3. `src/data/tahoe/tahoe_dataset.py` (or extend loader)
   - Load Tahoe H5AD and apply drug-based condition splits.
4. `data/processed/tahoe/README.md`
   - Record input file, preprocessing params, split summary.

## Verification
- **Normalization checks**:
  - `X` is float log1p; `layers["counts"]` remains integer-like.
  - `layers["ctrl_norm"]` matches log1p scale of `X`.
- **Split checks**:
  - No overlap between train/val/test conditions.
  - Drug sets are disjoint across splits.
  - All conditions for unseen drugs are test-only.
- **Reporting**:
  - Count conditions/cells per split and per test stratum.
  - Record seed and `unseen_drug_fraction`.
