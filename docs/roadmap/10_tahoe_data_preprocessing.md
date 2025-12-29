# Tahoe Drug-Condition Preprocessing Plan

Align Tahoe preprocessing and splitting with the Norman pipeline, using a
drug-based condition split that mirrors Norman's unseen-gene logic.

## Goals
- Normalize Tahoe counts to match Norman (log1p + library-size normalized).
- Keep condition-level splits (no cell-level leakage).
- Use **unseen single-target genes** as the split rule (test-only).
- Preserve reproducibility with fixed seeds and saved split artifacts.

## Inputs & Assumptions
- Input H5AD: `data/tahoe/tahoe.h5ad` (raw counts in `X`, control counts in `layers["ctrl"]`).
- `obs` has: `condition`, `target_gene`, `drug`, `control`, `cell_line_id`, `plate`.
- Condition format: `target_gene+drug` (single target per condition).
- Tahoe has multi-target drugs; split strata are based on target lists.

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

## Drug-Based Condition Split (Norman-Style Logic)
**Split unit = condition** (all cells of a condition stay together).

### 1) Single-target drugs (unseen genes)
1. **Filter low-coverage conditions**:
   - Drop conditions with `< min_cells_per_condition` (default 5).
2. **Identify single-target drugs**:
   - Drugs with exactly one unique target gene.
   - Unique single-target genes ≈ 25 (Tahoe has 28 single-target drugs).
3. **Select unseen genes** (`G_unseen`) from unique single-target genes:
   - Use `unseen_single_gene_fraction` (default 0.25 → 6 genes).
   - Or explicit `n_unseen_single_genes` (recommended 6–8).
   - Stratify by gene cell count quartiles to avoid skew.
4. **Assign single-target drugs**:
   - If drug's target gene ∈ `G_unseen` → **test only**.
   - Otherwise → **train/val only** (split by `single_seen_val_ratio`, default 0.1).

### 2) Multi-target drugs (Norman-style seen tiers)
Let `u(d) = |targets(d) ∩ G_unseen|` for multi-target drug `d`.

1. **Test-only if unseen genes appear**:
   - If `u(d) ≥ 1` → **test only** (no leakage via combos).
2. **Split remaining multi-target drugs**:
   - `M_seen = {d | u(d)=0}` split into train/val/test
   - Ratios by drug count: **train 60–70%**, **val 10–15%**, **test 20–30%**
   - Default: `multi_train_ratio=0.65`, `multi_val_ratio=0.1`
   - Stratify by target-count buckets (2,3,4,5,6,7,13) for balance

### 3) Test strata
- `single_unseen`: single-target drugs with unseen genes
- `multi_unseen`: multi-target drugs containing unseen genes
- `multi_seen_holdout`: multi-target drugs from `M_seen` held out for test

## Artifacts
- `data/processed/tahoe/splits/tahoe_drug_split_seed{seed}.json`
  - `train_conditions`, `val_conditions`, `test_conditions`
  - `test_strata` (`single_unseen`, `multi_unseen`, `multi_seen_holdout`)
  - `unseen_genes`, `seed`
- `data/processed/tahoe/README.md` documenting preprocessing + split stats.

## Implementation Tasks (Tahoe-Only)
1. `src/data/tahoe/preprocess_tahoe.py`
   - Normalize/log1p, optional HVG selection, write processed H5AD.
2. `src/data/tahoe/drug_condition_splits.py`
   - `TahoeConditionSplit` dataclass + Norman-style `TahoeDrugSplitter`.
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
  - All conditions for `G_unseen` single-target drugs are test-only.
  - Multi-target drugs with any unseen gene are test-only.
- **Reporting**:
  - Count conditions/cells per split and per test stratum.
  - Record seed and `unseen_single_gene_fraction` (or `n_unseen_single_genes`).
