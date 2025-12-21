# Data: Norman Dataset Splits and scGPT Fine-Tuning Input

## Norman split logic (current)

**Cell-level split (within-condition)**  
Implemented in `src/data/splits.py` and used by `load_perturb_data()` in
`src/data/perturb_dataset.py`.

- Scope: per perturbation condition, excluding `ctrl`.
- Uses only perturbed cells (`obs[control] == 0`).
- If a condition has fewer than `min_cells_per_condition` cells,
  it is dropped.
- Cells are shuffled with a fixed seed, then split into:
  - reference: `n_ref = n_cells - n_query`
  - query: `n_query = max(min_query_cells, int(n_cells * query_fraction))`
- Every kept condition appears in both ref and query (exact retrieval setting).
- Saved as JSON to the configured `split.output_path`
  (e.g. `data/norman/splits/cell_split_seed42.json`).
- Train/val/test are determined by the condition split; cell-level split
  defines ref/query within each condition.

**Condition-level split (generalization tracks)**  
Implemented in `src/data/condition_splits.py` and triggered by `--track`
in `src/main.py`.

- `in_dist`: random condition holdout by ratios.
- `unseen_combo`: all single-gene conditions in train; some gene-pair
  conditions held out for val/test only if both genes appear in train singles.
- `unseen_gene`: hold out a set of genes; any condition containing those genes
  goes to test; remaining conditions split into train/val.
- Saved as JSON to `condition_split.output_path`
  (e.g. `data/norman/splits/condition_split_in_dist_seed42.json`).
- Train/val/test:
  - `in_dist`: random condition split by `train_ratio/val_ratio/test_ratio`.
  - `unseen_combo`: train = all singles + remaining pairs; val/test = held-out pairs.
  - `unseen_gene`: test = any condition with held-out genes; train/val = remaining
    conditions by `train_ratio/val_ratio`.

## scGPT fine-tuning data expectations

Fine-tuning code path: `src/train/finetune.py` + `src/model/scgpt.py`.

**Required AnnData layout**
- `adata.X`: expression matrix (dense or sparse).
- `adata.obs["condition"]`: perturbation labels (e.g. `GENE1`, `GENE1+GENE2`).
- `adata.obs["control"]`: control indicator (1 for control, 0 for perturbed).
- `adata.var["gene_name"]` or `adata.var.index`: gene symbols used to map
  into scGPT vocab (non-matching genes are dropped).

**What gets fed into scGPT**
- `ScGPTEncoder.encode_adata(adata)` uses gene names to build token IDs and
  computes CLS embeddings via scGPT.
- Fine-tuning uses **reference cells only**:
  `dataset.get_ref_adata_for_conditions(dataset.all_conditions)`.
- Labels are taken from `adata.obs["condition"]`.

## Code interface (current)

**Load dataset and cell split**
```python
from src.data import load_perturb_data

dataset = load_perturb_data(
    h5ad_path="data/norman/perturb_processed.h5ad",
    split_path="data/norman/splits/cell_split_seed42.json",
    min_cells_per_condition=50,
    query_fraction=0.2,
    min_query_cells=10,
    seed=42,
)
```

**Condition split (required for train/val/test splits)**
```python
from src.data import ConditionSplitter

splitter = ConditionSplitter(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42)
cond_split = splitter.split(dataset.all_conditions, track="in_dist")  # or unseen_combo/unseen_gene
dataset.apply_condition_split(cond_split)
```
