# Data Contract

This repository currently targets Norman-style CRISPR Perturb-seq data for
inverse perturbation retrieval. Tahoe-specific preprocessing has been removed
from the active source tree.

## AnnData Input

The expected input is configured by `data.h5ad_path`, for example:

```yaml
data:
  h5ad_path: data/norman/perturb_processed.h5ad
```

The AnnData object must provide:

- `X`: cell-by-gene expression matrix.
- `var_names`: gene symbols used as candidate perturbation genes.
- `obs.condition`: perturbation label such as `ctrl`, `GENE+ctrl`, or
  `GENE1+GENE2`.

Optional fields used when available:

- `obs.batch`
- `obs.cell_type`

These can be listed in `data.control_match_keys` so scGPT training/evaluation
samples control cells from compatible batches or cell types.

## Condition Labels

Condition strings are parsed as `+`-separated gene names. The token `ctrl` is
ignored when extracting target genes.

Examples:

```text
ctrl              -> {}
FOSB+ctrl         -> {FOSB}
CNN1+MAPK1        -> {CNN1, MAPK1}
```

## Split Artifact

The condition-level split is configured under `condition_split` and saved to
`condition_split.output_path`.

```yaml
condition_split:
  output_path: data/norman/splits/norman_condition_split_hard_seed42.json
  seed: 42
```

The split is condition-based, not cell-random. Cells from the same perturbation
condition must not appear across train, validation, and test splits.

The active split supports:

- train conditions
- validation conditions
- test conditions
- optional test strata metadata for seen/unseen gene-combination regimes

## Model Inputs

Simple baselines consume expression features and condition-derived target gene
sets.

The scGPT gene-score model consumes:

- perturbed query cell expression
- optional matched control cells
- target labels derived from `obs.condition`
- scGPT token ids mapped from `var_names`

For leakage-sensitive evaluation, target gene expression can be masked before
ranking. This is controlled by each model config's `evaluation.mask` field.
