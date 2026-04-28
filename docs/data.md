# Data Contract

This repository currently targets Norman-style CRISPR Perturb-seq data for
inverse perturbation retrieval. Tahoe-specific preprocessing has been removed
from the active source tree.

## AnnData Input

The expected input is configured by `data_config.h5ad_path`, for example:

```yaml
data_config:
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

scGPT uses `data_config.control_n_samples` to sample control cells for
delta-style cell embeddings.

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

The split is condition-based, not cell-random. Cells from the same perturbation
condition must not appear across train, validation, and test splits.

Small experiments can define the condition-level split inline under
`data_config.condition_split`.

```yaml
data_config:
  condition_split:
    train: ["A"]
    validation: ["B"]
    test: ["A+B"]
```

For Norman runs, the default scGPT config uses a generated gene-held-out split
artifact because the source `.h5ad` does not need to contain train/test columns:

```yaml
run_config:
  stages: ["prepare", "train", "evaluate"]

data_config:
  condition_split_path: results/scgpt/norman_gene_heldout_split.yaml
  condition_split:
    train: []
    validation: []
    test: []
  split_config:
    strategy: gene_heldout
    train_gene_fraction: 0.7
    validation_gene_fraction: 0.1
    test_gene_fraction: 0.2
    min_cells_per_condition: 1
```

`src/scgpt/prepare.py` reads unique non-control conditions from
`obs.condition`, normalizes labels such as `GENE+ctrl` to `GENE`, partitions
perturbation genes into 70% train, 10% validation, and 20% test, then assigns
conditions by held-out gene membership:

- all genes in the train partition -> train
- contains a validation gene and no test gene -> validation
- contains a test gene -> test

The generated artifact stores both the gene partition and the induced condition
split. It also records final condition fractions and deltas from the target
70/10/20 ratio, because graph edges can make condition counts differ from gene
counts. Later stages read the artifact through `data_config.condition_split_path`
when inline split lists are empty.

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
ranking when the model implements masking. Evaluation options live under
`evaluation_config`.
