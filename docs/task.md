# Task Definition

The active task is inverse perturbation retrieval:

```text
observed perturbed cell state -> ranked candidate perturbation genes
```

Given a post-perturbation expression profile, the system ranks genes that could
explain the observed state. For multi-gene perturbations, the target label is the
set of perturbed genes parsed from the condition string.

Evaluation is single-cell by contract: each perturbed cell is one retrieval
query, and aggregate metrics are averaged over cells. Training stages may use
condition-level pseudobulk representations for simple baselines, but evaluation
must score individual held-out cells.

## Scope

The active codebase focuses on direct inverse retrieval models and baselines.
The deprecated forward-generation retrieval route has been removed.

In scope:

- gene-level ranking
- multi-label target gene supervision
- optional composition of ranked genes into candidate perturbation combinations
- leakage-aware evaluation with target-gene masking
- simple ML baselines and scGPT gene-score modeling

Out of scope in the active tree:

- Tahoe-specific preprocessing and training
- condition-aware forward expression generation
- reference database construction from predicted forward profiles
- forward-profile retrieval over generated candidate profiles

## Biological Semantics

The intended biological question is:

```text
Which perturbation genes are most consistent with this observed cellular state?
```

This preserves the reverse perturbation prediction semantics while allowing
models that do not reproduce the original scGPT paper's forward-model retrieval
mechanism.
