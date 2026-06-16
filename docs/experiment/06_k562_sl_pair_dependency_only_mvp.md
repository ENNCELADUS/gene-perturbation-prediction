# K562 SL Pair Dependency-Only MVP

Run status: design spec only on 2026-06-16; not implemented.

## Scope

This experiment establishes the first baseline for **gene-pair synthetic-lethality
(SL) link prediction** on the K562-mappable benchmark. It is the floor that all
later observed-B and frozen-AIVC pair features must beat.

The supervised object is an **undirected gene pair**, not a single gene:

```text
(gene_a, gene_b) -> P(SL) in [0, 1]
```

This is **not** DepMap GeneEffect regression. The existing pipeline (experiments
01-05, Tracks 1 and 2) predicts the per-gene scalar `C = GeneEffect(K562, g)`.
This experiment predicts the per-pair label `D = sl_label(gene_a, gene_b)`, and
*consumes* `C`-style evidence as input features. See [`CONTEXT.md`](../../CONTEXT.md)
glossary entries `C` and `D`.

The MVP uses **only** the two genes' K562 DepMap GeneEffect scalars as input.
Observed-B, predicted-B, and frozen-AIVC features are explicitly out of scope and
are added in later experiments against this same benchmark and metrics.

## Data Contract

Canonical input: [`data/k562_SL_benchmark_minimal.csv`](../../data/k562_SL_benchmark_minimal.csv)
(verified 2026-06-16). Built by
[`scripts/build_k562_sl_benchmark.py`](../../scripts/build_k562_sl_benchmark.py)
from the official `CV1_1.npy` Rand 1:1 split cache, filtered to pairs whose two
genes both have a numeric K562 DepMap GeneEffect value for `ACH-000551`.

| Property | Value |
| --- | --- |
| Rows | 331,730 |
| Split | `CV1` only (pair-level holdout) |
| Folds | 5 (`fold_id` 0-4), each with its own `split_role` train/test |
| Balance | 1:1 (165,865 positive / 165,865 negative) |
| Negatives | `Rand` (random unknown pairs; noisy non-SL, not confirmed) |
| Union genes | 9,471 |
| Pair ordering | Stored one way only; treat as undirected |

Required columns consumed by the MVP:

```text
pair_id, fold_id, split_role, sl_label,
gene_a_symbol, gene_b_symbol,
gene_a_k562_gene_effect, gene_b_k562_gene_effect
```

**CV1 honesty constraint:** in CV1 a test pair's two genes may appear in other
training pairs. Results are pair-level only and are **not** evidence of
held-out-gene generalization. No SL biological claim is made; this is a
benchmark-adapter floor.

## Model Task Definition

- **Input (MVP):** the two scalars `ea = gene_a_k562_gene_effect`,
  `eb = gene_b_k562_gene_effect`.
- **Output:** `P(SL | pair)`, a single float used for both thresholded
  classification and ranking.
- **Symmetry requirement:** the pair is undirected and stored in one ordering,
  so the model must be swap-invariant **by feature construction**, never by
  relying on the stored order.
- **Evaluation unit:** one fold. Fit on `split_role == "train"`, score
  `split_role == "test"`, aggregate mean +/- std across the 5 folds.

## Feature Construction

All features are symmetric functions of `(ea, eb)` so that swapping the two genes
leaves the input unchanged:

```text
f_min     = min(ea, eb)
f_max     = max(ea, eb)
f_sum     = ea + eb
f_product = ea * eb
f_absdiff = |ea - eb|
```

Five features. Standardized (zero mean, unit variance) using **train-fold
statistics only**; the same transform is applied to the test fold. No raw `ea`,
`eb`, or pair ordering is exposed as a feature.

## Models

Three models in one results table. A is the MVP; C is a mandatory leakage/degree
control shipped alongside it; B is a nonlinear second row.

| Id | Model | Inputs | Role |
| --- | --- | --- | --- |
| `A` | Symmetric logistic regression | 5 symmetric features (standardized) | Minimal honest floor: "two GeneEffect values -> probability". |
| `B` | XGBoost gradient-boosted trees | same 5 symmetric features | Captures nonlinear "both-essential" interactions. |
| `C` | Preferential-attachment frequency probe | per-gene positive-degree from train fold | Control: how gameable is CV1 from pair-degree alone? |

**Model A / B** see no gene identity, so they cannot exploit CV1 pair-degree
structure. That is intentional: it keeps the floor clean.

**Model C** is not a real predictor. For each fold it counts, using **training
positives only**, how often each gene appears (`pos_degree[g]`). It scores a test
pair by the preferential-attachment product:

```text
score(a, b) = pos_degree[a] * pos_degree[b]
```

Genes unseen in the train fold get degree 0. If `C` approaches `A`, CV1 results
are dominated by pair-frequency structure rather than dependency biology, and any
later feature-based gain must be read against that.

## Metrics

Every model emits a float per pair, so the same scores feed classification and
ranking.

**Classification (per fold):**

- AUROC
- AUPR (average precision)
- F1 at threshold 0.5

**Ranking (per fold, pair-level):** rank all test pairs in the fold by score and
treat positives as relevant:

- NDCG@{10, 20, 50}
- Recall@{10, 20, 50}
- Precision@{10, 20, 50}

These are **pair-level** ranking metrics over the flat test-pair list. This
differs from the official benchmark's per-gene-anchor candidate-partner ranking;
the divergence is recorded in the run manifest and is not claimed equivalent.
The frequency probe `C` has many tied scores (e.g. all degree-0 pairs); ties are
broken deterministically by `pair_id` so ranking metrics are reproducible.

All metrics are reported as mean +/- std across the 5 folds.

## Evaluation Protocol

For each `fold_id` in 0..4:

1. Slice `train` and `test` rows for the fold from the minimal CSV.
2. Build the 5 symmetric features; fit the standardizer on train only.
3. Fit `A` and `B` on train features and `sl_label`.
4. Build `C` positive-degree map from train positives only.
5. Score `test` rows with each model.
6. Compute classification and ranking metrics per model.

Aggregate to mean +/- std across folds. Determinism via fixed seeds for `A`
(solver), `B` (tree construction), and any sampling.

## Module Layout

A lean standalone module under `src/`, separate from the `dependency_baseline`
pipeline (this predicts `D`, not `C`):

```text
src/sl_benchmark_baseline/
  __init__.py
  config.py     # SLBaselineConfig frozen dataclass; defaults centralized, no magic numbers
  data.py       # load minimal CSV, per-fold train/test slicing, schema validation
  features.py   # symmetric pair features + train-fit standardizer
  models.py     # A logreg, B xgboost, C frequency-probe specs
  metrics.py    # classification + pair-level ranking metrics
  evaluate.py   # per-fold CV loop, aggregation, manifest
  __main__.py   # thin entrypoint: uv run python -m sl_benchmark_baseline
```

Config is a small frozen dataclass with defaults (input CSV path, model
hyperparameters, ranking `k` values, seed, output dir); CLI flags override. A
YAML loader is trivial to add later but is not required for v0.

## Outputs

Written under a run directory (gitignored, alongside other experiment artifacts):

- `fold_metrics.csv` - one row per (model, fold, metric).
- `summary.csv` - mean +/- std per (model, metric) across folds.
- `manifest.json` - config, seed, input CSV checksum, leakage-scope notes,
  ranking-semantics caveat.

## Non-Goals (scope-creep guard)

- No CV2 / CV3 -> **no held-out-gene generalization claims** (minimal CSV is CV1
  only).
- No observed-B, predicted-B, or frozen-AIVC (State / Tahoe-X1) features yet.
- No `Exp` / `Dep` negative sampling (both use DepMap and would leak against the
  DepMap GeneEffect feature). `Rand` only.
- No graph / knowledge-graph / SynLethDB-graph models (the official heavy track).
- No new `vcc-dep-baseline` subcommand; standalone module only.
- No new cell lines and no SL biological claims (pair-level adapter only).
- No reproduction of the official conda / PyTorch-Geometric environment.

## Leakage Notes

- Using `GeneEffect(K562, g)` as the input feature against `Rand` negatives is
  **low** leakage risk: random negatives are not selected via DepMap signal.
- This becomes **high** risk if/when `Exp` or `Dep` negative sampling is used,
  because those negatives are chosen using DepMap expression or dependency
  correlation. The manifest flags this explicitly; do not switch negative
  sampling without re-evaluating leakage.
- The `C` frequency probe is the explicit check on CV1 pair-degree gameability.

## Success Criteria

v0 success is **not** a high AUROC. It is:

1. A no-leakage, reproducible per-fold loop over the canonical benchmark.
2. A clean three-model floor (`A`, `B`, `C`) with classification and ranking
   metrics in one table.
3. An interpretable read of how much of CV1 is explained by dependency-only
   signal (`A`/`B`) versus pair-degree structure (`C`).

Any later observed-B or frozen-AIVC pair model must beat this floor on the same
folds and metrics before a transcriptomic or virtual-cell gain is claimed.
