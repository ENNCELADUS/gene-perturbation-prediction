# DDGCN K562 SL-Pair Reproduction — Design (exp10)

**Date:** 2026-06-19
**Status:** Approved (design); implementation pending
**Owner:** dependency-prediction track

## Goal

Reproduce the DDGCN model (Dual-Dropout Graph Convolutional Network; Cai et
al. 2020, *Bioinformatics*, doi 10.1093/bioinformatics/btaa211; official repo
https://github.com/CXX1113/Dual-DropoutGCN) and evaluate it on the K562 SL-pair
benchmark under **our** CV1/CV2/CV3 protocol and official per-anchor metrics —
the same protocol used by exp06/07/08/09 — so DDGCN slots into the existing
scoreboard as a new graph-model row.

Pinned hyperparameters (official defaults, confirmed against the repo and the
vendored port): `dropout=0.5`, `lr=0.01`.

## Scope

In scope:
- A clean PyTorch port of DDGCN (model, loss, training loop) in `src/ddgcn/`.
- One DDGCN trained per `(CV-split, fold)` on that fold's train-positive SL
  adjacency; produces a `9471 x 9471` score matrix scored with the official
  metrics.
- Run all of CV1/CV2/CV3 x 5 folds (30 training runs).
- Artifacts matching the exp08 layout (per-split subdirs + combined summary +
  manifest), plus a comparison table vs the exp06 floor.

Out of scope:
- Reproducing the paper's own Table 2 numbers (different gene universe,
  negatives, and metric — not comparable to our scoreboard).
- Hyperparameter tuning beyond the two pinned values.
- Adding gene features (GeneEffect, ESM2, transcripts) — DDGCN stays
  featureless; that would make it a different model.
- Slurm wrapper / multi-GPU. Local single-GPU via `uv run python -m ddgcn`.
- Internal validation split / best-epoch checkpointing — see Stopping Policy.

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Reproduction target | Model under OUR protocol, not paper numbers | Only apples-to-apples comparison with exp06-09 |
| Framework | PyTorch (port from `data/SL_benchmark/src/models/ddgcn.py`) | Official repo is PyTorch 1.1.0; vendored port matches defaults line-for-line; our env is torch 2.12 |
| Candidate universe | 9,471 K562-filtered genes (NOT DDGCN's 9,845) | Matches exp06-09 universe; rebuild adjacency on our index |
| Eval metric | `sl_benchmark_baseline.metrics.official_*` (per-anchor) | Single-sourced; NOT DDGCN-native flat F1@0.987 |
| Negatives | Rand 1:1, CV1/2/3 | Our CSVs are Rand-only; matches protocol |
| Train/test partition | `fold_split()` from our CSVs | NOT DDGCN's pickled folds |
| Stopping policy | Faithful loss-plateau, no val split, final-epoch score matrix | Zero leakage; test pairs never touch training |
| Compute | Local single GPU, `uv run python -m ddgcn` | Per user; no Slurm wrapper |
| Gene key | `gene_a_unified_id` / `gene_b_unified_id` | Same key `_build_gene_universe` uses (integer ids) |
| Adjacency symmetry | Symmetrize (`adj + adj.T`, clip to 1) | SL is symmetric; CSV stores each pair once |

## Architecture (ported, faithful)

DDGCN is a Graph Auto-Encoder for SL link prediction. Per fold:

- **Nodes** = the 9,471 genes. **Features** = identity matrix `eye(N)` (stream
  x1) and the train-positive adjacency itself (stream x2). Featureless,
  transductive.
- **Encoder** (shared weights across both streams): `gc1` `N -> 512` ReLU,
  `gc2` `512 -> 256`. Kipf-style `D^-0.5 (A+I) D^-0.5 X W`, no bias, Kaiming init.
- **Dual-dropout**: input `F.dropout(p=0.5)`; a single shared Bernoulli mask
  (inverted-dropout scaled) applied to both streams after `gc1`+ReLU and again
  in the decoder.
- **Decoder**: inner product `Z Z^T` per stream -> two logit matrices.
- **Fusion**: weighted geometric mean
  `(sigmoid(l1) * sigmoid(l2)^rho)^(1/(1+rho))`, `rho=1.0`, diagonal zeroed.
- **Loss**: class-weighted BCE-with-logits per stream over the
  `pair_mask = train_pos + train_neg` adjacency, `pos_weight=(N^2-E)/E`,
  `norm=N^2/(2(N^2-E))`, total `= loss1 + rho*loss2`.
- **Optimizer**: `Adam(lr=0.01, amsgrad=True)`.
- **Schedule**: `EPOCH=2000` max, `TOLERANCE_EPOCH=1000` (min epochs before
  early-stop check), early-stop when `|Δloss/loss| < STOP_THRESHOLD=1e-5`,
  `EVAL_INTER=50`. Final-epoch score matrix is used (no val-based selection).

## Port deltas vs the vendored code

The vendored `data/SL_benchmark/src/models/ddgcn.py` is the reference, but is
Python 3.7 / torch 1.5 / wandb / `cuda:0`-hardcoded and ranks via its own
pickled folds. The port must:

1. Replace deprecated `torch.sparse.FloatTensor(...)` with
   `torch.sparse_coo_tensor(...)` (env is torch 2.12).
2. Drop wandb; use `logging`.
3. Device-agnostic: `cuda` if available, else CPU (the loss currently moves
   logits to CPU for BCE; keep that — it's cheap and avoids a 9471x9471 dense
   target on GPU).
4. Build adjacency from our `fold_split()` train rows + `index_by_key` from
   `_build_gene_universe`, not from pickled folds / hardcoded `num_node=9845`.
5. Use the official `sl_benchmark_baseline` metric path, not `cal_metrics` /
   `Evaluator` / `ChecktoSave`.
6. Python 3.11 type hints, Google-style docstrings, functions < 50 lines.

## Module layout (`src/ddgcn/`)

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker |
| `config.py` | `DdgcnConfig` frozen dataclass + `load_config(yaml)` (mirrors `sl_dl_model/config.py`) |
| `model.py` | `GraphConvolution`, `GCNEncoder`, `InnerProductDecoder`, `GraphAutoEncoder`, `objective_weights()` |
| `graph.py` | `build_fold_adjacency()`, `normalize_adj()`, `to_torch_sparse()`, `identity_features()` |
| `train.py` | `train_fold()` -> `np.ndarray` score matrix (one fold); seeding, loss-plateau loop |
| `scoring.py` | `DdgcnProducer` with `score_matrix(symbols, gene_effects)`; `run_fold_ddgcn()` wrapping the protocol harness |
| `evaluate.py` | `run_cv(config)` -> summary DataFrame; artifact writing (exp08 layout) |
| `__main__.py` | `uv run python -m ddgcn run-cv --config ... [--split-type CV1] [--log-file ...]` |

Reused verbatim (imported, never reimplemented): `load_benchmark`,
`fold_split` (`sl_benchmark_baseline.data`); `GeneUniverse`,
`_build_gene_universe`, `_pair_indices`, `_metric_rows`, `_summarize`
(`sl_benchmark_baseline.evaluate`); `official_classification_metrics`,
`official_ranking_metrics` (`sl_benchmark_baseline.metrics`).

The `DdgcnProducer.score_matrix(symbols, gene_effects) -> np.ndarray` signature
intentionally matches the exp08 `hasattr(producer, "score_matrix")` seam so the
data flow mirrors `sl_dl_model/scoring.py`.

## Data flow (one fold)

```
all-CV CSV --load_benchmark--> frame
frame --_build_gene_universe--> universe (9471 genes, index_by_key)
frame --fold_split(split,fold)--> (train_df, test_df)
train_df[pos] --build_fold_adjacency(universe)--> A_pos (NxN, symmetric)
train_df[neg] --build_fold_adjacency(universe)--> A_neg
A_pos --normalize_adj/to_torch_sparse--> adj_norm ;  identity_features(N) --> x1 ; A_pos --> x2
train_fold(adj_norm, x1, x2, A_pos, A_neg, config) --> score_matrix (NxN, diag 0)
test_pos/test_neg/train_pos --_pair_indices--> pos_index/neg_index/seen_index
_metric_rows(score_matrix, pos_index, neg_index, seen_index) --> rows
```

## Determinism

Seed (`config.seed`, default 456 to match the port; configurable) sets
`torch.manual_seed`, `torch.cuda.manual_seed_all`, `numpy.random.seed` at the
start of each fold. Acceptance: two seeds agree within AUROC < 0.01 on one
CV2 fold (sanity, not a hard gate).

## Artifacts

Under `results/experiments/10_k562_sl_pair_ddgcn/run/`:
- `fold_metrics.csv` — `split_type, model, fold_id, slice, metric, value`
  (model = `"ddgcn"`, slice = `"full_universe"`).
- `summary.csv` — `split_type, model, slice, metric, mean, std`.
- `CV1/`, `CV2/`, `CV3/` per-split subdirs (each with fold_metrics/summary/manifest).
- `official_metrics_summary.csv` — combined summary across splits.
- `manifest.json` — input CSV + sha256, candidate_gene_count (assert 9471),
  seed, hyperparams (dropout, lr, rho, epochs, tolerance_epoch, stop_threshold,
  hidden dims), per-fold train edge counts, torch version, git commit, official
  metric source string.

## Acceptance criteria

1. All 30 folds train and produce a metric row set; CV1/CV2/CV3 all reported.
2. `candidate_gene_count == 9471` asserted in the run; score-matrix diagonal
   zeroed; metrics come only from `official_*_metrics`.
3. Comparison table written vs exp06 floor: exp06 XGB CV2 (AUROC 0.704, AUPR
   0.732, NDCG@10 0.042, MAP@10 0.034), CV3 (AUROC 0.596, NDCG@10 0.002), and
   degree-probe CV1 NDCG@10 0.197.
4. Unit tests pass for: model forward shapes + symmetry of fused output;
   dual-dropout active only in train mode; adjacency build + symmetrization;
   objective weights formula; producer score-matrix shape/diagonal; a
   tiny-synthetic end-to-end fold producing finite metrics.
5. `ruff check` + `ruff format` clean; `uv run python -m ddgcn run-cv --help`
   works without importing torch.
6. Honest framing in the experiment doc: DDGCN is transductive/featureless ≈ a
   learned degree probe; expect strong CV1 (topology-gameable, cf. degree probe
   NDCG@10 0.197) and near-floor CV3 cold-start. CV2/CV3 are the meaningful
   surfaces; a CV1-only win is a null finding.

## Key risk

DDGCN learns from the SL adjacency itself with no biological features. On CV1
(genes seen in train) it behaves like a learned nonlinear degree probe and may
top the board for the wrong reason; on CV3 (both genes cold-start) it has no
training edges touching test genes, so ranking collapses toward floor — exactly
the exp06 degree-probe pattern. The result is only credible if CV2/CV3 are
reported and contextualized, never CV1 alone.

