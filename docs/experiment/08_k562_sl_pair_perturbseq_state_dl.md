# Experiment 08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

**Status:** Implementation complete (Phase 0–4 code + unit tests). Phase 2/3
cluster artifacts are now available through 2026-06-22; the table below records
the selected best split-validation fold for each CV1/CV2/CV3 phase. These rows
are **single selected folds, not 5-fold means**, and several rerun directories
still have missing final `_fold_results/*.result.json` files.
**Design spec:** `docs/superpowers/specs/2026-06-17-exp08-state-dl-sl-ranking-design.md`.
**Implementation plan:** `docs/superpowers/plans/2026-06-17-exp08-state-dl-sl-ranking.md`.
**Orchestration plan:** `docs/superpowers/plans/2026-06-18-exp08-fold-parallel-orchestration.md`.
**Configs:** `configs/experiments/08_k562_sl_pair_state_dl/`.
**Package:** `src/sl_dl_model/`.

## Rationale

exp06 (dependency-only) and exp07 (real-bag features) establish the SL-pair baseline.
exp08 asks: can a frozen-STATE encoder + a trainable adapter fed by ESM2 gene
embeddings produce a transcriptomic signal that beats exp06 on CV2/CV3 per-anchor
ranking, and generalizes to held-out genes?

The local STATE checkpoint is a closed-vocabulary one-hot model (2,024 perturbation
genes, 16.3% of the SL universe). exp08 replaces STATE's one-hot `pert_encoder` with
a trainable adapter that consumes ESM2 protein embeddings, keeping the 8-layer Llama
backbone frozen. All 9,471 genes flow through one coordinate system; real gwps bags
supervise the covered train genes only (leakage-free CV2/CV3).

<!-- BODY -->

## Task Definitions

Identical to exp06/07 (spec §1). Classification: `(a,b) → sl_label`. Ranking
(primary): anchor `a` → rank all 9,471 candidate partners, evaluate against held-out
positives, with seen/train pairs masked and the diagonal zeroed.

## Data

- SL pairs: `data/SL_benchmark/derived/k562_depmap_rand_1to1/all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv`
  (CV1/CV2/CV3 Rand 1:1, 9,471-gene universe).
- ESM2 embeddings: `data/esm2/k562_sl_universe_esm2_650M.npz` (precomputed via
  `scripts/precompute_esm2_embeddings.py`, UniProt + HF `esm2_t33_650M_UR50D`).
- gwps bags: `data/exp08_cache/k562_gwps_bags.npz` (cached from
  `data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad`,
  6,070 covered genes).
- STATE checkpoint: `model/checkpoints/state/ST-HVG-Replogle/fewshot/k562/`.

## Model

**Architecture:** `PertAdapter` (ESM2 1280-d → 328-d pert token) → frozen STATE
backbone (8-layer Llama) → predicted response bag → `MeanStdPool` (bag → e_g) →
`SymmetricPairHead` (`[e_a+e_b, |e_a−e_b|, e_a⊙e_b]` + GeneEffect block → logit).

**Training:** 3-part loss (SL BCE + adapter token-distill + real-bag supervision),
warm-up schedule, Adam on trainable parameters only, and no RankNet term
(`lambda_rank=0.0`, recorded-but-unconsumed). Pairs are batched by gradient
accumulation (`batch_pairs`, default 1024): one optimizer step per batch, batch
loss reduced as the mean of per-pair losses. Accelerate launches use fold-level
task parallelism via `PartialState`/`gather_object`: each rank owns distinct
`(split_type, fold_id)` jobs, with no DDP gradient all-reduce. Bag supervision
uses covered train genes only (leakage-free held-out gene eval). Seed 17, max 20
epochs, lr 1e-3.

**Early stopping:** each epoch the model is validated by pair-AUROC over the
fold's **own test split** (SynLethDB `valid_rat=0` style), best-epoch weights are
restored, and `early_stop_patience` (default 5) epochs without improvement stops
training. Best-epoch selection begins after `warmup_epochs`; the reported
official metric is best-epoch only. **Honesty note:** because best-epoch
selection reads the test fold, exp08-vs-exp06 is **selection-matched to the
SynLethDB benchmark protocol, not a strict embedding-only ablation** against
exp06 (which fits to convergence with no epoch selection). Per-epoch
train/val curves and peak GPU memory are written per fold to
`<output_dir>/<split>/epoch_metrics_fold{N}.csv` and to per-rank `train_rank{N}.log`.

## Evaluation

Reuses `sl_benchmark_baseline.metrics.official_*_metrics` verbatim. Primary: CV2/CV3
per-anchor NDCG@k and MAP@k. Honesty checks: covered-pair diagnostic slice,
coverage-flag ablation, effect-size ± std reporting.

## Results to Beat (exp06 XGB, 5-fold mean)

| Split | AUROC | AUPR | NDCG@10 | MAP@10 |
| --- | ---: | ---: | ---: | ---: |
| CV2 | 0.704 | 0.732 | 0.042 | 0.034 |
| CV3 | 0.596 | — | 0.002 | — |

exp08 must beat these on the full-universe slice with lift concentrated on the
covered-pair slice. Within-noise lift is null. A negative result is publishable.

## Phase 2/3 Selected Best-Fold Results (2026-06-22)

Selection rule: for each phase and split, select the completed fold-result JSON
whose fold has the best split-validation `val_pair_auroc` in
`<output_dir>/<split>/epoch_metrics_fold{N}.csv`. Folds with epoch CSVs but no
final `_fold_results/<split>_foldN.result.json` are treated as incomplete and are
not used for the selected row.

Full-universe slice:

| Phase | Split | Source directory | Result JSONs | Selected fold / epoch | AUROC | AUPR | F1 | NDCG@10 | MAP@10 | NDCG@50 | MAP@50 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 BCE | CV1 | `phase2_bce` | 5/5 | fold 2 / epoch 18 | 0.6483 | 0.6498 | 0.6708 | 0.0008 | 0.0006 | 0.0105 | 0.0022 |
| Phase 2 BCE | CV2 | `phase2_bce` | 2/5 | fold 2 / epoch 6 | 0.6667 | 0.6754 | 0.6723 | 0.0050 | 0.0035 | 0.0286 | 0.0072 |
| Phase 2 BCE | CV3 | `phase2_bce_cv2_cv3_lr3e4_ep30` | 3/5 | fold 1 / epoch 6 | 0.5866 | 0.5406 | 0.6777 | 0.0015 | 0.0010 | 0.0049 | 0.0015 |
| Phase 3 bag supervision | CV1 | `phase3_bag_sup` | 1/5 | fold 2 / epoch 3 | 0.5000 | 0.7500 | 0.6667 | 0.0017 | 0.0004 | 0.0048 | 0.0013 |
| Phase 3 bag supervision | CV2 | `phase3_bag_sup_cv2_cv3_lr3e4_ep30` | 5/5 | fold 1 / epoch 3 | 0.5744 | 0.5756 | 0.6667 | 0.0040 | 0.0027 | 0.0121 | 0.0039 |
| Phase 3 bag supervision | CV3 | `phase3_bag_sup_cv2_cv3_lr3e4_ep30` | 1/5 | fold 0 / epoch 0 | 0.5654 | 0.5498 | 0.6677 | 0.0000 | 0.0000 | 0.0035 | 0.0007 |

Covered-pair diagnostic slice:

| Phase | Split | AUROC | AUPR | NDCG@10 | MAP@10 | NDCG@50 | MAP@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 BCE | CV1 | 0.6483 | 0.7320 | 0.0010 | 0.0007 | 0.0099 | 0.0022 |
| Phase 2 BCE | CV2 | 0.6504 | 0.7444 | 0.0072 | 0.0050 | 0.0252 | 0.0075 |
| Phase 2 BCE | CV3 | 0.5396 | 0.5851 | 0.0021 | 0.0014 | 0.0061 | 0.0020 |
| Phase 3 bag supervision | CV1 | 0.5000 | 0.8000 | 0.0017 | 0.0003 | 0.0049 | 0.0013 |
| Phase 3 bag supervision | CV2 | 0.5824 | 0.6670 | 0.0061 | 0.0042 | 0.0133 | 0.0051 |
| Phase 3 bag supervision | CV3 | 0.5638 | 0.6408 | 0.0000 | 0.0000 | 0.0050 | 0.0009 |

Readout: no selected Phase 2/3 row clears the exp06 CV2/CV3 gate. Phase 2 CV2
is the strongest completed selected row, but it remains below exp06 XGB on
full-universe AUROC/AUPR and top-k ranking. Phase 3 bag supervision did not
improve the selected CV2/CV3 full-universe ranking metrics in this artifact set.

## Implementation Phases

See `configs/experiments/08_k562_sl_pair_state_dl/README.md` for per-phase commands
and gates. Code and unit tests landed under `src/sl_dl_model/` and
`tests/sl_dl_model/`; the Phase 2/3 gates run on the cluster against the gitignored
ESM2 cache, STATE checkpoint, and gwps bags.

## Terminology Guardrail

exp08 is a benchmark-adapter extension of exp06. No "SL target" / biological SL
claims; `Rand` negatives are unconfirmed non-SL.
