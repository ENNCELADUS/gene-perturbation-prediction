# Experiment 08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

**Status:** Implementation complete (Phase 0–4 code + unit tests). Real-data gates
(Phase 2/3) pending ESM2 cache + STATE checkpoint + gwps bags.
**Design spec:** `docs/superpowers/specs/2026-06-17-exp08-state-dl-sl-ranking-design.md`.
**Plan:** `docs/superpowers/plans/2026-06-17-exp08-state-dl-sl-ranking.md`.
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
warm-up schedule, HuggingFace Accelerate/DDP + tqdm. Bag supervision for covered
train genes only (leakage-free held-out gene eval). Seed 17, max 20 epochs, lr 1e-3.

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

## Implementation Phases

See `configs/experiments/08_k562_sl_pair_state_dl/README.md` for per-phase commands
and gates. Code and unit tests landed under `src/sl_dl_model/` and
`tests/sl_dl_model/`; the Phase 2/3 gates run on the cluster against the gitignored
ESM2 cache, STATE checkpoint, and gwps bags.

## Terminology Guardrail

exp08 is a benchmark-adapter extension of exp06. No "SL target" / biological SL
claims; `Rand` negatives are unconfirmed non-SL.
