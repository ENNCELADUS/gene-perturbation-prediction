# Experiment 08 — STATE-Adapter DL Model for K562 SL-Pair Ranking

**Status:** Implementation complete (Phase 0–4 code + unit tests). Phase 0 parity +
Phase 2 BCE pending re-run. **Phase 3 (bag supervision) is BLOCKED on a NaN crash**
(2026-06-20): the first cluster run died at epoch-0 validation
(`_validate_auroc` → `roc_auc_score` "Input contains NaN"). Root cause is under a TDD
fix — see "Phase 3 NaN Blocker" below.
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

## Implementation Phases

See `configs/experiments/08_k562_sl_pair_state_dl/README.md` for per-phase commands
and gates. Code and unit tests landed under `src/sl_dl_model/` and
`tests/sl_dl_model/`; the Phase 2/3 gates run on the cluster against the gitignored
ESM2 cache, STATE checkpoint, and gwps bags.

## Phase 3 NaN Blocker (2026-06-20)

The first cluster Phase 3 run (`phase3_bag_supervision.yaml`, `lambda_bag=1.0`,
`lr=1e-3`) crashed at the end of epoch 0 in `_validate_auroc` →
`roc_auc_score(... )` with `ValueError: Input contains NaN`. `sigmoid(logit)` is NaN
only when `logit` is NaN, so the model parameters were already non-finite **before**
validation ran. Validation is only the *first detector*: `src/sl_dl_model/` has no
grad-clipping, `isfinite`, or `nan_to_num` guards anywhere, so a NaN injected early in
epoch 0 propagates silently to the epoch-end check.

Ranked root-cause hypotheses:

- **H1 — `_energy_distance` `torch.cdist` (phase3-only term via `bag_loss`).** Most
  likely. H1a: matmul-mode cdist computes `d² = |x|²+|y|²−2xy`, float error yields a
  small negative under the `sqrt` → NaN in the loss *value*. H1b: `cdist(x, x)`
  self-distance has a `0/0` gradient on the zero diagonal → NaN in the *gradient* only
  (forward value looks fine; `clamp_min(0)` guards the value, not the grad). Phase 2
  has no `bag_loss` and never reaches `_energy_distance`, which is why phase 2 did not
  crash.
- **H2 — bag-path gradient magnitude × `lr=1e-3` → adapter blow-up.** The bag term
  backprops the full predicted bag into the adapter (larger-magnitude path than the
  pooled `embed_gene`), energy distance is unnormalized, and `λ_bag=1.0` adds it
  directly; weights can diverge to inf→NaN over steps even without a cdist NaN.
- **H3 — `MeanStdPool.std(unbiased=False)` `sqrt(0)` gradient** when a bag feature is
  constant. Latent risk; present in phase 2 too, so not the phase-3 differentiator.
- **H4 — frozen STATE forward overflow** once the adapter output is pushed out of
  range (downstream symptom of H2).

Fix is TDD-driven defense-in-depth (grad-clip + `isfinite`/`nan_to_num` guards +
NaN-safe energy distance), tracked in
`docs/superpowers/plans/2026-06-20-exp08-phase3-nan-fix.md`. Phase 3 is the primary
gate; if the fixed run trains cleanly through CV2/CV3 the pipeline is unblocked.

## Terminology Guardrail

exp08 is a benchmark-adapter extension of exp06. No "SL target" / biological SL
claims; `Rand` negatives are unconfirmed non-SL.
