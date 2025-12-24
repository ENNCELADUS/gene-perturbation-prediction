# 08_scgpt_performance_recovery

## Goal
Close the performance gap between scGPT retrieval and strong baselines
(logreg, PCA) on the in_dist, cell-level split with masking enabled.

Target: scGPT exact_hit@1 >= 0.35 and mrr >= 0.45 (mask=True).

## Current Gap (Dec 2025 runs)
- logreg: exact_hit@1 ~0.35, mrr ~0.48 (mask=True)
- scgpt frozen: exact_hit@1 ~0.07, mrr ~0.13
- scgpt head-only InfoNCE: exact_hit@1 ~0.13, mrr ~0.23
- scgpt head-only classification: exact_hit@1 ~0.19, mrr ~0.29
- scgpt LoRA classification: exact_hit@1 ~0.03, mrr ~0.07 (collapsed)

## Priority Actions (Next Runs)
1. [done] Enable balanced sampling for classification fine-tuning.
   - Implemented in `src/configs/scgpt_finetune_classification.yaml`.
   - Next: re-run cls head-only with balanced batches.
2. [done] Increase adaptation capacity safely.
   - Unfreeze last 2 transformer layers + split LR (head 1e-4, backbone 1e-5).
   - Add grad clipping for stability (max_grad_norm=1.0).
3. [done] Restrict LoRA to last layers only.
   - Apply LoRA only to last 2 transformer blocks to avoid early-layer drift.
   - Keep rank=8 for now; increase only after stability is confirmed.
4. [in-progress] Early metric tracking on exact_hit@1/mrr.
   - [done] Log val Accuracy@1 for classification runs.
   - [todo] Add retrieval metric tracking for InfoNCE runs.

## Urgent Changes (Do First)
1. [done] Switch scGPT to the official preprocessing pipeline using raw counts.
   - [done] Use `layers["counts"]` as raw input, normalize total, log1p, then bin to discrete values.
   - [done] Ensure masking happens on the raw counts layer before preprocessing.
2. [done] Fix batch composition for InfoNCE.
   - [done] Balanced batch sampler (multiple cells per condition) or larger batch size.
3. [done] Run classification-loss fine-tuning as a sanity check.
   - [done] Add classification finetune config for quick runs.
   - [done] Wire classification mode into the scGPT run script.
   - [done] Align objective with top-k retrieval metric used by logreg baseline.

## Phase 1: Data Integrity And Alignment
- [done] Inspect `data/norman/perturb_processed.h5ad` for:
  - [done] Layer availability: counts/raw, normalized/log1p indicators.
  - [done] Data scale, dtype, sparsity, per-cell library sizes.
- [todo] Ensure scGPT uses the same gene set and masking policy as baselines.
- [done] Quantify scGPT vocab coverage.
- [todo] Map gene aliases if coverage < 95% (needs external mapping table).
- [done] Validate binned value distribution after log1p (check for collapsed bins).
- [done] Enforce evaluation parity (split seed, masking policy, query/library config).
- [done] Populate confidence metrics for classification finetune runs (config enabled).

Deliverables:
- [done] Data audit note with evidence of counts vs normalized scale.
- [done] Vocab coverage report and alias mapping summary.
- [done] Binning distribution snapshot for counts/log1p inputs.
- [done] Evaluation parity checklist with split seed and masking confirmation.
- [todo] Updated results report with confidence AUC + coverage curves.
- [done] Code/config change to select counts layer for scGPT.

### Phase 1 Evidence (Dec 2025 Audit)
- `layers["counts"]` present (float32 integer counts), `X` is log1p/normalized.
- Counts sparsity ~8.1% nonzero; library size p5/p50/p95 ~ 1370/2971/5063.
- scGPT vocab coverage 90.13% (4547/5045); missing mostly RP*/AC*/AL* lncRNA-like symbols.
- Binning check (log1p + 51 bins): all bins used; zero bin ~91.8% of entries.
- Split parity confirmed (seed 42, same split paths, mask_perturbed true, query_split test).

## Phase 2: Training Objective And Sampling
- [in-progress] Compare losses:
  - [done] Classification loss baseline confirmed.
  - [todo] Re-run classification with balanced sampling and partial unfreeze.
- [done] Add balanced sampler:
  - [done] Each batch contains 4-8 cells per condition across 4-8 conditions.
- [todo] Add early metric tracking on exact_hit@1 via a small validation subset.

Deliverables:
- [todo] New finetune config(s) and training logs with sampling details.
- [todo] Ablation table: {loss, sampler, batch size} -> metrics.

## Phase 3: Capacity And Adaptation
- [in-progress] Increase adaptation capacity:
  - [done] Unfreeze last 2 transformer layers in head-only mode.
  - [done] Use separate LR for backbone vs head (1e-5 vs 1e-4).
  - [todo] Re-evaluate with LoRA rank 16/32 after stability.
- [todo] Evaluate retrieval head depth/width and output dim.

Deliverables:
- [todo] Best checkpoint with clear uplift on validation metrics.
- [todo] Training curves that show stable convergence without collapse.

## Phase 4: Retrieval And Evaluation Tuning
- [todo] Library strategy:
  - [todo] Compare `bootstrap` vs `mean` vs `raw_cell` for scGPT embeddings.
- [todo] Query strategy:
  - [todo] Evaluate pseudo-bulk queries (50-100 cells) to reduce noise.
- [todo] Confidence metrics:
  - [todo] Recompute confidence AUC after improvements.

Deliverables:
- [todo] Updated results report in `results/reports`.
- [todo] Recommendation of best library/query settings.

## Success Criteria
- scGPT exact_hit@1 >= 0.35 (mask=True) on in_dist split.
- scGPT within 5% of logreg across mrr and ndcg@5.
- Improvements are reproducible across seeds (>=2 runs).

## Notes
- Keep data splits identical across models (cell split seed 42).
- Log data slice and seed for each experiment.
