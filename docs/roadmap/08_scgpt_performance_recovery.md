# 08_scgpt_performance_recovery

## Goal
Close the performance gap between scGPT retrieval and strong baselines
(logreg, PCA) on the in_dist, cell-level split with masking enabled.

Target: scGPT exact_hit@1 >= 0.35 and mrr >= 0.45 (mask=True).

## Current Gap (Dec 2025 runs)
- logreg: exact_hit@1 ~0.35, mrr ~0.48 (mask=True)
- scgpt frozen/head-only/LoRA: exact_hit@1 ~0.06-0.08, mrr ~0.12-0.15

## Urgent Changes (Do First)
1. [done] Switch scGPT to the official preprocessing pipeline using raw counts.
   - [done] Use `layers["counts"]` as raw input, normalize total, log1p, then bin to discrete values.
   - [done] Ensure masking happens on the raw counts layer before preprocessing.
2. [done] Fix batch composition for InfoNCE.
   - [done] Balanced batch sampler (multiple cells per condition) or larger batch size.
3. [done] Run classification-loss fine-tuning as a sanity check.
   - [done] Add classification finetune config for quick runs.
   - [done] Wire classification mode into the scGPT run script.
   - [todo] Align objective with top-k retrieval metric used by logreg baseline.

## Phase 1: Data Integrity And Alignment
- [todo] Inspect `data/norman/perturb_processed.h5ad` for:
  - [todo] Layer availability: counts/raw, normalized/log1p indicators.
  - [todo] Data scale, dtype, sparsity, per-cell library sizes.
- [todo] Ensure scGPT uses the same gene set and masking policy as baselines.
- [todo] Quantify scGPT vocab coverage; map gene aliases if coverage < 95%.

Deliverables:
- [todo] Data audit note with evidence of counts vs normalized scale.
- [done] Code/config change to select counts layer for scGPT.

## Phase 2: Training Objective And Sampling
- [todo] Compare losses:
  - [todo] InfoNCE (current) vs classification loss (closed-set).
- [todo] Add balanced sampler:
  - [todo] Each batch contains 4-8 cells per condition across 4-8 conditions.
- [todo] Add early metric tracking on exact_hit@1 via a small validation subset.

Deliverables:
- [todo] New finetune config(s) and training logs with sampling details.
- [todo] Ablation table: {loss, sampler, batch size} -> metrics.

## Phase 3: Capacity And Adaptation
- [todo] Increase adaptation capacity:
  - [todo] Unfreeze last N transformer layers OR raise LoRA rank to 16/32.
  - [todo] Use separate LR for backbone vs head (e.g., 1e-5 vs 1e-4).
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
