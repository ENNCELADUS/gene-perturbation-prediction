# Stage-one architecture stage

Task 3 of the [stage-one design](../../docs/specs/2026-10-16-pie-geneeffect-ladder-design.md) (plan Wave 1A, protocol
§16): every architecture arm trained on the 981 frame lines, main run plus five folds, one config each under
`configs/stage_one/architecture/`, reported by `configs/stage_one/architecture/report_3a.yaml`
(`hpc/run.sh stage-one-queue`, then `hpc/run.sh stage-one-report`). Run on the H20 container on port 30838, outputs
in `outputs/stage_one/architecture/` there; the full report is [report_3a_results.md](report_3a_results.md) (a copy
of the host's `report_3a/results.md`, beside its `results.json`). Seed 0. Nothing here is SL evidence.

## What ran

| Step | Finished (UTC) | Notes |
| --- | --- | --- |
| first queue | 2026-10-20 | every row; `direct_readout_warm`, `film_dE_full` and `multi_dE_full` (18 runs) failed (fixed in `05006e7`) |
| retry queue | 2026-10-21 21:28 | those 18 runs, none failed |
| hybrid queue | 2026-10-22 00:20 | `hybrid_dE` (added in `f5fefbe` once ΔE read best), main plus five folds |
| report | 2026-10-22 01:40 | code at `f5fefbe` |

Every trained row except `hybrid_dE` chose `best.pt` on validation selective Spearman; `hybrid_dE` trained after the
selector moved to the per-line residual Spearman (`33784bf`, protocol §16). The report's first column is the per-line
selector for every row.

## Validation and test

Per-line residual Spearman (the selector) and within-gene R²; every row in the full report.

| Row | Val per-line | Test per-line | Val within-gene R² | Test within-gene R² |
| --- | ---: | ---: | ---: | ---: |
| default prior refitted on the frame | 0.2146 | 0.2283 | 0.0462 | 0.0474 |
| FiLM on ΔE, selective genes | 0.1774 | 0.1716 | 0.0195 | 0.0089 |
| FiLM on ΔE, full panel | 0.1737 | 0.1891 | 0.0130 | 0.0243 |
| multi-encoder on ΔE, full panel | 0.1721 | 0.2022 | 0.0201 | 0.0332 |
| multi-encoder on ΔE, selective genes | 0.1545 | 0.1579 | 0.0204 | 0.0147 |
| FiLM on ΔE | 0.1457 | 0.1531 | −0.0092 | −0.0165 |
| hybrid on ΔE | 0.1396 | 0.1734 | −0.0431 | −0.0311 |
| shared components reader | 0.1267 | 0.1556 | −0.0096 | 0.0057 |
| multi-encoder on ΔE | 0.1240 | 0.1374 | −0.0071 | −0.0161 |
| lineage only | 0.1090 | 0.1433 | 0.0041 | 0.0087 |
| ΔH ridge, penalty 10 | 0.1087 | 0.1207 | −0.0249 | −0.0293 |
| FiLM on ΔH | 0.1057 | 0.0916 | −0.0292 | −0.0451 |
| direct readout | 0.0935 | 0.1276 | −0.0425 | −0.0509 |
| Cellosaurus text only | 0.0828 | 0.1030 | −0.0754 | −0.0614 |
| FiLM on ΔL | 0.0809 | 0.0936 | −0.0383 | −0.0265 |
| multi-encoder on ΔL | 0.0719 | 0.0979 | −0.0518 | −0.0508 |
| multi-encoder on ΔH | 0.0643 | 0.0793 | −0.0420 | −0.0426 |
| no-Δ FiLM | 0.0383 | 0.0550 | −0.0538 | −0.0700 |
| own expression only | 0.0301 | 0.0455 | −0.0130 | −0.0155 |
| direct readout, PIE warm start | 0.0211 | 0.0400 | −0.0094 | −0.0123 |
| line-permuted Δ rows (six) | 0.013–0.030 | 0.025–0.052 | −0.070 to −0.013 | −0.072 to −0.013 |
| base | 0.0098 | 0.0475 | −0.0024 | 0.0017 |

The selective-gene and full-panel rows predict the base outside the 4,269 selective panel genes, which the metrics
here are scored over anyway; the other trained rows cover the whole panel.

## Contrasts

Left minus right, per-line residual Spearman, 95% paired line-bootstrap intervals.

| Contrast | Validation | Test |
| --- | --- | --- |
| FiLM on ΔE − prior | −0.069 [−0.099, −0.042] | −0.075 [−0.104, −0.048] |
| hybrid on ΔE − prior | −0.075 [−0.098, −0.055] | −0.055 [−0.073, −0.038] |
| multi-encoder on ΔE − prior | −0.091 [−0.113, −0.070] | −0.091 [−0.116, −0.063] |
| components reader − prior | −0.088 [−0.107, −0.068] | −0.073 [−0.093, −0.054] |
| FiLM on ΔE − FiLM on ΔL | +0.065 [+0.019, +0.108] | +0.060 [+0.021, +0.097] |
| multi-encoder on ΔE − multi-encoder on ΔL | +0.052 [+0.016, +0.088] | +0.040 [−0.009, +0.094] |
| FiLM on ΔH − FiLM on ΔL | +0.025 [−0.010, +0.054] | −0.002 [−0.023, +0.015] |
| FiLM on ΔH − ΔH ridge 100 | +0.003 [−0.042, +0.044] | −0.038 [−0.065, −0.010] |
| FiLM on ΔE − direct readout | +0.052 [+0.005, +0.098] | +0.026 [−0.024, +0.078] |
| multi-encoder on ΔE − FiLM on ΔE | −0.022 [−0.042, −0.001] | −0.016 [−0.033, +0.001] |
| multi-encoder on ΔE − no-Δ FiLM | +0.086 [+0.046, +0.125] | +0.082 [+0.040, +0.123] |
| multi-encoder on ΔE − permuted | +0.094 [+0.054, +0.134] | +0.086 [+0.043, +0.125] |
| FiLM on ΔE − own expression only | +0.116 [+0.081, +0.151] | +0.108 [+0.071, +0.147] |
| warm − cold direct readout | −0.072 [−0.110, −0.036] | −0.088 [−0.136, −0.037] |
| FiLM on ΔE, full − selective | −0.004 [−0.027, +0.018] | +0.018 [−0.014, +0.053] |
| multi-encoder on ΔE, full − selective | +0.018 [−0.007, +0.041] | +0.044 [+0.020, +0.073] |
| hybrid on ΔE − FiLM on ΔE | −0.006 [−0.028, +0.014] | +0.020 [−0.005, +0.048] |

## Caveats read off the runs

- **The line-permuted FiLM on ΔE collapsed.** Its predictions are constant across lines within 17,553 of the 17,607
  panel genes, so its selective Spearman is defined on 3 validation and 5 test genes, and its `best.pt` (epoch 6) was
  chosen off a 3-gene score under the old selector. Its per-line residual Spearman (0.014 / 0.027) reads as a per-gene
  constant, not as a permuted-input fit; the other five permuted rows vary within gene.
- **Training to validation gap.** The ΔE rows reach training selective Spearman 0.35–0.40 against 0.13–0.16 on
  validation, and within-gene R² 0.14–0.20 against −0.01 to +0.02.
- **The hybrid ran to the epoch cap.** Its best epoch is 28 of 30; the others stopped early (best epochs 3–21).
- Out of fold over the 377 single-cell training lines (telemetry) keeps the same order: prior 0.192, multi-encoder on
  ΔE full panel 0.168, FiLM on ΔE full panel 0.159.

## Reading

No trained row reaches the default prior refitted on the frame: every listed contrast with the prior is below zero on
both splits, and the best trained rows on validation (FiLM on ΔE, selective or full panel, 0.177 / 0.174) sit about
0.04 under it, with within-gene R² less than half the prior's. Among the frozen PIE's deltas ΔE carries the line
signal: it beats ΔL in both families and its line-permuted and no-Δ controls, and beats own expression only, while
ΔH does not beat a per-gene ridge on ΔH itself, and ΔL does no better than the Cellosaurus text it reads. Adding a
context encoder (multi-encoder) does not help FiLM, PIE's warm start hurts the direct readout, the full-panel profile
delta does not separate from the 512-component one on validation (the multi-encoder's +0.044 is on test only), and
the hybrid matches FiLM on ΔE.
