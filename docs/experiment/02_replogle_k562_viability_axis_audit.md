# Replogle K562 Viability-Axis Audit

Run date: 2026-05-15

Canonical remote result paths after the 2026-05-26 artifact reorganization:

- Main audit:
  `results/experiments/02_replogle_k562_viability_axis_audit/runs/viability_axis_5x1_main_20260515`
- Signal decomposition:
  `results/experiments/02_replogle_k562_viability_axis_audit/runs/signal_decomposition_5x1_main_20260515`

Note: on the remote PC inspected on 2026-05-26, the previous per-config result
roots were present as empty placeholders and the documented run artifacts were
not present.

## Run Setup

- Task: audit whether the Replogle K562 B->C pseudobulk signal mostly learns a
  generic cell-death/proliferation or response-burden axis.
- Dataset: Replogle K562 CRISPRi essential Perturb-seq.
- Input features: perturbation pseudobulk delta expression.
- Label: DepMap K562 CRISPR GeneEffect.
- Evaluation: `internal_cv_all`, 5-fold CV x 1 repeat, seed `42`,
  unweighted.
- Viability anchors: 2019 NAR Achilles and CTRP cell-death/proliferation
  coefficients from `bence-szalai/Cell-death-signatures`.
- Main config:
  `configs/experiments/02_replogle_k562_viability_axis_audit/viability_axis_main.yaml`.
- Follow-up config:
  `configs/experiments/02_replogle_k562_viability_axis_audit/signal_decomposition_main.yaml`.

## Feature QA

| Item | Value |
| --- | ---: |
| Modeling rows | 1917 |
| Expression genes | 8563 |
| NAR coefficients per model | 978 |
| Matched NAR genes | 728 |
| Missing NAR genes | 250 |
| Matched fraction | 0.744 |

## 5-Fold Validation Results

| Comparison | Feature set | Model | Spearman | AUROC GE < -1.0 | RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| NAR score only | `nar_viability_scores` | `nar_score_ridge` | 0.244 | 0.623 | 0.680 |
| NAR score + burden | `nar_viability_scores_plus_burden` | `nar_score_plus_burden_ridge` | 0.443 | 0.714 | 0.643 |
| Best pseudobulk baseline | `delta_all` | `pca50_random_forest` | 0.494 | 0.742 | 0.612 |
| Best NAR-residualized transcriptome | `nar_resid_delta_all` | `nar_resid_pca50_random_forest` | 0.503 | 0.744 | 0.610 |
| NAR+burden-residualized transcriptome | `nuisance_resid_delta_all` | `nuisance_resid_pca50_random_forest` | 0.469 | 0.731 | 0.620 |
| Residual PCs + nuisance scores | `nuisance_resid_delta_all` | `nuisance_resid_pca50_plus_scores_ridge` | 0.491 | 0.740 | 0.620 |
| Sparse residualized model | `nuisance_resid_delta_all` | `nuisance_resid_lasso` | 0.113 | 0.559 | 0.739 |

## Main Readout

- The NAR viability score alone does not explain the B->C signal: Spearman
  `0.244` vs `0.494` for the best pseudobulk baseline.
- Response burden explains a large fraction of the signal, reaching `0.443`
  Spearman when added to the NAR score.
- Fold-local NAR residualization preserves PCA50 RandomForest performance
  (`0.503` vs `0.494`), arguing against a pure generic-viability explanation.
- Adding response burden to the residualizer lowers residualized PCA50
  RandomForest to `0.469`, while residual PCs plus nuisance scores recover
  near-baseline performance.
- The current signal is best interpreted as generic response burden plus
  residual transcriptomic structure.

## Files

- Main audit: `summary_metrics.csv`, `fold_metrics.parquet`,
  `predictions.parquet`, and `viability_axis_report.md` under the main audit run.
- Signal decomposition: `summary_metrics.csv`, `fold_metrics.parquet`, and
  `predictions.parquet` under the signal-decomposition run.
- Baseline comparison rows align with
  `docs/experiment/01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer.md`.
