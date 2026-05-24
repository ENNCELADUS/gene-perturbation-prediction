# Replogle K562 NAR Viability-Axis Audit Metrics

Run date: 2026-05-15

Remote workspace:

```bash
ssh -p 2222 richard@10.20.246.163
cd ~/gene-perturbation-prediction
```

Remote result paths:

- `results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515`
- `results/replogle_k562_signal_decomposition_5x1_main/runs/signal_decomposition_5x1_main_20260515`

## Run Setup

- Task: audit whether the Replogle K562 B->C baseline mostly learns a generic cell-death/proliferation or viability axis rather than perturbation-specific dependency biology.
- Dataset: Replogle K562 CRISPRi essential screen.
- Input features: perturbation pseudobulk delta expression.
- Label: DepMap K562 CRISPR GeneEffect.
- Main audit config: `configs/replogle_k562_viability_axis_audit_5x1_main.yaml`.
- Signal-decomposition follow-up config: `configs/replogle_k562_signal_decomposition_5x1_main.yaml`.
- Evaluation: `internal_cv_all`, 5-fold CV x 1 repeat, random seed `42`, unweighted.
- Viability anchors: 2019 NAR Achilles and CTRP cell-death/proliferation coefficients from `bence-szalai/Cell-death-signatures`.
- Leakage and bias checks were intentionally not rerun; see the baseline README for target-masking and cell-count controls.

Feature QA:

| Item | Value |
| --- | ---: |
| Modeling rows | 1917 |
| Expression genes | 8563 |
| NAR coefficients per model | 978 |
| Matched NAR genes | 728 |
| Missing NAR genes | 250 |
| Matched fraction | 0.744 |

## Main Audit Results

| Comparison | Feature set | Model | Spearman | AUROC GE < -1.0 | RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| NAR score only | `nar_viability_scores` | `nar_score_ridge` | 0.244 | 0.623 | 0.680 |
| NAR score + burden | `nar_viability_scores_plus_burden` | `nar_score_plus_burden_ridge` | 0.443 | 0.714 | 0.643 |
| Best transcriptome baseline | `delta_all` | `pca50_random_forest` | 0.494 | 0.742 | 0.612 |
| Best NAR-residualized transcriptome | `nar_resid_delta_all` | `nar_resid_pca50_random_forest` | 0.503 | 0.744 | 0.610 |
| NAR-residualized linear transcriptome | `nar_resid_delta_all` | `nar_resid_pca50_ridge` | 0.400 | 0.698 | 0.640 |

## Signal-Decomposition Follow-Up

| Comparison | Feature set | Model | Spearman | AUROC GE < -1.0 | RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| NAR + burden nuisance scores | `nuisance_scores` | `nuisance_score_ridge` | 0.443 | 0.714 | 0.643 |
| Program scores only | `program_scores` | `program_score_ridge` | 0.410 | 0.700 | 0.652 |
| Program scores + burden | `program_scores_plus_burden` | `program_score_ridge` | 0.451 | 0.720 | 0.638 |
| NAR+burden-residualized transcriptome | `nuisance_resid_delta_all` | `nuisance_resid_pca50_random_forest` | 0.469 | 0.731 | 0.620 |
| Residual PCs + nuisance scores | `nuisance_resid_delta_all` | `nuisance_resid_pca50_plus_scores_ridge` | 0.491 | 0.740 | 0.620 |
| Residual PCs + nuisance scores RF | `nuisance_resid_delta_all` | `nuisance_resid_pca50_plus_scores_random_forest` | 0.484 | 0.738 | 0.616 |
| Sparse residualized model | `nuisance_resid_delta_all` | `nuisance_resid_lasso` | 0.113 | 0.559 | 0.739 |

## Main Readout

- NAR viability score alone does not explain the B->C signal: Spearman `0.244` vs the best transcriptome baseline at `0.494`.
- Response burden accounts for a large fraction of the signal: NAR score + burden reaches Spearman `0.443` and AUROC `0.714`.
- Fold-local NAR residualization preserves PCA50 RandomForest performance (`0.503` vs `0.494` baseline), arguing against a pure generic-viability explanation.
- Adding response burden to the residualizer lowers residualized PCA50 RandomForest to `0.469`, while residual PCs plus nuisance scores recover near-baseline performance (`0.491` Ridge), suggesting mixed generic response-burden and residual transcriptomic signal.
- Sparse residualized Lasso performs poorly (`0.113` Spearman), so raw sparse-gene modeling is not the next default direction without more structured features.

## Files

- Main audit: `summary_metrics.csv`, `fold_metrics.parquet`, `predictions.parquet`, and `viability_axis_report.md` under `results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515`.
- Signal decomposition: `summary_metrics.csv`, `fold_metrics.parquet`, and `predictions.parquet` under `results/replogle_k562_signal_decomposition_5x1_main/runs/signal_decomposition_5x1_main_20260515`.
- Baseline comparison rows are aligned with `docs/experiment/baselines/replogle_k562_b_to_c_baseline/README.md`.
