# Replogle K562 B->C Baseline Metrics

Imported from `richard@10.20.161.54:/home/richard/projects/VCC/results/replogle_k562_b_to_c_baseline`.

## Run Setup

- Task: predict DepMap K562 CRISPR GeneEffect from Replogle K562 perturbation pseudobulk delta expression.
- Main baseline evaluation: 5-fold CV x 1 repeat, random seed `42`, 10 GeneEffect quantile bins.
- PCA RandomForest ablation: `pca_rf_unweighted_20260514`, same split policy, unweighted, `delta_all` and `delta_mask_target` only.
- Values below are fold means. Standard deviations and all rows are in `summary_metrics.csv`.

## Main Experiment Results

| Scope | Model / ablation | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_cv_all | Mean label baseline | NA | NA | 0.703 | 0.548 | -0.000 | 0.500 | 0.899 |
| internal_cv_all | response burden + Ridge | 0.426 | 0.387 | 0.649 | 0.497 | 0.148 | 0.705 | 1.557 |
| internal_cv_all | full delta + Ridge | 0.348 | 0.356 | 0.724 | 0.553 | -0.061 | 0.670 | 1.816 |
| internal_cv_all | full delta + ElasticNet | 0.282 | 0.294 | 0.788 | 0.607 | -0.256 | 0.640 | 1.675 |
| internal_cv_all | PCA20 delta + Ridge | 0.480 | 0.463 | 0.623 | 0.477 | 0.215 | 0.736 | 1.911 |
| internal_cv_all | PCA50 delta + Ridge | 0.485 | 0.474 | 0.619 | 0.472 | 0.224 | 0.738 | 1.982 |
| internal_cv_all | PCA100 delta + Ridge | 0.481 | 0.474 | 0.619 | 0.472 | 0.224 | 0.736 | 2.006 |
| internal_cv_all | full delta + RandomForest | 0.476 | 0.481 | 0.616 | 0.472 | 0.231 | 0.734 | 2.006 |
| internal_cv_all | PCA20 delta + RandomForest | 0.492 | 0.490 | 0.613 | 0.466 | 0.238 | 0.742 | 1.958 |
| internal_cv_all | PCA50 delta + RandomForest | 0.494 | 0.493 | 0.612 | 0.462 | 0.242 | 0.742 | 2.077 |
| internal_cv_all | PCA100 delta + RandomForest | 0.491 | 0.483 | 0.616 | 0.463 | 0.232 | 0.742 | 1.936 |
| internal_cv_all | PCA50 delta + Ridge, n_cells weighted | 0.484 | 0.471 | 0.621 | 0.472 | 0.220 | 0.736 | 1.982 |

## Leakage And Bias Checks

| Scope | Check | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_cv_all | n_cells-only negative control | 0.000 | 0.092 | 0.700 | 0.546 | 0.007 | 0.498 | 1.416 |
| internal_cv_all | target-gene delta only | 0.242 | 0.284 | 0.674 | 0.525 | 0.080 | 0.622 | 1.912 |
| internal_cv_all | full delta + Ridge | 0.348 | 0.356 | 0.724 | 0.553 | -0.061 | 0.670 | 1.816 |
| internal_cv_all | target-masked full delta + Ridge | 0.343 | 0.353 | 0.726 | 0.556 | -0.068 | 0.666 | 1.793 |
| internal_cv_target_index_valid | target-valid full delta + Ridge | 0.313 | 0.331 | 0.723 | 0.559 | -0.058 | 0.648 | 1.652 |
| internal_cv_target_index_valid | target-valid masked delta + Ridge | 0.310 | 0.327 | 0.727 | 0.562 | -0.071 | 0.646 | 1.677 |

## Main Readout

- Best main-model row shown here: `delta_all + pca50_random_forest (unweighted)`, Spearman `0.494`, Pearson `0.493`, RMSE `0.612`, MAE `0.462`, AUROC at GeneEffect `< -1.0` `0.742`.
- PCA50 RandomForest improves over full-delta RandomForest: Spearman `+0.018` and AUROC GE `< -1.0` `+0.008`.
- In the direct Ridge leakage check, target masking changes Spearman only slightly: `-0.005` on `internal_cv_all` and `-0.003` on the target-index-valid subset.
- `n_cells_only` is near-random by Spearman/AUROC, while response-burden features retain a moderate signal, so the useful signal is not explained by cell-count alone.

## Follow-Up Audit

- 2026-05-15 NAR viability-axis audit:
  [`docs/experiment/replogle_k562_viability_axis_audit_5x1_main.md`](../../replogle_k562_viability_axis_audit_5x1_main.md).
- Main conclusion: the NAR viability score alone is not enough to explain the
  B->C signal, but response burden accounts for a large fraction of it.
  Fold-local NAR residualization preserves PCA50 RandomForest performance
  (`0.503` Spearman vs `0.494` baseline), while PCA50 Ridge drops (`0.400` vs
  `0.485`), suggesting the linear model relies more on the generic viability
  axis than the nonlinear PCA RandomForest.
- 2026-05-15 next-step implementation:
  `configs/replogle_k562_signal_decomposition_5x1_main.yaml` adds the planned
  NAR+response-burden nuisance residualization audit, curated biological
  program-score features, residualized PCA+score models, and sparse Lasso /
  ElasticNet checks for the main 5-fold x 1 setup.

## Files

- `cv_config.json`: original remote CV configuration.
- `summary_metrics.csv`: curated aggregate metrics across the original baseline and PCA RandomForest ablation rows.
- `fold_metrics.parquet`: per-fold metrics, including PCA RandomForest rows.
- `predictions.parquet`: per-gene held-out predictions, including PCA RandomForest rows.
- `metrics_table.csv`: selected rows from `summary_metrics.csv` used in the table above.
