# Replogle K562 B->C Baseline Metrics

Imported on 2026-05-14 from `richard@10.20.161.54:/home/richard/projects/VCC/results/replogle_k562_b_to_c_baseline/cv`.

## Run Setup

- Task: predict DepMap K562 CRISPR GeneEffect from Replogle K562 perturbation pseudobulk delta expression.
- Evaluation: repeated stratified CV with `5` folds x `1` repeat, random seed `42`, `10` GeneEffect quantile bins.
- Source feature artifact on remote: `/home/richard/projects/VCC/results/replogle_k562_b_to_c_baseline/replogle_k562_delta_features.npz`.
- Values below are fold means. Standard deviations and all model rows are in `summary_metrics.csv`.

## Concise Metrics Table

| Scope | Model / ablation | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_cv_all | Mean label baseline | NA | NA | 0.703 | 0.548 | -0.000 | 0.500 | 0.899 |
| internal_cv_all | n_cells-only negative control | 0.000 | 0.092 | 0.700 | 0.546 | 0.007 | 0.498 | 1.416 |
| internal_cv_all | target-gene delta only | 0.242 | 0.284 | 0.674 | 0.525 | 0.080 | 0.622 | 1.912 |
| internal_cv_all | response burden + Ridge | 0.426 | 0.387 | 0.649 | 0.497 | 0.148 | 0.705 | 1.557 |
| internal_cv_all | full delta + Ridge | 0.348 | 0.356 | 0.724 | 0.553 | -0.061 | 0.670 | 1.816 |
| internal_cv_all | full delta + ElasticNet | 0.282 | 0.294 | 0.788 | 0.607 | -0.256 | 0.640 | 1.675 |
| internal_cv_all | PCA20 delta + Ridge | 0.480 | 0.463 | 0.623 | 0.477 | 0.215 | 0.736 | 1.911 |
| internal_cv_all | PCA50 delta + Ridge | 0.485 | 0.474 | 0.619 | 0.472 | 0.224 | 0.738 | 1.982 |
| internal_cv_all | PCA100 delta + Ridge | 0.481 | 0.474 | 0.619 | 0.472 | 0.224 | 0.736 | 2.006 |
| internal_cv_all | full delta + RandomForest | 0.476 | 0.481 | 0.616 | 0.472 | 0.231 | 0.734 | 2.006 |
| internal_cv_all | PCA50 delta + Ridge, n_cells weighted | 0.484 | 0.471 | 0.621 | 0.472 | 0.220 | 0.736 | 1.982 |
| internal_cv_all | target-masked PCA50 delta + Ridge | 0.484 | 0.473 | 0.619 | 0.472 | 0.224 | 0.737 | 1.982 |
| internal_cv_all | target-masked delta + RandomForest | 0.474 | 0.479 | 0.617 | 0.472 | 0.229 | 0.734 | 2.053 |
| internal_cv_target_index_valid | target-valid full delta + Ridge | 0.313 | 0.331 | 0.723 | 0.559 | -0.058 | 0.648 | 1.652 |
| internal_cv_target_index_valid | target-valid masked delta + Ridge | 0.310 | 0.327 | 0.727 | 0.562 | -0.071 | 0.646 | 1.677 |
| internal_cv_target_index_valid | target-valid PCA50 delta + Ridge | 0.459 | 0.448 | 0.629 | 0.482 | 0.197 | 0.721 | 1.850 |
| internal_cv_target_index_valid | target-valid masked PCA50 + Ridge | 0.459 | 0.448 | 0.629 | 0.482 | 0.197 | 0.721 | 1.850 |
| internal_cv_target_index_valid | target-valid delta + RandomForest | 0.445 | 0.446 | 0.629 | 0.485 | 0.196 | 0.719 | 1.923 |
| internal_cv_target_index_valid | target-valid masked delta + RandomForest | 0.442 | 0.443 | 0.630 | 0.486 | 0.194 | 0.717 | 1.923 |

## Main Readout

- Best internal row by Spearman: `delta_all + pca50_ridge (unweighted)`, Spearman `0.485`, Pearson `0.474`, RMSE `0.619`, MAE `0.472`, AUROC at GeneEffect `< -1.0` `0.738`.
- Target masking on target-index-valid genes does not materially reduce PCA50 Ridge performance: Spearman change `-0.0001` (`0.459` full vs `0.459` masked).
- `n_cells_only` is near-random by Spearman/AUROC, while response-burden features retain a moderate signal, so the useful signal is not explained by cell-count alone.

## Files

- `cv_config.json`: remote CV configuration.
- `summary_metrics.csv`: all aggregate metrics for every scope, feature set, model, and weighting.
- `fold_metrics.csv`: per-fold metrics.
- `predictions.csv`: per-gene held-out predictions for each CV job.
- `metrics_table.csv`: selected rows from `summary_metrics.csv` used in the table above.
