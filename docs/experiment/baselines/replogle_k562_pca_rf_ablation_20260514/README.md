# Replogle K562 PCA RandomForest Ablation

Imported on 2026-05-14 from `richard@10.20.161.54:/home/richard/projects/VCC/results/replogle_k562_b_to_c_baseline/runs/pca_rf_unweighted_20260514`.

## Run Setup

- Task: compare fold-internal PCA RandomForest against full-delta RandomForest and PCA50 Ridge.
- Evaluation: 5-fold CV x 1 repeat, unweighted only, same Replogle K562 feature artifact and split policy as the first baseline.
- Run id: `pca_rf_unweighted_20260514`.
- Models: `random_forest`, `pca20_random_forest`, `pca50_random_forest`, `pca100_random_forest`, `pca50_ridge`.
- Features/scopes: `delta_all` and `delta_mask_target` on `internal_cv_all` and `internal_cv_target_index_valid`.

## Metrics Table

| Scope | Feature | Model | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | Top 5% enrich GE < -1.0 | Mean fit sec/fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| internal_cv_all | delta_all | pca50_random_forest | 0.494 | 0.493 | 0.612 | 0.462 | 0.242 | 0.742 | 2.077 | 3.383 |
| internal_cv_all | delta_all | pca20_random_forest | 0.492 | 0.490 | 0.613 | 0.466 | 0.238 | 0.742 | 1.958 | 2.421 |
| internal_cv_all | delta_all | pca100_random_forest | 0.491 | 0.483 | 0.616 | 0.463 | 0.232 | 0.742 | 1.936 | 4.776 |
| internal_cv_all | delta_all | pca50_ridge | 0.485 | 0.474 | 0.619 | 0.472 | 0.224 | 0.738 | 1.982 | 2.679 |
| internal_cv_all | delta_all | random_forest | 0.476 | 0.481 | 0.616 | 0.472 | 0.231 | 0.734 | 2.006 | 127.408 |
| internal_cv_all | delta_mask_target | pca50_random_forest | 0.496 | 0.492 | 0.612 | 0.463 | 0.241 | 0.742 | 2.031 | 3.570 |
| internal_cv_all | delta_mask_target | pca100_random_forest | 0.491 | 0.483 | 0.616 | 0.464 | 0.232 | 0.742 | 1.982 | 5.029 |
| internal_cv_all | delta_mask_target | pca20_random_forest | 0.490 | 0.487 | 0.614 | 0.467 | 0.235 | 0.739 | 1.934 | 2.443 |
| internal_cv_all | delta_mask_target | pca50_ridge | 0.484 | 0.473 | 0.619 | 0.472 | 0.224 | 0.737 | 1.982 | 2.684 |
| internal_cv_all | delta_mask_target | random_forest | 0.474 | 0.479 | 0.617 | 0.472 | 0.229 | 0.734 | 2.053 | 126.753 |
| internal_cv_target_index_valid | delta_all | pca50_random_forest | 0.467 | 0.469 | 0.621 | 0.475 | 0.218 | 0.724 | 1.873 | 3.173 |
| internal_cv_target_index_valid | delta_all | pca20_random_forest | 0.464 | 0.467 | 0.622 | 0.477 | 0.215 | 0.726 | 1.850 | 2.261 |
| internal_cv_target_index_valid | delta_all | pca100_random_forest | 0.463 | 0.465 | 0.623 | 0.475 | 0.213 | 0.727 | 1.800 | 4.483 |
| internal_cv_target_index_valid | delta_all | pca50_ridge | 0.459 | 0.448 | 0.629 | 0.482 | 0.197 | 0.721 | 1.850 | 2.712 |
| internal_cv_target_index_valid | delta_all | random_forest | 0.445 | 0.446 | 0.629 | 0.485 | 0.196 | 0.719 | 1.923 | 114.630 |
| internal_cv_target_index_valid | delta_mask_target | pca50_random_forest | 0.466 | 0.468 | 0.621 | 0.475 | 0.216 | 0.725 | 1.924 | 3.234 |
| internal_cv_target_index_valid | delta_mask_target | pca100_random_forest | 0.464 | 0.466 | 0.622 | 0.473 | 0.215 | 0.728 | 1.875 | 4.511 |
| internal_cv_target_index_valid | delta_mask_target | pca20_random_forest | 0.464 | 0.467 | 0.622 | 0.477 | 0.215 | 0.726 | 1.850 | 2.310 |
| internal_cv_target_index_valid | delta_mask_target | pca50_ridge | 0.459 | 0.448 | 0.629 | 0.482 | 0.197 | 0.721 | 1.850 | 2.648 |
| internal_cv_target_index_valid | delta_mask_target | random_forest | 0.442 | 0.443 | 0.630 | 0.486 | 0.194 | 0.717 | 1.923 | 114.127 |

## Main Readout

- Best internal full-delta row: `pca50_random_forest`, Spearman `0.494`, Pearson `0.493`, RMSE `0.612`, MAE `0.462`, AUROC GE `< -1.0` `0.742`.
- PCA50 RF beats full-delta RF by Spearman `+0.018` and AUROC GE `< -1.0` `+0.008`.
- PCA50 RF beats PCA50 Ridge by Spearman `+0.009` and lowers RMSE by `0.008`.
- Target masking does not hurt PCA50 RF: internal Spearman change `+0.001`, target-valid Spearman change `-0.001`.
- Runtime improves sharply with PCA: full-delta RF mean fit `127.4` sec/fold vs PCA50 RF `3.4` sec/fold.

## Files

- `summary_metrics.csv`: aggregate metrics for all rows in this ablation.
- `fold_metrics.csv`: per-fold metrics.
- `model_manifest.csv`: checkpoint paths and fit seconds.
- `ranking_summary.csv`: ranking aggregate metrics.
- `topk_candidates.csv`: top-k predicted dependency candidate rows.
- `predictions.csv`: held-out per-gene predictions.
- `metrics_table.csv`: concise table source used above.
- `run_manifest.json` and `cv_config.json`: resolved run metadata.
