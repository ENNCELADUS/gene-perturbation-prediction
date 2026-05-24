# Adamson External Ensemble Metrics

Run date: 2026-05-24

Remote workspace:

```bash
ssh -p 2222 richard@10.20.246.163
cd ~/gene-perturbation-prediction
```

Remote result path:

- `results/replogle_k562_adamson_ensemble/runs/replogle_train_val_adamson_ensemble_20260524`

## Run Setup

- Task: train dependency predictors on Replogle K562 and test same-cell-line
  dataset transfer on Adamson K562.
- Training data: Replogle K562 CRISPRi essential Perturb-seq, 5-fold CV x 1
  repeat, random seed `42`.
- External data: combined Adamson K562 pilot, UPR epistasis, and UPR Perturb-seq.
- Input features: perturbation pseudobulk delta expression, `delta_all` only.
- Label: DepMap K562 CRISPR GeneEffect.
- Weighting: unweighted only.
- Primary external readout: Adamson gene-level ensemble Spearman from mean
  prediction across the five Replogle fold models.
- Configs: `configs/replogle_k562_adamson_ensemble.yaml` and
  `configs/adamson_k562_external_features.yaml`.

## Feature QA

| Item | Value |
| --- | ---: |
| Replogle modeling rows | 1917 |
| Replogle expression genes | 8563 |
| Adamson source perturbation rows | 116 |
| Adamson gene-level rows | 85 |
| Adamson observed Replogle-reference genes | 7904 |
| Adamson missing Replogle-reference genes | 659 |
| Adamson median cells per gene | 606 |

Adamson source contribution after gene-level aggregation:

| Source dataset | Genes | Source rows | Cells |
| --- | ---: | ---: | ---: |
| `adamson_pilot` | 6 | 6 | 3421 |
| `adamson_pilot;adamson_upr_epistasis` | 1 | 2 | 563 |
| `adamson_upr_epistasis;adamson_upr_perturb_seq` | 9 | 29 | 8942 |
| `adamson_upr_perturb_seq` | 69 | 79 | 45341 |

## Replogle CV Selection

Best variant per model family, selected on Replogle `internal_cv_all` Spearman.
Adamson was not used for model selection.

| Family | Selected model | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | `ridge_alpha100` | 0.359 | 0.369 | 0.709 | 0.541 | -0.018 | 0.675 |
| PCA Ridge | `pca50_ridge_alpha100` | 0.485 | 0.474 | 0.619 | 0.472 | 0.224 | 0.738 |
| PCA RandomForest | `pca50_random_forest_leaf3` | 0.499 | 0.495 | 0.611 | 0.462 | 0.244 | 0.743 |
| XGBoost | `xgboost_depth4_lr0p03` | 0.473 | 0.481 | 0.616 | 0.473 | 0.230 | 0.736 |
| Strict MLP raw | `mlp_raw` | 0.376 | 0.392 | 0.678 | 0.505 | 0.069 | 0.683 |
| Strict MLP PCA50 | `mlp_pca50` | 0.431 | 0.421 | 0.667 | 0.487 | 0.096 | 0.707 |
| Strict MLP PCA100 | `mlp_pca100` | 0.364 | 0.373 | 0.700 | 0.511 | 0.006 | 0.678 |

## Adamson Ensemble Results

Primary scope: `external_ensemble:adamson_k562`. Each Adamson gene is scored by
the mean raw prediction across all five eligible Replogle fold models.

| Model | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | AUPRC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pca50_ridge_alpha100` | 0.500 | 0.559 | 0.605 | 0.486 | 0.263 | 0.886 | 0.743 | 4.048 |
| `ridge_alpha100` | 0.436 | 0.517 | 1.007 | 0.879 | -1.041 | 0.895 | 0.770 | 4.048 |
| `pca50_random_forest_leaf3` | 0.385 | 0.359 | 0.754 | 0.639 | -0.143 | 0.791 | 0.546 | 2.429 |
| `mlp_raw` | 0.315 | 0.490 | 0.671 | 0.547 | 0.093 | 0.833 | 0.743 | 4.048 |
| `mlp_pca50` | 0.325 | 0.391 | 0.796 | 0.700 | -0.276 | 0.762 | 0.602 | 3.238 |
| `mlp_pca100` | 0.042 | -0.001 | 1.030 | 0.907 | -1.133 | 0.496 | 0.270 | 0.810 |
| `xgboost_depth4_lr0p03` | 0.024 | 0.173 | 0.763 | 0.641 | -0.170 | 0.586 | 0.440 | 2.429 |

## Target-Heldout Sensitivity

Scope: `external_ensemble_target_heldout:adamson_k562`. This stricter analysis
uses only fold predictions from models whose Replogle training split did not
contain the same Adamson target gene. Mean ensemble size is `2.65` fold models
per gene, with min `1` and max `5`.

| Model | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | AUPRC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pca50_ridge_alpha100` | 0.490 | 0.538 | 0.612 | 0.488 | 0.247 | 0.863 | 0.713 | 4.048 |
| `ridge_alpha100` | 0.349 | 0.442 | 1.052 | 0.896 | -1.226 | 0.821 | 0.677 | 4.048 |
| `pca50_random_forest_leaf3` | 0.307 | 0.263 | 0.769 | 0.656 | -0.190 | 0.695 | 0.521 | 2.429 |
| `mlp_raw` | 0.201 | 0.385 | 0.723 | 0.574 | -0.052 | 0.702 | 0.608 | 4.048 |
| `mlp_pca50` | 0.225 | 0.330 | 0.809 | 0.697 | -0.318 | 0.661 | 0.486 | 3.238 |
| `mlp_pca100` | 0.049 | 0.067 | 1.032 | 0.912 | -1.143 | 0.490 | 0.302 | 0.810 |
| `xgboost_depth4_lr0p03` | -0.014 | 0.081 | 0.773 | 0.656 | -0.202 | 0.542 | 0.387 | 1.619 |

## Main Readout

- Best Adamson transfer model: `pca50_ridge_alpha100`, primary Spearman `0.500`
  and target-heldout Spearman `0.490`.
- Replogle CV winner by Spearman was `pca50_random_forest_leaf3` (`0.499`), but
  it transferred worse to Adamson (`0.385` primary Spearman), so internal CV
  winner is not automatically the best external-transfer model.
- Ridge-family models transfer better than PCA RandomForest and XGBoost on this
  Adamson benchmark. The PCA Ridge result also has positive Adamson R2 (`0.263`),
  unlike the selected RandomForest and XGBoost rows.
- The 2026-05-24 strict MLP follow-up is suggestive only: MLP rows did not beat
  `pca50_ridge_alpha100` on Adamson transfer.
- Target-heldout performance is close to the primary ensemble for PCA Ridge
  (`0.490` vs `0.500` Spearman), arguing that the Adamson result is not purely a
  target-exposure artifact.
- Caveat: Adamson is small (`85` gene-level rows) and UPR-biased. This result
  supports same-cell-line dataset transfer for observed transcriptome -> DepMap
  dependency prediction, but it is not yet a broad SL validation.

## Files

- `results/summary_metrics.csv`: Replogle CV and per-fold external summaries.
- `results/external_ensemble_metrics.csv`: Adamson ensemble and target-heldout
  metrics.
- `artifacts/external_ensemble_predictions.parquet`: Adamson gene-level ensemble
  predictions with ensemble size per gene.
- `artifacts/external_ensemble_metrics.parquet`: machine-readable ensemble
  metrics.
- `artifacts/fold_metrics.parquet`: per-fold internal and external metrics.
- `artifacts/predictions.parquet`: internal CV and per-fold external
  predictions.
- `artifacts/splits.parquet`: Replogle split membership used for target-heldout
  sensitivity.
