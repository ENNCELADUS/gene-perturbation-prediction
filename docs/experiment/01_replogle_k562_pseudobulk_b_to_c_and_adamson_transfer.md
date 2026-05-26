# Replogle K562 Pseudobulk B->C Baseline and Adamson Transfer

Run dates: 2026-05-14 to 2026-05-24

Remote result paths:

- Replogle CV baseline: `results/replogle_k562_b_to_c_baseline`
- Adamson external ensemble:
  `results/replogle_k562_adamson_ensemble/runs/replogle_train_val_adamson_ensemble_20260524`

## Run Setup

- Task: predict DepMap K562 CRISPR GeneEffect from observed perturbation
  transcriptome features.
- Training data: Replogle K562 CRISPRi essential Perturb-seq.
- Validation: Replogle `internal_cv_all`, 5-fold CV x 1 repeat, seed `42`.
- External test: combined Adamson K562 pilot, UPR epistasis, and UPR
  Perturb-seq, aggregated to gene-level rows.
- Input features: perturbation pseudobulk delta expression.
- Primary external scope: `external_ensemble:adamson_k562`, mean prediction
  across the five Replogle fold models.
- Sensitivity scope: `external_ensemble_target_heldout:adamson_k562`, averaging
  only fold models whose Replogle train split did not contain the Adamson target
  gene.

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

## 5-Fold Validation and Adamson Test

Adamson was not used for model selection.

| Model / check | Replogle CV Spearman | Replogle CV RMSE | Adamson Spearman | Adamson RMSE | Adamson AUROC GE < -1.0 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pca50_ridge_alpha100` | 0.485 | 0.619 | 0.500 | 0.605 | 0.886 | Best external-transfer row. |
| `pca50_ridge_alpha100`, target-heldout | - | - | 0.490 | 0.612 | 0.863 | Similar transfer after removing fold models that trained on the same target. |
| `pca50_random_forest_leaf3` | 0.499 | 0.611 | 0.385 | 0.754 | 0.791 | Best Replogle CV row, weaker Adamson transfer. |
| `ridge_alpha100` | 0.359 | 0.709 | 0.436 | 1.007 | 0.895 | Transfers by rank but is poorly calibrated. |
| `response_burden + ridge` | 0.426 | 0.649 | - | - | - | Generic response magnitude carries substantial Replogle signal. |
| `n_cells_only` | 0.000 | 0.700 | - | - | - | Cell count does not explain the signal. |
| Target-masked Ridge vs full Ridge | -0.005 Spearman change | - | - | - | - | Direct target-gene expression leakage is not the main driver. |

## Main Readout

- The pseudobulk B->C bridge is viable: Replogle CV reaches about `0.49`
  Spearman with PCA Ridge / PCA RandomForest and Adamson transfer reaches `0.50`
  Spearman with PCA Ridge.
- The Replogle CV winner is not the best external-transfer model. PCA
  RandomForest wins internal CV, but the simpler PCA Ridge transfers better to
  Adamson.
- Cell count and direct target-expression leakage are insufficient explanations,
  while response burden is a meaningful component of the signal.
- Adamson is a small, UPR-biased test set with `85` genes, so this is a
  same-cell-line transfer gate, not a broad synthetic-lethality validation.

## Files

- Replogle CV: `summary_metrics.csv`, `fold_metrics.parquet`,
  `predictions.parquet`, and model checkpoints under
  `results/replogle_k562_b_to_c_baseline`.
- Adamson transfer: `results/summary_metrics.csv`,
  `results/external_ensemble_metrics.csv`,
  `artifacts/external_ensemble_predictions.parquet`,
  `artifacts/external_ensemble_metrics.parquet`,
  `artifacts/fold_metrics.parquet`, `artifacts/predictions.parquet`, and
  `artifacts/splits.parquet` under the Adamson ensemble run.
