# Replogle K562 Single-Cell Deep Sets and Adamson Transfer

Run dates: 2026-05-25 to 2026-05-26

Canonical remote result path after the 2026-05-26 artifact reorganization:

- `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/runs/20260525_185722_nogit`

Config:

- `configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/deepsets_cv_and_adamson.yaml`

## Run Setup

- Task: predict DepMap K562 CRISPR GeneEffect from observed single-cell
  perturbation-response bags.
- Training data: Replogle K562 CRISPRi essential Perturb-seq.
- External test: combined Adamson K562 pilot, UPR epistasis, and UPR
  Perturb-seq.
- Representation: each perturbation gene is one bag of post-perturbation cells.
  Cell features are PCA coordinates of cell-level delta expression in the
  Replogle HVG/PCA space.
- Model: Deep Sets regression, `deepsets_pca128_meanpool`, with a shared cell
  encoder and mean pooling over each gene-level bag.
- Validation: Replogle `internal_cv_all`, 5-fold CV x 1 repeat, seed `42`.
- External evaluation: existing fold checkpoints only; no Replogle retraining.
- Primary external scope: `external_ensemble:adamson_k562`.
- Sensitivity scope: `external_ensemble_target_heldout:adamson_k562`, averaging
  only fold models whose Replogle train split did not contain the Adamson target
  gene.

## Feature QA

| Item | Value |
| --- | ---: |
| Replogle bags | 1917 |
| Replogle cells | 280066 |
| Adamson source perturbation rows | 116 |
| Adamson gene-level bags | 85 |
| Adamson cells | 58267 |
| PCA dimensions | 128 |
| HVGs used for PCA | 2000 |
| Adamson median cells per gene | 606 |

## 5-Fold Validation and Adamson Test

| Scope | Weighting | Genes | Mean ensemble size | Spearman | Pearson | RMSE | MAE | R2 | AUROC GE < -1.0 | AUPRC GE < -1.0 | Top 5% enrich GE < -1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Replogle CV | unweighted | 1917 | - | 0.484 | 0.473 | 0.622 | 0.472 | 0.217 | 0.736 | - | 1.935 |
| Replogle CV | sqrt_n_cells | 1917 | - | 0.474 | 0.463 | 0.628 | 0.479 | 0.200 | 0.733 | - | 1.935 |
| Adamson ensemble | unweighted | 85 | 5.00 | 0.505 | 0.487 | 0.676 | 0.552 | 0.082 | 0.854 | 0.639 | 3.238 |
| Adamson ensemble | sqrt_n_cells | 85 | 5.00 | 0.482 | 0.444 | 0.697 | 0.566 | 0.023 | 0.847 | 0.575 | 2.429 |
| Adamson target-heldout | unweighted | 85 | 2.65 | 0.431 | 0.423 | 0.692 | 0.571 | 0.036 | 0.813 | 0.572 | 3.238 |
| Adamson target-heldout | sqrt_n_cells | 85 | 2.65 | 0.432 | 0.348 | 0.723 | 0.588 | -0.050 | 0.809 | 0.545 | 2.429 |

## Main Readout

- The first observed single-cell Deep Sets baseline is competitive with the
  pseudobulk Replogle CV baseline but does not clearly beat it: unweighted
  Replogle CV Spearman is `0.484`, close to pseudobulk PCA Ridge / RandomForest
  at about `0.49`.
- Adamson transfer is strong for the primary unweighted ensemble: Spearman
  `0.505`, AUROC at GeneEffect `< -1.0` `0.854`, and top-5% enrichment `3.238`.
- Target-heldout Adamson remains positive but drops to `0.431` Spearman, so some
  of the primary transfer signal may depend on fold models that saw the same
  target in Replogle training.
- `sqrt_n_cells` weighting does not improve this run; unweighted is the primary
  comparison row.
- This result passes the "single-cell bags are usable" gate, but it does not yet
  prove that bag-level heterogeneity improves over pseudobulk mean-delta
  summaries. The next experiment should add stronger set pooling / attention,
  burden and state-composition controls, and direct paired comparison against
  frozen pseudobulk models.

## Files

- Replogle bags:
  `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/features/single_cell_bags/replogle_k562_single_cell_bags.npz`.
- Adamson bags:
  `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/features/external/adamson_k562_single_cell_bags/adamson_k562_single_cell_bags.npz`.
- Checkpoint manifest: `artifacts/model_manifest.parquet`.
- Replogle and per-fold Adamson metrics: `artifacts/fold_metrics.parquet`.
- Replogle and per-fold Adamson predictions: `artifacts/predictions.parquet`.
- Adamson ensemble metrics: `artifacts/external_ensemble_metrics.parquet`.
- Adamson ensemble predictions: `artifacts/external_ensemble_predictions.parquet`.
