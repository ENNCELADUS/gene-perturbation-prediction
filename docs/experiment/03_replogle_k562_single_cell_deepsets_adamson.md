# Replogle K562 Single-Cell MIL and Adamson Transfer

Run dates: 2026-05-25 to 2026-05-26

## Scope

This experiment asks whether observed Replogle K562 single-cell perturbation
bags can predict DepMap K562 CRISPR GeneEffect and transfer to Adamson K562
Perturb-seq. The matched prediction key is perturbation gene. Each gene is one
bag of post-perturbation surviving cells; the label is the population-level
DepMap GeneEffect for that gene.

Training and validation use Replogle K562 CRISPRi essential Perturb-seq with
5-fold `internal_cv_all`, seed `42`. External evaluation uses combined Adamson
K562 pilot, UPR epistasis, and UPR Perturb-seq bags. Adamson metrics are
checkpoint-only ensembles: no retraining on Adamson labels.

## Compared Runs

| Purpose | Config | Result path |
| --- | --- | --- |
| First PCA Deep Sets baseline | `configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/deepsets_cv_and_adamson.yaml` | `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/runs/20260525_185722_nogit` |
| Controlled PCA/scVI/HVG mean-pool vs attention matrix | `configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding.yaml` | `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding/runs/attention_mil_20260526_180353` |

The comparison table below uses the controlled multi-embedding run so PCA Deep
Sets, scVI Deep Sets, HVG Deep Sets, and their attention counterparts are
compared under the same CV/external logic.

## Feature QA

| Feature pack | Replogle bags | Replogle cells | Adamson bags | Adamson cells | Dimension | Adamson median cells/gene |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA128 delta | 1917 | 280066 | 85 | 58267 | 128 | 606 |
| scVI128 delta | 1917 | 280066 | 85 | 58267 | 128 | 606 |
| HVG2000 delta | 1917 | 280066 | 85 | 58267 | 2000 | 606 |

scVI external encoding is reference-only: Adamson cells are encoded with the
frozen Replogle scVI model via `SCVI.load(model_dir, adata=query)`. No Adamson
query fine-tuning is used.

## Primary Results

Primary rows use `unweighted` training. `Adamson heldout` averages only fold
models whose Replogle train split did not contain the Adamson target gene.

| Feature | Pooling model | Replogle CV Spearman | Replogle CV AUROC | Adamson Spearman | Adamson AUROC | Adamson AUPRC | Adamson heldout Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA128 delta | Deep Sets mean | 0.484 | 0.736 | 0.504 | 0.854 | 0.639 | 0.431 |
| PCA128 delta | Gated attention | 0.471 | 0.731 | 0.485 | 0.856 | 0.637 | 0.435 |
| scVI128 delta | Deep Sets mean | 0.489 | 0.744 | 0.545 | 0.889 | 0.714 | 0.514 |
| scVI128 delta | Gated attention | 0.478 | 0.739 | 0.552 | 0.893 | 0.725 | 0.509 |
| HVG2000 delta | Deep Sets mean | 0.480 | 0.735 | 0.412 | 0.781 | 0.540 | 0.369 |
| HVG2000 delta | Gated attention | 0.478 | 0.735 | 0.410 | 0.756 | 0.538 | 0.354 |

## Sensitivity: `sqrt_n_cells`

| Feature | Pooling model | Replogle CV Spearman | Adamson Spearman | Adamson heldout Spearman |
| --- | --- | ---: | ---: | ---: |
| PCA128 delta | Deep Sets mean | 0.475 | 0.481 | 0.433 |
| PCA128 delta | Gated attention | 0.465 | 0.470 | 0.374 |
| scVI128 delta | Deep Sets mean | 0.488 | 0.509 | 0.481 |
| scVI128 delta | Gated attention | 0.472 | 0.505 | 0.457 |
| HVG2000 delta | Deep Sets mean | 0.468 | 0.410 | 0.299 |
| HVG2000 delta | Gated attention | 0.468 | 0.396 | 0.286 |

## Readout

- scVI128 is the best representation in this matrix. It improves Adamson
  transfer over PCA128 and HVG2000, with the best primary Adamson row from
  scVI128 gated attention: Spearman `0.552`, AUROC `0.893`, AUPRC `0.725`.
- Attention pooling does not consistently beat mean pooling within embedding.
  Unweighted Spearman deltas for attention minus mean are: PCA `-0.013`
  Replogle / `-0.019` Adamson / `+0.004` heldout; scVI `-0.011` Replogle /
  `+0.007` Adamson / `-0.005` heldout; HVG `-0.002` Replogle / `-0.002`
  Adamson / `-0.016` heldout.
- The original PCA Deep Sets baseline remains a useful floor: Adamson transfer
  is strong (`0.504` Spearman), but scVI128 gives the clearest gain.
- HVG2000 does not help in this configuration despite more input dimensions,
  suggesting the MIL head benefits from a compressed reference embedding.
- `sqrt_n_cells` weighting is not preferred; it lowers the main Spearman
  comparisons for most rows.
- Attention weights are exported for diagnostics of prediction relevance only.
  They should not be interpreted as causal attribution for cell states.

## Artifacts

All paths below are relative to
`results/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding/`.

| Artifact | Path |
| --- | --- |
| PCA128 Replogle bags | `features/single_cell_bags/replogle_k562_single_cell_bags.npz` |
| scVI128 Replogle bags | `features/single_cell_bags/single_cell_scvi_delta/replogle_k562_single_cell_scvi_delta_bags.npz` |
| HVG2000 Replogle bags | `features/single_cell_bags/single_cell_hvg_delta/replogle_k562_single_cell_hvg_delta_bags.npz` |
| PCA128 Adamson bags | `features/external/adamson_k562_single_cell_bags/adamson_k562_single_cell_bags.npz` |
| scVI128 Adamson bags | `features/external/adamson_k562_single_cell_scvi_delta_bags/adamson_k562_single_cell_scvi_delta_bags.npz` |
| HVG2000 Adamson bags | `features/external/adamson_k562_single_cell_hvg_delta_bags/adamson_k562_single_cell_hvg_delta_bags.npz` |
| Fold metrics | `runs/attention_mil_20260526_180353/artifacts/fold_metrics.parquet` |
| Predictions | `runs/attention_mil_20260526_180353/artifacts/predictions.parquet` |
| External ensemble metrics | `runs/attention_mil_20260526_180353/artifacts/external_ensemble_metrics.parquet` |
| External ensemble predictions | `runs/attention_mil_20260526_180353/artifacts/external_ensemble_predictions.parquet` |
| Attention weights | `runs/attention_mil_20260526_180353/artifacts/single_cell_attention_weights.parquet` |
| scVI/HVG external resume log | `logs/attention_mil_20260526_180353_external_resume_20260526_224836.log` |
