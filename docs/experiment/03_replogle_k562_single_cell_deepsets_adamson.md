# Replogle K562 Single-Cell MIL and Adamson Transfer

Run dates: 2026-05-25 to 2026-05-27

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
| Multi-head gated attention MIL | `configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/multihead_attention_mil.yaml` | `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding/runs/multihead_attention_mil_20260526_233442` |
| Distribution / prototype regression | `configs/experiments/03_replogle_k562_single_cell_deepsets_adamson/distribution_prototype_regression.yaml` | `results/experiments/03_replogle_k562_single_cell_deepsets_adamson/distribution_prototype_regression/runs/distribution_proto_20260527_013948` |

The comparison table below uses the controlled multi-embedding run for mean
pooling and single-head attention, plus the follow-up multi-head run for the
advanced MIL baseline and the completed distribution/prototype run for frozen
GMM and CloudPred-like distribution regression. The multi-head and distribution
runs reuse the same PCA/scVI/HVG bag logic, run only `unweighted` models, and
use the same CV/external checkpoint ensemble logic. The distribution run
completed PCA128 and scVI128 Adamson evaluation; HVG2000 CloudPred training was
stopped by GPU OOM after fold 0 frozen-GMM rows, so HVG distribution rows are not
used for the primary comparison.

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
| PCA128 delta | Multi-head gated attention | 0.469 | 0.732 | 0.466 | 0.848 | 0.602 | 0.450 |
| PCA128 delta | GMM prototype K64 deltap Ridge alpha1 | 0.414 | 0.774 | 0.488 | 0.810 | 0.738 | 0.487 |
| scVI128 delta | Deep Sets mean | 0.489 | 0.744 | 0.545 | 0.889 | 0.714 | 0.514 |
| scVI128 delta | Gated attention | 0.478 | 0.739 | 0.552 | 0.893 | 0.725 | 0.509 |
| scVI128 delta | Multi-head gated attention | 0.474 | 0.736 | 0.541 | 0.895 | 0.717 | 0.507 |
| scVI128 delta | GMM prototype K64 deltap Ridge alpha1 | 0.639 | 0.899 | 0.664 | 0.911 | 0.785 | 0.639 |
| scVI128 delta | GMM prototype K64 centered Ridge alpha100 | 0.636 | 0.898 | 0.663 | 0.908 | 0.793 | 0.668 |
| scVI128 delta | CloudPred-like K32 centered | 0.601 | 0.899 | 0.602 | 0.902 | 0.734 | 0.593 |
| HVG2000 delta | Deep Sets mean | 0.480 | 0.735 | 0.412 | 0.781 | 0.540 | 0.369 |
| HVG2000 delta | Gated attention | 0.478 | 0.735 | 0.410 | 0.756 | 0.538 | 0.354 |
| HVG2000 delta | Multi-head gated attention | 0.458 | 0.726 | 0.387 | 0.724 | 0.503 | 0.293 |

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

- scVI128 is the best representation in this matrix. The best overall Adamson
  transfer row is now scVI128 GMM prototype distribution regression with K64
  deltap Ridge alpha1: Spearman `0.664`, AUROC `0.911`, AUPRC `0.785`, and
  heldout Spearman `0.639`. The best target-heldout row is the adjacent scVI128
  K64 centered Ridge alpha100 model: primary Adamson Spearman `0.663`, heldout
  Spearman `0.668`, AUROC `0.912`, and AUPRC `0.799`.
- Distribution / prototype regression is the first single-cell method in this
  experiment to clearly beat the previous Adamson transfer gate. Relative to the
  earlier best scVI128 single-head gated attention row (`0.552` Spearman,
  `0.893` AUROC, `0.725` AUPRC), the best scVI GMM row improves Adamson
  Spearman by `+0.112` and AUPRC by `+0.060`. It also clears the planned
  success threshold of Adamson Spearman `>=0.572` and heldout Spearman `>=0.494`.
- Attention pooling does not consistently beat mean pooling within embedding.
  Unweighted Spearman deltas for attention minus mean are: PCA `-0.013`
  Replogle / `-0.019` Adamson / `+0.004` heldout; scVI `-0.011` Replogle /
  `+0.007` Adamson / `-0.005` heldout; HVG `-0.002` Replogle / `-0.002`
  Adamson / `-0.016` heldout.
- The strongest distribution rows are simple frozen GMM occupancy features plus
  Ridge, not the end-to-end CloudPred-like head. The best CloudPred-like scVI row
  reaches Adamson Spearman `0.602` and heldout Spearman `0.593`, which is better
  than attention but below the frozen scVI GMM Ridge rows.
- Multi-head gated attention with 4 heads and orthogonality regularization does
  not improve the primary transfer metric in this v1. The best multi-head row is
  scVI128 with Adamson Spearman `0.541`, AUROC `0.895`, AUPRC `0.717`, and
  heldout Spearman `0.507`, close to but below the single-head scVI row on
  Spearman/AUPRC. PCA multi-head slightly improves heldout Spearman over
  single-head attention (`0.450` vs `0.435`) but lowers Adamson overall Spearman
  (`0.466` vs `0.485`).
- The original PCA Deep Sets baseline remains a useful floor: Adamson transfer
  is strong (`0.504` Spearman). PCA distribution regression is mixed: the best
  PCA target-heldout distribution row reaches heldout Spearman `0.487` with
  primary Adamson Spearman `0.488`, but it does not improve over the original
  PCA Deep Sets primary Adamson Spearman.
- HVG2000 does not help in this configuration despite more input dimensions,
  suggesting the MIL head benefits from a compressed reference embedding. This
  remains true for multi-head attention, where HVG2000 drops to `0.387` Adamson
  Spearman and `0.293` heldout Spearman. The distribution run did not complete
  HVG2000 CloudPred because of CUDA OOM, and only fold 0 frozen-GMM HVG rows are
  available, so HVG distribution is not interpreted.
- `sqrt_n_cells` weighting is not preferred; it lowers the main Spearman
  comparisons for most rows.
- Attention weights are exported for diagnostics of prediction relevance only.
  Multi-head attention additionally exports per-head entropy, effective cell
  counts, and head-similarity diagnostics. These weights should not be
  interpreted as causal attribution for cell states.

## Artifacts

Attention/MIL paths below are relative to
`results/experiments/03_replogle_k562_single_cell_deepsets_adamson/attention_mil_multi_embedding/`.
Distribution/prototype paths are relative to
`results/experiments/03_replogle_k562_single_cell_deepsets_adamson/distribution_prototype_regression/`.

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
| Multi-head fold metrics | `runs/multihead_attention_mil_20260526_233442/artifacts/fold_metrics.parquet` |
| Multi-head predictions | `runs/multihead_attention_mil_20260526_233442/artifacts/predictions.parquet` |
| Multi-head external ensemble metrics | `runs/multihead_attention_mil_20260526_233442/artifacts/external_ensemble_metrics.parquet` |
| Multi-head external ensemble predictions | `runs/multihead_attention_mil_20260526_233442/artifacts/external_ensemble_predictions.parquet` |
| Multi-head attention weights | `runs/multihead_attention_mil_20260526_233442/artifacts/single_cell_attention_weights.parquet` |
| Multi-head attention head diagnostics | `runs/multihead_attention_mil_20260526_233442/artifacts/single_cell_attention_head_diagnostics.parquet` |
| Distribution PCA/scVI fold metrics | `runs/distribution_proto_20260527_013948/artifacts/fold_metrics.parquet` |
| Distribution PCA/scVI predictions | `runs/distribution_proto_20260527_013948/artifacts/predictions.parquet` |
| Distribution PCA/scVI external ensemble metrics | `runs/distribution_proto_20260527_013948/artifacts/external_ensemble_metrics.parquet` |
| Distribution PCA/scVI external ensemble predictions | `runs/distribution_proto_20260527_013948/artifacts/external_ensemble_predictions.parquet` |
| Distribution PCA/scVI external log | `logs/adamson_pca_scvi_external_20260527_170826.log` |
