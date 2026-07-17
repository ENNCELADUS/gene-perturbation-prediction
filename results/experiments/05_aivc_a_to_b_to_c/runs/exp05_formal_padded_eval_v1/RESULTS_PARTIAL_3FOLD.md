# exp05 formal training: partial three-fold results

**Run ID:** `exp05_formal_padded_eval_v1`

**Code revision:** `9eff7e4`

**Status:** Stopped by user after folds 0, 1, and 2 completed. Fold 3 was interrupted and is excluded. These are partial three-fold results, not the planned five-fold result.

**Evaluation scope:** Frozen-checkpoint `internal_outer_test` only. All means and sample standard deviations below use the completed folds 0, 1, and 2 (`n = 3`).

## GeneEffect prediction

| Fold | C loss | Spearman | Pearson | RMSE | MAE | R2 | Generation loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.219824 | 0.462714 | 0.571025 | 0.468854 | 0.283345 | 0.285952 | 0.277073 |
| 1 | 0.190214 | 0.436057 | 0.538048 | 0.436135 | 0.295270 | 0.269461 | 0.222290 |
| 2 | 0.221430 | 0.386225 | 0.542138 | 0.470563 | 0.310904 | 0.275598 | 0.262554 |
| **Mean** | **0.210489** | **0.428332** | **0.550404** | **0.458517** | **0.296506** | **0.277004** | **0.253972** |
| **Sample SD** | 0.017577 | 0.038825 | 0.017976 | 0.019403 | 0.013821 | 0.008335 | 0.028382 |

## Dependency classification

| Threshold | AUROC | AUPRC | Top-5% enrichment |
|---|---:|---:|---:|
| GeneEffect < -0.5 | 0.827262 +/- 0.017870 | 0.590845 +/- 0.028712 | 4.337638 +/- 0.283453 |
| GeneEffect < -1.0 | 0.851332 +/- 0.015762 | 0.452512 +/- 0.026156 | 6.249890 +/- 0.333679 |

## Artifact status

For each completed fold, the remote run produced non-empty `fold_metrics.csv`, `predictions.csv`, `fit_access_audit.csv`, and `external_alignment_qa.csv`. The interrupted fold 3 was not included in any aggregate above.
