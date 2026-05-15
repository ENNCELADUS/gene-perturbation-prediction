# Replogle K562 NAR Viability-Axis Audit, 5x1 Main Setup

Run date: 2026-05-15

Remote run:

```text
richard@10.20.161.54:/home/richard/projects/VCC/results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515
```

Local config:

```text
configs/replogle_k562_viability_axis_audit_5x1_main.yaml
```

## Question

Does the Replogle K562 B->C baseline mostly learn a generic
cell-death/proliferation or viability axis, rather than perturbation-specific
dependency biology?

This audit uses the public 2019 NAR cell-death/proliferation coefficient models
as external viability-axis anchors. The coefficients come from
`bence-szalai/Cell-death-signatures`:

- `models/achilles.csv`
- `models/ctrp.csv`

The run computes NAR Achilles, CTRP, and mean scores from Replogle pseudobulk
delta expression, then compares score-only models, score-plus-burden models,
the original transcriptome baseline, and transcriptome models after fold-local
NAR-score residualization.

## Setup

- Dataset: Replogle K562 CRISPRi essential screen.
- Input features: perturbation pseudobulk delta expression.
- Label: DepMap K562 CRISPR GeneEffect.
- Evaluation: `internal_cv_all`, 5-fold CV x 1 repeat, random seed `42`.
- Weighting: unweighted only.
- Leakage and bias checks were intentionally not rerun. This means no
  `delta_mask_target`, no `internal_cv_target_index_valid`, and no target-valid
  only subset.
- Main baseline rows were aligned with
  `docs/experiment/baselines/replogle_k562_b_to_c_baseline/README.md`.

Feature QA:

| Item | Value |
| --- | ---: |
| Modeling rows | 1917 |
| Expression genes | 8563 |
| NAR coefficients per model | 978 |
| Matched NAR genes | 728 |
| Missing NAR genes | 250 |
| Matched fraction | 0.744 |

## Key Results

| Comparison | Feature set | Model | Spearman | AUROC GE < -1.0 | RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| NAR score only | `nar_viability_scores` | `nar_score_ridge` | 0.244 | 0.623 | 0.680 |
| NAR score + burden | `nar_viability_scores_plus_burden` | `nar_score_plus_burden_ridge` | 0.443 | 0.714 | 0.643 |
| Best transcriptome baseline | `delta_all` | `pca50_random_forest` | 0.494 | 0.742 | 0.612 |
| Best NAR-residualized transcriptome | `nar_resid_delta_all` | `nar_resid_pca50_random_forest` | 0.503 | 0.744 | 0.610 |
| NAR-residualized linear transcriptome | `nar_resid_delta_all` | `nar_resid_pca50_ridge` | 0.400 | 0.698 | 0.640 |

Main-style baseline rows from the same run:

| Feature set | Model | Spearman | Pearson | RMSE | R2 | AUROC GE < -1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `response_burden` | `ridge` | 0.426 | 0.387 | 0.649 | 0.148 | 0.705 |
| `delta_all` | `ridge` | 0.348 | 0.356 | 0.724 | -0.061 | 0.670 |
| `delta_all` | `elastic_net` | 0.300 | 0.314 | 0.758 | -0.164 | 0.649 |
| `delta_all` | `pca20_ridge` | 0.480 | 0.463 | 0.623 | 0.215 | 0.736 |
| `delta_all` | `pca50_ridge` | 0.485 | 0.474 | 0.619 | 0.224 | 0.738 |
| `delta_all` | `pca100_ridge` | 0.481 | 0.474 | 0.619 | 0.224 | 0.736 |
| `delta_all` | `random_forest` | 0.476 | 0.481 | 0.616 | 0.231 | 0.734 |
| `delta_all` | `pca20_random_forest` | 0.492 | 0.490 | 0.613 | 0.238 | 0.742 |
| `delta_all` | `pca50_random_forest` | 0.494 | 0.493 | 0.612 | 0.242 | 0.742 |
| `delta_all` | `pca100_random_forest` | 0.491 | 0.483 | 0.616 | 0.232 | 0.742 |

## Correlation Diagnostics

| Pair | Spearman | Pearson |
| --- | ---: | ---: |
| `y_true` vs `nar_achilles_score` | 0.244 | 0.254 |
| `y_true` vs `nar_ctrp_score` | 0.121 | 0.095 |
| `y_true` vs `nar_mean_score` | 0.230 | 0.237 |
| `y_true` vs `response_burden_delta_l2` | -0.421 | -0.345 |
| `y_true` vs held-out baseline prediction | 0.496 | 0.492 |
| `nar_achilles_score` vs held-out baseline prediction | 0.470 | 0.504 |
| `nar_mean_score` vs held-out baseline prediction | 0.436 | 0.453 |
| `response_burden_delta_l1_mean` vs held-out baseline prediction | -0.800 | -0.638 |

Interpretation of signs: more negative GeneEffect means stronger dependency.
The response-burden features are therefore negatively correlated with `y_true`,
while prediction columns are on the GeneEffect scale.

## Takeaway

The NAR viability axis alone does not explain the current Replogle B->C signal.
`nar_score_ridge` reaches only Spearman `0.244`, far below the best transcriptome
baseline at `0.494`.

Adding response-burden features closes much of the gap: `nar_score + burden`
reaches Spearman `0.443` and AUROC `0.714`. This means a substantial part of the
baseline signal is generic response magnitude, stress, or viability-like biology.

The strongest counterpoint is the residualized RandomForest result. After
fold-local residualization of each delta-expression gene against NAR Achilles and
CTRP scores, `nar_resid_delta_all + pca50_random_forest` retains and slightly
improves performance: Spearman `0.503` vs the original `delta_all +
pca50_random_forest` Spearman `0.494`. This argues against the simple claim that
the current B->C model is mostly just the NAR generic viability axis.

The linear PCA Ridge result is more sensitive: `pca50_ridge` drops from Spearman
`0.485` to `0.400` after NAR residualization. The linear model appears to rely
more on a viability-like axis, while the nonlinear PCA RandomForest can still use
additional transcriptomic structure after that axis is removed.

## Next Experiments

1. Treat response burden as a first-class nuisance axis, not only NAR scores.
   Run fold-local residualization against both NAR scores and the top response
   burden summaries.
2. Add pathway-level stress, cell-cycle, translation, ribosome, interferon,
   DNA-damage, and apoptosis scores to separate generic burden from interpretable
   biological programs.
3. Compare Lasso/ElasticNet feature selection against Ridge only after nuisance
   residualization. Lasso may help if the remaining signal is sparse, but it
   should not be used as the first generic-noise control because response burden
   is already a strong low-dimensional confounder.
4. Repeat the same audit for any future external perturbation-response datasets
   before treating cross-dataset performance as perturbation-specific dependency
   biology.

## Reproduction Commands

On `richard@10.20.161.54`:

```bash
cd /home/richard/projects/VCC

uv run vcc-dep-baseline build-features \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml \
  --run-id viability_axis_5x1_main_20260515 \
  --feature-set delta_all \
  --model mean_label \
  --model ridge \
  --model elastic_net \
  --model random_forest \
  --model pca20_ridge \
  --model pca50_ridge \
  --model pca100_ridge \
  --model pca20_random_forest \
  --model pca50_random_forest \
  --model pca100_random_forest

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml \
  --run-id viability_axis_5x1_main_20260515 \
  --resume \
  --feature-set response_burden \
  --model ridge

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml \
  --run-id viability_axis_5x1_main_20260515 \
  --resume \
  --feature-set nar_viability_scores \
  --model nar_score_ridge

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml \
  --run-id viability_axis_5x1_main_20260515 \
  --resume \
  --feature-set nar_viability_scores_plus_burden \
  --model nar_score_plus_burden_ridge

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_viability_axis_audit_5x1_main.yaml \
  --run-id viability_axis_5x1_main_20260515 \
  --resume \
  --feature-set nar_resid_delta_all \
  --model nar_resid_pca50_ridge \
  --model nar_resid_pca50_random_forest

uv run vcc-dep-baseline viability-axis-report \
  --run-dir results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515 \
  --features results/replogle_k562_viability_axis_audit_5x1_main/features/replogle_k562_delta_features.npz \
  --output results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515/results/viability_axis_report.md \
  --baseline-predictions results/replogle_k562_viability_axis_audit_5x1_main/runs/viability_axis_5x1_main_20260515/artifacts/predictions.parquet
```

Verification completed on the remote machine:

```text
uv run ruff check .       passed
uv run python -m pytest   5 passed, 2 warnings
```
