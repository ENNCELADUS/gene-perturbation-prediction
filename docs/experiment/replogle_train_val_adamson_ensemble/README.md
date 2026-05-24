# Replogle 5-Fold Ensemble -> Adamson External Test

Status date: 2026-05-24

## Run Setup

- Task: train dependency predictors on Replogle K562 5-fold CV and evaluate the
  fold-model ensemble on Adamson K562.
- Training source: Replogle K562 CRISPRi essential Perturb-seq.
- External test source: combined Adamson K562 pilot, UPR epistasis, and UPR
  Perturb-seq h5ad files.
- Label: DepMap K562 CRISPR GeneEffect.
- Feature set: `delta_all`.
- Weighting: unweighted only.
- Primary external metric: Adamson gene-level Spearman from the mean prediction
  across eligible fold models.

Configs:

- `configs/replogle_k562_adamson_ensemble.yaml`
- `configs/adamson_k562_external_features.yaml`

## Model Selection

Tune variants on Replogle `internal_cv_all`, 5-fold x 1 repeat, seed `42`.
Select the best variant per model family by:

```text
Spearman mean desc
AUROC GE < -1.0 mean desc
RMSE mean asc
```

Model families:

- `ridge_alpha{1,10,100}`
- `pca{20,50,100}_ridge_alpha{1,10,100}`
- `pca{20,50,100}_random_forest_leaf{3,5,10}`
- `xgboost_depth{2,3,4}_lr{0p03,0p1}`

## External Evaluation

The existing pipeline still writes per-fold external predictions under
`external:adamson_k562`. The primary Adamson result is the ensemble table:

```text
external_ensemble:adamson_k562
```

The stricter sensitivity analysis is:

```text
external_ensemble_target_heldout:adamson_k562
```

This target-heldout scope only uses fold predictions from models whose Replogle
training split did not contain the same perturbation gene.

## Output Files

After `run-cv`, key files live under:

```text
results/replogle_k562_adamson_ensemble/runs/<run_id>/
```

Important tables:

- `results/summary_metrics.csv`: Replogle CV and per-fold external summaries.
- `results/external_ensemble_metrics.csv`: primary Adamson ensemble metrics.
- `artifacts/external_ensemble_predictions.parquet`: Adamson gene-level ensemble
  predictions.
- `artifacts/external_ensemble_metrics.parquet`: machine-readable ensemble
  metrics.
- `artifacts/predictions.parquet`: internal CV and per-fold external
  predictions.
- `artifacts/splits.parquet`: Replogle train/test split membership used for
  target-heldout sensitivity.

## Reproduction Commands

```bash
uv run vcc-dep-baseline build-features \
  --config configs/replogle_k562_adamson_ensemble.yaml

uv run vcc-dep-baseline build-external-features \
  --config configs/adamson_k562_external_features.yaml \
  --reference-features results/replogle_k562_adamson_ensemble/features/replogle_k562_delta_features.npz \
  --external-name adamson_k562

uv run vcc-dep-baseline run-cv \
  --config configs/replogle_k562_adamson_ensemble.yaml \
  --run-id replogle_train_val_adamson_ensemble_20260524
```

## Interpretation Caveat

Primary Adamson ensemble metrics test same-cell-line dataset transfer from
Replogle to Adamson. They are not strict unseen-target generalization unless
supported by the target-heldout sensitivity scope. Adamson is much smaller and
more pathway/UPR-biased than Replogle, so weak external performance should be
interpreted as transfer evidence, not as a full rejection of the B->C task.
