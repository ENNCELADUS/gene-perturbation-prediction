# K562 Linear Predicted-B to Dependency Pilot

Run status: implemented on 2026-06-04; full real-data run not completed locally.

## Scope

This experiment tests a conservative fold-local A->B->C baseline before using
AIVC or other virtual-cell models. The supervised unit remains the matched
K562 perturbation gene. The pipeline is:

```text
control-cell expression + perturbation target
    -> predicted post-perturbation single-cell expression bag B_hat
    -> scVI128 GMM/prototype Ridge features
    -> DepMap K562 GeneEffect
```

The C-side architecture is fixed to the observed-B winner family, but the C
head is trained inside each fold on predicted `B_hat` features. The experiment
does not reuse observed-B checkpoints.

## Implemented Baselines

| Baseline | A->B training target | Perturbation encoding | Main caveat |
| --- | --- | --- | --- |
| `mean_delta_ridge` | Gene-level mean expression delta | Masked control mean only | Predicts mean shift, not cell-level heterogeneity. |
| `pseudo_pair_ridge` | Pseudo-paired cell expression delta | Masked control cell only | Pairing is random; optional strata are used only if obs fields exist. |

Both baselines generate `B_hat` from a deterministic capped control-cell panel.
The default cap is `512` cells per perturbation gene to avoid using test-gene
observed cell counts.

## Fold-Local Evaluation

For each Replogle CV fold:

1. Train A->B only on train genes.
2. Generate `B_hat_train` and `B_hat_test` from control cells.
3. Fit scVI128 using controls plus observed train-gene cells only.
4. Transform `B_hat_train` and `B_hat_test` into scVI delta bags.
5. Fit K64 centered GMM prototypes on `B_hat_train` plus controls.
6. Train Ridge alpha `300` on `B_hat_train` features and train GeneEffect labels.
7. Predict GeneEffect for test genes.

Primary outputs are split train/test A->B reconstruction metrics, `B_hat` bag
metadata, scVI/GMM QA, C fold metrics, predictions, and a run manifest with
leakage-scope notes. Test-gene observed cells are read only after `B_hat_test`
is generated, and only for `a_to_b_test_mean_rmse` /
`a_to_b_test_mean_mae`; they are not used for A->B fitting, scVI fitting, GMM
fitting, or C-head fitting.

## Configuration

Default config:
`configs/experiments/04_k562_linear_predicted_b_to_c/linear_predicted_b_cv.yaml`.

Run command:

```bash
uv run vcc-dep-baseline run-predicted-b-cv \
  --config configs/experiments/04_k562_linear_predicted_b_to_c/linear_predicted_b_cv.yaml
```

## Readout

The v0 success criterion is not to beat observed-B or pseudobulk baselines. The
goal is to establish a no-leakage predicted-B evaluation loop with interpretable
forward-model and dependency-ranking metrics. Any future AIVC result should use
this baseline as the minimum comparison before claiming that virtual-cell
predictions improve dependency ranking.
