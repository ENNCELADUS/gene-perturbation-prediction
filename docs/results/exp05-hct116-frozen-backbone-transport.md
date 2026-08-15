# Exp05 HCT116 Frozen-Backbone Transport Closeout

**Status:** completed one-shot single-gene external audit; negative for frozen
K562 response-to-GeneEffect transport. This is backbone evidence only, not a
pairwise SL or held-out-cell-line SL result.
**Evaluation date:** 2026-07-21.
**Primary scope:** `E_shared_train` (1,652 genes).

## 1. Experimental result

The experiment passed HCT116 observed perturbation responses through the frozen
K562 exp05 response encoder, GMM pooler, and GeneEffect head. HCT116 Non-Targeting
cells supplied sample-matched controls. Inputs used the frozen relative-control
normalization contract; no HCT-to-K562 control mean/variance matching or STATE
batch remapping was applied.

The prediction contract was amended before label access to reproduce K562
`pad_short=True` behavior: each `sample x gene` bag used every real cell, and a
short final 64-cell window was completed by sampling with replacement from the
same bag. Predictions were averaged first within `sample x gene`, then equally
across samples for each gene. The full label-blind run produced 4,111 gene
predictions from 20,499 `sample x gene` groups and 185,542 real response cells.
The independently copied prediction artifacts were sealed before the HCT116
GeneEffect merge.

The label merge yielded 3,982 evaluated genes. The primary comparison used the
1,652 `E_shared_train` genes that also had a K562 GeneEffect transfer value.

| System | Spearman (95% CI) | Pearson | RMSE | MAE | R2 | AUROC, GE < -0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen K562 response head on HCT116 | -0.001 (-0.050, 0.049) | -0.017 | 0.419 | 0.280 | -0.053 | 0.475 |
| Direct K562 GeneEffect transfer | 0.554 (0.517, 0.591) | 0.751 | 0.296 | 0.200 | 0.474 | 0.945 |
| K562-train mean | undefined | undefined | 0.420 | 0.292 | -0.058 | 0.500 |

On the paired primary cohort, the frozen response head was worse than direct
K562 GeneEffect transfer by `+0.0808` MAE (95% CI 0.0657 to 0.0954) and
`+0.1229` RMSE (95% CI 0.0879 to 0.1585). The frozen response head also showed
no ranking signal on `E_double_shift` (Spearman -0.004, 2,059 genes) or `E_all`
(Spearman -0.001, 3,982 genes).

The comparison includes the K562-train mean and the decisive direct K562
GeneEffect transfer baseline. The preregistered ESM2-only and response-only
baselines were not run, so the complete baseline ladder is not closed. The
result nevertheless rejects the narrower claim that the frozen response head
adds HCT116 value beyond direct K562 GeneEffect transfer.

Primary artifacts:

- [`metrics.csv`](../../results/experiments/05_aivc_a_to_b_to_c/runs/hct116_formal_relative_z_evaluation_v1/metrics.csv)
- [`bootstrap_ci.csv`](../../results/experiments/05_aivc_a_to_b_to_c/runs/hct116_formal_relative_z_evaluation_v1/bootstrap_ci.csv)
- [`paired_comparisons.csv`](../../results/experiments/05_aivc_a_to_b_to_c/runs/hct116_formal_relative_z_evaluation_v1/paired_comparisons.csv)
- [`evaluation_manifest.json`](../../results/experiments/05_aivc_a_to_b_to_c/runs/hct116_formal_relative_z_evaluation_v1/evaluation_manifest.json)

The evaluation manifest binds prediction-manifest SHA-256
`893ceacc937712f96cf0a179b0631ae309071ff57d42846c1da032a061228eb1`
and prediction-seal SHA-256
`96e0a0e31f25d9787a1c05368fdaab3737c2c082f4fe64b07aba162ceb221584`.

## 2. Analysis and interpretation

The direct K562 GeneEffect value transferred strongly to HCT116 (Spearman
0.554), whereas the frozen response head was uncorrelated with HCT116 GeneEffect.
This separates a conserved gene-dependency prior from the failed
response-to-fitness transport path.

On the same 1,652-gene primary cohort, the frozen response prediction had a
standard deviation of only 0.059, compared with 0.409 for HCT116 GeneEffect and
0.428 for K562 GeneEffect. The response head therefore collapsed toward a narrow
output range rather than reproducing the target-line dependency distribution.

The registered `delta_spearman` computed
`corr(pred_HCT - GE_K562, GE_HCT - GE_K562)` and returned a positive value.
Both quantities contain the same subtracted K562 baseline, so this statistic is
mathematically coupled and cannot establish independent HCT116 context signal.
A follow-up analysis that controls K562 GeneEffect directly gives partial
Spearman -0.006 (approximately -0.005), and partial Pearson -0.026. This
analysis explains the shared-baseline artifact.

The bound interpretation is therefore:

- C1 fails because the primary Spearman confidence interval includes zero;
- C2 fails because the response head is materially worse than direct K562
  GeneEffect transfer on paired error;
- the failed component is `HCT116 observed response -> frozen K562 C-head ->
  absolute GeneEffect`;
- the result does not show that the gene-dependency prior fails to transfer;
- the result does not evaluate pairwise SL, Bridge A, Bridge B, or cross-cell-line
  SL generalization; and
- HCT116 may be assigned to a declared GeneEffect training, development, or
  evaluation role; the completed result remains unchanged.
