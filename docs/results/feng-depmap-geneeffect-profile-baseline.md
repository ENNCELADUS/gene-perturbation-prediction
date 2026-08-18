# Feng2024 DepMap GeneEffect profile baseline

**Status:** completed 2026-07-23. Two ground-truth DepMap single-gene
GeneEffect profiles contain substantial supervised signal for the sampled Feng2024
pair labels, but simple co-dependency correlations are close to random and the
signal weakens materially after removing pan-essential pairs. The output is
`P(sampled Feng label = 1 | balanced benchmark)`, not a biological SL probability
and not a double-knockout interaction estimate.

## Question and fixed evaluation

The experiment asks for the empirical ceiling available when a pair `(a, b)` is
represented only by the measured DepMap GeneEffect vector for `a` and the measured
vector for `b`. It uses the official Feng2024 Rand 1:1 train/test arrays for CV1,
CV2, and CV3, all five folds, without using an SL graph in the feature path.

- **Data.** DepMap Public 26Q1 `CRISPRGeneEffect.csv`: 1,208 cell lines. Of the
  9,845 Feng genes, 9,650 have an exact symbol match and 9,635 have at least 20
  finite cell-line measurements. Uncovered profiles are represented by a neutral
  zero vector plus an explicit coverage feature.
- **Strict features.** The supervised models see only the two queried profiles:
  swap-invariant per-gene summaries, pairwise profile statistics, or the raw
  `[x_a + x_b, |x_a - x_b|]` vector. They do not use other candidate profiles,
  graph edges, or test labels.
- **Model ladder.** Absolute Pearson and Spearman co-dependency; a summary-feature
  L2 logistic regression; summary-feature XGBoost; and raw-profile elastic-net SGD
  logistic regression. The ladder and hyperparameters were fixed before the final
  run; all models are reported rather than selecting one on test folds.
- **Controls.** `missingness_only` tests the continuous DepMap coverage shortcut.
  `cell_line_residual_pearson_abs` subtracts each cell line's median across all
  covered Feng candidates before correlation; because this uses the complete
  unlabeled candidate panel, it is reported separately as a transductive control
  and is not included in any supervised feature set.
- **Metrics.** Average precision (AP) and AUROC are primary. The repository's
  Feng-compatible `aupr` is also emitted, but it is trapezoidal PR-AUC and can be
  approximately 0.74 even for the low-resolution missingness control; it should
  not be interpreted against prevalence like AP. `f1_oracle` maximizes its
  threshold on the test fold and is reported separately.

CV2/CV3 mean genes unseen during Feng SL-pair/graph training. They do **not** mean
unseen to DepMap: the ground-truth auxiliary GeneEffect profiles for all mapped
genes are deliberately available here.

## Classification result

Mean +/- standard deviation across the five official folds is shown below. The
full-universe test sets are balanced at prevalence 0.50.

| Model | CV1 AP / AUROC | CV2 AP / AUROC | CV3 AP / AUROC |
| --- | --- | --- | --- |
| Missingness only | 0.512 +/- 0.001 / 0.522 +/- 0.002 | 0.510 +/- 0.001 / 0.520 +/- 0.003 | 0.511 +/- 0.003 / 0.522 +/- 0.006 |
| Pearson co-dependency | 0.540 +/- 0.007 / 0.530 +/- 0.008 | 0.539 +/- 0.010 / 0.529 +/- 0.009 | 0.539 +/- 0.031 / 0.532 +/- 0.031 |
| Spearman co-dependency | 0.544 +/- 0.006 / 0.535 +/- 0.006 | 0.543 +/- 0.009 / 0.534 +/- 0.009 | 0.541 +/- 0.026 / 0.534 +/- 0.023 |
| Summary logistic | 0.781 +/- 0.006 / 0.750 +/- 0.006 | 0.774 +/- 0.028 / 0.743 +/- 0.024 | **0.754 +/- 0.079 / 0.727 +/- 0.073** |
| Summary XGBoost | **0.832 +/- 0.005 / 0.800 +/- 0.004** | **0.792 +/- 0.015 / 0.763 +/- 0.013** | 0.729 +/- 0.043 / 0.714 +/- 0.050 |
| Raw-profile SGD | 0.733 +/- 0.093 / 0.759 +/- 0.086 | 0.693 +/- 0.061 / 0.682 +/- 0.073 | 0.564 +/- 0.071 / 0.530 +/- 0.097 |

The bold cells identify the largest mean AP within the frozen supervised ladder;
they are descriptive, not a post-hoc selected deployment model. The compact
summary models are much more stable than the 2,416-dimensional raw-profile model.

## Non-pan-essential slice

The primary confounding check excludes any pair containing a gene dependent in at
least half of DepMap lines. Its mean positive prevalence is 0.432 (CV1), 0.431
(CV2), and 0.422 (CV3), so AP must be compared with these slice-specific nulls.

| Model | CV1 AP / AUROC | CV2 AP / AUROC | CV3 AP / AUROC |
| --- | --- | --- | --- |
| Spearman co-dependency | 0.453 +/- 0.008 / 0.511 +/- 0.009 | 0.453 +/- 0.024 / 0.512 +/- 0.012 | 0.437 +/- 0.075 / 0.507 +/- 0.023 |
| Summary logistic | 0.668 +/- 0.003 / 0.682 +/- 0.005 | 0.654 +/- 0.049 / 0.673 +/- 0.028 | 0.590 +/- 0.149 / 0.643 +/- 0.093 |
| Summary XGBoost | 0.738 +/- 0.003 / 0.733 +/- 0.004 | 0.691 +/- 0.023 / 0.697 +/- 0.013 | 0.590 +/- 0.131 / 0.635 +/- 0.075 |
| Raw-profile SGD | 0.685 +/- 0.110 / 0.758 +/- 0.073 | 0.633 +/- 0.080 / 0.674 +/- 0.100 | 0.454 +/- 0.092 / 0.521 +/- 0.117 |

The summary models remain above the slice prevalence, but the reduction is large,
especially in CV3. This supports a candidate-prioritization use of marginal
essentiality profiles while showing that a material portion of the full-benchmark
signal is tied to marginal/pan-essentiality structure rather than pair-specific
interaction.

## Full-candidate ranking control

The three label-free correlation scores were also evaluated by scoring the full
9,845 x 9,845 candidate matrix and masking each official test fold. NDCG@10 is
near-null: Pearson is 0.0097 (CV1), 0.0122 (CV2), and 0.0082 (CV3); Spearman and
cell-line-residualized Pearson are essentially identical. Full-candidate ranking
was not run for the supervised summary/raw models because materializing and scoring
the 97-million-pair feature matrix exceeded the safe local resource envelope; their
result is classification-only.

## Interpretation and boundary

Ground-truth single-gene profiles can predict the balanced Feng labels surprisingly
well after supervised calibration, with summary XGBoost reaching AP 0.832/0.792 on
CV1/CV2 and summary logistic reaching AP 0.754 on CV3. However, the near-random
co-dependency ranking, the non-pan-essential degradation, and the absence of any
observed joint perturbation mean this is not evidence that two marginal effects
identify biological synthetic lethality. No function `psi(single_a, single_b)` can
recover an interaction residual without a joint outcome or pairwise supervision;
here the Feng labels provide that supervision and the resulting probabilities are
benchmark-calibrated candidate scores only. Rand negatives remain unconfirmed
non-SL pairs.

## Reproduction and artifacts

Run:

```bash
uv run python -m sl_profile_baseline \
  --config configs/experiments/11_feng_depmap_geneeffect_profiles/depmap_profile_cv.yaml
```

Configuration: `configs/experiments/11_feng_depmap_geneeffect_profiles/depmap_profile_cv.yaml`,
deleted at `873c99c` — check out `a7e2c91` to re-run.
Implementation: [`../../src/sl_profile_baseline/`](../../src/sl_profile_baseline/) (still present).
Gitignored outputs:
[`../../results/experiments/11_feng_depmap_geneeffect_profiles/run/`](../../results/experiments/11_feng_depmap_geneeffect_profiles/run/).
The manifest records the resolved configuration, DepMap and source SHA-256 hashes,
feature/probability semantics, and ranking boundary. Final integrity checks found
1,800 finite fold-metric rows, 360 summary rows, 45 slice-count rows, 9,845 gene
mapping rows, and exactly five folds per metric group. The adjacent benchmark and
new profile-baseline tests pass (20/20); ruff passes.
