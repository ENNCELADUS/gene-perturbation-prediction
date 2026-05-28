# Model Card: Replogle K562 Pseudobulk B->C Baseline and Adamson Transfer

Source experiment:
[`01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer.md`](../01_replogle_k562_pseudobulk_b_to_c_and_adamson_transfer.md)

## Purpose

This model card summarizes the pseudobulk model families used to predict
DepMap K562 CRISPR GeneEffect from observed Replogle K562 CRISPRi
perturbation responses, then transfer the fold checkpoints to Adamson K562
perturbation responses.

The common prediction unit is one perturbation-level response vector:

$$
x_g = \mu_g - \mu_{\mathrm{control}}
$$

Here, $g$ is the perturbed gene, $\mu_g$ is the pseudobulk mean expression
under perturbation $g$, and $\mu_{\mathrm{control}}$ is the control mean
expression in the same feature space. The supervised target is:

$$
f(x_g) \to y_g
$$

where $y_g$ is the population-level DepMap K562 GeneEffect label for gene
$g$. Unlike the single-cell experiment, there is no bag of cells at model time:
each perturbation is represented by one aggregate response vector.

## Model Families

### Mean-label baseline

The mean-label baseline ignores the transcriptome response and predicts the
training-label mean:

$$
\hat{y}_g =
\frac{1}{|\mathcal{G}_{\mathrm{train}}|}
\sum_{h \in \mathcal{G}_{\mathrm{train}}} y_h
$$

This is the no-signal floor. It checks whether the pipeline and metrics behave
sensibly, but it cannot rank target genes because every perturbation receives
the same prediction.

### Ridge on pseudobulk delta expression

Ridge uses the full pseudobulk delta vector directly:

$$
\hat{y}_g = x_g^T\beta + b
$$

The coefficients are fitted with squared loss and L2 shrinkage:

$$
\min_{\beta,b}
\sum_{g \in \mathcal{G}_{\mathrm{train}}}
\left(y_g - x_g^T\beta - b\right)^2
+ \alpha \lVert\beta\rVert_2^2
$$

The intuition is that every transcript feature can contribute to the
dependency prediction, but the L2 penalty prevents the high-dimensional linear
fit from depending too strongly on any single noisy coefficient. Full-feature
Ridge transferred by rank to Adamson but was poorly calibrated: the experiment
table reports Adamson Spearman `0.436` and RMSE `1.007`.

### PCA Ridge

PCA Ridge first compresses the perturbation response into a low-dimensional
linear coordinate system:

$$
z_g = W_k^T(x_g - \bar{x})
$$

Then it fits Ridge on the principal-component scores:

$$
\hat{y}_g = z_g^T\beta + b
$$

The fitting objective is the same Ridge objective, but with $z_g$ replacing
$x_g$. PCA removes noisy gene-level dimensions and lets the linear predictor
learn broad response axes rather than individual transcript features.

This was the best external-transfer family in the experiment.
`pca50_ridge_alpha100` reached Replogle CV Spearman `0.485` and Adamson
Spearman `0.500`; the target-heldout ensemble stayed similar at `0.490`. The
heldout result argues that the model is not only memorizing targets seen in
the Replogle folds.

### RandomForest on pseudobulk delta expression

RandomForest fits an ensemble of regression trees on the raw pseudobulk delta
features:

$$
\hat{y}_g =
\frac{1}{T}\sum_{t=1}^{T} T_t(x_g)
$$

Each tree $T_t$ partitions the feature space into nonlinear regions, and the
forest prediction is the average across trees. The motivation is to capture
nonlinear feature interactions without manually specifying response axes.

Direct RandomForest variants were part of the full model ladder, but the main
reported comparison is driven by the PCA version. The broader result suggests
that nonlinear flexibility can fit internal structure but does not guarantee
cross-dataset transfer.

### PCA RandomForest

PCA RandomForest applies the same tree ensemble to principal-component scores:

$$
z_g = W_k^T(x_g - \bar{x})
$$

$$
\hat{y}_g =
\frac{1}{T}\sum_{t=1}^{T} T_t(z_g)
$$

PCA reduces the transcriptome response to broad linear axes, while the forest
allows nonlinear splits across those axes. This was the best Replogle CV row
but did not transfer as well as PCA Ridge. `pca50_random_forest_leaf3` reached
Replogle CV Spearman `0.499` but only Adamson Spearman `0.385`, pointing to
internal-CV overfitting or dataset-specific nonlinear structure that does not
survive the Adamson shift.

### Response-burden Ridge

Response-burden models replace the full response vector with one or more
magnitude summaries. A simple form is:

$$
r_g = \lVert x_g\rVert
$$

or a vector of related response-size statistics:

$$
q_g = [r_{g1}, r_{g2}, \ldots, r_{gm}]
$$

Ridge is then fitted on those low-dimensional burden features:

$$
\hat{y}_g = q_g^T\beta + b
$$

The audit question is whether stronger dependency mostly appears as a larger
transcriptional disturbance. Response burden reached Replogle CV Spearman
`0.426`, which is substantial but below the best pseudobulk models. It
explains part of the signal but not all of it.

### Target-masked Ridge

Target-masked Ridge uses the same linear model as full Ridge, but first removes
features that directly correspond to the perturbed target gene:

$$
\tilde{x}_g = M_g x_g
$$

$$
\hat{y}_g = \tilde{x}_g^T\beta + b
$$

Here, $M_g$ is a masking operator that drops or zeros the target-gene feature
for perturbation $g$ when that feature is present. This tests whether the
model wins by reading direct target-gene expression rather than a broader
perturbation response. Target masking changed Ridge Spearman by only `-0.005`,
so direct target-expression leakage is not the main driver of the B->C signal.

### ElasticNet

ElasticNet is a sparse-shrinkage linear alternative to Ridge:

$$
\min_{\beta,b}
\sum_{g}
\left(y_g - x_g^T\beta - b\right)^2
+ \lambda_1\lVert\beta\rVert_1
+ \lambda_2\lVert\beta\rVert_2^2
$$

The L1 term encourages feature selection, while the L2 term stabilizes the fit
under correlated transcript features. It was included to test whether a smaller
selected gene set improves transfer, but it did not drive the reported
conclusion.

### XGBoost

XGBoost is a boosted-tree regressor:

$$
\hat{y}_g =
\sum_{m=1}^{M} \eta_m t_m(x_g)
$$

Each new tree $t_m$ is fitted to improve the current ensemble's residuals under
regularized boosting. The motivation is to test a stronger nonlinear model
than RandomForest. It was configured in the full ladder, but the source
experiment's transfer readout centers on PCA Ridge and PCA RandomForest.

### Strict MLP

The strict MLP is a small neural regressor over raw or PCA features:

$$
h_g = \phi_\theta(v_g)
$$

$$
\hat{y}_g = \rho_\theta(h_g)
$$

where $v_g$ is either $x_g$ or a compressed representation such as $z_g$. The
intended role is to test whether a learned nonlinear feature map improves over
linear or tree baselines under stricter validation. It did not drive the
documented Adamson transfer gate.

## Result Readout

The main working family is PCA Ridge: it is simple, compressed, and the most
robust Adamson transfer model in this experiment. PCA RandomForest is stronger
inside Replogle CV but weaker on Adamson, so internal CV alone is not
sufficient for selecting the bridge model. Response burden is meaningful, while
cell count and direct target-expression leakage do not explain the result.

| Model family | Input representation | What it tests | Main result |
| --- | --- | --- | --- |
| Mean-label baseline | No transcriptome input | No-signal floor | Sanity baseline only |
| Ridge | Full $x_g$ | Dense linear response-to-dependency map | Adamson Spearman `0.436` |
| PCA Ridge | PC scores $z_g$ | Broad linear response axes | Best Adamson transfer, `0.500` |
| PCA RandomForest | PC scores $z_g$ | Nonlinear splits over response axes | Best Replogle CV, weaker Adamson transfer |
| Response-burden Ridge | Magnitude summaries $q_g$ | Generic response-size signal | Replogle CV Spearman `0.426` |
| Target-masked Ridge | Masked response $\tilde{x}_g$ | Direct target-expression leakage | Ridge Spearman changed by only `-0.005` |

## Configured But Not Conclusion-Driving

- ElasticNet, XGBoost, and strict MLP were included to cover sparse linear,
  boosted-tree, and neural nonlinear alternatives.
- Their role is negative or supporting evidence: the report-facing conclusion
  remains that PCA Ridge gives the cleanest Adamson transfer in this matrix,
  while PCA RandomForest shows that stronger internal CV can fail to transfer.

## Caveats

Adamson has only `85` gene-level rows and is UPR-biased, so this is a
same-cell-line transfer check, not broad synthetic-lethality validation. The
matched key is still perturbation gene in K562; context specificity remains a
later-stage requirement.
