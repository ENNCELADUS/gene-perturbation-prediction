# Model Card: Replogle K562 Viability-Axis Audit

Source experiment:
[`02_replogle_k562_viability_axis_audit.md`](../02_replogle_k562_viability_axis_audit.md)

## Purpose

This model card summarizes the model families used to audit whether the
Replogle K562 B->C pseudobulk signal is mostly a generic viability,
cell-death/proliferation, or response-burden axis. The audit uses the same
perturbation-level pseudobulk response as Experiment 01, then adds external
NAR viability scores, response-burden features, and fold-local residualized
transcriptome features.

The common starting point is:

$$
x_g = \mu_g - \mu_{\mathrm{control}}
$$

where $x_g$ is the pseudobulk delta response for perturbation gene $g$. The
supervised target remains:

$$
f(\cdot) \to y_g
$$

where $y_g$ is the population-level DepMap K562 GeneEffect label. The goal is
not just to maximize prediction accuracy. The goal is to decompose what kind
of signal is being used: external viability-axis signal, generic response
burden, or residual transcriptomic structure.

## Model Families

### NAR viability score Ridge

The NAR score models project each perturbation response onto external Achilles
and CTRP cell-death/proliferation coefficient vectors from the NAR viability
signature resource:

$$
s_{gm} = x_g^T c_m + a_m
$$

Here, $m$ indexes an external signature model, $c_m$ is its gene-level
coefficient vector aligned to the experiment's transcript features, and $a_m$
is the exported intercept. Missing genes receive zero contribution after
alignment. The resulting low-dimensional score vector is:

$$
s_g = [s_{g1}, s_{g2}, \ldots, s_{gM}]
$$

Ridge is then fitted only on these viability-axis scores:

$$
\hat{y}_g = s_g^T\beta + b
$$

This asks whether the B->C task is mostly learning a known generic
cell-death/proliferation axis. NAR score only reached Spearman `0.244`, far
below the best pseudobulk baseline at `0.494`. The external viability axis is
real signal, but it does not explain most of the dependency-prediction signal.

### NAR plus response-burden Ridge

This model concatenates NAR viability scores with response-burden features:

$$
u_g = [s_g; q_g]
$$

where $q_g$ contains response-size summaries such as norms or related burden
statistics of $x_g$. Ridge is fitted on the combined low-dimensional feature
vector:

$$
\hat{y}_g = u_g^T\beta + b
$$

The audit question is whether a known viability axis plus overall response
magnitude is enough to approach the full transcriptome models. NAR plus burden
reached Spearman `0.443`, much stronger than NAR alone and close to the full
pseudobulk baseline. Response magnitude therefore carries a large fraction of
the B->C signal.

### Pseudobulk PCA RandomForest

The reference transcriptome model projects pseudobulk delta expression to PCA
space:

$$
z_g = W_k^T(x_g - \bar{x})
$$

and trains a RandomForest regressor:

$$
\hat{y}_g =
\frac{1}{T}\sum_{t=1}^{T} T_t(z_g)
$$

This model is the audit comparator: it can use broad transcriptomic structure
beyond the externally defined viability scores and response-burden summaries.
It reached Spearman `0.494` and AUROC `0.742`, remaining stronger than
NAR-only and slightly stronger than NAR plus burden.

### NAR-residualized PCA models

NAR-residualized models first remove the linear component of the transcriptome
that is explained by the NAR score matrix. Let $S$ be the fold-local matrix of
NAR score columns for training perturbations. For each transcript feature, fit:

$$
X = \mathbf{1}b^T + S\Gamma + R
$$

The residualized response is:

$$
\tilde{X}_{\mathrm{NAR}} = R
$$

For a perturbation $g$, the model uses the corresponding residual vector
$\tilde{x}_{g,\mathrm{NAR}}$, then applies PCA and Ridge or RandomForest:

$$
z_g = W_k^T(\tilde{x}_{g,\mathrm{NAR}} - \bar{\tilde{x}})
$$

$$
\hat{y}_g = h(z_g)
$$

where $h$ is Ridge or RandomForest. If the external viability axis explains the
important variation, this residualization should collapse performance. It did
not: NAR-residualized PCA RandomForest reached Spearman `0.503`, slightly
above the unresidualized baseline.

### Nuisance-residualized PCA models

Nuisance-residualized models use a broader nuisance matrix that includes NAR
scores and burden-like response summaries:

$$
N_g = [s_g; q_g]
$$

Across the training fold, each transcript feature is residualized against the
nuisance matrix:

$$
X = \mathbf{1}b^T + N\Gamma + R
$$

The residualized response is:

$$
\tilde{X}_{\mathrm{nuisance}} = R
$$

The model then applies PCA and Ridge or RandomForest to
$\tilde{x}_{g,\mathrm{nuisance}}$. This is a stricter audit because it removes
both known viability-score structure and generic response magnitude before
testing whether transcriptome residuals still predict GeneEffect.

Nuisance-residualized PCA RandomForest fell to Spearman `0.469`, below the
baseline but still meaningful. Burden is important, but the remaining residual
transcriptome still carries dependency signal.

### Residual PCs plus nuisance scores

This family combines both sides of the decomposition: residualized
transcriptome PCs and the nuisance scores themselves.

$$
z_g = W_k^T(\tilde{x}_{g,\mathrm{nuisance}} - \bar{\tilde{x}})
$$

$$
u_g = [z_g; s_g; q_g]
$$

Then Ridge or RandomForest predicts:

$$
\hat{y}_g = h(u_g)
$$

This tests whether the best representation is additive: generic nuisance axes
plus residual transcriptomic structure. Residual PCs plus nuisance scores
recovered near-baseline performance with Spearman `0.491`, supporting the
interpretation that both components matter.

### Sparse residualized Lasso

Sparse residualized Lasso fits an L1-penalized linear model on residualized
features:

$$
\hat{y}_g = \tilde{x}_g^T\beta + b
$$

with objective:

$$
\min_{\beta,b}
\sum_g
\left(y_g - \tilde{x}_g^T\beta - b\right)^2
+ \alpha \lVert\beta\rVert_1
$$

The intuition is that if only a small number of residual genes matter, Lasso
should retain them while discarding noise. Sparse residualized Lasso reached
only Spearman `0.113`, so the useful residual signal does not appear to be a
tiny sparse linear signature in this setup.

### Program-score variants

Program-score variants summarize each transcriptome response into predefined
or derived biological program scores:

$$
p_g = \psi(x_g)
$$

Then a supervised head predicts GeneEffect:

$$
\hat{y}_g = h(p_g)
$$

where $h$ can be Ridge, ElasticNet, or RandomForest. Program scores are an
intermediate representation between raw transcriptome features and generic
viability/burden scores. They were part of the signal-decomposition grid, but
the documented conclusion is driven by the NAR, burden, residualized PCA, and
residual-plus-score comparisons.

## Result Readout

The viability-axis audit argues against a pure generic-viability explanation.
NAR scores alone are weak, response burden is strong, and fold-local
residualized transcriptome models remain predictive. The best interpretation
is that the B->C signal combines generic response burden with residual
transcriptomic structure.

| Model family | Representation | What it tests | Main result |
| --- | --- | --- | --- |
| NAR score Ridge | External viability scores $s_g$ | Known cell-death/proliferation axis alone | Spearman `0.244` |
| NAR plus burden Ridge | Scores and burden $[s_g; q_g]$ | Viability axis plus response magnitude | Spearman `0.443` |
| Pseudobulk PCA RandomForest | PCA of full $x_g$ | Full transcriptome comparator | Spearman `0.494` |
| NAR-residualized PCA | PCA of $\tilde{x}_{g,\mathrm{NAR}}$ | Signal after removing NAR axis | Spearman `0.503` |
| Nuisance-residualized PCA | PCA of $\tilde{x}_{g,\mathrm{nuisance}}$ | Signal after removing NAR plus burden | Spearman `0.469` |
| Residual PCs plus scores | $[z_g; s_g; q_g]$ | Additive nuisance plus residual structure | Spearman `0.491` |
| Sparse residualized Lasso | Sparse $\tilde{x}_g$ model | Tiny sparse residual signature | Spearman `0.113` |

## Configured But Not Conclusion-Driving

- Mean-label, Ridge, ElasticNet, PCA Ridge, and full RandomForest families were
  present as baseline infrastructure inherited from the broader pseudobulk
  package.
- The signal-decomposition config also included program-score Ridge,
  ElasticNet, and RandomForest variants. They provide coverage for alternative
  low-dimensional summaries but do not change the main audit conclusion.

## Caveats

The audit is fold-local and K562-specific. It can rule out a simple
known-viability-only explanation for this experiment, but it does not prove
that the residual transcriptomic signal is causal or synthetic-lethal.
