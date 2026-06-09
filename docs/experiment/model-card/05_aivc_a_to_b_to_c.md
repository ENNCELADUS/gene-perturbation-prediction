# Model Card: AIVC A->B->C Forward Modules for K562 Dependency Ranking

Source experiment:
[`05_aivc_a_to_b_to_c.md`](docs/experiment/05_aivc_a_to_b_to_c.md)

## Purpose

This model card defines the AIVC-style A->B->C formulation for predicting
DepMap K562 CRISPR GeneEffect from predicted post-perturbation single-cell
response distributions. It is a model definition for the next forward-model
experiment, not an implementation report.

The experiment compares two primary A->B forward-module families:

1. State-style State Transition (`ST`).
2. Tahoe-x1-derived representations with a comparable ST adapter.

Both families share the same downstream interface:

```text
K562 control scVI latent bag + perturbation gene representation
    -> A->B forward module
    -> predicted post-perturbation scVI latent bag B_hat
    -> GMM distribution features
    -> MLP B_hat->C head
    -> DepMap K562 GeneEffect
```

The purpose is not to show that a virtual-cell model can reconstruct expression
in isolation. The purpose is to test whether the predicted response distribution
preserves enough dependency-relevant signal for target-gene ranking.

## Prediction Unit

The supervised unit is one held-out K562 perturbation gene. The train,
validation, and test partitions are disjoint sets of perturbation genes:

$$
\mathcal{G}_{\mathrm{train}}
\cap
\mathcal{G}_{\mathrm{val}}
= \emptyset,
\quad
\mathcal{G}_{\mathrm{train}}
\cap
\mathcal{G}_{\mathrm{test}}
= \emptyset,
\quad
\mathcal{G}_{\mathrm{val}}
\cap
\mathcal{G}_{\mathrm{test}}
= \emptyset
$$

There is no random cell-level split in this formulation. Cells are grouped by
perturbation gene so that a test gene's observed response distribution is not
seen during training or test-time inference.

Let the K562 control-cell latent bag be:

$$
Z^0 =
\{z^0_1, z^0_2, \ldots, z^0_{n_0}\}
$$

where each $z^0_i$ is a frozen scVI128 latent vector for a control cell. For
perturbation gene $g$, let $r_g$ be the perturbation-gene representation used by
the A->B module. The upstream input is:

$$
A_g = (Z^0, r_g)
$$

The observed post-perturbation latent response bag is:

$$
B^z_g =
\{z^1_{g1}, z^1_{g2}, \ldots, z^1_{gn_g}\}
$$

The forward module predicts:

$$
F_\theta(A_g) \to \hat{B}^z_g
$$

where:

$$
\hat{B}^z_g =
\{\hat{z}^1_{g1}, \hat{z}^1_{g2}, \ldots, \hat{z}^1_{gm_g}\}
$$

The supervised dependency target is:

$$
h_\omega(\phi_B(\hat{B}^z_g)) \to y_g
$$

where $y_g$ is the DepMap K562 GeneEffect label for perturbation gene $g$,
$\phi_B$ is the shared GMM distribution featureizer, and $h_\omega$ is the MLP
dependency head.

## Model Families

### State-style ST A->B module

The State-style ST family treats A->B as a set transition from the K562 control
latent bag to a predicted perturbed latent bag. It receives the control bag
$Z^0$ and a perturbation representation $r_g$:

$$
\hat{B}^z_g =
F^{\mathrm{StateST}}_\theta(Z^0, r_g)
$$

A generic cell-level transition form is:

$$
u_{gi} =
\mathrm{SetTransition}_\theta(z^0_i, r_g, c_{\mathrm{K562}})
$$

$$
\hat{z}^1_{gi} =
z^0_i + \Delta_\theta(u_{gi})
$$

where $c_{\mathrm{K562}}$ is the K562 context representation. In the current
single-context setup this context may be constant, but the symbol is kept so the
model definition remains compatible with future multi-context extension.

The important modeling choice is that the output remains a bag of predicted
scVI128 latent cells, not a pseudobulk mean and not a direct GeneEffect
prediction. This lets the same GMM distribution featureizer summarize both
observed and predicted response bags.

### Tahoe-x1 representation plus comparable ST adapter

The Tahoe-x1 family uses Tahoe-x1-derived representations as the foundation-model
input side, then maps them through a comparable ST adapter that also predicts a
post-perturbation scVI latent bag:

$$
e_g^{\mathrm{Tx1}} =
E_{\mathrm{Tx1}}(g)
$$

$$
e_0^{\mathrm{Tx1}} =
E_{\mathrm{Tx1}}(Z^0)
$$

$$
\hat{B}^z_g =
F^{\mathrm{Tx1ST}}_\theta(Z^0, e_0^{\mathrm{Tx1}}, e_g^{\mathrm{Tx1}})
$$

This makes Tahoe-x1 a directly comparable A->B experiment object rather than a
separate direct-C predictor. The Tx1-derived branch can provide gene, cell, or
context representations, but its output interface is still $\hat{B}^z_g$.

Keeping the output in scVI latent space is intentional. It lets the State-style
and Tahoe-x1 branches use the same downstream GMM and MLP dependency head, so
the comparison focuses on whether each forward module produces a useful
post-perturbation response distribution.

### Shared GMM featureizer and MLP B_hat->C head

Both A->B modules feed the same B->C model. First, a GMM summarizes each latent
bag as a distribution over shared cell-state prototypes:

$$
p(z) =
\sum_{k=1}^{K}\pi_k
\mathcal{N}(z;\mu_k,\Sigma_k)
$$

For a predicted cell $\hat{z}^1_{gi}$, the soft assignment to prototype $k$ is:

$$
\hat{r}_{gik} =
\frac{
\pi_k\mathcal{N}(\hat{z}^1_{gi};\mu_k,\Sigma_k)
}{
\sum_{\ell=1}^{K}
\pi_\ell\mathcal{N}(\hat{z}^1_{gi};\mu_\ell,\Sigma_\ell)
}
$$

The predicted occupancy vector is:

$$
\hat{p}_{gk} =
\frac{1}{m_g}
\sum_{i=1}^{m_g}\hat{r}_{gik}
$$

or:

$$
\hat{p}_g =
[\hat{p}_{g1}, \hat{p}_{g2}, \ldots, \hat{p}_{gK}]
$$

The C-side feature vector is built from distribution features of
$\hat{B}^z_g$, such as occupancy, delta occupancy relative to controls, latent
mean, latent variance, entropy, and related bag-level summaries:

$$
\phi_B(\hat{B}^z_g) =
[
\hat{p}_g;
\hat{p}_g - p^0;
\bar{\hat{z}}^1_g;
\mathrm{Var}(\hat{B}^z_g);
H(\hat{p}_g)
]
$$

The MLP dependency head predicts GeneEffect:

$$
\hat{y}^{\mathrm{pred}}_g =
h_\omega(\phi_B(\hat{B}^z_g))
$$

The C head does not directly consume gene identity or the perturbation-gene
embedding. That restriction is deliberate: it reduces the chance that the C
model bypasses the predicted response distribution and learns a target-gene
prior instead.

### Observed-B anchor path

Observed post-perturbation bags are used during training as supervision and as
an upper-bound anchor. For train and validation genes, the observed path is:

$$
\hat{y}^{\mathrm{obs}}_g =
h_\omega(\phi_B(B^z_g))
$$

This path can stabilize the C-side representation and quantify the gap between
observed-B->C and predicted-B->C. It does not make test-gene observed bags
available at inference time.

For test genes, observed $B^z_g$ is only used after prediction for offline A->B
error analysis, such as latent reconstruction error, distribution distance, or
GMM occupancy error. It is not used to train the A->B module, fit the GMM, train
the MLP C head, or produce the test-time GeneEffect prediction.

## Training and Evaluation Readout

The recommended training objective combines A->B response supervision with C-side
dependency supervision:

$$
\mathcal{L} =
\lambda_{\mathrm{dist}}\mathcal{L}_{\mathrm{dist}}
+
\lambda_{\Delta}\mathcal{L}_{\Delta}
+
\lambda_{\mathrm{occ}}\mathcal{L}_{\mathrm{occ}}
+
\alpha_{\mathrm{pred}}\mathcal{L}^{\mathrm{pred}}_C
+
\alpha_{\mathrm{obs}}\mathcal{L}^{\mathrm{obs}}_C
$$

where the A->B terms may include latent distribution distance, mean-delta error,
and occupancy error:

$$
\mathcal{L}_{\Delta} =
\left\|
(\bar{\hat{z}}^1_g - \bar{z}^0)
-
(\bar{z}^1_g - \bar{z}^0)
\right\|_2^2
$$

$$
\mathcal{L}_{\mathrm{occ}} =
D(\hat{p}_g, p_g)
$$

The C-side losses compare predicted GeneEffect with the DepMap label:

$$
\mathcal{L}^{\mathrm{pred}}_C =
\ell(\hat{y}^{\mathrm{pred}}_g, y_g)
$$

$$
\mathcal{L}^{\mathrm{obs}}_C =
\ell(\hat{y}^{\mathrm{obs}}_g, y_g)
$$

The readout should separate forward-model quality from downstream utility:

| Readout family | Example metrics | What it tests |
| --- | --- | --- |
| A->B latent reconstruction | mean latent delta RMSE or MAE | Whether predicted cells move in the right latent direction |
| A->B distribution reconstruction | MMD, energy distance, or related set distance | Whether the predicted bag matches the observed bag distribution |
| GMM occupancy reconstruction | occupancy RMSE, KL, or Jensen-Shannon distance | Whether predicted bags preserve prototype-level response structure |
| C ranking utility | Spearman, top-k overlap or stability | Whether predicted responses preserve dependency ranking |
| C threshold utility | AUROC or AUPRC for dependency thresholds | Whether predicted responses support dependency classification |
| Upper-bound gap | predicted-B->C versus observed-B->C | How much downstream signal is lost through A->B prediction |

The primary success criterion is not best expression reconstruction alone. A
useful 05 model should narrow the gap between predicted-B->C and observed-B->C
while staying ahead of simple forward-model comparators.

## Configured/Future Comparators

The primary model-card definitions above are State-style ST and Tahoe-x1 plus a
comparable ST adapter. The broader model ladder should remain visible as
comparators without being treated as primary 05 model families.

| Comparator | Role in 05 | Expected interface | Main caution |
| --- | --- | --- | --- |
| No-change/control | A->B floor | Reuse or lightly perturb $Z^0$ as $\hat{B}^z_g$ | Cannot model perturbation-specific response |
| Linear A->B baseline | Conservative forward baseline | Predict scVI latent or expression delta before GMM/MLP | Useful minimum comparator before FM claims |
| Observed-B upper bound | C-side anchor | Use $\phi_B(B^z_g)$ instead of $\phi_B(\hat{B}^z_g)$ | Not a deployable test-time predictor |
| GEARS | Classical perturbation-response baseline | Generate or embed a predicted response before shared B->C | Usually expression-centric rather than scVI-distribution-centric |
| scGPT-style model | Single-cell FM comparator | Use embeddings or predicted response through adapter | Must beat simple controls, not just internal FM baselines |
| X-Cell | High-capacity future teacher/comparator | Map generated response to scVI latent distribution | Potential prior leakage and heavier implementation cost |
| Stack | Future in-context comparator | Adapt in-context response prediction to $\hat{B}^z_g$ | Not part of the primary two-family 05 definition |

## Caveats

Tahoe-x1 has a specific interpretation risk because the reported model family is
connected to DepMap essentiality-style benchmarks. In this 05 formulation, that
is treated as a caveat rather than a hard exclusion: any Tx1 component used for
the main result should be audited for whether it was supervised on DepMap
GeneEffect or closely related essentiality labels.

The endpoint is DepMap K562 GeneEffect. It is a population-level dependency
score, not a single-cell death label and not a synthetic-lethality label. This
model card should therefore use dependency-ranking language, not SL-target
claims.

This card defines the intended model formulation and evaluation boundary. It
does not claim that the State-style or Tahoe-x1 A->B modules have been
implemented, trained, or benchmarked in this repository.

The train/validation/test split is by perturbation gene. Test-gene observed
post-perturbation bags may be used only for final offline A->B diagnostics, not
for fitting the forward module, fitting GMM prototypes, training the MLP C head,
or making test-time predictions.
