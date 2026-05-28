# Model Card: Replogle K562 Single-Cell MIL and Adamson Transfer

Source experiment:
[`03_replogle_k562_single_cell_deepsets_adamson.md`](../03_replogle_k562_single_cell_deepsets_adamson.md)

## Purpose

This model card summarizes the single-cell model families used to predict
DepMap K562 CRISPR GeneEffect from observed Replogle K562 perturbation bags and
transfer fold checkpoints to Adamson K562 perturbation bags.

The common prediction unit is a perturbation-level bag:

$$
B_g = \{z_{g1}, z_{g2}, \ldots, z_{gn_g}\}
$$

Here, $B_g$ is the bag of surviving cells observed under perturbation gene
$g$, $z_{gi}$ is the feature vector for cell $i$, and $n_g$ is the number of
sampled cells for that perturbation. The supervised target is:

$$
f(B_g) \to y_g
$$

where $y_g$ is the population-level DepMap GeneEffect label for gene $g$.
Cells do not have labels in this setup; only the bag has a label.

## Model Families

### Feature encoders: PCA128, scVI128, and HVG expression

The feature encoder is not the final dependency predictor. It only maps each
cell's measured expression response into a feature vector:

$$
z_{gi} = e(x_{gi})
$$

The downstream Deep Sets, attention MIL, GMM regression, and CloudPred-like
models then operate on the bag $B_g$.

PCA128 is a linear projection of centered expression or delta-expression
features onto the top training-set principal components:

$$
z_i = W_{128}^T(x_i - \mu)
$$

This gives a compact linear geometry. It is useful as a stable baseline, but it
can discard nonlinear or count-model structure in single-cell expression.

scVI128 uses the posterior latent mean from a variational single-cell model:

$$
z_i = \mathbb{E}_{q_\theta(u \mid x_i)}[u]
$$

This gives a denoised nonlinear latent representation. In this experiment
matrix, scVI128 is the strongest feature space for distribution-regression
models.

HVG expression keeps the selected highly variable genes directly, optionally as
delta expression against the control reference:

$$
z_i = x_{i,\mathrm{HVG}} - \mu_{\mathrm{control},\mathrm{HVG}}
$$

This preserves more raw gene-level resolution than PCA or scVI, but it is also
higher-dimensional and noisier. In the observed runs, HVG2000 underperformed
and caused GPU OOM for the CloudPred-like branch.

### Deep Sets mean pooling

Deep Sets is the simplest permutation-invariant set model. Each cell feature is
encoded by a shared cell network:

$$
h_{gi} = \phi(z_{gi})
$$

The bag representation is the average cell embedding:

$$
b_g = \frac{1}{n_g}\sum_{i=1}^{n_g} h_{gi}
$$

The bag-level head predicts GeneEffect:

$$
\hat{y}_g = \rho(b_g)
$$

Equivalently:

$$
\hat{y}_g =
\rho\left(
\frac{1}{n_g}\sum_{i=1}^{n_g}\phi(z_{gi})
\right)
$$

The modeling assumption is that the average learned cell-state response is
sufficient to predict the population-level dependency label. This is simple,
stable, and naturally supports variable-size bags, but it can lose
subpopulation structure. In the results, Deep Sets is a strong baseline:
PCA128 mean pooling reached Adamson Spearman `0.504`, and scVI128 mean pooling
reached `0.545`.

### Single-head gated attention MIL

Single-head gated attention MIL replaces uniform averaging with a learned
cell weight. It first computes shared cell embeddings:

$$
h_{gi} = \phi(z_{gi})
$$

Then it computes a gated attention logit:

$$
e_{gi} =
w^T\left[
\tanh(Vh_{gi}) \odot \sigma(Uh_{gi})
\right]
$$

The logits are normalized within the same perturbation bag:

$$
a_{gi} =
\frac{\exp(e_{gi})}
{\sum_{j=1}^{n_g}\exp(e_{gj})}
$$

The bag embedding is a weighted sum:

$$
b_g = \sum_{i=1}^{n_g} a_{gi}h_{gi}
$$

The final prediction is:

$$
\hat{y}_g =
\rho\left(
\sum_{i=1}^{n_g}a_{gi}\phi(z_{gi})
\right)
$$

The intuition is that if only some surviving cells carry
dependency-relevant signal, attention can assign larger weights to those cells.
Empirically, attention did not consistently beat mean pooling. scVI128 gated
attention improved Adamson Spearman slightly over scVI mean pooling,
`0.552` versus `0.545`, but PCA and HVG attention were weaker or less stable.
Attention weights should be read as diagnostics of prediction behavior, not as
causal attribution.

### Multi-head gated attention MIL

Multi-head gated attention MIL extends the single-head model by learning
multiple attention maps over the same bag:

$$
a_{gi}^{(1)}, a_{gi}^{(2)}, \ldots, a_{gi}^{(H)}
$$

Each head produces its own pooled bag embedding:

$$
b_g^{(k)} = \sum_{i=1}^{n_g} a_{gi}^{(k)}h_{gi}
$$

The head-specific embeddings are concatenated:

$$
b_g =
\left[
b_g^{(1)};
b_g^{(2)};
\ldots;
b_g^{(H)}
\right]
$$

The prediction head then maps the concatenated embedding to GeneEffect:

$$
\hat{y}_g = \rho(b_g)
$$

The configured version also penalizes head similarity with an off-diagonal
cosine-squared regularizer, so different heads are encouraged to attend to
different response modes. The intended use case is a bag containing multiple
cell-state modes, such as stress-like cells, cycling cells, escape-like cells,
or dying-like survivors.

The v1 result did not improve transfer. The strongest multi-head row,
scVI128, reached Adamson Spearman `0.541`, below single-head scVI attention
at `0.552` and below scVI mean pooling.

### Frozen GMM prototype regression

Frozen GMM prototype regression explicitly models each bag as a distribution
over shared cell-state prototypes. First, a fold-local diagonal Gaussian
mixture is fitted on training cells and controls:

$$
p(z) =
\sum_{k=1}^{K}\pi_k\mathcal{N}(z;\mu_k,\Sigma_k)
$$

Each Gaussian component is a latent cell-state prototype. For each cell in a
perturbation bag, the model computes posterior responsibility for prototype
$k$:

$$
r_{gik} =
p(k \mid z_{gi}) =
\frac{
\pi_k\mathcal{N}(z_{gi};\mu_k,\Sigma_k)
}{
\sum_{\ell=1}^{K}
\pi_\ell\mathcal{N}(z_{gi};\mu_\ell,\Sigma_\ell)
}
$$

The bag is summarized by prototype occupancy:

$$
p_{gk} =
\frac{1}{n_g}\sum_{i=1}^{n_g} r_{gik}
$$

or as a vector:

$$
p_g = [p_{g1}, p_{g2}, \ldots, p_{gK}]
$$

For the `deltap` view, the model subtracts the control occupancy:

$$
\Delta p_g = p_g - p_{\mathrm{control}}
$$

The supervised head then predicts GeneEffect from occupancy features and
optional distribution statistics:

$$
\hat{y}_g = \beta^T u_g + b
$$

$$
u_g = [p_g; \Delta p_g; s_g]
$$

Here, $s_g$ can include entropy, effective number of components,
assignment-confidence summaries, or negative log-likelihood summaries. The
head can be Ridge, RandomForest, or MLP; the strongest current rows use Ridge.

This family asks a different question from pooling models: instead of learning
a single pooled embedding, it asks how the perturbation shifts the surviving
cell population across shared cell-state prototypes. That distributional view
matches the population-level GeneEffect label better in the observed runs. The
best scVI128 frozen GMM Ridge rows reached Adamson Spearman about `0.665` to
`0.666`, with heldout Spearman up to `0.669`, clearly above the scVI attention
rows.

### CloudPred-like trainable distribution regressor

The CloudPred-like model keeps the distribution-regression structure but makes
the prototype assignment trainable. It initializes Gaussian prototypes from GMM
parameters:

$$
\{\mu_k,\Sigma_k\}_{k=1}^{K}
$$

Then it learns soft assignments from cells to prototypes:

$$
r_{gik} =
\mathrm{softassign}(z_{gi}, \mu_k, \Sigma_k)
$$

The bag-level occupancy is:

$$
p_{gk} =
\frac{1}{n_g}\sum_{i=1}^{n_g} r_{gik}
$$

The final head is an end-to-end trainable MLP:

$$
\hat{y}_g = \mathrm{MLP}(p_g)
$$

The intended advantage is flexibility: prototypes and the supervised mapping
can adapt to the GeneEffect objective instead of staying fixed after GMM
fitting. In these runs, that extra flexibility did not beat the simpler frozen
GMM plus Ridge approach. The best scVI CloudPred-like row reached Adamson
Spearman `0.602` and heldout Spearman `0.593`, better than attention but below
the frozen scVI GMM Ridge rows.

## Result Readout

The best current single-cell method is scVI128 frozen GMM prototype regression
with a Ridge head. It is the first single-cell family in this experiment to
clearly beat the earlier Adamson transfer gate. Mean pooling remains a strong
floor; attention and multi-head attention do not reliably improve transfer; HVG
features do not help in this configuration.

| Model family | Bag representation | What it learns | Main strength | Main limitation |
| --- | --- | --- | --- | --- |
| Deep Sets | Mean pooled learned embedding | Average response | Stable strong baseline | Loses subpopulation structure |
| Attention MIL | Weighted cell embedding | Important cells | Can focus on rare signal-bearing cells | Attention is unstable and not causal attribution |
| Multi-head attention MIL | Multiple weighted embeddings | Multiple response modes | Can represent several cell-state modes | Did not improve transfer in v1 |
| Frozen GMM regression | Prototype occupancy or delta occupancy | Response distribution shift | Best current transfer; aligns with population-level label | Prototypes are not causal cell states |
| CloudPred-like | Trainable prototype occupancy | Task-adapted distribution | More flexible than frozen prototypes | Underperformed frozen GMM Ridge in this matrix |

In short: Deep Sets asks whether the average cell response predicts
GeneEffect; attention MIL asks which cells should matter more; GMM regression
asks which shared cell-state prototypes the surviving-cell population moved
toward. For this task, the third question is currently the closest match to the
population-level label.

## Configured But Not Conclusion-Driving

- Frozen GMM RandomForest heads were included as nonlinear supervised heads, but
  the strongest and most stable rows came from Ridge heads.
- Frozen GMM MLP heads were configured for PCA/scVI tokens at primary K values,
  but they did not drive the documented result.
- Adamson validation sweep variants over K, view, and Ridge alpha checked
  stability. The sweep reproduced the distribution-regression gain but did not
  clear the more aggressive primary sweep target.

## Caveats

The Adamson validation sweep is Adamson-guided and should not be treated as an
untouched external-test claim. Labels remain population-level DepMap
GeneEffect, not single-cell death labels. Attention and prototype assignments
are useful diagnostics for prediction behavior but are not causal cell-state
attributions.
