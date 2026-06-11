# Model Card: AIVC A->B->C STATE Pipeline for K562 Dependency Ranking

Source experiment:
[`05_aivc_a_to_b_to_c.md`](../05_aivc_a_to_b_to_c.md)

## Purpose

This model card describes the implemented AIVC-style A->B->C pipeline in
`src/aivc_model` for predicting DepMap K562 CRISPR GeneEffect from predicted
post-perturbation single-cell response distributions.

The implemented 05 surface currently covers one A->B family: a STATE/ST-style
adapter loaded from the ST-HVG-Replogle checkpoint. Tahoe-x1 is not implemented
in `src/aivc_model`; it remains a future comparator rather than a primary
implemented model family.

The implemented pipeline is:

```text
control expression/HVG cell-set chunks + perturbation vector
    -> STATE/ST forward module
    -> predicted expression/HVG response
    -> expression-to-scVI latent projector
    -> predicted post-perturbation latent bag B_hat
    -> fixed GMM distribution features
    -> MLP B_hat->C head
    -> DepMap K562 GeneEffect
```

The purpose is not to show that STATE reconstructs expression in isolation. The
purpose is to test whether a predicted response distribution preserves enough
dependency-relevant signal for perturbation-gene ranking.

## Prediction Unit

The supervised unit is one K562 perturbation gene. The primary training split is
by matched Replogle K562 perturbation gene. The current configs use Replogle
`train_fraction: 0.9`, `val_fraction: 0.1`, and internal `test_fraction: 0.0`.
Adamson K562 is configured as external evaluation.

The current external loader evaluates all matched Adamson genes and does not
filter genes that overlap Replogle train or validation genes. Therefore current
Adamson readouts should be described as external assay-transfer readouts, not as
guaranteed held-out perturbation-gene generalization.

For perturbation gene $g$, the implemented input is:

$$
A_g = (X^0_{S_g}, r_g)
$$

where $X^0_{S_g}$ is a sampled control expression/HVG cell-set chunk and $r_g$
is the STATE perturbation vector. The observed post-perturbation response bag is:

$$
B_g = \{x^1_{g1}, x^1_{g2}, \ldots, x^1_{gn_g}\}
$$

During A->B training, the sampled control chunk is intentionally matched to the
target response chunk's optional batch labels. `make_cell_set_chunks` records
those target batch labels, samples matching control cells when control batch
annotations exist, and passes the corresponding batch labels through the
STATE/ST adapter. The target response chunk is used here as set-level A->B
supervision.

The STATE/ST adapter predicts an expression/HVG response:

$$
F_\theta(A_g) \to \hat{X}^1_g
$$

The learned linear projector maps this output into the scVI latent space:

$$
P_\psi(\hat{X}^1_g) \to \hat{B}^z_g
$$

The C-side head predicts the dependency label:

$$
\hat{y}^{\mathrm{pred}}_g =
h_\omega(\phi_B(\hat{B}^z_g))
$$

where $y_g$ is the DepMap K562 GeneEffect label, $\phi_B$ is the fixed GMM
distribution featureizer, and $h_\omega$ is the MLP dependency head.

Epoch validation and final internal/external test evaluation are
prediction-only. They compute `y_pred` from all available control cells plus
perturbation identity, using one same-gene STATE call per evaluated perturbation
gene. They do not use the evaluated gene's observed response bag, target cell
count, or target batch labels. `models/best/` is selected by `val_spearman` from
this prediction-only validation path.

## Implemented Model Components

### STATE/ST A->B adapter

The `StateForwardAdapter` wraps an ArcInstitute STATE transition model. It
passes control cells, a perturbation vector, perturbation names, and optional
batch labels into the checkpoint's `predict_step` or forward method. The output
is treated as predicted expression/HVG cell-level response.

The ranknet/freeze-state config freezes this adapter and trains downstream
parameters. The default config leaves the adapter trainable.

### Perturbation representation

Known perturbation vectors are loaded from the configured ST-HVG-Replogle
perturbation map. Genes without a known vector receive trainable
mean-initialized missing-vector parameters. This allows external Adamson genes
to enter the same model object, but `test_predictions.csv` records whether the
perturbation had a known vector.

### Expression-to-latent projector

The projector is a ridge-initialized linear map from STATE expression/HVG output
to scVI latent space. It is fit on Replogle train genes plus controls, then used
inside the differentiable model. It can remain trainable according to config.

### Fixed GMM featureizer

The GMM is fit on train-gene scVI latent bags plus controls. It is fixed during
AIVC training and summarizes predicted or observed latent bags through occupancy,
delta occupancy, latent mean, latent variance, and entropy.

The prototype density is:

$$
p(z) =
\sum_{k=1}^{K}\pi_k
\mathcal{N}(z;\mu_k,\Sigma_k)
$$

For predicted cell $\hat{z}^1_{gi}$, the posterior responsibility for component
$k$ is:

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

The implemented feature vector is:

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

where $p^0$ is the control occupancy vector.

### MLP C head and observed-B anchor

The MLP C head predicts GeneEffect from GMM distribution features. It does not
directly consume gene identity or the perturbation vector.

The same C head is also evaluated on observed post-perturbation latent bags:

$$
\hat{y}^{\mathrm{obs}}_g =
h_\omega(\phi_B(B^z_g))
$$

This observed-B anchor is a supervised training diagnostic for measuring the gap
between observed-B->C and predicted-B->C during development. It is not used for
validation checkpoint selection, is not a deployable test-time predictor, and is
omitted from validation and final test artifacts.

## Objective

For each training perturbation gene, the implemented loss combines set-level A->B
supervision with C-side dependency supervision:

$$
\mathcal{L} =
\lambda_{\mathrm{hvg},\Delta}\mathcal{L}_{\mathrm{hvg},\Delta}
+
\lambda_{\mathrm{hvg},E}\mathcal{L}_{\mathrm{hvg},E}
+
\lambda_{z,\Delta}\mathcal{L}_{z,\Delta}
+
\lambda_{z,E}\mathcal{L}_{z,E}
+
\lambda_{\mathrm{occ}}\mathcal{L}_{\mathrm{occ}}
+
\alpha_{\mathrm{pred}}\mathcal{L}^{\mathrm{pred}}_C
+
\alpha_{\mathrm{obs}}\mathcal{L}^{\mathrm{obs}}_C
+
\alpha_{\mathrm{rank}}\mathcal{L}_{\mathrm{rank}}
$$

The mean-delta terms compare perturbation-induced shifts against the control
mean. For latent bags:

$$
\mathcal{L}_{z,\Delta} =
\left\|
(\bar{\hat{z}}^1_g - \bar{z}^0)
-
(\bar{z}^1_g - \bar{z}^0)
\right\|_2^2
$$

The energy terms are set distances between predicted and observed cell sets.
The occupancy term compares predicted and observed GMM occupancy vectors:

$$
\mathcal{L}_{\mathrm{occ}} =
\left\|\hat{p}_g - p_g\right\|_2^2
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

The optional RankNet term ranks predicted GeneEffect values within the local
gene batch.

## Training and DDP Pipeline

The training entry point is:

```bash
uv run python src/aivc_model/train.py --config <config.yaml>
```

The Slurm entry point is `scripts/state.sh`, which launches the same script
through Accelerate with four processes and bf16 mixed precision.

Accelerate DDP wraps the model and optimizer, pads per-rank gene loaders to keep
step counts synchronized, enables `find_unused_parameters=True` for sparse
per-gene perturbation-vector parameters, gathers fixed-shape tensors for metrics
and predictions, and restricts artifact/checkpoint writes to rank0.

When `projector.teacher: scvi`, rank0 materializes the scVI teacher latent cache
before the main epoch loop. The cache subprocess removes distributed environment
variables and narrows `CUDA_VISIBLE_DEVICES` to one GPU. Other ranks wait for a
validated cache before entering DDP training. Cache validation includes train
genes, primary/external dataset shapes, feature names, latent dimension, seed,
and scVI teacher settings.

## Readout Contract

Current outputs include:

| Readout | Current artifact |
| --- | --- |
| Train diagnostics and prediction-only validation metrics | `train_log.csv` |
| Final predicted-B->C regression/ranking metrics | `test_metrics.csv` |
| Per-gene prediction-only outputs | `artifacts/test_predictions.csv` |
| Split membership | `artifacts/gene_splits.csv` |
| External source QA | `artifacts/external_test_qa.json` |
| Run-local projector/GMM fit caches | `artifacts/ridge_projector_fit/`, `artifacts/fixed_gmm_fit/` |
| Checkpoints | `models/best/`, `models/final/` |

The intended final readout should also separate forward-model quality from
downstream utility:

| Readout family | Intended metric examples |
| --- | --- |
| A->B mean reconstruction | HVG and latent mean-delta error |
| A->B distribution reconstruction | energy distance or related set distance |
| GMM occupancy reconstruction | occupancy RMSE, KL, or Jensen-Shannon distance |
| C ranking utility | Spearman, top-k overlap, rank stability |
| C threshold utility | AUROC or AUPRC for dependency thresholds |
| Upper-bound gap | predicted-B->C versus observed-B->C |

The current final artifact schema does not yet export occupancy-specific
diagnostic artifacts, top-k overlap, or fold-local summaries.

## Current Implementation Deltas

- The implemented model family is STATE/ST only. Tahoe-x1 remains a future
  comparator.
- Adamson external evaluation is not filtered to genes absent from Replogle
  train/validation, so it is not yet a guaranteed held-out perturbation-gene
  result.
- RankNet pairs are local to the DDP rank and local gene batch; there is no
  cross-rank RankNet synchronization.
- STATE checkpoint and mapping files are loaded from local torch/pickle assets
  and must be treated as trusted local files.

## Caveats

The endpoint is DepMap K562 GeneEffect. It is a population-level dependency
score, not a single-cell death label and not a synthetic-lethality label. Results
from this model card should therefore use dependency-ranking language unless
separate context-specific SL evidence is added.

Observed-B->C is an anchor and diagnostic path. It should not be presented as a
deployable predicted-transcriptome result.

No local `results/experiments/05_aivc_a_to_b_to_c` artifacts were present during
the 2026-06-10 documentation review, so this card describes implementation and
evaluation contracts rather than completed benchmark results.
