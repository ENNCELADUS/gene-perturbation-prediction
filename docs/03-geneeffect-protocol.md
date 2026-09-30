# Experiment Protocol: Held-Out-Cell-Line GeneEffect Prediction

Updated 2026-09-30. This is the protocol for the **implemented** GeneEffect track under
[the research blueprint](01-blueprint.md) §3–§4 and §6. Implementation settings live in
the [joint-training design](specs/2026-09-06-modular-joint-training-design.md) and
operating commands in the [runbook](../hpc/README.md). One run has completed: seed 0,
trained, tested and baselined ([result](results/joint_geneeffect_seed0/README.md)). The
current best model is that backbone frozen under an explicit gene-specific context-slope
head (§7, validation only). A numeric-space defect in response preparation was found and
corrected. After the correction, the response pathway learns the cell lines it is adapted
on but does not transfer to a held-out line
(§8, [result](results/p1_response_pathway_diagnostics/README.md)). The
[SL ranking protocol](04-sl-ranking-protocol.md) builds on this backbone; nothing here
is SL evidence.

## 1. Objective and prediction unit

For a held-out cell line $c$ and gene $g$, predict the DepMap GeneEffect $y_{g,c}$
from the basal single-cell transcriptome of $c$ and the protein sequence of $g$.
The model predicts the residual over a fixed training gene mean,

$$
\hat y_{g,c}=\mu_{\text{train}}(g)+\hat\delta(g,c),\qquad
\mu_{\text{train}}(g)=\operatorname{mean}_{c\in\mathcal C_{\text{train}}}y_{g,c},
$$

so that the context-blind term is fixed preprocessing and the residual is the
quantity under test. The generalization axis is the cell line, not the gene. No
GeneEffect measurement of $c$ enters the deployable feature path. GeneEffect is a
single-gene dependency: it is not an SL label and estimates no genetic interaction.

The question this protocol answers is whether a perturbation-response model learns
context information that improves held-out-line dependency prediction beyond the
gene mean and simple context predictors. It is a diagnostic for the backbone, scored
on its own benchmark, and does not substitute for the SL protocol's pair evaluation.

## 2. Benchmark

The [fixed split](../configs/benchmarks/cell_line_geneeffect_226_split.json) is the
sole membership authority; the [data card](data/cell-line-geneeffect-226.md)
documents its construction.

| Cohort | Members | Labeled | Role |
| --- | ---: | ---: | --- |
| train | 172 | 170 | supervised fitting; PC9 and HeLa have no 26Q1 GeneEffect row |
| validation | 27 | 27 | once-per-epoch evaluation, checkpoint selection, early stopping |
| test | 27 | 27 | one-shot final evaluation |

All 47 original basal/SL-context union lines are fixed in train. The 179 atlas lines
follow a patient-grouped partition: no patient's lines are split across cohorts, and
GeneEffect values did not determine membership. The 226-line benchmark and the
nine-context SL split never substitute for each other.

## 3. Inputs and supervision

### 3.1 Basal cells

Use each line's basal cells as raw UMI counts, never CPM: Tx1 does not read CPM the way
it reads the counts underneath, and the shift survives pooling
([measured](results/exp13_stage0/README.md)). A fixed random sample of up to 128 basal
cells per line is encoded once by the frozen Tx1-3B foundation model.

### 3.2 Response anchors

Four labeled training lines carry genetic-perturbation response data: K562, HCT116,
Jurkat and HepG2, joined by DepMap ModelID. Use genetic-perturbation responses and
non-targeting controls; a response condition need not carry a GeneEffect observation.
A fixed 10% of each anchor's response conditions is withheld from optimization and
scored at validation and test. This is condition holdout on training lines, not
cell-line or unseen-gene holdout.

### 3.3 Dependency labels

Use the pinned DepMap 26Q1 GeneEffect release, joined by ModelID. Fit the gene mean, the
variable-gene set and all normalization on the 170 labeled training lines only, and
reuse them unchanged in every later evaluation. Test values never enter preparation or
fitting. Missing labels stay missing.

## 4. Model

![](../figures/geneeffect_architecture.svg)

*Figure 1. (a) Frozen Tx1-3B encodes the sampled basal cells of line $c$ and frozen ESM-2
encodes gene $g$. Both are computed once and cached. (b) A trainable adapter turns the
gene embedding into a perturbation token. The trainable STATE transition model,
initialised from a published Replogle checkpoint, predicts the line's post-perturbation
expression. (c) Response descriptors summarise how predicted expression differs from
basal expression. With the pooled context embedding, the gene embedding and basal
statistics of $g$ in $c$, they feed a small residual head that predicts the GeneEffect
residual. Arrows carry the tensors passed between modules, shaped for one gene–line
condition (cells × features, no batch dimension); boxes state the width change inside
each module. The adapter emits 2024 values because that is the width of STATE's
perturbation vocabulary, which STATE projects, like the 2560-wide Tx1 embeddings, to its
hidden size of 328. Notation: $H_c$, Tx1 cell embeddings of line $c$; $e_g$ and $p_g$, the gene
embedding and the perturbation token; $X_c$ and $\hat Y_{g,c}$, basal and predicted
post-perturbation expression; $\Delta$ and $s$, the projected expression shift and scalar
response summaries; $q_{g,c}$, basal statistics of $g$ in $c$; $z_c$, the pooled context
embedding; $\mu_{\text{train}}(g)$, the training gene mean; $\hat\delta$, the predicted
residual. The training objective is given in §5.*

Predicted and basal expression are compared in one shared gene space. Cell distributions
are supervised as distributions, without pairing individual control and perturbed
cells. Covariates that are unavailable for a gene or line are masked, never zero-filled.

## 5. Training and selection

A single joint training loop runs. Every update minimizes the mean GeneEffect Huber
loss on $\hat\delta$ against $y-\mu_{\text{train}}$. Every fourth update also adds a
response batch balanced across the four anchors:

$$
L_t=L_{GE}+\mathbf{1}[t\bmod4=0]\,\lambda\,\big(L_{\text{mean-shift MSE}}+L_{\text{energy distance}}\big),\qquad \lambda=1.
$$

The pretrained STATE model is updated most cautiously and the new residual head fastest.
Validation runs once per epoch over all 27 validation lines and the withheld response
conditions. **Only validation GeneEffect Huber loss selects the checkpoint and controls
early stopping**; total loss, response loss and correlations are reported diagnostics.
Learning rates, batch sizes, patience and seeds are fixed in the
[joint-training design](specs/2026-09-06-modular-joint-training-design.md).

## 6. Evaluation

Evaluate the selected checkpoint with the training-side gene mean and normalization
restored. Train and validation use their own observed cell-line/gene pairs and the same
train-defined variable-gene set. Let $y_{cg}$ be the observed GeneEffect, $\mu_g$ the
fitted training gene mean, $r_{cg}=y_{cg}-\mu_g$ the target residual and $\hat r_{cg}$
the predicted residual.

### Residual Pearson ↑

For each variable gene, compute Pearson correlation across cell lines, then take
an equal-weight mean over genes with defined correlations:

$$
\rho_g=\operatorname{Pearson}_c(r_{cg},\hat r_{cg}),\qquad
\rho_{\mathrm{macro}}=\frac{1}{|G_\rho|}\sum_{g\in G_\rho}\rho_g.
$$

This measures how well predictions follow gene-specific context variation. The reported
set contains 4,447 train-defined variable genes; undefined correlations are excluded
and counted. A context-blind predictor, such as the gene mean, is constant per gene, so
its residual correlation is undefined rather than zero.

### SD ratio

For each variable gene, divide prediction SD by target residual SD across cell lines,
then average the defined ratios equally:

$$
q_g=\frac{\operatorname{SD}_c(\hat r_{cg})}{\operatorname{SD}_c(r_{cg})},\qquad
q_{\mathrm{macro}}=\frac{1}{|G_q|}\sum_{g\in G_q}q_g.
$$

A ratio below 1 indicates shrinkage; 1 indicates matching amplitude.

### GeneEffect Huber ↓

For $e_{cg}=\hat r_{cg}-r_{cg}$, use Huber loss with $\delta=1$:

$$
\ell(e)=\begin{cases}
\tfrac12 e^2,& |e|\le1,\\
|e|-\tfrac12,& |e|>1,
\end{cases}
\qquad
L_{\mathrm{GE}}=\frac{1}{|\Omega|}\sum_{(c,g)\in\Omega}\ell(e_{cg}).
$$

$\Omega$ contains all labeled pairs in the split, including genes outside the
variable-gene set, each with equal weight. It excludes response loss and is the
selection criterion of §5.

## 7. Results

The best model to date is the selected seed-0 joint backbone, frozen, read out by a
residual head that adds an explicit gene-specific context slope to the §4 head:

$$
\hat\delta(g,c)=\mathrm{MLP}(F_{g,c})+w_g^{\top}u_c,
$$

where $u_c$ holds the first eight principal components of the pooled Tx1 context
embedding, fitted on training lines, and $w_g$ is one regularised slope per
training-covered gene. The head sees the context embedding, gene embedding and basal
covariates; the backbone's response block is excluded. Selection follows §5. The model
is validation-selected and carries **no test number**: the test split was spent once on
the joint backbone, which improved on the gene mean by 0.08% Huber and trailed the
context ridge baselines
([joint result](results/joint_geneeffect_seed0/README.md)), and it has not been opened
for any readout. Provenance and the full numbers:
[readout and response diagnostics](results/p1_response_pathway_diagnostics/README.md).
All result figures are drawn from tracked evidence by
[`plot_geneeffect_protocol.py`](figures/plot_geneeffect_protocol.py).

### 7.1 Learning curves

![](figures/geneeffect_readout_learning_curves.svg)

*Figure 2. Readout training on the frozen backbone, head seed 0, one point per epoch. (a)
Validation GeneEffect Huber loss, the selection criterion; the ring marks the selected
epoch. (b) Residual Pearson over the 4,447 train-defined variable genes, validation
(solid) and training (dashed); the dotted line is the eight-component context-PCA ridge
on validation. Both readouts start from identical MLP weights and see identical cached
features; the explicit slope is the only difference.*

The explicit slope reaches the level of the linear context baseline within three epochs;
the shared MLP stays at less than half of it. Training correlation keeps rising with no
further validation gain, so the readout fits gene-by-context structure that does not
transfer. Validation Huber turns upward after the selected epoch while residual Pearson
holds.

### 7.2 Validation comparison with controls

![](figures/geneeffect_validation_comparison.svg)

*Figure 3. Four methods on the same 479,084 observed cell-line/gene pairs over the 27
validation lines; the readouts consume identical cached features. (a) Residual Pearson;
open circles are two further head initialisations. (b) Differences in residual Pearson
from the explicit slope, 1,000 paired bootstrap resamples of the 27 lines, 95%
intervals.*

The explicit slope recovers more than twice the context signal of the shared MLP and of
the joint backbone's own readout, and it reproduces at every head initialisation. It is
indistinguishable from the context-PCA ridge. It also lowers Huber loss the most (0.75%
below the gene mean, against 0.22% for the ridge), while predicting with smaller
amplitude than the ridge (SD ratio 0.19 against 0.27). Adding the backbone's response
block changes nothing under either readout (−0.003 [−0.010, +0.002] with the slope,
+0.000 [−0.020, +0.016] with the shared MLP).

This is a validation-only readout result on a backbone trained before the numeric-space
correction of §8. It licenses no context-modelling claim beyond an eight-component
linear context baseline and no SL claim.

## 8. Where the model stalls: response-pathway diagnostics

Three closed, validation-only diagnostics on the selected backbone locate the
bottleneck; designs are under [`specs/`](specs/) and evidence in the
[result note](results/p1_response_pathway_diagnostics/README.md). The seed-0 backbone
was trained on response data cached as raw counts, whereas STATE expects log-normalised
expression, so its response block made no detectable use of perturbation identity. Preparation now
works in STATE's own space (§3.4), and every diagnostic below uses the corrected
preparation.

![](figures/geneeffect_interface_transfer.svg)

*Figure 4. Four interfaces between the Tx1 context and the response model, each adapted on
three anchor lines and scored on the fourth. The x axis is the response loss on the
held-out line divided by the no-change reference; below 1 beats no-change. Markers are
held-out lines; the black tick is the pooled ratio, whose 95% interval is narrower than
the tick. The untrained released checkpoint is the reference row.*

No interface transfers to a held-out line. The more of STATE's basal pathway is trained
on Tx1 input, the larger the held-out error; the expression-residual interface avoids the
failure but carries almost no perturbation-specific effect. With the readout already at
the linear context baseline (§7.2), the bottleneck is upstream of the head: the pooled
Tx1 context holds about eight components of usable signal, and the response model does
not generalise beyond its three adaptation lines. Progress needs a response model that
holds up on a held-out line or a richer context representation.

## 9. Future work

The [readout objective and selection](specs/2026-09-10-readout-objective-and-selection-design.md)
plan works on the frozen backbone and cached features, in order:

1. **Objective and selection.** Select on validation residual Pearson rather than Huber
   (an in-place amendment to blueprint §4, by the owner), and try correlation-aligned
   objectives, per-gene calibration and head-seed ensembles.
2. **Context representation.** Basal-expression components, learned pooling of per-cell
   Tx1 embeddings and gene-neighbourhood covariates, each against an expression-only
   control.
3. **Response model.** Re-enters the feature path only once it beats no-change on a
   held-out anchor.
4. **Test split.** Reopens once, only for a model that beats the context ridge by at
   least +0.02 residual Pearson, with an interval excluding 0, at each of three training
   seeds.
