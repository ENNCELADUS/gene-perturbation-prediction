# Experiment Protocol: Held-Out-Cell-Line GeneEffect Prediction

Updated 2026-10-02. This is the protocol for the **implemented** GeneEffect track under
[the research blueprint](01-blueprint.md); it holds the model, expression space, training,
metrics and results. The design behind the current wiring is the
[expression-space and `all`-run design](specs/2026-10-02-expression-space-and-all-pipeline-design.md),
and operating commands are in the [runbook](../hpc/README.md). One run has completed: seed 0,
trained, tested and baselined ([result](results/joint_geneeffect_seed0/README.md)). It
predates the expression-space change of §3.4 (STATE was fed raw counts) and its numbers are
not compared like for like with the current pipeline. The best validation model to date is
that backbone frozen under an explicit gene-specific context-slope head (§7, validation
only). Response-pathway diagnostics (§8, [result](results/p1_response_pathway_diagnostics/README.md))
found that the response pathway learns the cell lines it is adapted on but does not
transfer to a held-out line; §9 specifies the comparison that measures what STATE adds and
the single command that runs the whole pipeline. The
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
cells per line is encoded once by the frozen Tx1-3B foundation model; the same cells,
in the expression space of §3.4, are STATE's basal input.

### 3.2 Response anchors

Four labeled training lines carry genetic-perturbation response data: K562, HCT116,
Jurkat and HepG2, joined by DepMap ModelID. Use genetic-perturbation responses and
non-targeting controls; a response condition need not carry a GeneEffect observation.
Response supervision is an auxiliary task that fine-tunes STATE: it uses all conditions of
all four anchors and withholds no condition. The 27 GeneEffect validation lines (§2) are the
only validation split, so the response task has no held-out score inside training; its
held-out behaviour is measured by the comparison of §9.

### 3.3 Dependency labels

Use the pinned DepMap 26Q1 GeneEffect release, joined by ModelID. Fit the gene mean, the
variable-gene set and all normalization on the 170 labeled training lines only, and
reuse them unchanged in every later evaluation. Test values never enter preparation or
fitting. Missing labels stay missing.

### 3.4 Expression space

Every expression quantity except Tx1's input uses one space, the one STATE's released
checkpoint was trained in (arc-state `preprocess_train`): normalise each cell's library to
a target total $T$, take $\log(1+\cdot)$, then slice STATE's 2,000 highly variable genes
(HVGs):

$$
x^{\text{log}}_{ig}=\log\!\Big(1+\frac{T\,x_{ig}}{L_i}\Big),\qquad L_i=\sum_{g'}x_{ig'}\ \text{over all genes of the cell's source matrix.}
$$

- $T$ is the median library size of the non-targeting cells in the Nadig 2025 Jurkat and
  HepG2 response sources, the data STATE's Replogle checkpoint was trained on. It is
  computed once at preparation, recorded in the prepared manifest as `expression_space`,
  and never configured.
- Library size is taken over **all** genes of each source matrix, not over the HVG panel.
  This is a known approximation: gene universes differ slightly between the Nadig,
  Replogle, X-Atlas-Orion and atlas matrices.
- Tx1 keeps raw UMI counts (§3.1). The transform covers STATE's basal bags for all 226
  lines, the response targets and anchor control bags, the basal statistics $q_{g,c}$
  (mean and variance; detected fraction is space-invariant), and the HVG features of the
  context baselines (§6).
- Preparation refuses a source matrix that is already normalised, and loading refuses a
  prepared root without `expression_space`, so a raw-count cache cannot reach STATE.

## 4. Model

![](../figures/geneeffect_architecture.svg)

*Figure 1. (a) Frozen Tx1-3B encodes the sampled basal cells of line $c$ from raw counts into
cell embeddings that are pooled into a context embedding; frozen ESM-2 encodes gene $g$.
Both are computed once and cached. The same cells in log-normalised HVG expression (§3.4) are
STATE's basal input. (b) A trainable adapter turns the gene embedding into a perturbation
token. The trainable STATE transition model, initialised from a published Replogle
checkpoint, reads the log-normalised basal cells through its own released basal encoder
and predicts the line's post-perturbation expression. (c) Response descriptors summarise
how predicted expression differs from basal expression. With the pooled Tx1 context
embedding, the gene embedding and basal statistics of $g$ in $c$, they feed a small
residual head that predicts the GeneEffect residual. Tx1 feeds only this head; it does not
enter STATE. Arrows carry the tensors passed between modules, shaped for one gene–line
condition (cells × features, no batch dimension); boxes state the width change inside each
module. The adapter emits 2024 values because that is the width of STATE's perturbation
vocabulary; STATE projects it, like the 2000-wide expression input, to its hidden size of
328. Notation: $H_c$, Tx1 cell embeddings of line $c$; $e_g$ and $p_g$, the gene
embedding and the perturbation token; $X_c$ and $\hat Y_{g,c}$, basal and predicted
post-perturbation log expression; $\Delta$ and $s$, the projected expression shift and
scalar response summaries; $q_{g,c}$, basal statistics of $g$ in $c$; $z_c$, the pooled
context embedding; $\mu_{\text{train}}(g)$, the training gene mean; $\hat\delta$, the
predicted residual. The training objective is given in §5.*

Tx1 is frozen and supplies cached basal-cell embeddings. STATE and an ESM2 adapter predict
perturbed expression, and the head uses five feature blocks: pooled expression change,
response dispersion statistics, gene-specific basal single-cell statistics, gene embedding
and basal context embedding. The gene mean $\mu_{\text{train}}$ is fixed preprocessing, not
a learned head.

Predicted and basal expression are compared in one shared gene space, the 2,000 log-space
HVGs of §3.4; basal Tx1 embeddings and predicted expression have different widths and are
never subtracted. Cell distributions are supervised as distributions, without pairing
individual control and perturbed cells. Covariates that are unavailable for a gene or line
are masked, never zero-filled.

## 5. Training and selection

A single joint training loop runs. Every update minimizes the mean GeneEffect Huber loss
(delta 1) on $\hat\delta$ against $y-\mu_{\text{train}}$. Every fourth update (0, 4, 8, …)
also adds a response batch balanced across the four anchors (K562, HepG2, Jurkat, HCT116),
sampled from all of their conditions:

$$
L_t=L_{GE}+\mathbf{1}[t\bmod4=0]\,\lambda\,\big(L_{\text{mean-shift MSE}}+L_{\text{energy distance}}\big),\qquad \lambda=1.
$$

Per rank, batches hold 1024 dependency conditions and 64 response conditions. AdamW has
three parameter groups: the new residual head at $10^{-4}$, the ESM2 adapter at $10^{-4}$
and the pretrained STATE model at $10^{-5}$, the most cautious. Training, cell-collation
and projection base seeds are all 0. Settings are fixed in
`configs/geneeffect_joint.yaml` and described in the
[joint-training design](specs/2026-09-06-modular-joint-training-design.md), with the
rates and basal path of the [current design](specs/2026-10-02-expression-space-and-all-pipeline-design.md).

Validation runs once per completed epoch over the 27 validation lines, the only validation
split. **Only minimum validation GeneEffect Huber loss selects `best.pt` and controls early
stopping** (patience 5, at most 50 epochs). Test restores the checkpoint's preprocessing
without refitting or optimizer updates; the seed-0 test has been observed once and must not
become a tuning or checkpoint-selection surface.

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

### Further metrics and controls

Let $\mathcal C_{eval}$ be the evaluated lines and $\mathcal G_{var}$ the train-defined
variable-gene set (4,447 genes in the seed-0 run). Only finite observed labels are scored.

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| RMSE, MAE | Error over all observed gene–line pairs | Accuracy, including prediction scale |
| Absolute Pearson/Spearman | For each line, correlate $\hat y$ and $y$ across genes; average lines equally | Cross-gene dependency ranking within a line |
| Residual Spearman | As residual Pearson, with rank correlation | Context variation for the same gene |

Residual and absolute correlations differ in the **axis of correlation**, not in
centering: for a fixed gene, subtracting its fixed mean does not change Pearson. Targets
are centred on the fold-fit training mean $\mu_g^{(-c)}$ and predictions on the
fold-independent $\bar\mu_g$; centring a prediction on the fold-fit mean scores Spearman
$+1$ by construction. Constant predictions have undefined correlation: gene-mean and
copy-prior residual correlations stay missing, with scored and undefined counts reported,
never zero-filled.

Every model is compared on identical observed keys with the controls fitted on training
lines only: gene mean, K562 copy prior, nearest line, and context-PCA ridge on both the Tx1
embedding and the log-space HVG mean and variance features (§3.4). High absolute
correlation alone cannot establish context learning, and response improvement alone cannot
establish dependency improvement.

## 7. Results

### 7.0 Seed-0 joint backbone

The seed-0 joint run completed eight epochs and selected epoch 3 (stored index 2). Epochs
1–2 used batch 256 per rank and epochs 3–8 continued at 1024, a mixed-batch continuation,
not a batch ablation. Test and all six baseline variants completed on 2026-09-07 with
identical 478,501 observed keys. STATE was fed raw counts in this run (§3.4), so the
numbers below predate the expression-space change.

| Test method | Huber ↓ | Absolute Pearson ↑ | Residual Pearson ↑ | Residual Spearman ↑ |
| --- | ---: | ---: | ---: | ---: |
| Joint best.pt | 0.01611905 | 0.91050 | 0.05414 | 0.05280 |
| Gene-mean | 0.01613152 | 0.91047 | undefined | undefined |
| Context-PCA-ridge, Tx1 | 0.01625512 | 0.90983 | 0.12161 | 0.11574 |
| Context-PCA-ridge, HVG | 0.01635306 | 0.90920 | 0.07736 | 0.08324 |

Joint Huber improves on gene-mean by only 0.0773%, while residual correlations trail the
context baselines. Response validation loss fell 45.57% from epoch 1 to 8 without
sustained GeneEffect validation improvement. This establishes a working training and
evaluation path, not a context-modelling advantage
([full result](results/joint_geneeffect_seed0/README.md)).

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
expression, so its response block made no detectable use of perturbation identity. The diagnostics
below ran in STATE's log-normalised space, with library size approximated by scaling
HVG-panel row sums to a target (a proxy). The formal pipeline now normalises by the whole
library (§3.4).

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

## 9. Response-model comparison and the `all` run

The diagnostics of §8 never replaced STATE with a plain model on the same inputs, so the
value of STATE's transformer, and of Tx1 as a representation for the response task, is
unmeasured. The comparison below measures both, in the expression space of §3.4. The
design and its decisions are in the
[expression-space and `all`-run design](specs/2026-10-02-expression-space-and-all-pipeline-design.md).

### 9.1 Six-arm response-model comparison

Leave-one-anchor-out over the four anchors: each fold trains on all conditions of three
anchors and scores every condition of the fourth.

| Arm | Basal input | Response model | Trained |
| --- | --- | --- | --- |
| No-change | none | basal bag copied | no |
| Global mean effect | none | mean shift of the training anchors, gene-blind | no |
| Released STATE checkpoint | log HVG | released STATE with its one-hot gene vocabulary, scored on genes in that vocabulary | no |
| STATE as in the joint model | log HVG | STATE with the ESM2 adapter, joint learning rates | yes |
| MLP on log HVG | log HVG cell | $x+f([x;a(e_g)])$, final layer zero-initialised | yes |
| MLP on Tx1 | Tx1 cell embedding $h$ | $x+f([h;a(e_g)])$, final layer zero-initialised | yes |

Trained arms use AdamW (new layers $10^{-4}$, STATE $10^{-5}$), 50 fixed epochs with no early
stopping, balanced anchor sampling and seed 0. The response loss is the mean-shift MSE plus
energy distance of §5. The reported statistic is the pooled held-out loss ratio to
no-change at the final epoch (below 1 beats no-change), with one 95% interval from a
1,000-resample gene bootstrap, pooled over all four folds and over the three folds without
HCT116. Identity share (loss increase under ten fixed gene shuffles, as a fraction of
loss), source-anchor training ratio and per-epoch held-out curves are reported for
reading only. Two verdicts: STATE as in the joint model against MLP on log HVG (what the
STATE transformer adds), and MLP on Tx1 against MLP on log HVG (what Tx1 adds as a
representation). A tie at no-change is an expected, informative outcome. The comparison has
four contexts, and STATE's own pretraining exposure (K562, HepG2, Jurkat) qualifies every
result.

### 9.2 The `all` run

`hpc/run.sh all configs/geneeffect_joint.yaml` runs, skipping any step whose output
exists so that a rerun with the same run id resumes: preparation (Tx1 cache reuse, one
pass computing $T$, log-space bags, $q_{g,c}$ and response cache); a sanity line scoring the
released STATE checkpoint on each anchor; the six-arm comparison; joint training on all
visible GPUs; validation evaluation of `best.pt`, the controls of §6 and the explicit
context-slope readout of §7 on the new backbone's cached features; and `summary.md`
(target total $T$, sanity line, comparison table and verdicts, validation table for the
joint model, readout and every control). `hpc/run.sh test CHECKPOINT` is the only route to
the test split; `all` never calls it, and the test split stays closed until a model beats
the context ridge by at least +0.02 residual Pearson, with an interval excluding 0, at each
of three training seeds.

Seed-0 numbers in §7 predate the expression-space change and are not compared like for
like with the `all` run. The [readout objective and selection](specs/2026-09-10-readout-objective-and-selection-design.md)
plan on the frozen backbone remains the follow-up for objective, selection and context
representation; the response model re-enters the feature path only once it beats no-change
on a held-out anchor.
