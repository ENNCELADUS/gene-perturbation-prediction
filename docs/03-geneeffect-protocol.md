# Experiment Protocol: Held-Out-Cell-Line GeneEffect Prediction

Updated 2026-09-09. This is the executable protocol for the **implemented** GeneEffect
track under [the research blueprint](01-blueprint.md) §3–§4 and §6; the
[joint-training design](specs/2026-09-06-modular-joint-training-design.md) is its
implementation specification and the [runbook](../hpc/README.md) its operator guide.
One run has completed: seed 0, trained, tested and baselined
([result](results/joint_geneeffect_seed0/README.md)). The current best model is that
backbone frozen under an explicit gene-specific context-slope head (§7, validation
only). The backbone's response pathway has been diagnosed as non-functional because
preparation feeds raw counts to a log-space decoder
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

All 47 original basal/SL-context union lines are fixed in train; the 179 atlas lines
follow a patient-grouped partition in which no PatientID crosses sides and GeneEffect
values did not determine membership. Every fit passes through the split's eligibility
guard; the two unlabeled train members are declared explicitly, and any other missing
label is a hard error. The 226-line benchmark and the nine-context SL split never
substitute for each other.

## 3. Inputs and supervision

### 3.1 Basal cells

Use the exact registered model's basal cells as raw UMI counts, never CPM: Tx1 does not
read CPM like the counts underneath it, and the shift survives pooling
([measured](results/exp13_stage0/README.md)). Select up to 128 basal cells per line
deterministically, without replacement, from the recorded preparation seed; a line with
fewer cells keeps all of them. Frozen Tx1-3B embeddings of these cells are cached once
at preparation time with their pretraining and preprocessing provenance.

### 3.2 Response anchors

| ModelID | Cell line |
| --- | --- |
| ACH-000551 | K562 |
| ACH-000971 | HCT116 |
| ACH-000995 | Jurkat |
| ACH-000739 | HepG2 |

All four are labeled training members. Use genetic-perturbation responses and
non-targeting controls; a response condition need not carry a GeneEffect observation.
A fixed 10% of each anchor's response conditions (holdout seed 13) contributes no
response targets to optimization and is scored at validation and test. Response
holdout is condition holdout on training lines, not cell-line or unseen-gene holdout.

### 3.3 Dependency labels

Use the pinned DepMap 26Q1 GeneEffect release joined by ModelID. Fit the gene mean,
the variable-gene set and all normalization on the 170 labeled training lines only,
store them in the checkpoint and restore them for every later evaluation. The common
evaluation gene panel is prepared once from label availability; test values never
enter preparation or fitting. Missing labels stay masked.

### 3.4 Preparation

Prepare fixed inputs once, in a single process, under the configured `prepared_root`:
Tx1 basal-embedding cache, per-gene basal statistics $q_{g,c}$, ESM-2 table, response
cache with gene order and the common gene panel. Training opens these caches and never
rebuilds them; a missing cache is an error, not a trigger.

## 4. Model

![Context-conditioned GeneEffect architecture](../figures/geneeffect_architecture.png)

*(a) Frozen Tx1-3B encodes 128 sampled basal cells of line $c$ into
$H_c\in\mathbb{R}^{128\times2560}$ and frozen ESM-2 encodes gene $g$ into
$e_g\in\mathbb{R}^{1280}$; both are cached at preparation time. (b) A trainable adapter maps
$e_g$ to the perturbation token $p_g\in\mathbb{R}^{2024}$, and the trainable STATE transition
model, initialised from the ST-HVG-Replogle checkpoint and run on 64-cell windows, predicts
post-perturbation HVG expression $\hat Y_{g,c}\in\mathbb{R}^{128\times2000}$. (c) Response
descriptors compare $\hat Y_{g,c}$ with the matched basal HVG expression $X_c$: the 4000-d
mean/variance shift $\Delta$ is reduced to 256 dimensions by one fixed seeded projection,
plus six scalar summaries $s$. Concatenated with the fixed covariates $q_{g,c}$ (3 basal
statistics of $g$ in $c$), $e_g$ and $z_c$ (mean- and variance-pooled $H_c$, 5120-d) and
three coverage masks, the 6668-d vector feeds the residual head, an MLP 6668 → 256 → 256 → 1
over train-fitted standardised blocks. (d) The joint objective of §5. 

Expression changes subtract basal and predicted bags in the same 2000-gene output
space; $H_c$ and $\hat Y_{g,c}$ have different widths and are never subtracted.
Cell distributions are supervised without pairing an individual control cell to an
individual perturbed cell. The STATE checkpoint load reports loaded and newly
initialised layers once; unexpected incompatibilities are errors. Coverage masks
declare whether $q_{g,c}$, the HVG-panel membership of $g$ and the own-gene shift are
available; nothing is zero-filled.

## 5. Training and selection

One joint loop. Every optimizer update minimizes the mean GeneEffect Huber loss
(delta 1) on $\hat\delta$ against $y-\mu_{\text{train}}$; updates 0, 4, 8, … add a
balanced response batch:

$$
L_t=L_{GE}+\mathbf{1}[t\bmod4=0]\,\lambda\,\big(L_{\text{mean-shift MSE}}+L_{\text{energy distance}}\big),\qquad \lambda=1.
$$

| Setting | Value |
| --- | --- |
| Dependency conditions per rank / replay conditions per rank | 1024 / 64, 16 per anchor |
| Learning rates STATE / adapter / head | $10^{-6}$ / $10^{-5}$ / $10^{-4}$ |
| Optimizer, weight decay, gradient clipping | AdamW, 0.01, 1.0 |
| Maximum epochs, validation patience | 50, 5 |
| Cell bag, STATE window | 128 cells, 64 cells |
| Training, cell-collation, projection base seeds | 0, 0, 0 |

Feature scales are initialised before the first update from up to 32 finite
dependency conditions per labeled training line, in evaluation mode on training data
only, and stored in the checkpoint. Validate exactly once at the end of every
completed epoch over all 27 validation lines and the held-out response conditions,
reporting the fields of the joint design §4. **Only minimum `val_geneeffect_loss`
selects `best.pt` and controls early stopping**; total loss, response loss and
correlations are reported diagnostics. Resume happens at epoch boundaries with the
checkpoint's configuration; a batch-size change needs a new run id and a documented
derived checkpoint, and produces a mixed continuation, not an ablation.

## 6. Evaluation

Evaluate the selected checkpoint in `eval()` mode without gradients, restoring its
fitted preprocessing. Train and validation use their own observed cell-line/gene
pairs and the same train-defined variable-gene set. Let $y_{cg}$ be the observed
GeneEffect, $\mu_g$ the fitted training gene mean, $r_{cg}=y_{cg}-\mu_g$ the target
residual and $\hat r_{cg}$ the predicted residual.

### Residual Pearson ↑

For each variable gene, compute Pearson correlation across cell lines, then take
an equal-weight mean over genes with defined correlations:

$$
\rho_g=\operatorname{Pearson}_c(r_{cg},\hat r_{cg}),\qquad
\rho_{\mathrm{macro}}=\frac{1}{|G_\rho|}\sum_{g\in G_\rho}\rho_g.
$$

This measures how well predictions follow gene-specific context variation.
The reported set contains 4,447 train-defined variable genes. Correlations use
pairwise-finite observations; undefined correlations are excluded and counted.
Absolute Pearson in Section 7 instead correlates across genes within each line,
then averages over lines.

### SD ratio

For each variable gene, divide prediction SD by target residual SD across the
same pairwise-finite cell lines, then average the defined ratios equally:

$$
q_g=\frac{\operatorname{SD}_c(\hat r_{cg})}{\operatorname{SD}_c(r_{cg})},\qquad
q_{\mathrm{macro}}=\frac{1}{|G_q|}\sum_{g\in G_q}q_g.
$$

SD uses `ddof=0`. A ratio below 1 indicates shrinkage; 1 indicates matching
amplitude. Fewer than two observations or zero target SD makes the ratio
undefined; zero prediction SD with positive target SD gives zero. Section 7.4
reports the median and percentiles of these per-gene ratios instead of the mean.

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

$\Omega$ contains all finite labeled pairs in the split, including genes outside
the variable-gene set. Each pair has equal weight. Adding the same fitted gene
mean to predictions and targets leaves this loss unchanged. It excludes response
loss; minimum validation GeneEffect Huber selects `best.pt`.

## 7. Measured results and learning curves

The current best model is **A2**: the selected seed-0 joint backbone frozen
(`joint_seed0_20260906T174818Z_b1024/best.pt`, epoch 3, SHA-256 `37405454…63c59`) with
a fresh residual head that adds a gene-specific context slope to the §4 MLP,

$$
\hat\delta(g,c)=\mathrm{MLP}(F_{g,c})+w_g^{\top}u_c,\qquad u_c=\text{train-fitted PCA8 scores of } z_c \text{ scaled to unit training SD},
$$

with one zero-initialised, L2-regularised $w_g\in\mathbb{R}^8$ per training-covered gene,
trained on the cached direct features D ($z_c$, $e_g$, $q_{g,c}$ and masks; no response
block) with head seed 0 and data seed 0. Selection follows §5: minimum validation
GeneEffect loss, patience 5. Head run `outputs/p1a/p1a_seed0_20260907T144045Z/heads/A2`,
`best.pt` SHA-256 `09352f21…209d`, code `afdd7fc`. It is a validation-selected model
with **no test number**: the test split was spent on the backbone (§7.3) and has not been
opened for any head. Design: [P1-A](specs/2026-09-07-p1a-fixed-backbone-head-diagnostics-design.md);
evidence: [P1 result](results/p1_response_pathway_diagnostics/README.md).

### 7.1 Training and validation curves

A2 head, batch 1024, learning rate $10^{-4}$, 2945 updates per epoch. Best epoch 3;
training stopped after epoch 8 with five epochs without a new minimum validation
GeneEffect loss. The shared-MLP head A0 (same features, same initial MLP weights) is the
comparison. Residual Pearson and SD ratio are macro means over the 4447 train-defined
variable genes.

| Epoch | Step | A2 train Huber ↓ | A2 val Huber ↓ | A2 train residual Pearson | A2 val residual Pearson ↑ | A2 val SD ratio | A0 val Huber | A0 val residual Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2945 | 0.01490 | 0.01629 | 0.2632 | 0.1112 | 0.1093 | 0.01645 | 0.0339 |
| 2 | 5890 | 0.01458 | 0.01626 | 0.2870 | 0.1293 | 0.1569 | 0.01644 | 0.0479 |
| 3 | 8835 | 0.01430 | 0.01619 | 0.3144 | 0.1315 | 0.1886 | 0.01637 | 0.0495 |
| 4 | 11780 | 0.01418 | 0.01623 | 0.3201 | 0.1308 | 0.2144 | 0.01638 | 0.0446 |
| 5 | 14725 | 0.01404 | 0.01629 | 0.3282 | 0.1332 | 0.2450 | 0.01641 | 0.0580 |
| 6 | 17670 | 0.01394 | 0.01625 | 0.3348 | 0.1362 | 0.2565 | 0.01640 | 0.0570 |
| 7 | 20615 | 0.01388 | 0.01632 | 0.3369 | 0.1312 | 0.2628 | 0.01640 | 0.0555 |
| 8 | 23560 | 0.01379 | 0.01635 | 0.3428 | 0.1344 | 0.2744 | 0.01643 | 0.0580 |

The explicit slope reaches validation residual Pearson 0.11 after one epoch and
0.13 by epoch 3, while A0 never exceeds 0.06. Train residual Pearson rises to 0.34 with
no validation gain after epoch 3: the head fits gene × context structure that does not
transfer beyond what the first epochs capture. The backbone's own training curves are in
the [joint result](results/joint_geneeffect_seed0/README.md).

### 7.2 Selected-checkpoint validation and controls

All methods share the 479084 observed `(ModelID, gene_symbol)` keys across the 27
validation lines (1165 missing labels excluded) and the same 4447 variable genes.
Heads and references use identical cached features; PCA8-ridge is the Tx1 context-PCA
ridge fitted on training lines.

| Validation method | Huber ↓ | Absolute Pearson ↑ | Residual Pearson ↑ | Residual Spearman ↑ | SD ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A2, explicit context slope on D** | 0.01619 | 0.9103 | 0.1315 | 0.1270 | 0.1886 |
| A3, explicit slope on D + R | 0.01620 | 0.9102 | 0.1280 | 0.1240 | 0.1864 |
| A0, shared MLP on D | 0.01637 | 0.9093 | 0.0495 | 0.0515 | 0.0929 |
| A1, shared MLP on D + R | 0.01637 | 0.9093 | 0.0496 | 0.0479 | 0.1270 |
| Joint best.pt head (P0) | 0.01632 | 0.9094 | 0.0404 | 0.0439 | 0.0931 |
| Context-PCA8 ridge, Tx1 | 0.01628 | 0.9097 | 0.1330 | 0.1233 | 0.2727 |
| Gene mean | 0.01632 | 0.9094 | undefined | undefined | 0 |

Paired 27-line bootstrap (1000 resamples, seed 0), residual Pearson unless stated:
A2 − A0 = +0.0820 [0.0577, 0.1047] and Huber −0.00018 [−0.00037, −0.00000];
A2 − P0 = +0.0911 [0.0661, 0.1154]; A2 − PCA8-ridge = −0.0015 [−0.0153, 0.0159] with
SD ratio −0.084 [−0.103, −0.070]; A1 − A0 = +0.0001 [−0.0204, 0.0159];
A3 − A2 = −0.0035 [−0.0103, 0.0025]. Head seeds 1 and 2 reproduce A2 at 0.1328 and
0.1344 with A2 − A0 = +0.0711 [0.0487, 0.0947] and +0.0802 [0.0551, 0.1038].

A2 beats the joint head and the shared MLP on every residual metric with intervals
excluding 0, ties Tx1 PCA8-ridge on correlation while predicting with less variance,
and gains nothing from the response block R. Huber differences are at the
$10^{-4}$ level: the context signal is small against the gene-mean block. This is a
validation-only readout result on a frozen backbone whose response pathway is
non-functional (§8); it licenses no context-modelling claim beyond the explicit PCA8
baseline and no SL claim.

### 7.3 Backbone test record

Run `joint_seed0_20260906T174818Z_b1024`, seed 0, `best.pt` epoch 3 (stored index 2),
optimizer step 13250, evaluated once on the test split with its baselines on 2026-09-07.
Sources: [run evidence](results/joint_geneeffect_seed0/evidence.json) and
[full result](results/joint_geneeffect_seed0/README.md). Epochs 1–2 used batch 256/rank
(5889 updates, 1473/1472 replays per epoch); epochs 3–8 used 1024/rank (1472 updates,
368 replays), a documented mixed continuation; training ended after epoch 8 on patience.
The test split is spent: it has not been and will not be used to choose or score heads.

All methods share 478501 observed `(ModelID, gene_symbol)` keys across 27 test lines;
1748 missing labels out of 480249 possible pairs are excluded. Residual correlations
use the same 4447 train-defined variable genes.

| Test method | Huber ↓ | Absolute Pearson ↑ | Residual Pearson ↑ | Residual Spearman ↑ |
| --- | ---: | ---: | ---: | ---: |
| Joint best.pt, epoch 3 | 0.01612 | 0.9105 | 0.0541 | 0.0528 |
| Gene mean | 0.01613 | 0.9105 | undefined | undefined |
| Context-PCA ridge, Tx1 | 0.01626 | 0.9098 | 0.1216 | 0.1157 |
| Context-PCA ridge, HVG | 0.01635 | 0.9092 | 0.0774 | 0.0832 |
| Nearest line, Tx1 | 0.02869 | 0.8458 | 0.0608 | 0.0597 |
| Nearest line, HVG | 0.02854 | 0.8460 | 0.0590 | 0.0613 |
| K562 copy-prior | 0.03390 | 0.8169 | undefined | undefined |

Seed 0 improves on the gene mean by 0.0773% Huber and trails both contextual ridge
baselines on residual correlation. Selected-checkpoint response MSE / energy distance /
total are 85.58490 / 370.35131 / 455.93621. These score the same response-condition
holdout on four training anchors, not measured responses on the 27 test lines.

### 7.4 Backbone test prediction diagnostics

Population standard deviations below are computed across observed test lines within each variable gene; SD ratios are computed
per gene before taking their median.

| Diagnostic, 4447 variable genes | Joint | Tx1 PCA-ridge |
| --- | ---: | ---: |
| Median prediction SD | 0.0129 | 0.0568 |
| Median true residual SD | 0.2283 | 0.2283 |
| Median prediction / true SD ratio | 5.7500% | 25.3100% |
| 10th–90th percentile SD ratio | 2.7600%–15.7000% | 14.4700%–40.8900% |
| Genes with positive residual Pearson | 61.4800% | 72.9000% |
| Huber on variable-gene subset | 0.04032 | 0.04047 |
| Gene-mean Huber on the same subset | 0.04042 | 0.04042 |

Tx1 PCA-ridge has higher per-gene Pearson on 2683/4447 genes (60.33%); joint has
higher Pearson on 1764. Median joint-minus-ridge Pearson is −0.06350.

For the 4315 variable genes with labels on all 27 test lines, the prediction spectrum
was measured after subtracting each gene's mean across those lines.

| Centered matrix diagnostic | Joint predictions | Tx1 ridge predictions | True residuals |
| --- | ---: | ---: | ---: |
| First singular direction, fraction of squared singular values | 58.9000% | 42.9200% | 11.0600% |
| First three directions, fraction of squared singular values | 83.4300% | 79.5300% | 25.3300% |
| Participation rank, $(\sum_i s_i^2)^2/\sum_i s_i^4$ | 2.5800 | 3.6600 | 20.2600 |

An identical cell-line offset shared across genes accounts for 6.05% of centered
joint prediction energy and 1.36% for Tx1 ridge. Source:
[diagnostic measurements](../autoresearch/reason-260907-residual-diagnosis/prediction_diagnostics.json)
and [calculation](../autoresearch/reason-260907-residual-diagnosis/diagnose_predictions.py).


### 7.5 Backbone train/validation diagnostics

The same selected checkpoint was evaluated on fixed cached inputs on 2026-09-07.
Pearson and SD ratio are macro means over the train-defined variable genes;
Huber covers all observed pairs.

| Metric | Train | Validation |
| --- | ---: | ---: |
| Residual Pearson ↑ | 0.1649 | 0.0404 |
| SD ratio | 0.1257 | 0.0931 |
| GeneEffect Huber ↓ | 0.01511 | 0.01632 |

Exports: `evaluation/best/{train,val}/` under the run directory on H20.
Training predictions already show weak context correlation and substantial
shrinkage; validation declines further. Prioritize the fitting bottleneck before
investigating the additional generalization gap.

## 8. Response-pathway diagnostics (P1)

Three seed-0 diagnostics on the selected checkpoint above — P1-A fixed-backbone heads,
P1-B interface adaptation, P1-C interface isolation — are closed. Evidence and provenance:
[result](results/p1_response_pathway_diagnostics/README.md); designs under
[`specs/`](specs/). Validation only; the test split was not used.

**Finding.** §3.4 preparation caches the HVG control bags and response targets as raw
UMI counts, while the STATE ST-HVG-Replogle checkpoint in §4 was trained on
`normalize_total` + `log1p` expression. The joint trainer therefore drove a log-space
decoder with count-space inputs and scored its outputs against count-space targets. As
trained, the seed-0 backbone's response block shows no perturbation-identity use and
scores about 21 × the no-change reference on its own anchors' condition holdout. Driven in
approximately its own space (log1p of HVG-panel row sums normalised to 3,500), the
released checkpoint beats no-change on HepG2 (0.84) and Jurkat (0.95) with identity
advantage 22–66% of loss, misses the pre-registered K562 gate (1.28 against 1.10) and
fails on HCT116 at every scale (8.9, an X-Atlas-Orion platform mismatch).

| Count-space interface, LOAO held-out ratio to no-change | jurkat | k562 | hepg2 | hct116 | pooled [95%] |
| --- | ---: | ---: | ---: | ---: | --- |
| V0, P1-B interface adaptation at 1e-4 | 3.96 | 19.9 | 10.2 | 35.3 | 17.3 [16.8, 17.9] |
| V1, null-subtracted expression residual | 1.02 | 1.04 | 1.00 | 1.06 | 1.031 [1.028, 1.033] |
| V2, native basal path + zero-init Tx1 context | 2.86 | 11.9 | 5.62 | 27.7 | 12.0 [11.6, 12.4] |

Nothing met the keep predicate. V1 removes the blow-up but its held-out interval
excludes 1 from above and its held-out identity advantage vanishes on HCT116. Interface
learning rate is monotone (1e-6 → 10.2, 1e-5 → 4.63, 1e-4 → 3.96 on the Jurkat fold)
but every arm fits a count-space offset, so the P0 rate of 1e-6 on the new basal encoder
is bounded, not isolated, as a cause.

**GeneEffect side.** With the backbone frozen, an explicit gene-specific PCA8 context
slope (A2) raises validation residual Pearson from 0.05 to 0.13 at head seeds 0, 1 and 2
(A2 − A0 = +0.071 to +0.082, intervals exclude 0), ties Tx1 PCA8-ridge (−0.001
[−0.015, 0.016]) and gains nothing from the response block R at any seed. A2 is the
head control for any future response representation.

**Consequence for this protocol.** Before any further response supervision or response
feature is used, §3.4 preparation must transform control bags and targets into the
response model's own numeric space and record the transform in the prepared bundle; the
loss, the references and the identity derangements are then computed in that space. The
count-space round is retained as the record of the defect and is never pooled with a
transformed round. The P0 number in §7 stands as the GeneEffect record; its response
losses in §7.3 were computed in count space and carry no response-quality meaning.

