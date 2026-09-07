# Experiment Protocol: Held-Out-Cell-Line GeneEffect Prediction

Updated 2026-09-07. This is the executable protocol for the **implemented** GeneEffect
track under [the research blueprint](01-blueprint.md) §3–§4 and §6; the
[joint-training design](specs/2026-09-06-modular-joint-training-design.md) is its
implementation specification and the [runbook](../hpc/README.md) its operator guide.
One run has completed: seed 0, trained, tested and baselined
([result](results/joint_geneeffect_seed0/README.md)). The
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
undefined; zero prediction SD with positive target SD gives zero. Section 7.3
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

Run `joint_seed0_20260906T174818Z_b1024`, seed 0. The selected checkpoint is
`best.pt`, epoch 3 (stored index 2), optimizer step 13250.
Sources: [run evidence](results/joint_geneeffect_seed0/evidence.json) and
[full result](results/joint_geneeffect_seed0/README.md).

### 7.1 Training and validation curves

![Training and validation curves](../outputs/analysis/30030_20260907/learning_curves.png)

Epochs 1–2 used batch 256/rank, with 5889 optimizer updates per epoch and 1473/1472
response replays respectively. Epochs 3–8 used batch 1024/rank, with 1472 updates
and 368 replays per epoch. This is a documented mixed continuation. Training ended
after epoch 8 with five epochs without a new minimum validation GE loss. The largest
observed validation residual Pearson was 0.04946 at epoch 6; selection used GE loss.

### 7.2 Selected-checkpoint test and controls

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

### 7.3 Existing-test prediction diagnostics

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


### 7.4 Selected-checkpoint train/validation diagnostics

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
