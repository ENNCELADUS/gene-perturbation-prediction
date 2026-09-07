# Research Blueprint: Context-Conditioned Synthetic-Lethality Ranking

Updated 2026-09-07. This document defines the research task and claim boundaries.
[Related work](02-literature-review.md) explains the prior art;
[SL protocol](03-experiment-protocol.md) defines the separate pair-label experiment;
[joint GeneEffect design](specs/2026-09-06-modular-joint-training-design.md)
defines the implemented intermediate task.

## 1. Task

Given basal single-cell transcriptomes from a cancer cell line and an unordered
pair of genes, produce a score for ranking experimental synthetic-lethal (SL)
hits in that line. At prediction time, no GeneEffect or SL measurements from the
query line are inputs. The generalization axis is the **cell line**, not the gene.
No SL graph enters the feature path.

The research question is whether a model trained to predict perturbation responses
and single-gene dependencies learns context information that improves SL ranking
beyond gene identity, pan-essentiality and simple context predictors. This is a
hypothesis to test, not an established biological mechanism.

| Task | Input and output | Current state |
| --- | --- | --- |
| Perturbation response | Basal cells + one gene → predicted expression distribution | Auxiliary supervision on four training lines |
| GeneEffect | Basal cells + one gene → single-gene dependency | Joint model and test baselines completed, seed 0 |
| SL ranking | Basal cells + two genes → pair ranking score | Separate proposal; no model run or SL result |

The implemented model predicts single-gene outcomes. It has no double-knockout
output and estimates no genetic-interaction quantity. A pair classifier is a
proposed downstream use, not an existing capability. A sigmoid score alone is
not a calibrated probability of biological synthetic lethality.

## 2. Data and generalization

The two benchmarks have different assignments and cannot substitute for each other.

| Benchmark | Training | Validation | Test | Authority |
| --- | --- | --- | --- | --- |
| GeneEffect, 226 members | 172 members, 170 labeled | 27 lines | 27 lines | [Fixed split](../configs/benchmarks/cell_line_geneeffect_226_split.json), [data card](data/cell-line-geneeffect-226.md) |
| SL, nine contexts | K562, Jurkat, OVCAR8, HAP1, HT29 | A549 | 22RV1, PC9, HeLa | [Fixed split](../configs/benchmarks/context_screen_v2_split.json), [data card](data/sl-context-screen.md) |

GeneEffect labels are from the pinned DepMap 26Q1 release. PC9 and HeLa are the
two unlabeled GeneEffect training members; neither participates in supervised
fitting. K562, Jurkat, HepG2 and HCT116 provide response supervision and belong
to the labeled training cohort. Their response-condition holdout is distinct
from the 27-line GeneEffect holdouts.

SL labels mean experimental screen hit or screened non-hit in a named context.
They are inferred from unanimous aggregate source rows (`silver_inferred`), not
independently reconstructed per-context evidence. The current SL table has
incomplete filter attribution, degenerate A549 validation labels and shared
PC9/HeLa aggregate labels. These limitations constrain any future evaluation.

## 3. Implemented GeneEffect model

For gene $g$ and context $c$, let $y_{g,c}$ be GeneEffect and $e_g$ the ESM2
embedding. Fit a fixed mean on the labeled training lines only:

$$
\mu_g=\operatorname{mean}_{c\in\mathcal C_{train}}y_{g,c},\qquad
r_{g,c}=y_{g,c}-\mu_g,\qquad
\hat y_{g,c}=\mu_g+\hat r_{g,c}.
$$

Tx1 is frozen and supplies cached basal-cell embeddings. Trainable STATE and an
ESM2 adapter predict perturbed expression; a residual head uses five feature
blocks: pooled expression change, response dispersion statistics, gene-specific
basal single-cell statistics, gene embedding and basal context embedding.
The gene mean is fixed preprocessing, not a learned head.

Expression changes subtract basal and predicted bags in the **same output gene
space**. Basal Tx1 embeddings and predicted HVG expression have different widths
and are not subtracted. Cell distributions are supervised without pairing an
individual control cell to an individual perturbed cell.

## 4. Training and selection

The current path is one joint optimization loop. Every update minimizes mean
GeneEffect Huber loss, delta 1. Updates 0, 4, 8, … also minimize response loss:

$$
L_t=L_{GE}+\mathbf{1}[t\bmod4=0]\,L_{response},\qquad
L_{response}=L_{mean\text{-}delta\ MSE}+L_{energy\ distance}.
$$

Response batches contain equal numbers of conditions from the four anchors.
Current per-rank batches are 1024 dependency conditions and 64 response conditions;
response weight is 1. Learning rates for STATE, adapter and head are respectively
$10^{-6}$, $10^{-5}$ and $10^{-4}$. Training, collation and projection base seeds are 0.
Configuration and execution details live in the
[joint design](specs/2026-09-06-modular-joint-training-design.md) and
[runbook](../hpc/README.md).

Validate once per completed epoch. **Only minimum `val_geneeffect_loss` selects
`best.pt` and controls early stopping**, with patience 5 and maximum 50 epochs.
Response loss, total loss and correlations are reported diagnostics. Test restores
checkpoint preprocessing without refitting or optimizer updates. The seed-0 test
has now been observed; it must not become a tuning or checkpoint-selection surface.

## 5. Proposed SL composition

The proposed head combines symmetric pair features derived from predicted
single-gene dependency profiles. Its reference cohort excludes all SL benchmark
contexts. Profiles use residuals; target-context features use absolute GeneEffect;
fixed training gene means expose pan-essentiality as a separate control.
The [SL protocol](03-experiment-protocol.md#6-sl-head-and-controls) records the
24-feature proposal and controls.

Every backbone-derived SL training vector must be generated out of fold, with
that row's context group excluded from response/dependency fitting and fitted
preprocessing. The standalone 226-line checkpoint does not meet this requirement
and is not eligible for direct held-out-SL claims. An SL run still needs its own
eligible backbone fits, complete feature construction and documented selection.

## 6. What the GeneEffect metrics measure

Let $\mathcal C_{eval}$ be the 27 validation or test lines and $\mathcal G_{var}$
the variable-gene set selected from training data (4,447 genes in the completed run).
Only finite observed labels are scored.

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| Huber, RMSE, MAE | Error over all observed gene–line pairs | Accuracy, including prediction scale |
| Absolute Pearson/Spearman | For each line, correlate $\hat y$ and $y$ across genes; average lines equally | Cross-gene dependency ranking within a line |
| Residual Pearson/Spearman | For each variable gene, correlate $\hat r$ and $r$ across lines; average genes equally | Context variation for the same gene |
| Response MSE and energy distance | Average conditions within each anchor, then average four anchors equally | Expression-distribution prediction on held-out anchor conditions |

Explicitly, residual Pearson is

$$
\frac{1}{|\mathcal G_{scored}|}\sum_{g\in\mathcal G_{scored}}
\operatorname{Pearson}_{c\in\mathcal C_{eval}}(\hat r_{g,c},r_{g,c}).
$$

For a fixed gene, subtracting its fixed mean does not change Pearson. The defining
distinction from absolute Pearson is the **axis of correlation**, not centering.
Constant predictions have undefined correlation: gene-mean and copy-prior residual
correlations remain missing, with scored/undefined counts reported, never zero-filled.

Compare gene-mean, K562 copy-prior, nearest-line and context-PCA-ridge on identical
observed keys; contextual baselines use both Tx1 and HVG mean/variance features.
High absolute correlation alone cannot establish context learning. Response
improvement alone cannot establish dependency or SL improvement.

## 7. Current scientific state

The seed-0 joint run completed eight epochs and selected epoch 3 (stored index 2).
Epochs 1–2 used batch 256/rank; epochs 3–8 continued with 1024/rank. This is a
mixed-batch continuation, not an independent batch ablation. Test and all six
baseline variants completed on 2026-09-07 with identical 478,501 observed keys.

| Test method | Huber ↓ | Absolute Pearson ↑ | Residual Pearson ↑ | Residual Spearman ↑ |
| --- | ---: | ---: | ---: | ---: |
| Joint best.pt | 0.01611905 | 0.91050 | 0.05414 | 0.05280 |
| Gene-mean | 0.01613152 | 0.91047 | undefined | undefined |
| Context-PCA-ridge, Tx1 | 0.01625512 | 0.90983 | 0.12161 | 0.11574 |
| Context-PCA-ridge, HVG | 0.01635306 | 0.90920 | 0.07736 | 0.08324 |

Joint Huber improves on gene-mean by only **0.0773%**, while residual correlations
trail simple context baselines. During training, response validation loss fell
45.57% from epoch 1 to 8 without sustained GeneEffect validation improvement.
This establishes a working training/evaluation path, not a useful context-modeling
advantage. There is no SL result.

[Full result, all baselines and provenance](results/joint_geneeffect_seed0/README.md)
are the evidence source. Further model decisions should use validation, including
residual-scale diagnostics and a matched-batch response-supervision ablation.