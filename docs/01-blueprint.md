# Research Blueprint: Context-Conditioned Synthetic-Lethality Ranking

Updated 2026-10-04. This document defines the research task and its claim boundaries
and nothing else. [Related work](02-literature-review.md) explains the prior art;
[GeneEffect protocol](03-geneeffect-protocol.md) holds the model, training, metrics and
results of the implemented single-gene track; [SL protocol](04-sl-ranking-protocol.md)
defines the separate pair-label experiment.

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

| Task | Input and output | Where it lives |
| --- | --- | --- |
| Perturbation response | Basal cells + one gene → predicted expression distribution | Auxiliary supervision in the GeneEffect protocol |
| GeneEffect | Basal cells + one gene → single-gene dependency | [GeneEffect protocol](03-geneeffect-protocol.md) |
| SL ranking | Basal cells + two genes → pair ranking score | [SL protocol](04-sl-ranking-protocol.md); proposal, no model run or SL result |

The implemented model predicts single-gene outcomes. It has no double-knockout
output and estimates no genetic-interaction quantity. A pair classifier is a
proposed downstream use, not an existing capability.

## 2. Data and generalization

The two benchmarks have different assignments and cannot substitute for each other.

| Benchmark | Training | Validation | Test | Authority |
| --- | --- | --- | --- | --- |
| GeneEffect, 226 members | 172 members, 170 labeled | 27 lines | 27 lines | [Fixed split](../configs/benchmarks/cell_line_geneeffect_226_split.json), [data card](data/cell-line-geneeffect-226.md) |
| SL, nine contexts | K562, Jurkat, OVCAR8, HAP1, HT29 | A549 | 22RV1, PC9, HeLa | [Fixed split](../configs/benchmarks/context_screen_v2_split.json), [data card](data/sl-context-screen.md) |

DepMap lines outside the 226 with 26Q1 bulk RNA may join the GeneEffect training side of
the linear context prior ([membership](../configs/benchmarks/extra_bulk_lines_26Q1.json),
[card](data/extra-bulk-lines-26q1.md)): their bulk RNA and other-omics labels fit context
encoders, and their GeneEffect labels fit the prior once the learning-curve rule passes.
Every line sharing a patient with a validation or test line is excluded from every fit.

GeneEffect labels are from the pinned DepMap 26Q1 release. PC9 and HeLa are the
two unlabeled GeneEffect training members; neither participates in supervised
fitting. K562, Jurkat, HepG2 and HCT116 carry genetic-perturbation response data and
belong to the labeled training cohort.

SL labels mean experimental screen hit or screened non-hit in a named context.
They are inferred from unanimous aggregate source rows (`silver_inferred`), not
independently reconstructed per-context evidence. The current SL table has
incomplete filter attribution, degenerate A549 validation labels and shared
PC9/HeLa aggregate labels. These limitations constrain any future evaluation.

## 3. SL composition (intent)

The proposed pair head combines symmetric features derived from predicted
single-gene dependency profiles of the two genes. Its reference cohort excludes all SL
benchmark contexts, and fixed training gene means expose pan-essentiality as a separate
control. Every backbone-derived SL training vector must be generated out of fold, with
that row's context group excluded from response and dependency fitting and from fitted
preprocessing. The standalone 226-line checkpoints do not meet this requirement and are
not eligible for direct held-out-SL claims. Features, controls and selection are
specified in the [SL protocol](04-sl-ranking-protocol.md#6-sl-head-and-controls).

## 4. Claim boundaries

- A GeneEffect result is single-gene dependency evidence. It is not an SL result and
  estimates no genetic interaction; no SL result exists.
- Context claims require residual evaluation against context-blind priors. High absolute
  correlation alone cannot establish context learning, and response improvement alone
  cannot establish dependency or SL improvement.
- A sigmoid score is not a calibrated probability of biological synthetic lethality.
- Fitting, normalisation and calibration use training lines only; no per-context
  prediction z-scoring; one label-independent pair universe; missing scores stay missing.
- Held-out-context results are qualified by the Tx1 Tahoe-100M pretraining exposure of
  the held-out lines, and response results by STATE's pretraining exposure to K562,
  HepG2 and Jurkat.
- Results that use extra lines are a training-data change, scored on the unchanged
  validation and test lines.
- Validation lines' bulk RNA is read only by the oracle diagnostic; test lines' bulk RNA is
  never read.
- The seed-0 GeneEffect test split has been observed once and is not a tuning or
  selection surface. Further model decisions use validation.

## 5. Current status

A joint GeneEffect model has been trained, tested and baselined once (seed 0). Its Huber
loss beats the context-blind gene mean by 0.08%, and its residual correlations trail
simple context baselines: a working path, not a result
([record](../results/joint_geneeffect_seed0/README.md)). That run predates the
expression-space change; the model, expression space, training, metrics, response-model
comparison and the single `all` run are specified in the
[GeneEffect protocol](03-geneeffect-protocol.md), and curated evidence is under
[`results/`](../results/).
