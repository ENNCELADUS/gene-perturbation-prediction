# Plan: PIE as the counterfactual predictor, and single cells in the line encoder

Date: 2026-11-12. This plan orders the redesign list of the model evaluation (2026-11-12, before the backbone was
built) with the results of the two backbone runs ([round zero](../../results/sl_backbone_round_zero_seed0/README.md),
[buffer-aware reader](../../results/sl_buffer_aware_reader_seed0/README.md)). It is a plan, not a protocol: each round
gets its own short spec when it starts.

## 1. Where the model is

- **The SL rule works only at the level of the pair.** A linear reader, with or without each gene's buffer, gives a
  pair the same score in the real line and in the average line (−0.004 on the partner pairs).
- **Stage one is far below what the data permit.** The reader has a per-line residual Spearman of 0.18 to 0.19 on
  the SL lines; two DepMap screens of one line agree at 0.246, so about 0.5 can be reached.
- **No deep learning arm reaches the linear model** (best 0.177 / 0.172 against 0.215 / 0.228). The arms had no
  per-gene parameter: on the same 128 components, the shared reader scores 0.127.
- **PIE was used as a feature extractor.** Its predicted change adds nothing beside the linear model that a
  line-permuted change does not. In PIE, the line's expression does not reach the latents; it enters only as each
  decoded gene's own value. With its DepMap source masked, PIE's Spearman is 0.45 on seen perturbations and 0.11 on
  unseen ones.
- **The single cells are used only as their mean.** Pseudo-bulk input is below bulk input for the same lines (0.205
  against 0.222), and per-gene statistics of the cells add nothing in linear form. The neural arms that read cells
  were designed and never run.

## 2. The model this plan builds

SL of (A, B) in a line = B's dependency in the line after A is lost, minus B's dependency in the line as it is.

| Part | Job | Today | Target |
| --- | --- | --- | --- |
| Line encoder | the line's state from its basal single cells | mean of the cells, mapped to bulk space | a reader of the cells that is not below bulk |
| Counterfactual predictor | the line's state after A is lost | A's own expression set low | PIE's predicted profile after A's loss |
| Dependency reader | a gene's dependency in a given state | linear model | per-gene reader on a shared line encoder, equal to the linear model at the start |
| SL score | the change of B's dependency | as the rule above | unchanged |

PIE has one job: the counterfactual. The reader has one job: dependency in a state. The single cells have one job:
the state of the line.

## 3. Rounds (one part each, each with a control and a decision)

### Round A. PIE as the counterfactual predictor

- **Question.** Does PIE's predicted profile after A's loss say which line needs B?
- **Measurement first, no training.** On PIE's own task (the four Replogle folds, one line held out each): PIE's
  predicted response against the mean measured response to the same perturbation in the other lines. If PIE is not
  above that mean, the frozen PIE holds no line-specific knowledge of a response.
- **Build.** The frozen cross-dataset PIE (DepMap source masked, as before) predicts the profile change of each SL
  line for each lost gene. The cost is about 7 ms for one line and gene on one GPU, so about 3 GPU hours. The
  existing reader scores B on the profile before and after. No other part changes.
- **Rows** (validation / test, with the paired differences of the harness):

  | Row | What it shows |
  | --- | --- |
  | A's own expression set low (existing) | the editor without PIE |
  | PIE's predicted profile | the module |
  | PIE's predicted profile of another line (lines permuted) | whether PIE's part is line-specific |
  | Measured profile after A's loss, own line against another line (reference) | the top for any counterfactual predictor |

  The reference row uses the lines with measured Perturb-seq and SL labels: K562 and Jurkat (Replogle with
  Horlbeck2018), HCT116 and HEK293T (X-Atlas/Orion with their screens). All these datasets are on the H20.
- **Decision.**

  | Result | Next |
  | --- | --- |
  | PIE's profile above the simple edit and above its permuted version | PIE keeps the job; go to round B |
  | PIE equals its permuted version, the measured own-line profile is above the other line's | PIE must become line-specific: round D before more use |
  | The measured own-line profile equals the other line's | the profile after A's loss does not carry the line effect through this reader: round B first, then repeat |

### Round B. The per-gene reader (the stage-one deep learning part)

- **Why.** The deep learning arms lost because they had no per-gene parameter, not because of the line encoder.
- **Build.** Per-gene readout weights on a shared line encoder, set equal to the linear model at the start, so the
  first epoch cannot be below it. Gene knowledge (function text, protein, PIE's perturbation embedding) predicts a
  starting value for a gene's weights, and each gene keeps a deviation that is shrunk toward it. Train on the scored
  quantity (the standardized residual of the selective genes), regularized for about 1,000 lines.
- **Then** add a nonlinear line encoder and keep it only where validation improves.
- **Control.** The linear model itself (the reader at its start), and the reader with lines permuted.
- **Target.** Above 0.215 / 0.228 on the stage-one frame; the room is up to about 0.5.
- **Decision.** If the trained reader is above the linear model, it replaces it in the backbone and round A's rows
  are scored again. If not, the reader stays linear and the work goes to rounds C and D.

### Round C. Single cells in the line encoder

1. **Do not lose against bulk.** Close the gap between pseudo-bulk and bulk input for the same lines, and read each
   line in the genes it measures (the reader edits both genes of 44% of the held-out rows today, about 75% then).
2. **A reader of the cells.** A set encoder over the line's cells feeds the round-B reader (the HVG-cell and
   Tx1-cell arms of the earlier design). Before any claim: the planted-signal test, then the shuffled-cells control
   (each gene's distribution kept, co-expression inside cells destroyed).
3. **One single-cell input for the SL score, with its hypothesis.** A and B expressed in the same cells: a buffer
   needs both genes in one cell, and a mean cannot show it. Control: shuffled cells.

A clean negative result here answers blueprint Task 2 and is reported as such.

### Round D. A line-specific PIE (only if round A asks for it)

- A light adapter lets the line's profile reach PIE's latents; the rest stays frozen.
- Fine-tune with two tasks together: PIE's own response loss on its perturbation data (the anchor), and GeneEffect
  through the round-B reader.
- **Test.** On held-out lines, the response prediction is above the mean response of the other lines. Then round A's
  rows are scored again.

## 4. Order

A first: it is frozen inference plus the existing reader, and its measurement decides whether round D is needed.
B second: it is the largest build and the fix of stage one. A and B do not depend on each other and can run at the
same time. C follows B, because the cell reader needs the per-gene reader. D is conditional.

## 5. Not planned

More variants of the linear reader, the pair's training labels as a model input, changes to the benchmark labels,
and copy number or mutation inputs (on the training lines they gave no more than expression).
