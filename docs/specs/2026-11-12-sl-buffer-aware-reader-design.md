# Buffer-aware reader: the next module of the SL backbone (dependency reader slot)

Date: 2026-11-12. Follows the backbone run ([record](../../results/sl_backbone_round_zero_seed0/README.md), SL protocol
§11). This spec changes one slot, the dependency reader. The line encoder, the context editor and the pair scorer stay
as they are.

## 1. What the backbone run shows

The record's numbers agree with the host's reports (`outputs/sl/conditional/conditional_seed0`,
`outputs/sl/benchmark_baselines/benchmark_backbone_seed0`). Three readings of the record change:

1. **The conditional score equals its context-ablated score by construction.** The reader is affine, so a term is the
   fall of A in the line times a response of B to A that is the same in every line. In an anchor list the fall is one
   number for the whole list, so the ranking cannot change with the line. This result is not evidence about the reader.
2. **The reader edits only 44% of the held-out rows because of the gene space, not because of the genes.** The space
   is the genes that every held-out line measures in pseudo-bulk (6,523). Each held-out line measures 8,295 to 18,768
   genes. If each row is read in the genes of its own line, 75% of the held-out rows have both genes.
3. **The stage-one ceiling can be measured.** `ScreenGeneEffect.csv` and `CRISPRScreenMap.csv` are on the H20 (since
   2026-10-08): 227 lines have two or more screens, 136 of them in the SL split, 12 held out.

A read-only probe on the host (2026-11-12; training lines only for the feature tables, no held-out label read with a
feature) gives the reason for the next module.

**Table 1.** Inside one study and one pair, how often a line that called the pair SL has the higher value than a line
that did not (concordance; 0.5 is no signal). Combinatorial screens of the training lines, pairs with both labels
(6,231 to 6,662 pairs), the mean of the pair and the mean of the line removed. 95% interval from a bootstrap of pairs.

| Value of A plus B in the line | Concordance |
| --- | ---: |
| Measured GeneEffect (lower is SL) | 0.385 [0.376, 0.394] |
| GeneEffect predicted by the reader, out of fold (lower is SL) | 0.415 [0.407, 0.425] |
| Bulk expression | 0.549 [0.538, 0.559] |
| Copy number (line mean not removed) | 0.541 [0.528, 0.552] |
| Damaging mutation (line mean not removed) | 0.499 [0.497, 0.500] |
| Random | 0.495 [0.487, 0.503] |

In the form of the benchmark (inside one training line's screen, among the pairs that no other training line of the
study called SL, ranked by the line's deviation from the pair's mean over those lines), the mean AUROC is 0.616 for the
measured GeneEffect, 0.580 for the reader's prediction and 0.566 for expression (52 to 55 screens).

**Table 2.** The chance that a second label of a pair is SL, given the first label (genetic screens, one label per
study, line and pair).

| Second label | First label SL | First label not SL |
| --- | ---: | ---: |
| Same study, other line | 0.275 | 0.027 |
| Other study, same line | 0.267 | 0.027 |
| Other study, other line | 0.273 | 0.040 |

Readings:

- The line-specific part of the labels follows the single-gene dependency of A and B in the line. The reader already
  predicts most of what the measured GeneEffect gives (0.585 against 0.615 after the direction is turned).
- **Most of that signal is single-gene dependency.** In the training lines, 49% of the SL rows have a gene that is a
  dependency alone in that line (DepMap GeneEffect below −0.5), against 27.5% of the other rows. Without those rows
  the measured GeneEffect falls from 0.630 to 0.563 (line mean not removed), and without any gene below −0.25 to
  0.515 [0.496, 0.534]. A gain of a score over its context-ablated score is thus context evidence only if it also
  holds on the rows without such a gene (section 3).
- The pair's training labels are not a model input. They say nothing about a new pair and hold no mechanism; the
  training SL fraction stays a control row.
- **The labels stay as they are (decided 2026-11-12).** A rule that leaves a row unlabelled when a gene is a
  dependency alone removes 16% of the genetic rows and 31% of the SL rows, and it was rejected. SLKB's rule reaches 9
  studies, replaces most of their SL rows and does not make two studies agree more. A label that differs between
  lines is the task. A label that differs between studies is kept (1.3% of the training rows) and is read only inside
  its own study.
- Copy number gives no more than expression, and damaging mutations give nothing. The copy number and mutation module
  is not next.

## 2. The biology and the module

A cell needs gene B only when no other gene does the work of B. The genes that can do it are the paralogs of B and
the other members of its complex: the buffer of B. A pair (A, B) is SL in a line when A is the buffer that the line
still has. It is not SL in a line where another paralog is expressed (a buffer stays after A is lost), or where A is
already low (B is then a dependency alone).

| Slot | Backbone | This module |
| --- | --- | --- |
| Dependency reader | 128 expression components and 50 data-selected genes; affine in expression | the same, plus the buffer of each gene in the line (the existing partner block) |
| Context editor | A's expression set to the training side's low level | unchanged |
| Pair scorer | B's predicted dependency with A lost minus as it is, both directions; then the backbone head | unchanged |

- **Buffer inputs** of gene B in a line (`src/linear_model/gene_features.py`, block `partners`): the lowest
  expression among its paralogs, their sum, the closest paralog's expression, the mean expression of its complex
  members, and the fraction of its partners that are low. The weights are fitted on GeneEffect over the training-side
  lines (block set `partners_and_selected`). No SL label is read.
- **Why the score now depends on the line.** The lowest paralog and the low fraction are not linear. When A is lost,
  the buffer of B changes much if A was the only expressed partner, and little if another partner is expressed.
  With the affine reader, the response of B to A was the same number in every line.
- **Limits known before the run.** The partner block does not change the accuracy of the predicted GeneEffect
  (−0.0004 / −0.0009 against the default reader, `results/default_linear_model_followups_seed0`); it is used here
  for its response to the edit. It acts only on pairs in the paralog and complex tables and on genes in the
  expression space (44% of the held-out rows).

## 3. Rows and contrasts (validation / test)

- Rows: conditional score of the buffer-aware reader; the same score context-ablated (training-side mean profile);
  the backbone head on the new terms and its context-ablated version; beside the backbone run's rows and the controls
  (training SL fraction, pan-essentiality).
- The subset where the module can act: pairs in which A is a paralog or a complex member of B. Every row is also
  given on this subset, with its row count.
- Paired difference per line, with the harness's line bootstrap: conditional score minus its context-ablated score,
  on all pairs and on the subset. The same difference is given on the rows where no gene of the pair is a
  dependency alone in the line (lines with GeneEffect).
- Stage-one metrics of the new reader beside the default reader, and the stage-one ceiling from two DepMap screens
  of one line (owed by the backbone run).

## 4. What the result decides

| Conditional score against its context-ablated score, on the partner pairs | Next module |
| --- | --- |
| Above (interval above zero on validation, same sign on test) | Line encoder: each line read in the genes it measures (coverage 44% to about 75%), so that the buffer reaches more rows |
| Equal | The expression of the buffer does not say which line calls the pair: context editor (PIE's profile after A's loss), then single-cell features |

## 5. Code

- A linear model config with the block set `partners_and_selected` as the reader, and an SL config that points to it.
- `src/experiments/sl/conditional.py`: the terms come from the reader's predictions of each line's edited rows (the
  affine shortcut and its check are removed, because the reader is no longer affine).
- `src/experiments/sl/benchmark_baselines.py`: the partner-pair subset and the paired difference named in the config.
- Tests on synthetic fixtures: when A is the only expressed paralog of B, the loss of A moves B's prediction; when
  another paralog is expressed, it moves less; the context-ablated score is equal in every line.
- Docs in the same change: SL protocol (new section), `hpc/README.md`, `src/README.md`, `README.md`, `CLAUDE.md`; a
  results record after the run.

## 6. Run (CPU, H20)

```bash
hpc/run.sh sl-conditional configs/sl/conditional_buffer.yaml --run-id conditional_buffer_seed0
hpc/run.sh sl-baselines configs/sl/benchmark_baselines.yaml --run-id benchmark_buffer_seed0
```

Check on the output: on the pairs outside the partner tables, the conditional score equals the backbone run's score
up to the refit of the reader.

## 7. Not in this module

The gene space of the line encoder, copy number and mutation, PIE's profile, and a continuous target. The pair's
training labels as a model input are not planned.
