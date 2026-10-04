# Research Blueprint: Context-Conditioned Synthetic-Lethality Ranking

Updated 2026-10-04. This document is the research statement: the question, why it is posed
on cell lines, and the route to an answer. The rules the experiments follow are in the
[GeneEffect protocol](03-geneeffect-protocol.md#11-rules) and the
[SL protocol](04-sl-ranking-protocol.md#8-rules); [related work](02-literature-review.md)
explains the prior art.

## 1. The question

Two genes are synthetic lethal (SL) when losing either one alone is tolerated and losing both
kills the cell. Which pairs are lethal depends on the cell: its lineage, its expression
programs, the paralogs and pathway partners it happens to express. Curated SL graphs record
which pairs scored in screens; they say little about a cell nobody has screened.

The task: given the basal single-cell transcriptome of a cancer cell line and a pair of genes,
rank the pairs that are synthetic lethal in that line. The model sees the line only through its
unperturbed cells. No dependency or SL measurement of the line is an input, and no SL graph
enters the features.

The research question is whether what a model learns about a cell's context, from
perturbation responses and single-gene dependencies, ranks SL partners better than gene
identity, pan-essentiality and simple context predictors do.

## 2. Why held-out cell lines

A pair that is lethal in one line and harmless in another can only be told apart by the
context of the line. Splitting by cell line makes the model earn its context signal: every
test line is a context the model has never seen, so memorising a gene, a pair or a line's
identity does not help. The unit that has to generalise is the cell line. Genes are shared
across train and test.

This is also the clinical shape of the problem: a new tumour or model line arrives with
expression data and no screen.

## 3. The approach

**Step one: a GeneEffect context model.** Predict each gene's DepMap single-gene dependency
in a held-out line from its basal single cells and the gene's protein sequence. A frozen
single-cell foundation model (Tx1) encodes the cells into a context; STATE, a perturbation-
response model, supplies the predicted expression change when the gene is knocked out; a head
combines them. The target is the residual over the training gene mean, so the part of a
dependency that every line shares is fixed and the part that depends on context is what is
learned. This track is implemented and is scored on a 226-line benchmark with 172 training,
27 validation and 27 test lines, against context-blind priors and context predictors of
increasing strength, including a closed-form linear prior from bulk and pseudo-bulk
expression.

**Step two: SL ranking built on it.** For a pair $(a,b)$ in line $c$, the SL head reads
symmetric features of the two genes' predicted dependency profiles across lines and their
predicted dependencies in $c$, and outputs a ranking score. Pair labels come from
experimental screens in nine named contexts, split by context. The comparison that matters
is against the same head with the context information removed.

Single-gene predictions are the input to step two; they are not themselves SL results.

## 4. Where the work stands

- **GeneEffect model.** Trained, tested and baselined once at seed 0, before the
  expression-space change; it is a working path whose residual correlations trail simple
  context baselines ([record](../results/joint_geneeffect_seed0/README.md)). The current
  revision (head, objective, STATE treatment; [design](specs/2026-10-03-geneeffect-revision-design.md),
  [head design](specs/2026-10-03-geneeffect-head-revision-design.md)) is specified in the
  [GeneEffect protocol](03-geneeffect-protocol.md).
- **Linear context prior.** A closed-form prior from expression alone scores selective
  Spearman 0.223 on validation and test against 0.130 and 0.121 for the Tx1 context ridge
  ([design](specs/2026-10-04-context-generalization-design.md),
  [results](../results/context_prior_seed0/README.md)). The next run compares remedies for
  the single-cell-to-bulk bridge that gene-level features need
  ([plan](specs/2026-10-04-bridge-remedies-plan.md)).
- **SL ranking.** Specified in the [SL protocol](04-sl-ranking-protocol.md); no head has
  been trained.

Curated evidence is under [`results/`](../results/); data cards are under
[`docs/data/`](data/).
