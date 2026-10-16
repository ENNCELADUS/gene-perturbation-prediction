# Design: a PIE-derived GeneEffect model and the cell-distribution ladder

**Status:** agreed 2026-10-16 (grill session after the review of the noise diagnosis); not built. Evidence it starts
from: the [noise diagnosis](2026-10-12-noise-diagnosis-design.md) and its records
([noise diagnosis](../../results/noise_diagnosis_seed0/README.md), [learning curve](../../results/noise_lines_seed0/README.md)).
PIE: Arc Institute's "Perturbation is Everything" (bioRxiv, 5 October 2026), code
[ArcInstitute/pie](https://github.com/ArcInstitute/pie), data and checkpoints in the Hugging Face
[PIE collection](https://huggingface.co/collections/arcinstitute/pie).

## 1. Purpose

The noise diagnosis tested one route: a line's cells → a few hand-made summaries → a linear model. Its nulls cannot say
whether the cell distribution holds no GeneEffect information, whether the summaries lost it, or whether the linear
reader could not use it. This design tells them apart as far as a capable reader and planted controls allow; no null
can show that the raw cells hold nothing. It answers, in order:

1. Does the trunk reading a line's mean expression (every gene, with gene knowledge) beat the linear prior trained on
   the same lines?
2. Does reading the line's cells add GeneEffect information beyond its mean, and does that gain need the joint
   structure of cells (co-expression within a cell) or only each gene's own distribution?
3. Does the Tx1 pipeline (pretrained encoder, raw-count input, its vocabulary) add over a standard HVG encoding of the
   same cells, under the same reader?
4. Can the reader recover planted distribution-only terms of known kind and size (a positive control)?

The reader is one trunk for every arm, taken from PIE and reproduced first, so the arms differ only in what they read.
It is also the model the project builds on afterwards.

## 2. Roadmap

| Task | What it produces | Needs | Compute |
| --- | --- | --- | --- |
| 1. PIE reproduction | Vendored PIE that re-scores the released checkpoints and retrains one fold | — | 2 × H20 |
| 2. Inputs | Cell bags, Tx1 embeddings of every cell, shuffled cells, pseudo-bulk, gene sources | — | H20 CPU + GPU |
| 3. GeneEffect trunk and gene query | PIE adapted to GeneEffect; mean-only arm; the gene query chosen | 1, 2 | 4 × H20 |
| 4. The ladder | Seven arms and three reference rows, every contrast | 3 | 4 × H20 |
| 5. Planted control | Recovery of planted distribution-only terms of known size | 3 | 4 × H20 |
| 6. Cell-count curve | Accuracy against cells per line, repeated draws | 4 | 1 × H20 |
| 7. Context text (deferred) | The mean-only arm with the line's Cellosaurus text | 3, an OpenAI key | 4 × H20 |

Tasks 1 and 2 run in parallel. Task 7 waits for the user's key and approval (it sends public Cellosaurus records to
OpenAI's embedding API).

## 3. Architecture (from PIE)

**PIE as published** (paper §3, `src/pie/model/pie.py`). A sample is a (context, perturbation) pair. The encoder's
inputs are knowledge-source tokens: the context's Cellosaurus text and the perturbation's sources (NCBI text, GO and
OMIM annotations, ESM2, STRING, DepMap Chronos). A latent array cross-attends to them, then self-attention blocks
process the latents (Perceiver IO). Each measured gene is decoded independently by a query built from its NCBI
description (`gene_text`, OpenAI `text-embedding-3-large`) plus a per-gene evidence vector. The vector holds the gene's
mean log-normalised expression in the context's control cells and its responses in other contexts or to other
perturbations. Gene queries do not attend to one another and there is no learned per-gene parameter, so any gene
decodes. Heads output DE probability, log fold change and a binned Δ.

**What it cannot read for GeneEffect.** The context's whole transcriptome reaches gene g only through g's own control
mean and the Cellosaurus text. GeneEffect of g depends on the rest of the line (paralogs, pathway state, lineage), and
PIE never reads cells.

**The GeneEffect trunk** keeps PIE's trunk, decoder and evidence encoders, trained from fresh weights on GeneEffect (no
PIE checkpoint is loaded), and changes five things:

- **Context tokens into the encoder.** The latents cross-attend to the line's expression, given as tokens (§4.3), in
  chunks with a running softmax: exact attention over every token, so a 58,324-cell bag fits in memory. Each channel
  has its own cross-attention, normalised over that channel's tokens only, and the channels' outputs are summed into
  the latents. Duplicating a channel's tokens leaves its output unchanged, so the cell channel's weight does not grow
  with the number of cells.
- **A sample is a line.** No perturbation tokens: the knocked-out gene is the decoded gene, so gene knowledge goes into
  the query (§4.4).
- **One head.** GeneEffect delta in residual-SD units replaces the DE, log fold change and Δ heads.
- **Evidence for gene g.** The gene's own mean in the line, and, in one arm, its spread across cells. No response
  evidence (no response data here).
- **Evidence dropout**, as PIE: during training a sample's per-gene evidence is zeroed at PIE's rate, so a gene the
  line's source does not list is a case the model has seen.

**Vendoring.** PIE's model imports its own `pie.data` and `pie.utils`, so the whole repository is copied at a pinned
commit into `third_party/pie/` (the `src/pie` package, its tests, `pyproject.toml`, `uv.lock` and `LICENSE.md`), with a
note naming the commit, and installed editable into the runs' environment (PIE needs Python 3.12). The copy and our
edits stay under CC BY-NC-SA 4.0. Task 1 runs the copy unchanged through its own CLI; Task 3 adds the GeneEffect model
beside `pie.model` and reuses its layers, trunk blocks and evidence encoders.

## 4. Shared setup

### 4.1 Frame

- **Lines.** Every arm trains on the 377 labelled single-cell training lines of the
  [final split](../data/cell-line-geneeffect-split.md), validates on its 27 and scores its 28 test lines once.
  Bulk-only lines enter only the base (below) and the default prior's reference row.
- **Target.** `y(c, g) = GeneEffect(c, g) − m_g`. `m_g` is the gene's mean measured GeneEffect over the 947 lines the
  default prior fits, built as `src/experiments/context_prior.py` builds them: single-cell training lines admitted by
  `assert_fit_eligible` and bulk-only labelled lines from `load_bulk_lines` (which refuses any overlap with the split's
  validation and test lines), lymphoid and myeloid lines dropped. The prediction is `m_g + ŷ(c, g)`. The loss is standardised
  MSE in units of the 377-line residual SD over every labelled gene; each step decodes a random 2,048 of a line's
  labelled genes, and evaluation decodes every gene.
- **Scoring.** The head outputs `z` in units of the 377-line residual SD `σ_g`. The scorer receives GeneEffect,
  `m_g + σ_g · z`, and re-centres it on the protocol's 377-line mean `μ_g` as every prediction is; it never receives
  `z`. The base alone is a row of its own: its within-gene R² is that of `m_g − μ_g`, not zero.
- **Metric.** The final split's (protocol §6): gene means, selective genes (4,323) and residual SD fitted on the 377
  lines. Headline: selective Spearman and within-gene R² (protocol §14). Beside them: AUPR lift, residual Pearson,
  per-line absolute Pearson and RMSE. Contrasts carry 95% paired line-bootstrap intervals (1,000 resamples).
- **Training.** AdamW, warmup then cosine, Accelerate on the 4 GPUs; `best.pt` is chosen on validation selective
  Spearman only (protocol rule) and scored once on test. One seed (0). Width, depth and latent count start at PIE's
  `replogle_wdataset` values, are set once on the mean-only arm's validation score and held for every arm.

### 4.2 Cells per line

Every line reads all of its cells (uncapped). Counts per line, from the raw h5ad:

| Split | Lines | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Training | 379 | 11 | 250 | 58,324 |
| Validation | 27 | 78 | 246 | 1,623 |
| Test | 28 | 128 | 307 | 1,573 |

Pooling is size-invariant per channel (§3); no sum and no cell-count input. Task 6 reports error
against bag size, since the large training bags come from the large extra deposits.

### 4.3 Context channels

| Channel | Tokens | Built from |
| --- | --- | --- |
| Mean | one per gene the line's source lists: projected gene embedding (the query's gene sources) + embedded value | `log1p` of the mean CP10K over the line's cells (log pseudo-bulk); a gene the source does not list is absent, not zero |
| Per-gene spread | none; enters gene g's evidence | the SD of `log1p` CP10K across the line's cells and the detected fraction |
| HVG cells | one per cell | an MLP over the cell's 2,000 HVGs (`log1p` of `normalize_total`, the protocol's space) |
| Tx1 cells | one per cell | a projection of the frozen Tx1-3B embedding of the cell (raw UMI, 2,048 tokens) |
| Mean as one cell | one token through the HVG cell MLP | `log1p` of the line's mean normalised HVG profile: the cell branch's capacity with no distribution |
| Shuffled HVG cells | one per cell, as HVG cells | each HVG's values in the consumed space (`log1p` of `normalize_total`) permuted independently across the line's cells (seed 0): every gene's within-line distribution in that space is kept exactly; co-expression within cells is destroyed |
| Shuffled Tx1 cells | one per cell, as Tx1 cells | the raw counts permuted per gene the same way, then embedded by Tx1. Tx1 reads raw counts, and the shuffle changes each cell's library size and Tx1's per-cell binning, so this control keeps raw-count marginals only |

The mean channel is `log1p` of the mean, not PIE's mean of `log1p`: a mean of logs moves with a gene's variance and
dropout, so it is not a mean-only input. The mean channel and the per-gene spread always come from the real cells,
also in the shuffled arms.

### 4.4 Gene query

The query for gene g is a projection of its gene sources plus its evidence vector; no per-gene parameter. Task 3
starts from PIE's `gene_text` and chooses one or two further sources:

| Source | Family | Genes | Origin |
| --- | --- | ---: | --- |
| `gene_text` | function text | 31,684 | PIE_sources (OpenAI `text-embedding-3-large` over NCBI Gene descriptions) |
| ESM C 600M | protein sequence | ~19,200 | computed on the H20 from the reviewed human UniProt sequences PIE's `esm2` source uses |
| `string_space` | interaction network | 19,699 | PIE_sources (STRING v12) |
| Gene2vec | co-expression | ~24,000 | the matrix Cilantro-SL's stage one reads; Cilantro's best gene prior (Hua et al. 2026, Fig. 2C) |

Excluded: PIE's `depmap_gene_effect` (it is the label, held-out lines included) and `ncbi_text`. A gene without a
source entry has that source masked.

## 5. Tasks

### Task 1. PIE reproduction

**Question.** Does the vendored PIE reproduce PIE's own numbers on our hardware?

**Definition.**
- **Re-score.** The four released `replogle_wdataset` checkpoints (`<held-out line>/best_auprc.ckpt`), run through the
  vendored code on their `unseen_ctx` test splits (PIE_splits), on the preprocessed Replogle–Nadig data and PIE_sources.
- **Retrain.** One fold, HepG2 held out, trained from the vendored code with its published recipe
  (`configs/experiment/replogle_wdataset.yaml`: one node, two devices, gradient accumulation 4) on two of the H20s,
  and scored the same way.
- **Targets.** The model card's `unseen_ctx` metrics (four folds: AUPRC 0.4155 ± 0.0286, Spearman log fold change
  0.6568 ± 0.0411) and each fold's released `metrics_best_auprc.csv`.

**Reading.** Re-scored checkpoints matching their released per-fold metrics to rounding show the copy and data path
are faithful; a mismatch is a copy or data error and is fixed before Task 3. The retrained fold is reported beside the
four-fold spread.

### Task 2. Inputs

**Question.** None; the fixed inputs every later task reads.

**Definition.** One preparation step on the H20 (the only reader of raw data) writes `data/geneeffect_joint/v2/ladder/`
with a manifest recording the expression space:
- per line: all cells as HVG `float16` (the protocol's space) and their shuffle in that space; the mean channel, the
  mean HVG profile and the per-gene spread over the source's listed genes;
- Tx1-3B embeddings of every cell and of every raw-count-shuffled cell (GPU, chunked per line, seeded collator as now);
- the gene sources of §4.4, aligned to the GeneEffect gene axis with a coverage table;
- the 947-line base `m_g` (§4.1).

**Reading.** Coverage per source and per line is reported; nothing is chosen from it.

### Task 3. GeneEffect trunk and gene query

**Question.** Does the trunk reading the mean beat the linear prior on the same lines, and which gene sources should
the query keep?

**Definition.** The trunk of §3 with the mean channel only (the **mean-only arm**). Gene-query rows on it:
`gene_text`; `gene_text` + ESM C; `gene_text` + STRING; `gene_text` + Gene2vec; `gene_text` + the best two others.
The **matched linear prior** is a reference row: a ridge on 128 components of the 377 lines' log pseudo-bulk, no bulk
rows and no bridge, its penalty set on validation.

**Reading.** The query that scores best on validation is kept for every later task (one or two sources besides
`gene_text`). The mean-only arm against the matched linear prior compares the two readers as built (every gene token,
gene knowledge and nonlinearity together against 128 components and a ridge), the first question of §1.

### Task 4. The ladder

**Question.** Questions 2 and 3 of §1.

**Arms** (every arm has the mean channel and the Task 3 query):

| Arm | Adds |
| --- | --- |
| Mean only | — (from Task 3) |
| Per-gene spread | each gene's SD across cells and detected fraction, in its evidence |
| Mean as one cell | one token of the mean HVG profile through the cell branch |
| HVG cells | HVG cell tokens |
| Shuffled HVG cells | HVG cell tokens of the shuffled cells |
| Tx1 cells | Tx1 cell tokens |
| Shuffled Tx1 cells | Tx1 cell tokens of the shuffled counts |

**Contrasts:**

| Contrast | Asks |
| --- | --- |
| Per-gene spread − mean only | does each gene's own distribution add? |
| HVG cells − mean only | do the cells add anything? |
| HVG cells − mean as one cell | do the cells add beyond the cell branch's capacity? |
| HVG cells − shuffled HVG cells | does joint structure within cells add? |
| Tx1 cells − HVG cells | does the Tx1 pipeline (pretraining, raw-count input, vocabulary) add over the HVG one? |
| Tx1 cells − shuffled Tx1 cells | does joint structure add when read through Tx1? |
| Shuffled HVG cells − per-gene spread | do cell tokens of independent genes read more than two statistics per gene? |

**Reference rows:** the 947-line gene mean (the base), the matched linear prior (Task 3) and the default prior
(`results/final_split_seed0/`, 947 lines through the bridge).

**Reading.** A contrast with a validation interval above zero is a gain of that kind of information for this reader;
test is reported beside it. A tie is read against Task 5: it is informative only for a kind of term the arm recovered
there at a comparable size.

### Task 5. Planted control

**Question.** Do the mean-only, HVG cells and shuffled HVG cells arms recover planted distribution-only terms of known
kind and size?

**Definition.** The real cells and labels, with a planted term added to the target of every line:
`y(c, g) + β · (w_g · h_marginal(c) + w′_g · h_joint(c))`.
- `h_marginal(c)`: the within-line SD, across the line's cells, of one HVG in the consumed space: the HVG whose
  within-line SD varies most across training lines. It is a single gene's distribution, so the HVG shuffle keeps it
  exactly.
- `h_joint(c)`: the within-line Pearson correlation, across the line's cells, of two HVGs: the pair, among the 100
  HVGs of largest across-line variance, whose within-line correlation varies most across training lines. The shuffle
  sets its expectation to zero; its realised value under the shuffle is reported.
- Both choices and fits use training lines only. Each term is residualised on the 128 pseudo-bulk components by a ridge
  fitted on training lines, then standardised.
- `w_g`, `w′_g`: fixed random normal gene loadings (seed 0).
- `β`: the planted terms make up 0.5%, 2% and 5% of the target's variance over training lines and selective genes.
- 3 arms × 3 `β` = 9 trained runs. Recovery per term, by projection on its loading: for each held-out line,
  `ĥ(c) = Σ_g w_g Δ(c, g) / (β Σ_g w_g²)` over selective genes, with `Δ` the run's prediction minus the same arm's
  unplanted prediction (Task 4); reported, the Pearson and the slope of `ĥ` on `h` over the 55 held-out lines, with
  line-bootstrap intervals (and the same with `w′` for `h_joint`).

**Reading.** HVG cells recovering both terms and shuffled HVG cells only `h_marginal`: this reader extracts
distribution-only terms of these kinds at that size; it says nothing about other kinds. The mean-only arm's recovery
measures how much of a term a nonlinear reader of the mean still sees after linear residualisation.

### Task 6. Cell-count curve

**Question.** How does each arm's accuracy depend on the number of cells read from a held-out line?

**Definition.** The trained mean-only, HVG cells and Tx1 cells arms (Task 4), unchanged, score validation and test
lines read from 10, 25, 50, 100 and 200 cells and from all of them, 16 random draws per size and line (a line with
fewer cells than the size reads them all and is counted). The mean channel is rebuilt from the same draw. Reported:
per size, the mean and spread over draws of every metric; each cell arm's paired gain over the mean-only arm on the
same draws, with its interval; and per-line error against the line's full bag size.

**Reading.** A cell arm's paired gain over the mean-only arm that grows with the cells read is consistent with
information beyond the mean (capacity is held by the mean-as-one-cell arm, channel weight by §3's per-channel
normalisation). The spread over draws is the sampling uncertainty the noise diagnosis did not measure.

### Task 7. Context text (deferred)

**Question.** How much does the line's Cellosaurus record add to the mean-only arm?

**Definition.** The mean-only arm plus one context token: the line's Cellosaurus 56.0 record rendered with PIE's
template and embedded with `text-embedding-3-large`. PIE_sources holds 88 contexts; our 434 lines need new embeddings.

**Reading.** Text is line metadata (disease, site, sex, sequence variants), not a basal profile; its gain is reported
on its own and never credited to expression.

## 6. Code

| Path | Change | Responsibility |
| --- | --- | --- |
| `third_party/pie/` | vendored, then extended | PIE's repository (CC BY-NC-SA 4.0, pinned commit), installed editable |
| `third_party/pie/src/pie/model/geneeffect.py` | new | context tokens, per-channel chunked encoder, gene queries and evidence, GeneEffect head |
| `src/data/prepare/prepare_ladder_inputs.py` | new | Task 2's caches and manifest |
| `src/data/gene_sources.py` | new | read and align the gene sources of §4.4 |
| `src/training/ladder_trainer.py` | new | the training loop of §4.1 |
| `src/experiments/ladder.py` | new | config validation, runs, scoring, contrasts, `results.md` |
| `src/experiments/pie_reproduction.py` | new | Task 1 |
| `configs/ladder/` | new | one config per run |
| `hpc/run.sh` | edit | `pie-reproduce` and `ladder` commands |
| `docs/03-geneeffect-protocol.md` | edit | a new section for the trunk and the ladder |

`data` and `model` import neither `training` nor `experiments`. Configs are strict (unknown or missing keys raise).
Checkpoints record the split membership and load strictly, the vendored code included.

## 7. Tests

On synthetic fixtures:
- chunked cross-attention equals unchunked attention;
- the HVG shuffle keeps every gene's values in the consumed space in each line and removes the correlation between
  genes;
- duplicating every cell token leaves the cell channel's output unchanged, with the mean channel present;
- the scorer receives `m_g + σ_g · z`, never `z`;
- a gene absent from training decodes (no per-gene parameter);
- the target is GeneEffect minus the 947-line mean, and the base and every planted-term fit reject validation and test
  lines;
- the vendored model loads a released checkpoint strictly;
- a two-line end-to-end run trains, selects and scores.

## 8. Out of scope, and follow-ups

Out of scope: SL pairs; perturbation-response data and evidence; PIE's chemical sources; any change to the split, the
metric or the default prior.

Follow-ups, each its own design when taken up:
- bulk-only lines as training lines for the trunk, once the ladder reads;
- a Tx1 cell arm from randomly initialised Tx1 weights, to separate pretraining from Tx1's input pipeline;
- a learning curve with the bridge held at all paired lines and several patient orders;
- a deposit batch-correction test: whether correcting the 99 lines' second deposits recovers accuracy.
