# Design: a PIE-based stage-one GeneEffect model, its architecture stage and the cell-distribution ladder

**Status:** agreed 2026-10-16 (two grill sessions after the review of the noise diagnosis; checked by a Codex
alignment review and an Opus review). Task 1 done 2026-10-16 ([record](../../results/pie_reproduction/README.md): the
released checkpoints reproduce to rounding); Tasks 2–7 not built. Evidence it starts from: the
[noise diagnosis](2026-10-12-noise-diagnosis-design.md) (its §6 lists the open corrections) and its records
([noise diagnosis](../../results/noise_diagnosis_seed0/README.md), [learning curve](../../results/noise_lines_seed0/README.md)).
PIE: Arc Institute's "Perturbation is Everything" (bioRxiv, 5 October 2026), code
[ArcInstitute/pie](https://github.com/ArcInstitute/pie), data and checkpoints in the Hugging Face
[PIE collection](https://huggingface.co/collections/arcinstitute/pie). Cilantro-SL: Hua, Haber and Ma 2026, official code
in `third_party/cilantro_sl/` on the H20.

## 1. Purpose

Stage one predicts a held-out line's GeneEffect from its basal single cells (blueprint §3). This design builds that
model on PIE used as a frozen virtual cell, establishes its architecture first, then asks what it can read from a
line's cells. In order:

1. **Architecture.** Which stage-one architecture predicts GeneEffect best from a line's mean expression: reading the
   line directly, reading the virtual cell's simulated knockdown of each gene (Cilantro's FiLM design on a trained
   virtual cell instead of Geneformer deletion), several encoders fused by FiLM, or a hybrid?
2. **Cells.** On that architecture, does reading the line's cells add GeneEffect information beyond its mean, and does
   the gain need joint structure within cells or only each gene's own distribution?
3. **Encoders.** Does the Tx1 pipeline (pretrained encoder, raw-count input, its vocabulary) add over a standard HVG
   encoding of the same cells?
4. **Sensitivity.** Can the model recover planted distribution-only terms of known kind and size?

Every arm trains on the same labelled lines as the default prior (§4.1), so its 0.2045 / 0.2025 is a like-for-like
bar.

The noise diagnosis tested one route (cells → a few summaries → a linear model), whose nulls cannot separate absent
information, information lost in summarising, and a reader that could not use it. Questions 2–4 tell these apart as
far as a capable reader and planted terms allow; no null can show that the raw cells hold nothing.

## 2. Roadmap

| Task | What it produces | Needs | Compute |
| --- | --- | --- | --- |
| 1. PIE reproduction | Vendored PIE that re-scores its released checkpoints | — | 1 × H20 |
| 2. Inputs and virtual-cell precompute | Cell bags, Tx1 embeddings, Cellosaurus text, gene sources, the frozen PIE's deltas for every line × gene | 1 | H20 CPU + 4 × H20 |
| 3. Architecture stage | Four architecture arms on the mean channel; the stage-one architecture | 2 | 4 × H20 |
| 4. Gene query | The query's gene sources, on that architecture | 3 | 4 × H20 |
| 5. Data ladder | Cell-reading arms and their controls, every contrast | 4 | 4 × H20 |
| 6. Planted control | Recovery of planted distribution-only terms | 5 | 4 × H20 |
| 7. Cell-count curve | Accuracy against cells per line, repeated draws | 5 | 1 × H20 |

Embedding the Cellosaurus text (Task 2) sends public Cellosaurus records to OpenAI's embedding API and needs the
user's key; it is asked for when Task 2 runs.

## 3. The model

### 3.1 PIE as published

A sample is a (context, perturbation) pair. The encoder's inputs are knowledge-source tokens: the context's
Cellosaurus text and the perturbation's sources (NCBI text, GO and OMIM annotations, ESM2, STRING, DepMap Chronos). A
latent array (512 × 768) cross-attends to them, then self-attention blocks process the latents (Perceiver IO). Every
gene the dataset measures is decoded, independently, by a query built from its NCBI description (`gene_text`) plus a
per-gene evidence vector: the gene's mean `log1p` CP10K over the context's control cells, and its responses in other
contexts or to other perturbations. Heads output DE probability, log fold change and a binned change in mean
expression. Gene queries do not attend to one another, there is no learned per-gene parameter, and the line's
expression never reaches the latents: it enters only as each decoded gene's own evidence.

### 3.2 The frozen virtual cell

- **Checkpoint.** `PIE_replogle_xdataset/best_auprc.ckpt`, frozen and never trained: trained on Tahoe-100M (50 cancer
  lines' control cells under 379 compounds), X-Atlas/Orion (genome-wide CRISPRi in HCT116 and HEK293T), Jiang's
  signalling screens and ARC VCC 25; Replogle held out. Of PIE's Perturb-seq lines, K562, HepG2, Jurkat and HCT116 are
  our training lines; no validation or test line has perturbation data in its corpus. 38 of our training lines are in
  Tahoe; test line SW620 shares a patient with Tahoe's SW480 (chemical exposure only).
- **DepMap source.** PIE reads each perturbed gene's Chronos scores across 1,186 lines in ModelID order (its
  provenance lists them; 375 of our 379 training lines and all 55 held-out lines are columns). For every (line c, gene
  g) pass, these columns are set to 0 (PIE's missing-value convention): the 55 held-out lines, every line sharing a
  patient with one of them (ACH-000023 with validation ACH-000022, ACH-000842 with test ACH-000651; protocol §11),
  and line c's own column with its patient siblings. A training pair then never sees its own label, and training and
  held-out inputs are built the same way. One row of Task 3 masks the source entirely.
- **Context.** For line `c`: PIE's control-mean evidence on one decode panel shared by every line: for a line with
  cells, its definition (the mean over cells of `log1p` CP10K); for a bulk-only line, the bulk profile rescaled to
  PIE's CP10K scale (a bulk profile has no cells to average logs over); (the protocol's GeneEffect genes with a `gene_text` entry), a gene the line's
  source does not list taking the training-line mean, as the prior fills it (PIE's frozen evidence encoder blanks a
  whole row on any NaN); and the line's Cellosaurus text where a record exists.
- **Knockdown of g.** The perturbation tokens of gene g; same-perturbation evidence from g's responses in the
  checkpoint's training data; same-context evidence empty (no responses in our lines).
- **No perturbation.** The same context inputs with every perturbation source and every response-evidence block
  absent (PIE's zero carrier token stands in for the missing perturbation sources). PIE never trained on this input,
  so the reference pass is outside its training distribution; ΔE needs no reference pass (below).
- **Three deltas per (line, gene)**, computed once:
  - **Decoder-feature delta ΔH**: PIE's last decoder-layer gene features, mean-pooled over the decode panel (768-d),
    knockdown pass minus reference pass. The direct analogue of Cilantro's cell-embedding delta.
  - **Profile delta ΔE**: PIE's predicted change in mean expression `Δ̂(c, g)` on the decode panel, read as the
    softmax expectation over its Δp bin centres (PIE's own readout takes the arg-max bin, which quantises small
    changes). The before-state is the line's PIE input `x_c`, the after-state `x_c + Δ̂(c, g)`. Both are stored as
    coordinates on 512 principal components fitted on `Δ̂` of the pilot's training pairs (below), so the basis keeps
    knockdown directions; the variance kept is reported, and a full-panel row on the selective genes checks what the
    compression loses. The state encoder `E` is trained in stage one on the stored states: `ΔE = E(after) − E(before)`.
  - **Latent delta ΔL**: the mean-pooled latent array, knockdown pass minus reference pass. The latents see no
    expression, so ΔL carries the line only through its Cellosaurus text: an expression-blind, text-conditioned
    control; without text it depends on g alone (the gene-identity control).
- **Own-gene slot.** Beside each pooled delta, the pair stores gene g's own decoder feature `h_g(c, g)`, its own
  predicted change `Δ̂_g(c, g)` and its basal value `x_cg`: mean pooling over ~18,000 genes weights g's own term by
  1/18,000, and "g is not expressed" is the most direct context signal a knockdown carries.
- **Storage and order.** Values are stored in fp32, centred within each gene (a line-specific component can sit two to
  three orders of magnitude below the gene component). The pass loops gene-major so g's sources, memory and evidence
  are built once per gene; the vendored model is called directly (`pie infer` needs the checkpoint's split hash).
- **Pilot.** Before the full pass: about 200 genes × every line, timed, reporting for each delta the variance across
  lines within a gene against the variance between genes. People read it before the full pass is launched.

### 3.3 Stage-one architecture arms (Task 3)

Every arm ends in **Cilantro's tail** (`→ 64 → ReLU → 32 → ReLU → 1`, from `FiLMNet128`); its 32-d layer is the
viability embedding `V(c, g)`, saved for every line × gene for stage two. Arms differ only before the tail.

| Arm | Before the tail |
| --- | --- |
| **Direct readout** | A PIE-architecture trunk trained from fresh weights: the line's tokens (§4.3) go into the encoder through per-channel cross-attention; gene g's query (gene encoder + own evidence) decodes a feature `h(c, g)` → Linear 256 → ReLU. It starts small (width 128–256, 32–64 latents, 1–2 blocks per stage, about 1–5M parameters; about 1,000 distinct line inputs feed it) and widens only on validation; 30–50% of input gene tokens dropped per step, weight decay 0.05–0.1, EMA weights, a zero-initialised output so the untrained model is a candidate. Row: shared modules warm-started from the checkpoint (at PIE's width). |
| **Virtual cell + FiLM** (Cilantro) | Δ → Linear 256 → ReLU → FiLM, `[γ, β] = Linear(gene encoder(g))`, `x = γ ⊙ x + β`. Rows: ΔH, ΔE, ΔL; and the best Δ with the DepMap source masked. |
| **Multi-encoder** | As virtual cell + FiLM, with the FiLM condition `gene encoder(g) ⊕ context encoder(c)`; the context encoder pools the line's tokens (§4.3) with the Cellosaurus text. Rows: ΔH, ΔE, ΔL. |
| **Hybrid** | `[h(c, g) ; FiLM(best Δ)]` → Linear 256 → ReLU. |

The gene encoder in Task 3 is a projection of `gene_text`; Task 4 chooses its other sources. Selection is on
validation selective Spearman; the winner carries to Tasks 4–7.

**Matched inputs.** The frozen PIE must read its own input, the mean over cells of `log1p` CP10K, which moves with a
gene's variance and dropout, plus the Cellosaurus text. So that architecture contrasts compare models rather than
inputs, every Task 3 arm reads the same line information: the mean channel's tokens carry both values (`log1p` of the
mean and the mean of `log1p`), and the Cellosaurus text is one more context token for the direct readout and the
multi-encoder's context encoder. Because the text carries the line's disease, site and mutations (PIE renders
Cellosaurus "Sequence Variation"), every arm with text has a text-off row, and text is never credited to expression.

### 3.4 Vendoring

PIE's model imports `pie.data` and `pie.utils`, so its repository is copied at a pinned commit into `third_party/pie/`
(as Cilantro's is in `third_party/cilantro_sl/`), with `LICENSE.md` (CC BY-NC-SA 4.0) and a note naming the commit. The
copy and our additions stay under that licence. PIE pins Python 3.12, `torch==2.10.0` (cu128) and pandas ≥ 3, which the
project's environment (with `arc-state`) cannot share, and the hybrid joins PIE's trunk with our heads in one process.
So the precompute and every stage-one run use one environment, `.venv-pie`: PIE installed editable plus the project's
stage-one modules, which stay free of `arc-state` and run under pandas 3. Task 1 checks the H20 driver (cu128 needs
570 or newer) and this environment. The direct-readout trunk adds `pie.model.geneeffect` beside `pie.model` and reuses
its layers, trunk blocks and evidence encoders. PIE's assets (Tahoe, Orion, sources including the 43 GB `ncbi_text`;
about 100 GB) come through hf-mirror.

## 4. Shared setup

### 4.1 Frame

- **Lines.** Every arm trains on the same lines: the 377 labelled single-cell training lines of the
  [final split](../data/cell-line-geneeffect-split.md) and the bulk-only labelled lines the default prior fits
  (lymphoid and myeloid bulk-only lines dropped, as it drops them); the count is reported in Task 2. It validates on
  the split's 27 lines and scores its 28 test lines once. Cell channels exist only on the 377 single-cell lines and
  are absent on bulk-only lines, so arms differ only in their channels and the default prior, trained on the same
  labelled lines, is a like-for-like bar. (The first grill chose a 377-line frame; the architecture grill moved to
  this one, because a frame of 377 lines starts near 0.17, the learning curve's value at that size, against the
  prior's 0.2045, and a per-gene base cannot change selective Spearman.)
- **Base and target.** `m_g` is the gene's mean measured GeneEffect over the 947 lines the default prior fits, built as
  `src/experiments/context_prior.py` builds them (single-cell training lines admitted by `assert_fit_eligible`,
  bulk-only labelled lines from `load_bulk_lines`, which refuses any overlap with validation and test; lymphoid and
  myeloid lines dropped). The target is `GeneEffect(c, g) − m_g`; the loss is MSE in units of the 377-line residual SD
  `σ_g` over every labelled gene.
- **Scoring.** The tail outputs `z`; the scorer receives GeneEffect `m_g + σ_g · z` and re-centres it on the protocol's
  377-line mean `μ_g` as every prediction is; it never receives `z`. The base is a row of its own (its within-gene R²
  is that of `m_g − μ_g`). One Task 3 row trains the winning family around the 377-line mean instead (target
  `GeneEffect − μ_g`, prediction `μ_g + σ_g · z`): without per-gene parameters the main rows must learn each gene's
  offset `μ_g − m_g`, which moves within-gene R², RMSE and calibration but not selective Spearman.
- **Metric.** The final split's (protocol §6): selective genes (4,323) and residual SD fitted on the 377 lines.
  Headline: selective Spearman and within-gene R² (protocol §14). Beside them: AUPR lift, residual Pearson, per-line
  absolute Pearson, RMSE and the calibration slope (measured residual regressed on predicted, pooled over selective
  genes). Contrasts carry 95% paired line-bootstrap intervals (1,000 resamples).
- **Training.** AdamW, warmup then cosine; `best.pt` chosen on validation selective Spearman only and scored once on
  test; one seed (0); train-versus-validation gaps are reported for every run.

### 4.2 Cells per line

Every line reads all of its cells. Counts per line, from the raw h5ad:

| Split | Lines | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Training | 379 | 11 | 250 | 58,324 |
| Validation | 27 | 78 | 246 | 1,623 |
| Test | 28 | 128 | 307 | 1,573 |

### 4.3 Context channels

Each channel has its own cross-attention, normalised over that channel's tokens only (PyTorch's scaled-dot-product
attention per channel with a padding mask, whose memory-efficient kernels are exact over 58,324 keys and have a native
backward); the channels' outputs are summed. Duplicating a channel's tokens leaves its
output unchanged; there is no sum pooling and no cell-count input.

| Channel | Tokens | Built from |
| --- | --- | --- |
| Mean | one per gene the line's profile lists: gene encoder + embedded value | the line's profile in bulk space, as the default prior reads it: bulk RNA for bulk-only lines; for single-cell training lines and held-out lines, `log1p` of the mean CP10K over the line's cells (log pseudo-bulk, not a mean of `log1p`, which moves with a gene's variance and dropout) mapped by the prior's affine bridge. Row: a separate bulk encoder instead of the bridge (below). An unlisted gene is absent, never a NaN in a shared row |
| Per-gene spread | none; enters gene g's evidence | the excess of the detected fraction and of the SD of `log1p` CP10K over their depth-expected values (`mean_i(1 − exp(−μ_g L_i / T))` for detection; the Poisson trend for SD) |
| Summaries | one token | the noise diagnosis's per-line features: 23 program fractions and means, cluster shares at 8 and 16 |
| QC and cycle cells | one per cell | the cell's log UMI, genes detected, mitochondrial and ribosomal fractions, S and G2/M scores, doublet score |
| HVG cells | one per cell | an MLP over the cell's 2,000 HVGs (`log1p` of `normalize_total`) |
| Mean as one cell | one token through the HVG cell MLP | `log1p` of the line's mean normalised HVG profile: the cell branch's capacity with no distribution |
| Shuffled HVG cells | one per cell, as HVG cells | each HVG's values in that space permuted independently across the line's cells (seed 0): every gene's within-line distribution kept exactly; co-expression within cells destroyed |
| Tx1 cells | one per cell | a projection of the frozen Tx1-3B embedding of the cell (raw UMI, 2,048 tokens) |
| Shuffled Tx1 cells | one per cell, as Tx1 cells | raw counts permuted per gene the same way, then embedded; this keeps raw-count marginals only (library size and Tx1's per-cell binning change) |

The mean channel and the per-gene spread always come from the real cells. **Separate bulk encoder (row):** bulk profiles
and pseudo-bulk profiles enter through two token encoders into one token space, with a source embedding; the 343
single-cell training lines that also have bulk RNA give both views of one line and align the two encoders; held-out
lines go through the pseudo-bulk encoder. Evidence dropout (new, not PIE's, which
drops response evidence only): during training a sample's own-gene evidence is zeroed at a fixed rate, so an unlisted
gene is a case the model has seen.

### 4.4 Gene sources

The gene encoder projects these and sums them; a gene without an entry has that source masked; no per-gene parameter.

| Source | Family | Genes | Origin |
| --- | --- | ---: | --- |
| `gene_text` | function text | 31,684 | PIE_sources (OpenAI `text-embedding-3-large` over NCBI Gene descriptions) |
| ESM C 600M | protein sequence | ~19,200 | computed on the H20 from the reviewed human UniProt sequences PIE's `esm2` uses |
| ESM3-open | sequence and structure | ~19,200 | the same sequences; a swap row for ESM C |
| `string_space` | interaction network | 19,699 | PIE_sources (STRING v12) |
| Gene2vec | co-expression | ~24,000 | the matrix Cilantro's stage one reads; its best gene prior (Fig. 2C) |

PIE's `depmap_gene_effect` is never a gene-encoder source; the self-masked training-side profile of Task 4 is the
only GeneEffect-derived gene source.

## 5. Tasks

### Task 1. PIE reproduction

**Question.** Does the vendored PIE reproduce PIE's released results on our hardware?

**Definition.** The released checkpoints run through the vendored code on PIE's preprocessed data, splits and sources:
`PIE_replogle_xdataset` on Replogle `test_seen` and `test_unseen`, and the four `replogle_wdataset` folds on their
`unseen_ctx` test splits. Targets: each checkpoint's released `metrics_best_auprc.csv` (the wdataset model card:
`unseen_ctx` AUPRC 0.4155 ± 0.0286, Spearman log fold change 0.6568 ± 0.0411 over four folds).

**Reading.** Matching the released metrics to rounding shows the copy, data path and sources are faithful; a mismatch
is a copy or data error, fixed before Task 2.

### Task 2. Inputs and virtual-cell precompute

**Question.** None; the fixed inputs every later task reads.

**Definition.** Preparation on the H20 (the only reader of raw data) writes `data/geneeffect_joint/v2/ladder/` with a
manifest recording the expression space:
- per line, from all cells: the channels of §4.3, their shuffles and per-cell QC values;
- Tx1-3B embeddings of every cell and of every raw-count-shuffled cell (about 2 × 10⁶ cells; roughly 1–2 days on
  4 × H20 by a FLOP estimate, timed on a sample first);
- the gene sources of §4.4 aligned to the GeneEffect gene axis, with coverage per source;
- Cellosaurus 56.0 records fetched on the Mac for every line with an RRID, rendered with PIE's template and embedded
  with `text-embedding-3-large` (the user's key), as PIE's `context_text`;
- PIE's evidence inputs: its datasets' responses for same-perturbation evidence, and the DepMap source with the 55
  held-out columns zeroed (the mapping from its columns to ModelIDs checked first);
- the frozen PIE's deltas and own-gene slots for every line × labelled gene (about 434 × 17,900 = 7.8M pairs; about
  7 hours on 4 × H20 in bf16 by a FLOP estimate, doubled by the masked-DepMap row), after the pilot of §3.2;
- the base `m_g` (§4.1).

**Reading.** Coverage per source, per line and per gene (genes with same-perturbation evidence among the selective
genes) is reported; nothing is chosen from it.

### Task 3. Architecture stage

**Question.** Question 1 of §1.

**Definition.** The arms and rows of §3.3, on the mean channel, with `gene_text` as the gene encoder.

Rows of the winning family: the separate bulk encoder in place of the bridge; and the family trained on the 377
single-cell lines alone (what the bulk-only lines add).

Reference rows:
- the base;
- the default prior, now on the same labelled lines and the same input: the like-for-like linear reference;
- a **shared nonlinear reader of the same 128 components**: one network conditioned on the gene encoder, no per-gene
  parameter, so the linear-to-nonlinear step is measured on fixed inputs;
- lineage only; Cellosaurus text only; own expression `x_cg` only.

Controls for the virtual-cell families:
- **no-Δ FiLM**: the multi-encoder with Δ replaced by a learned constant (the gene × context model without PIE);
- **line-permuted Δ**: for each gene, Δ shuffled across lines (keeps gene identity and each gene's spread of Δ,
  removes its line content);
- a per-gene ridge on line-centred ΔH (is its line signal linearly decodable?);
- ΔL without text (gene identity).

Beside validation, every Task 3 row reports patient-grouped five-fold out-of-fold scores over the 377 single-cell
training lines (377 scored lines against 27), as telemetry.

**Reading.** The arm and row best on validation is the stage-one architecture. Virtual cell + FiLM against direct
readout: does the simulated knockdown add over reading the line? Its ΔL row against its ΔH and ΔE rows: how much of
it needs no expression (gene identity and line text). The hybrid against both: are the two complementary? The masked-DepMap row: what keeping that
source costs or gives.

### Task 4. Gene query

**Question.** Which gene sources should the gene encoder keep?

**Definition.** On the Task 3 architecture: `gene_text`; then + ESM C; + ESM3-open; + STRING; + Gene2vec; then
`gene_text` with the best two of ESM C, STRING and Gene2vec (ESM3-open only ever swaps for ESM C). One more row:
`gene_text` + g's GeneEffect profile over the 947 training-side lines, with line c's own column and its patient
siblings masked for every pair (held-out lines and their siblings never enter); it is how the prior specialises per
gene.

**Reading.** The query best on validation is kept for Tasks 5–7, `gene_text` alone included.

### Task 5. Data ladder

**Question.** Questions 2 and 3 of §1.

**Definition.** The Task 3 architecture with the Task 4 query. Where each channel enters, by the winning family:

| Winner | Token channels (mean, summaries, QC, cells) | Per-gene spread |
| --- | --- | --- |
| Direct readout | its encoder | gene g's query evidence |
| Hybrid | its direct trunk's encoder | gene g's query evidence |
| Multi-encoder | its context encoder | the FiLM condition (gene g's own spread) |
| Virtual cell + FiLM | it has no context path: the ladder runs on the multi-encoder with the winning delta, whose Task 3 row is then the ladder's mean-only baseline | the FiLM condition |

In the virtual-cell families the frozen PIE's deltas always read the mean of `log1p` CP10K and the text, so their
"mean only" already carries that summary; their contrasts measure additions beyond it. The gene-to-cell arm needs a
decoder and runs only when the direct readout or the hybrid wins. Arms, each with the mean channel:

| Arm | Adds |
| --- | --- |
| Mean only | — |
| Per-gene spread | depth-corrected spread in each gene's evidence |
| Summaries | the noise diagnosis's summaries as one token |
| QC and cycle cells | per-cell QC and cycle tokens |
| Mean as one cell | the mean HVG profile through the cell branch |
| HVG cells | HVG cell tokens |
| Shuffled HVG cells | HVG cell tokens of the shuffled cells |
| Tx1 cells | Tx1 cell tokens |
| Shuffled Tx1 cells | Tx1 cell tokens of the shuffled counts |
| HVG cells, gene to cell | HVG cells, plus one decoder cross-attention block from gene queries to the cell tokens (direct-readout family only) |

| Contrast | Asks |
| --- | --- |
| Summaries − mean only | do the noise diagnosis's summaries carry information a capable reader uses? |
| HVG cells − summaries | was information lost in summarising? |
| Per-gene spread − mean only | does each gene's own distribution add, beyond depth? |
| HVG cells − mean as one cell | do the cells add beyond the cell branch's capacity? |
| HVG cells − shuffled HVG cells | does joint structure within cells add? |
| HVG cells − QC and cycle cells | is the cell gain more than depth, quality and cycle? |
| Tx1 cells − HVG cells | does the Tx1 pipeline add over the HVG one? |
| Tx1 cells − shuffled Tx1 cells | joint structure as Tx1 encodes it (its shuffled cells are off its training manifold) |
| Gene to cell − HVG cells | do states specific to gene g add? |

Per-line QC covariates (median UMI, genes per cell, mitochondrial and ribosomal fractions, doublet and ambient scores,
chemistry, pooled flag) are reported beside every contrast, with each contrast's per-line gain against them.

**Reading.** A contrast with a validation interval above zero is a gain of that kind for this model; test is reported
beside it. Summaries ≈ mean only with cells above both is consistent with the noise diagnosis's null coming from
summarising; summaries above mean only, with it coming from the linear reader (the fitting setup differs as well). A tie is informative only for a kind of term Task 6 recovered at a
comparable size.

### Task 6. Planted control

**Question.** Question 4 of §1.

**Definition.** The real cells and labels, with planted terms added to the target of every line with cells (bulk-only
lines keep their real target; they have no cells to carry a term):
`y(c, g) + β · (w_g · h_marginal(c) + w′_g · h_joint(c) + w″_g · h_rare(c))`.
- `h_marginal(c)`: the within-line SD of one HVG (the one whose SD varies most across training lines), kept exactly by
  the HVG shuffle.
- `h_joint(c)`: the within-line correlation of two HVGs (the pair, among the 100 most variable, whose correlation varies
  most across training lines); the shuffle sets its expectation to zero, and its realised shuffled value is reported.
- `h_rare(c)`: the share of the line's cells in a small cluster fitted on training cells (2–5% prevalence), loading on
  about 1% of genes (`w″` sparse).
- Each term is residualised on the 128 pseudo-bulk components and on its own gene's mean–variance and depth trend
  (ridges fitted on training lines), then standardised. All choices and fits use training lines only.
- Loadings: a random linear function of the gene-query sources (decodable from the query); one row with random normal
  loadings instead.
- `β`: the planted terms make up 0.5%, 2% and 5% of the target's variance over training lines and selective genes.
- Arms: mean only, HVG cells and shuffled HVG cells at all three `β` (9 runs); Tx1 cells at 2% (1 run).
- Recovery, jointly: for each held-out line, regress `Δ(c, ·)` over selective genes on the three loadings
  `[w, w′, w″]` (least squares), with `Δ` the run's prediction minus the same arm's unplanted prediction (Task 5); the
  coefficients divided by `β` are `ĥ_marginal(c)`, `ĥ_joint(c)`, `ĥ_rare(c)`. Reported per term: the Pearson and slope
  of `ĥ` on `h` over the 55 held-out lines, with line-bootstrap intervals.

**Reading.** HVG cells recovering all three terms and shuffled HVG cells only `h_marginal`: the model extracts
distribution-only terms of these kinds at that size; it says nothing about other kinds. The mean-only arm's recovery
measures what a nonlinear reader of the mean still sees after residualisation.

### Task 7. Cell-count curve

**Question.** How does accuracy depend on the number of cells read from a line?

**Definition.**
- Held-out lines: the trained mean-only, HVG cells and Tx1 cells arms, and the default prior, score validation and
  test lines read from 10, 25, 50, 100 and 200 cells and from all of them, 16 random draws per size and line (a line
  with fewer cells reads them all and is counted). Every input that depends on the bag is rebuilt from the same draw:
  the mean channel, the cell tokens and, for a virtual-cell-family model, the frozen PIE's deltas (recomputed for 4 of
  the 16 draws and on the selective genes only, to bound the cost).
- Training-bag stability (evaluation only): on training lines with at least 1,000 cells, each arm's prediction from all
  cells against 16 draws of 250 and 500 cells.
- Reported: per size, the mean and spread over draws of every metric; each cell arm's paired gain over the mean-only arm
  on the same draws, with its interval; per-line error against full bag size; the training-bag drift.

**Reading.** A cell arm's paired gain that grows with the cells read is consistent with information beyond the mean
(capacity is held by the mean-as-one-cell arm, channel weight by §4.3's per-channel normalisation). Training-bag drift
shows how much the model leans on statistics only large bags provide.

## 6. Code

| Path | Change | Responsibility |
| --- | --- | --- |
| `third_party/pie/` | vendored, then extended | PIE's repository (CC BY-NC-SA 4.0, pinned commit), its own environment |
| `third_party/pie/src/pie/model/geneeffect.py` | new | the direct-readout trunk: per-channel chunked encoder, gene queries and evidence, gene-to-cell block |
| `src/data/prepare/prepare_ladder_inputs.py` | new | Task 2's caches and manifest |
| `src/data/prepare/pie_precompute.py` | new | Task 2's frozen-PIE deltas (runs in PIE's environment) |
| `src/data/prepare/cellosaurus_context.py` | new | Cellosaurus records and their embeddings |
| `src/data/gene_sources.py` | new | read and align §4.4's sources |
| `src/model/stage_one.py` | new | Cilantro's FiLM layer and tail, the gene and context encoders, the virtual-cell, multi-encoder and hybrid arms |
| `src/training/stage_one_trainer.py` | new | the training loop of §4.1 |
| `src/experiments/stage_one.py` | new | config validation, runs, scoring, contrasts, `results.md` |
| `src/experiments/pie_reproduction.py` | new | Task 1 |
| `configs/stage_one/` | new | one config per run |
| `hpc/run.sh` | edit | `pie-reproduce`, `pie-precompute` and `stage-one` commands |
| `docs/03-geneeffect-protocol.md` | edit | a new section for the stage-one model |

`data` and `model` import neither `training` nor `experiments`. Configs are strict. Checkpoints record the split
membership and load strictly, the vendored PIE included.

## 7. Tests

On synthetic fixtures:
- chunked cross-attention equals unchunked attention, per channel;
- duplicating every cell token leaves the cell channel's output unchanged, with the mean channel present;
- the HVG shuffle keeps every gene's values in its space in each line and removes the correlation between genes;
- an unlisted gene masks only its own evidence;
- a gene absent from training decodes (no per-gene parameter);
- Cilantro's FiLM layer and tail match `third_party/cilantro_sl/nn_helpers` on the same weights;
- the target is GeneEffect minus the 947-line mean; the scorer receives `m_g + σ_g · z`;
- the base, the zeroed DepMap columns and every planted-term fit reject validation and test lines;
- the vendored PIE loads a released checkpoint strictly;
- a two-line end-to-end run precomputes, trains, selects and scores.

## 8. Out of scope, and follow-ups

Out of scope: training or fine-tuning PIE; SL pairs (stage two reads the saved `V(c, g)` later); PIE's chemical
sources as gene sources; any change to the split, the metric or the default prior.

Stage two (its own design) needs, from stage one: out-of-fold `V(c, g)` and predictions for training lines (in-sample
values for training lines would leak into the SL classifier, as the prior's export avoids), and predictions on edited
contexts (gene A's expression set to zero, or the after-state of knocking A down) for the SL definition's "B's
dependency with A lost"; the gene-only virtual-cell arm cannot provide the latter.

Follow-ups, each its own design when taken up:
- bulk-only lines as training lines for stage one;
- a Tx1 cell arm from randomly initialised Tx1 weights, to separate pretraining from Tx1's input pipeline;
- a learning curve with the bridge held at all paired lines, five patient orders and the penalty re-chosen per scale;
- batch: per-line and per-gene distributions of the deposit swap's differences, and a prior row with per-deposit
  centring or a per-deposit bridge.
