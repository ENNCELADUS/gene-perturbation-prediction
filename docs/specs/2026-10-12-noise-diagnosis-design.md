# Design: diagnosing what limits cross-line GeneEffect prediction

**Status:** agreed 2026-10-12 (grill session after the Cilantro-SL review); implemented and run on the H20 on
2026-10-12 ([record](../../results/noise_diagnosis_seed0/README.md)). Task 1 of the research plan ([blueprint §5](../01-blueprint.md#5-research-plan)).
Protocol §14 holds the executable description.

## 1. Why

Every model that reads single cells has trailed the default linear prior (GeneEffect protocol §10): the joint STATE model
trailed the Tx1 context ridge, every trained epoch of the single-cell correction lowered validation, and the single-cell
statistics stack (detected fraction and variance beside the mean) found nothing reliable. The suspicion is that scRNA-seq
technical noise defeats them. Cilantro-SL ([reproduction](../../results/cilantro_reproduction_seed0/README.md)) offers no
remedy: it reads bulk RNA, and what survives its reductions (top-2,048 rank-value encoding, frozen Geneformer, deletion
delta, 32-d bottleneck) is gene identity, not context. The default prior already averages a line's cells into pseudo-bulk,
bridges it onto bulk per gene and fits 128 expression components, so per-cell dropout is largely averaged away before any
fit. This design measures which noise is left and whether it binds, **before** any denoising model is built.

## 2. Shared setup

- **Reference.** The default prior of the final split (run `default_prior_final_split_20261012`): 0.2045 validation /
  0.2025 test selective Spearman, on its own metric.
- **Reporting.** Every row reports validation and test; gains carry 95% paired line-bootstrap intervals; nothing is chosen
  in code (protocol §11). Each task's reading is applied by a person to the tables.
- **Cell order.** A line's cells are ranked by `sha256(ModelID|cell)` (the order `select_context_cells` uses; no seed).
  Half A takes the even ranks, half B the odd ones; the `n`-cell draw takes the first `n`, so draws nest. A line with no
  more cells than the draw reads its full pseudo-bulk and is counted as capped.
- **Pseudo-bulk.** Summed raw UMI of the chosen cells, CPM over their library, log1p; then the prior's space: its genes, a
  gap filled with the training-line mean, quantile-normalised to the training side's profile.

## 3. Tasks

### 3.1 Cell sampling (`cell_sampling`)

**Question.** Do a line's cells sample its expression well enough?

**Definition.**
- *Reliability.* Per space gene, the Pearson across split lines (two or more cells) between half A and half B, over values
  both halves measure, and its Spearman–Brown full-sample value `2r/(1+r)`. Beside it, the bridge's agreement (per gene,
  out-of-fold bridged pseudo-bulk against bulk on the paired lines) and the paired lines' own reliability; their ratio
  `bridge_r / sqrt(reliability)` is the share of the sampling ceiling the bridge reaches. Quartiles over all space genes
  and the selective genes in the space; per-gene table `cell_sampling_genes.csv`.
- *Task effect.* The prior, fitted once on the whole training side (it must reproduce the export's validation and test
  rows), is queried with each draw of the validation and test lines (halves; 10, 25, 50, 100, 200, 400 cells). Scores and
  gains over the full pseudo-bulk; between the halves, per selective gene the Pearson across lines of the two predictions.

**Reading.** Scores flat from 50 to 100 cells up, and reliability well above the bridge's agreement: per-cell sampling of
the pseudo-bulk mean does not bind **for the default prior**, and the bridge's loss is systematic.
<span style="color:red">**TODO (open, 2026-10-14):** the draws query a prior that reads only the mean through a linear
map, so this does not show that more cells carry no more dependency information. A model that reads the distribution of
cells (subpopulations or cell types within a line, rare states, co-expression within cells, nonlinear functions of the
cell mix) could use what a mean discards. `cell_states` (§3.5) and the single-cell statistics stack (protocol §13) test
only linear use of a few per-line summaries, chosen by hand or fitted without labels. Open: does a model that reads
heterogeneous cells directly (a set or attention encoder over cells, cell-type-aware pooling) gain from more cells per
line? Testing it needs that model's own draw curve (10 to all cells) against the default prior.</span>

**Outcome.** Reliability 0.939 (selective 0.947); the bridge reaches 0.484 of that ceiling. Ten cells score −0.0030 /
−0.0060 (intervals span zero); the halves' predictions agree at 0.990. Sampling does not bind; the bridge gap is systematic.

### 3.2 Learning curve (`learning_curve`)

**Question.** Does using up DepMap's labelled lines make the prior predict GeneEffect more accurately, above all in
the part it adjusts per line (beyond the training gene mean)? Redesigned 2026-10-14 (grill session): the first run
scored selective Spearman only and had no baseline per fraction.

**Definition.**
- *Fractions.* Patients of the training side are put in a seeded order; fraction `f` keeps the lines of the first
  `ceil(f · patients)`, so fractions nest: 1/8, 1/4, 1/2, 3/4, 7/8 and the full prior (947 labelled lines: every
  labelled DepMap training-side line with bulk RNA outside the dropped lymphoid and myeloid lineages). The prior is
  refitted on each (bridge, components, data-selected genes; the quantile profile and the metric stay those of the full
  run).
- *Gene mean per fraction.* The prior's own intercept on the fraction's labelled lines (its context-blind part), so
  each row's context gain is measured against a baseline fitted on the same lines.
- *Line-specific part* (each gene's deviation from the training gene mean), the headline: selective Spearman (ranks
  lines per gene; gain over the full prior) and **within-gene R²**, `1 − Σ(predicted − measured residual)² / Σ
  measured residual²` pooled over the selective genes and the scored lines in residual-SD units (the training gene mean
  scores 0; negative when deviations are too large), with paired line-bootstrap gains over the row's gene mean and over
  the full prior. Beside them: AUPR lift for dependent lines and residual Pearson.
- *Whole prediction* (secondary; the gene mean dominates it): per-line Pearson across genes and RMSE, prior and gene
  mean.
- *Cilantro-SL.* The released stage-one checkpoint of `as_released_gc95M`, not retrained
  (`hpc/cilantro/stage_one_predictions.py`), predicts the validation and test pairs it covers (a gene among the line's
  top 2,048 expressed). It trained on every labelled line, so these predictions are in-sample: an upper bound for it.
  It, the full prior and its gene mean are scored on those pairs only.

**Reading.** The step from 7/8 to all lines, with an interval above zero on selective Spearman and on within-gene R²:
using up DepMap still helps, and the curve's slope at the end says how much another source of labelled lines might
add. Within-gene R² above zero over the gene mean at every fraction: the prior's per-line adjustment is right in size,
not only in order. Cilantro at or below the gene mean even in-sample: it carries no line-specific GeneEffect to
compare against.

**Outcome** (2026-10-14, [record](../../results/noise_lines_seed0/README.md)). Selective Spearman climbs from 0.1132
/ 0.1331 at an eighth of the lines to 0.2051 / 0.2007 at 7/8 and then stops: the last eighth (167 lines, 108
labelled) adds −0.0006 [−0.0062, +0.0050] / +0.0018 [−0.0034, +0.0067] and within-gene R² +0.0001 / +0.0009
(intervals spanning zero). The full prior's within-gene R² is 0.043 / 0.042 against −0.003 / +0.002 for its gene mean,
above it at every fraction; whole-prediction Pearson moves only 0.910 → 0.914. Cilantro, in-sample on its covered
pairs, scores 0.169 / 0.196 selective Spearman and R² 0.039 / 0.049 against the prior's 0.215 / 0.221 and 0.052 /
0.051 (prior minus Cilantro R² +0.013 [−0.010, +0.038] / +0.002 [−0.015, +0.022]); it does carry line-specific signal
(R² over its gene mean +0.042 / +0.046). More DepMap lines no longer help this prior.

### 3.3 Deposit confound (`deposit_confound`)

**Question.** Does a training line's deposit explain what the prior gets wrong on it, beyond lineage?

**Definition.** Out-of-fold error `E = truth − export` on the labelled single-cell training lines, over the selective
genes. The deposit is the final split table's `expression_source` (the 226's names mapped to the identity map's source
keys, so `Tahoe-100M DMSO` and `tahoe100m` are one deposit); deposits with fewer than `min_lines` lines pool as `other`.
Per gene, η² (the share of `E`'s variance between deposits), averaged over genes; the null permutes deposits among lines
of one lineage (`permutations`, seed 0). Reported beside it: η² by lineage (null permuting among all lines), the same for
the residual itself (a label no deposit touched), and each deposit's error mean square relative to all lines.

**Reading.** Deposit η² at its null: deposit batch does not explain the error. An excess on the error that the residual
also shows comes from which lines each deposit holds, not from batch.

**Outcome.** Error η² 0.0365 against a null of 0.0315 (p 0.001); the residual shows nearly the same excess (0.0045 against
0.0050). Lineage explains far more (0.0927 against 0.0640). Per-deposit error runs from 0.73 to 1.29 of the average.

### 3.4 Deposit swap (`deposit_swap`)

**Question.** Does the prior's prediction for a line change when its cells come from another deposit, more than between
two halves of one deposit, and does its accuracy?

**Definition.** Lines: labelled single-cell training lines with a second deposit (§4). Per patient-grouped fold of the
export, the prior is refitted without the fold and queried with the fold's lines read four ways: primary pseudo-bulk (must
reproduce the export), second deposit, half A and half B, each through the bridge fitted on the primary pseudo-bulk of the
paired lines outside the fold. Reported: each reading's score over these lines and its gain over primary; per selective
gene the Pearson across lines between primary and second predictions against that between the halves; the same for the
space genes' normalised inputs, over measured values only.

**Reading.** Second-deposit predictions agreeing with primary as well as the halves agree: deposit batch does not bind.
If they agree less but score no worse, batch moves the prediction without costing accuracy. If they score worse, batch
correction, not cell denoising, is the lever.

**Outcome.** 99 lines. Inputs agree at 0.375 against 0.865 for the halves, predictions at 0.764 against 0.982, yet the
second deposit scores +0.0066 [−0.0052, +0.0184] over primary. Batch moves the input and the prediction, not the accuracy.

### 3.5 Cell states (`cell_states`)

**Question.** Does a line's distribution of cell states carry dependency information that its mean does not?

**Definition.** Per-line features, fitted on the labelled single-cell training lines without labels:
- *Programs.* Tirosh et al. 2016's S and G2/M sets and the 21 of Gavish et al. 2023's 41 malignant meta-programs with no
  cell-line study among their sources (the other 20 count Kinker et al. 2020's CCLE pools, whose cells include validation
  and test lines; Tirosh's sets cite synchronised HeLa, a test line, and are kept and noted in
  `configs/context_prior/reference/provenance.json`). Per cell, the mean log-normalised expression (`log1p(x · T / L_cell)`)
  of a program's genes minus the cell's mean over the scoring set. The scoring set is the pseudo-bulk's genes, one set for
  every line; a gene a source does not list scores as unexpressed (some builders drop zero-count genes, and scoring over
  each source's own list moved identical cells by build policy). The share of the set each source lists is reported, and
  lines under 90% are named. Per program, a threshold at the 90th percentile of the training lines' pooled scores (lines
  weighted equally): `program_fractions` is a line's fraction of cells above it, `program_means` (the control) its mean
  score.
- *Clusters.* The prepared context cells (`basal_hvg`, 128 per line, repeated when fewer), centred on their line's mean,
  projected on the leading principal components of the training lines' centred cells; K-means (seed 0) fitted on training
  cells; `clusters_<k>` is a line's share of cells nearest each centroid.
- *Stack.* Per arm and penalty, a ridge of every gene on the line's features, fitted on what the out-of-fold export leaves
  on the training lines (a missing label counts as zero), added to the export on validation and test. Gains over the prior,
  and `program_fractions` over `program_means` at the same penalty.

**Reading.** `program_fractions` above `program_means` with a validation interval above zero: state heterogeneity carries
information beyond the mean, and the second-wave distributional models start from that arm.

**Outcome.** Best row +0.0010 on validation; fractions over means +0.0000 [−0.0009, +0.0009]. Nothing added.

### 3.6 Cilantro encodings

**Question.** Do Cilantro-SL's input reductions (rank-value encoding, top-2,048 truncation) clean the prior's input?
Geneformer's deletion delta is not tested (GPU, and no line-specific signal in the reproduction).

**Definition.** A bridge remedy `rank_value` (`configs/context_prior/cilantro_encodings.yaml`, `hpc/run.sh prior`). Both
sources go through one transform before the affine bridge: back to linear scale (`expm1`), divided by each gene's nonzero
median over the training-side bulk rows (Geneformer's normalisation), ranked within the line. `encoding: rank_value` gives
`1 − rank/G` (no `top_k`) or `max(0, 1 − rank/k)` (top `k`); `encoding: expression` with `top_k` keeps the normalised
expression of the same top `k` genes and zeroes the rest, so the two differ only in what the kept genes carry. Out-of-fold
bridge diagnostics take medians from the bulk outside each fold. Rows: rank alone, rank at `k = 2048`, truncation alone at
`k = 2048`.

**Reading.** An encoding above the reference row: rank or truncation cleans the input; otherwise it does not.

**Outcome.** Rank alone ties (−0.0022 [−0.0076, +0.0033]); both top-2,048 rows lose on validation (−0.0192, −0.0142).

## 4. Data steps

- **Pinned programs** (Mac): `src/data/prepare/build_state_programs.py` writes `cell_cycle.tsv`, `meta_programs.tsv` and
  `meta_program_sources.tsv` under `configs/context_prior/reference/`; the config names the programs it reads.
- **Second deposits** (plan on the Mac, build on the H20): `src/data/prepare/plan_second_sources.py` writes
  `configs/data/second_source_lines_26Q1.json`, one accepted raw-count source per labelled single-cell training line other
  than its own and from another accession, chosen by the membership policy (condition class, raw counts, cells, name). It
  plans 99 lines (36 Kinker to MIX-seq, 13 Tahoe-100M to MIX-seq, the rest over GEO; 89 from wave-1 sources, 10 from
  wave 2). `build_extra_single_cell_sources --second-source PLAN` builds them into their own directory, one wave at a time;
  the training caches do not change.
- **Noise preparation** (H20, CPU, called by the runner): `prepare_noise_inputs` writes `<prepared_root>/noise/`:
  `subsamples.parquet` (halves and draws of every split line) and `programs/<ModelID>.npz` (per-cell program scores) with
  their manifest; `prepare_second_source` writes `noise/second_source/` from the second-deposit directory, growing as files
  are added. Preparation stays the only raw-data reader.

## 5. Out of scope

Any new model, any change to the training caches or the split, Geneformer, and SL. A diagnosis result is single-gene
dependency evidence on held-out lines; it estimates no genetic interaction.
