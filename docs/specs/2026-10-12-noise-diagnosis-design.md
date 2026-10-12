# Design: diagnosing what limits cross-line GeneEffect prediction

**Status:** agreed 2026-10-12 (grill session after the Cilantro-SL review); implemented on branch
`feat/noise-diagnosis`; not yet run on the H20. Task 1 of the research plan ([blueprint §5](../01-blueprint.md#5-research-plan)):
before any single-cell model with its own noise handling is built, measure which noise limits the default linear prior
(GeneEffect protocol §10), the bar every model has to clear. Protocol §14 holds the executable description.

## 1. Why

Every model that reads single cells has trailed the default prior: the joint STATE model trailed the Tx1 context ridge, every
trained epoch of the single-cell correction lowered validation, and the single-cell statistics stack (detected fraction and
variance beside the mean) found nothing reliable. The suspicion is that scRNA-seq technical noise over thousands of genes
per cell defeats them. Cilantro-SL ([reproduction](../../results/cilantro_reproduction_seed0/README.md)) gives no remedy:
it reads bulk RNA, not single cells; its rank-value encoding keeps each line's top 2,048 genes, a frozen Geneformer pools them,
a deletion delta and a 32-d supervised bottleneck shrink them, and what survives is gene identity: its stage one equals a gene
mean from other lines, and its delta alone has R² −0.255. The default prior already averages a line's cells into pseudo-bulk,
maps it onto bulk with a per-gene affine bridge and fits 128 expression components, so per-cell dropout is largely averaged
away before any fit. Which noise is left, and whether it binds, is measured here before anything is built against it.

## 2. Decisions (2026-10-12)

1. **Diagnose first.** No denoising model until the diagnosis is read.
2. **Noise sources measured:** per-cell sampling, deposit batch, too few lines, and noise in the GeneEffect target.
3. **Cell-state heterogeneity** is measured beside them: does a line's distribution of cell states carry dependency
   information its mean does not?
4. **Reference programs** are the main state arm: Tirosh et al. 2016's S and G2/M sets and the 21 of Gavish et al. 2023's 41
   malignant meta-programs with no cell-line study among their sources. The other 20 count Kinker et al. 2020's CCLE pools
   among their sources, whose cells are those of validation and test lines (the exposure ruled out for Kinker's own
   programs). Tirosh's sets cite synchronised HeLa (a test line) among earlier sources; kept and noted
   (`configs/context_prior/reference/provenance.json`, `checks`).
5. **K-means clusters** on line-centred cells, fitted on training lines, are the second state arm.
6. **Cilantro's input reductions** are tested on the prior's input: rank-value encoding and top-2,048 truncation. Geneformer's
   deletion delta is not (GPU, and no line-specific signal in the reproduction).
7. **Scope:** every layer, including the data steps (second-deposit builds, per-cell program scores, the per-screen
   GeneEffect download).

## 3. What is measured

Everything is read against the default prior of the final split (run `default_prior_final_split_20261012`: 0.2045
validation / 0.2025 test selective Spearman) on its own metric. Every row reports validation and test; gains carry 95%
paired line-bootstrap intervals; nothing is chosen in code (protocol §11).

| Section | Question | Data |
| --- | --- | --- |
| `cell_sampling` | Do a line's cells sample its expression well enough? Split-half reliability of pseudo-bulk per gene, against the bridge's agreement with bulk; the prior's validation and test scores from half the cells and from 10 to 400 cells | subsampled pseudo-bulk (new preparation) |
| `learning_curve` | Would more lines help? The prior refitted on nested patient-grouped fractions of the training side | existing |
| `deposit_confound` | Does a training line's deposit explain what the prior gets wrong on it? η² of the deposit on the out-of-fold error, against permutations within lineage | existing |
| `deposit_swap` | Does the prior's prediction for a line change when its cells come from another deposit, more than between two halves of one deposit? | second-deposit pseudo-bulk (new builds) |
| `cell_states` | Do cell-state proportions add to the prior, and beyond the state means? Ridge on per-line state features, stacked on the out-of-fold export | per-cell program scores (new preparation), prepared context cells |
| `screen_replicates` | How much line-specific signal can the label carry? Agreement of two screens of one line, against the prior | DepMap per-screen GeneEffect (manual download) |
| Cilantro encodings | Does Cilantro's rank-value encoding or top-k truncation of the input clean it? | existing (prior runner) |

## 4. Definitions

**Cell order.** A line's cells are ranked by `sha256(ModelID|cell)`, the order `select_context_cells` uses: deterministic,
no seed. Half A takes the even ranks, half B the odd ones; the `n`-cell draw takes the first `n` (nested). A draw with at
least as many cells as the line has is not written; the line then reads its full pseudo-bulk and is counted as capped.
Every draw is pseudo-bulk in the prior's transform (summed raw UMI of the draw's cells, CPM over their library, log1p),
then the prior's space: the space's genes, a gap filled with the training-line mean, quantile-normalised to the training
side's profile.

**`cell_sampling`.**
- *Reliability:* per space gene, Pearson across every split line with two or more cells between half A and half B, over
  the values both halves measure (a gap's training-mean fill would pair two equal values), and its Spearman–Brown
  full-sample value `2r/(1+r)`; beside it, the bridge's per-gene correlation (out-of-fold bridged pseudo-bulk against bulk
  on the paired lines) and the paired lines' own split-half reliability. If the bridge's agreement sits far below
  `sqrt(reliability)` the bridge's loss is systematic, not sampling. Quartiles over all space genes and the selective genes
  in the space, plus a per-gene table (`cell_sampling_genes.csv`).
- *Task effect:* the reference prior fitted once on the whole training side (checked against the export's validation and
  test rows, which it must reproduce), queried with each draw of the validation and test lines: metrics of §6 and the gain
  over the full pseudo-bulk; and, between the halves, per selective gene the Pearson across lines of the two predictions.

**`learning_curve`.** Patients of the training side are put in a seeded order; fraction `f` keeps the lines of the first
`ceil(f · patients)`, so fractions nest. The reference row is refitted on them (bridge, components, data-selected genes;
the quantile profile and the metric's definitions stay those of the full run) and scored with its gain over the full prior.

**`deposit_confound`.** Out-of-fold error `E = truth − export` on the labelled single-cell training lines, over the
selective genes; the deposit is the final split table's `expression_source` (the 226's names mapped to the identity
map's source keys, so `Tahoe-100M DMSO` and `tahoe100m` are one deposit), deposits with fewer than `min_lines` lines
pooled as `other`. Per gene the share of `E`'s variance between deposits (η²), macro-averaged; the null permutes deposit
labels among lines of one lineage (`permutations`, seed 0); lineage is reported beside it, its null permuting among all
lines. The same for the residual itself, and the error's mean square per deposit relative to all lines.

**`deposit_swap`.** Lines: labelled single-cell training lines with a second deposit (§5). Per patient-grouped fold of the
export, the reference prior is refitted without the fold and queried with the fold's lines read four ways: primary
pseudo-bulk (must reproduce the export), second deposit, half A and half B, each through the bridge fitted on the primary
pseudo-bulk of the paired lines outside the fold. Reported: scores of each reading over these lines, gains over primary,
and per selective gene the Pearson across lines between primary and second predictions against that between the halves;
the same for the space genes' normalised inputs, over measured values only.

**`cell_states`.** Per-line features, fitted on the labelled single-cell training lines, label-free:
- *Programs:* per cell, the mean log-normalised expression (`log1p(x · T / L_cell)`, the prepared space) of a program's
  member genes, minus the cell's mean over every gene of the scoring set. The scoring set is the pseudo-bulk's genes
  (DepMap's protein-coding symbols), one set for every line; members outside it (an IG gene, a lncRNA) are not part of the
  program here, and the manifest records how many are in. A gene the source does not list scores as unexpressed: the
  breast-atlas and CCLA builders drop each line's zero-count genes, and scoring over each source's own list moved identical
  cells by build policy (0.84 against 0.07 in the review's example). The share of the scoring set and of each program a
  line's source lists is reported, and lines listing under 90% of the set are named. Per program, a threshold at the 90th percentile of the
  training lines' pooled scores, each line weighted equally; `program_fractions` is the fraction of a line's cells above
  it, `program_means` (the control) the line's mean score.
- *Clusters:* the prepared context cells (`basal_hvg`, 128 per line, repeated when a line has fewer), centred on their
  line's mean, projected on the leading principal components of the training lines' centred cells, then K-means fitted on
  the training cells (seed 0); `clusters_<k>` is the share of a line's cells nearest each centroid.
- *Stack:* per arm and penalty, a ridge of every gene on the line's features (the prior's shared ridge), fitted on what the
  out-of-fold export leaves on the training lines (a missing label counts as zero) and added to the export on validation
  and test. Gains over the prior alone, and `program_fractions` over `program_means` at the same penalty.

**`screen_replicates`.** DepMap's per-screen gene effect (`ScreenGeneEffect.csv`) and its map to models
(`CRISPRScreenMap.csv`, columns `ScreenID` and `ModelID`), downloaded by hand: the DepMap portal now sits behind a browser
check. Lines with two or more screens, outside validation and test, pair their first two screens by `ScreenID`. Per selective
gene the Spearman across lines between the two screens; its macro mean is the replicate agreement of one screen, and
`sqrt(2r/(1+r))` approximates the ceiling a perfect predictor reaches against a two-screen label. Reported beside the
prior's validation score. The two file schemas are checked when read; a missing column stops the section.

**Cilantro encodings.** A bridge remedy, `rank_value` (`configs/context_prior/cilantro_encodings.yaml`, run with
`hpc/run.sh prior`). Both sources go through one transform before the affine bridge: values back to linear scale
(`expm1`), divided by each gene's nonzero median over the training-side bulk rows (Geneformer's normalisation), then ranked
within the line. `encoding: rank_value` gives each gene `1 − rank/G` (no `top_k`) or `max(0, 1 − rank/k)` (top `k`);
`encoding: expression` with `top_k` keeps the normalised expression of the same top `k` genes and zeroes the rest
(truncation alone), so the two encodings differ only in what the kept genes carry. The out-of-fold bridge diagnostics take
medians from the bulk outside each fold. Settings: rank alone, rank with `k = 2048`, truncation alone with `k = 2048`.

## 5. Data steps

- **Pinned programs** (Mac): `src/data/prepare/build_state_programs.py` writes `cell_cycle.tsv`, `meta_programs.tsv` and
  `meta_program_sources.tsv` under `configs/context_prior/reference/`. The config names the programs it reads.
- **Second deposits** (plan on the Mac, build on the H20): `src/data/prepare/plan_second_sources.py` writes
  `configs/data/second_source_lines_26Q1.json`: for every labelled single-cell training line, one accepted raw-count source
  of the identity map other than the one it was built from (the final split table's `expression_source`) and from another
  accession, chosen by the membership policy (condition class, raw counts, cells, name), among the extras spec's sources.
  It plans 99 lines (36 Kinker to MIX-seq, 13 Tahoe-100M to MIX-seq, the rest over GEO deposits; 89 from wave-1 sources,
  10 from wave 2). `build_extra_single_cell_sources --second-source PLAN` builds them into their own directory, one wave
  at a time, lines of the 226 included; nothing in the training caches changes.
- **Noise preparation** (H20, CPU, called by the runner): `prepare_noise_inputs` writes `<prepared_root>/noise/`:
  `subsamples.parquet` (halves and cell draws of every split line) and `programs/<ModelID>.npz` (per-cell program scores
  over every basal cell) with their manifest; `prepare_second_source` writes `noise/second_source/` from the
  second-deposit directory, growing as files are added. Preparation stays the only raw-data reader.
- **Per-screen GeneEffect** (by hand): `ScreenGeneEffect.csv` and `CRISPRScreenMap.csv` of DepMap Public 26Q1, downloaded in
  a browser, copied to `data/sl_dependency_v0/raw/depmap/` on the H20 with their SHA-256 and added to the DepMap card.

## 6. Reading (not code)

- Scores flat from 50 to 100 cells up, and split-half reliability well above the bridge's agreement: per-cell sampling does
  not bind; the bridge's loss is systematic.
- Second-deposit predictions agreeing with primary as well as the halves agree with each other, and no deposit η² above its
  null: deposit batch does not bind. Otherwise batch correction, not cell denoising, is the lever.
- A learning curve still rising at the full training side: more labelled lines are the lever.
- The prior's 0.20 against the replicate ceiling: how much of the reachable signal the prior already has.
- `program_fractions` above `program_means` by an interval above zero on validation: state heterogeneity carries dependency
  information beyond the mean; then the second-wave distributional models start from the arm that carries it.
- A Cilantro encoding above the reference row: rank or truncation cleans the input; otherwise it does not.

## 7. Out of scope

Any new model, any change to the training caches or the split, Geneformer, and SL. A diagnosis result is single-gene
dependency evidence on held-out lines; it estimates no genetic interaction.
