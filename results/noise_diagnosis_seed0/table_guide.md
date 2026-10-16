# Noise diagnosis tables, row by row

A guide to [`noise_20261012.results.md`](noise_20261012.results.md) and
[`cilantro_encodings_20261012.results.md`](cilantro_encodings_20261012.results.md): for every row, what it does and
what its number is. The findings are in the [README](README.md).

## Terms

- **The prior.** The default linear context prior (GeneEffect protocol §10, `configs/context_prior/default_prior.yaml`).
  It predicts each line's GeneEffect residual (GeneEffect minus the gene's training mean) from one expression profile
  per line. The profile is bulk RNA for bulk-only lines, and for single-cell lines (every validation and test line)
  their pseudo-bulk (all basal cells summed, CPM, log1p) mapped onto the bulk scale by a per-gene affine bridge.
  Two ridge stages: 128 expression components (penalty 1), then per gene its 50 data-selected genes (penalty 10).
  It is fitted on 947 labelled lines. The **export** is its out-of-fold prediction of every training line (five
  patient-grouped folds).
- **Selective Spearman.** Per selective gene (4,323 genes with a dependent tail among the 377 labelled training
  lines), the Spearman across the scored lines between predicted and measured GeneEffect; then the mean over genes.
  Validation has 27 lines, test 28. The prior scores 0.2045 / 0.2025.
- **Gain `x [lo, hi]`.** A row's selective Spearman minus a reference's on the same lines. The bracket is a 95% paired
  line bootstrap (1,000 resamples of the lines, the same draw for both).
- **Median `[Q25, Q75]`.** For agreements and reliabilities, the bracket is the quartiles over genes, not an interval.

## Cell sampling

What the section does: tests whether a line's cells sample its expression well enough, by reading lines from fewer
cells.

**Gene table** (per gene, over the 8,737 genes of the expression space, or the 2,928 selective genes in it).

| Row | What it does | Number |
| --- | --- | ---: |
| `reliability` | Splits each of 434 lines' cells into two halves; per gene, the Pearson across lines between the halves' pseudo-bulks, corrected to the full cell count (2r/(1+r)) | 0.939 (selective 0.947) |
| `paired_reliability` | The same on the 343 lines that also have bulk RNA | 0.933 |
| `bridge_r` | Per gene, the Pearson across those 343 lines between bridged pseudo-bulk (out of fold) and bulk RNA | 0.453 (selective 0.400) |
| `bridge_over_ceiling` | `bridge_r` over √`paired_reliability`, the most sampling noise would allow | 0.484: the bridge reaches under half of what sampling noise allows; the rest of the gap is systematic |

**Draw table.** The prior, fitted once, is queried with fewer cells of each validation and test line. Cells are
ordered by a hash of line and cell barcode; draws are nested.

| Row | What it does | Number (val / test; gain over all cells) |
| --- | --- | --- |
| `half_a` | The even-ranked half of each line's cells | 0.2049 / 0.2025; +0.0004 / +0.0001 |
| `half_b` | The odd-ranked half | 0.2032 / 0.2013; −0.0013 / −0.0012 |
| `cells_10` | The first 10 cells | 0.2015 / 0.1965; −0.0030 / −0.0060, intervals span zero |
| `cells_25` | The first 25 | 0.2033 / 0.2013; −0.0012 / −0.0012 |
| `cells_50` | The first 50 | 0.2044 / 0.2016; −0.0001 / −0.0008 |
| `cells_100` | The first 100 | 0.2047 / 0.2007; +0.0002 / −0.0018 |
| `cells_200` | The first 200 | 0.2050 / 0.2006; +0.0005 / −0.0018 (test interval below zero) |
| `cells_400` | The first 400 | 0.2047 / 0.2015; +0.0002 / −0.0010 (test interval below zero) |

`capped` counts lines with fewer cells than the draw; they read all their cells. The line under the table: per
selective gene, the Pearson across lines between the predictions from the two halves, 0.990 on both splits.

## Learning curve

What the section does: refits the prior on less of the training side, to see whether more lines would help. Patients
are put in a seeded order, a fraction keeps the lines of its first patients (so fractions nest), and bridge, components
and data-selected genes are all refitted. Columns: `Training side` are all lines with expression (the components are
fitted on them); `Labelled` lines fit the ridges; `Paired` lines fit the bridge. Gains are against the full prior.

| Row | What it does | Number (val / test) |
| --- | --- | --- |
| 0.125 | 173 lines, 126 labelled, 41 paired | 0.1132 / 0.1331 |
| 0.25 | 354 lines, 251 labelled, 86 paired | 0.1617 / 0.1412 |
| 0.5 | 708 lines, 486 labelled, 165 paired | 0.1778 / 0.1753 |
| 0.75 | 1,060 lines, 721 labelled, 246 paired | 0.1930 / 0.1903; −0.0115 / −0.0122, intervals below zero |
| 1 | The full prior: 1,399 lines, 947 labelled, 343 paired | 0.2045 / 0.2025 |

## Deposit confound

What the section does: asks whether the deposit a training line's cells come from explains what the prior gets
wrong on it, beyond lineage. Over the 377 labelled single-cell training lines, per selective gene, η² is the share of
a quantity's variance across lines that lies between groups; the table gives its mean over genes. Deposits with fewer
than 5 lines are pooled as `other` (10 groups); lineages give 25.

| Row | What it does | Number |
| --- | --- | ---: |
| error × deposit | η² of the prior's error (measured residual minus the export) by deposit; the null shuffles deposits among lines of one lineage (1,000 times) | 0.0365 against null 0.0315 (95th percentile 0.0332, p 0.001) |
| error × lineage | The same by lineage; the null shuffles lineages among all lines | 0.0927 against 0.0640 |
| residual × deposit | η² of the label itself (GeneEffect minus the gene mean), which no single-cell deposit touches | 0.0384 against 0.0339 |
| residual × lineage | The same by lineage | 0.1093 against 0.0640 |

Columns: `Null mean` and `Null 95%` are the mean and 95th percentile of η² over the shuffles; `p` is the share of
shuffles reaching the observed value (0.001 is the smallest possible).

Reading (an inference, not a test): the label alone, which no deposit touched, shows about the same excess over its
null (0.0045) as the error (0.0050). So the error's grouping by deposit is about what the choice of lines in each
deposit already produces in the label, which leaves little room for a batch effect on top.

**Per-deposit table.** Per deposit, the mean squared error over its lines and the selective genes, divided by that of
all lines (1.0 = average).

| Row | Lines | Number |
| --- | ---: | ---: |
| `breast_cell_line_atlas` | 8 | 1.220 |
| `ccla_omix005191` | 7 | 0.922 |
| `emtab13204` | 16 | 1.285 (the largest: errors 29% above average) |
| `figshare33273600` | 13 | 0.731 (the smallest) |
| `gse229224` | 8 | 1.092 |
| `gse311521` | 13 | 1.101 |
| `kinker_scp542` (Kinker et al. 2020's CCLE pools) | 110 | 0.969 |
| `mixseq` (McFarland et al. 2020's MIX-seq) | 83 | 0.917 |
| `other` (deposits with fewer than 5 lines) | 81 | 1.060 |
| `tahoe100m` (Tahoe-100M controls) | 38 | 1.029 |

## Deposit swap

What the section does: reads the same lines from two deposits. It uses the 99 labelled training lines that have a
second public deposit, each predicted out of fold (the prior refitted without the line's fold). Scores are selective
Spearman over these 99 lines, not the held-out lines, so they sit below 0.2045.

| Row | What it does | Number (gain over primary) |
| --- | --- | --- |
| `primary` | The line read from the cells the prior was built on; reproduces the export | 0.1828 |
| `second` | The line read from another deposit's cells, through the same bridge | 0.1894; +0.0066 [−0.0052, +0.0184] |
| `half_a` | Half of the primary deposit's cells | 0.1813; −0.0015 [−0.0030, −0.0002] |
| `half_b` | The other half | 0.1812; −0.0016 [−0.0029, −0.0002] |

**Agreement table.** Per gene, the Pearson across the 99 lines between two readings: `Predictions` over the
selective genes, `Inputs` over the space genes' normalised pseudo-bulk (measured values only).

| Row | What it does | Number (predictions; inputs) |
| --- | --- | --- |
| `primary_second` | Primary deposit against second deposit | 0.764; 0.375 |
| `half_a_half_b` | Two halves of one deposit | 0.982; 0.865 |

## Cell states

What the section does: tests whether a line's mix of cell states adds to the prior. Each arm gives each line a few
features, fits a ridge from them to what the export leaves on the training lines, adds that to the prior on validation
and test, and scores it. The penalty is the ridge's: at 1 the added part is large, at 1,000 nearly nothing (the row
then equals the prior).

| Arm | What it does | Number (best penalty, val; test) |
| --- | --- | --- |
| `program_fractions` | 23 programs: Tirosh's S and G2/M and 21 Gavish meta-programs. Per cell, a program score (its genes' mean expression minus the cell's mean); per line, the fraction of cells above the training lines' 90th percentile. This is the heterogeneity arm | 0.2055 at 100: +0.0010 [−0.0001, +0.0021]; test +0.0010 |
| `program_means` | Per line, the mean program score over its cells. This is the control: no distribution, only a mean | 0.2055 at 100: +0.0010 [+0.0001, +0.0018]; test +0.0009 [−0.0007, +0.0024] |
| `clusters_8` | 128 cells per line, centred on the line's mean, 50 principal components, K-means with 8 clusters fitted on training cells; per line, the share of its cells in each cluster | 0.2047 at 100: +0.0002 |
| `clusters_16` | The same with 16 clusters | 0.2051 at 100: +0.0006 |

`Val over means` / `Test over means`, on the `program_fractions` rows: fractions minus means at the same penalty,
which is the heterogeneity test. At 100 it is +0.0000 [−0.0009, +0.0009]; at no penalty is it above zero. The line under
the table: the share of the scoring set (the pseudo-bulk's genes, over which program scores are computed) that each
line's source lists (median 0.911). 81 lines list under 90%, and an unlisted gene counts as unexpressed.

## Cilantro encodings

What the section does: puts Cilantro-SL's input reductions in front of the prior. Both pseudo-bulk and bulk are
transformed the same way before the bridge: each gene is divided by its nonzero median over training bulk
(Geneformer's normalisation), then ranked within the line. Every row keeps the prior's components penalty 1 and gene
penalty 10. Gains are against the first row.

| Row | What it does | Number (val / test; val gain) |
| --- | --- | --- |
| `affine`, `-` | The prior itself (the reference) | 0.2045 / 0.2025 |
| `rank_value, top_k None` | Every gene's value replaced by its within-line rank, 1 − rank/G | 0.2023 / 0.2037; −0.0022 [−0.0076, +0.0033] |
| `rank_value, top_k 2048` | Only the line's top 2,048 genes keep a rank value (1 − rank/2048); the rest are 0 | 0.1853 / 0.1948; −0.0192 [−0.0300, −0.0092] |
| `expression, top_k 2048` | The same top 2,048 genes keep their normalised expression; the rest are 0 (truncation without ranking) | 0.1904 / 0.1996; −0.0142 [−0.0201, −0.0067] |

Other columns:

- **AUPR lift.** Per selective gene with both dependent and non-dependent lines in the split, the average precision of
  ranking lines by predicted GeneEffect for the dependent ones (GeneEffect below −0.5), minus their share; then the
  mean. The reference scores 0.1963 / 0.1827.
- **Residual Pearson.** Per variable gene, the Pearson across lines between predicted and measured residual; then the
  mean. The reference scores 0.2220 / 0.2221.
- **Oracle.** The same fit queried with validation lines' bulk RNA instead of their cells. It is not a model; it shows
  what the bridge costs. The reference scores 0.2216, and every encoding scores lower (0.2190, 0.2111, 0.2149).

**Bridge diagnostics.** Per gene, the Pearson across the 343 paired lines between bridged pseudo-bulk (out of fold)
and bulk, as median and quartiles. Then counts of genes at or above 0.3 / 0.5 / 0.7, among all 8,737 space genes,
the 2,928 selective genes in the space and their 2,394 paralogs. `Gene space` `all` means no gene was gated out.

| Row | Median | Genes ≥ 0.3 / 0.5 / 0.7 (all) |
| --- | ---: | --- |
| Reference (log expression) | 0.453 | 6,831 / 3,538 / 1,062 |
| Rank value, every gene | 0.466 | 7,313 / 3,657 / 654 |
| Rank value, top 2,048 | 0.302 | 4,391 / 1,572 / 222 |
| Expression, top 2,048 | 0.252 | 3,460 / 1,001 / 215 |

Ranking every gene raises the median agreement slightly but loses the most faithful genes (654 against 1,062 at 0.7).
Truncating to 2,048 genes lowers agreement for most genes.
