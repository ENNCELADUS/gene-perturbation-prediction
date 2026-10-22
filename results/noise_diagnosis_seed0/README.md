# Noise diagnosis of the default prior, seed 0

Runs of 2026-10-12 on the H20 container on port 30838, CPU only, branch `feat/noise-diagnosis`, on the
[final split](../../docs/data/cell-line-geneeffect-split.md) and its default prior (`default_prior_final_split_20261012`:
0.2045 validation / 0.2025 test, [record](../final_split_seed0/README.md)). Design:
[noise diagnosis](../../docs/specs/2026-10-12-noise-diagnosis-design.md); protocol §14. Nothing here is
synthetic-lethality evidence.

- `noise_20261012`: `hpc/run.sh prior-noise` with `configs/context_prior/noise_diagnosis.yaml`
  ([table](noise_20261012.results.md)). `learning_curve`, `cell_sampling` and `cell_states` at
  `97a3066`; `deposit_confound` and `deposit_swap` at `a1bdadc`, which reads Tahoe-100M as one deposit (the first
  `deposit_confound`, at `97a3066`, pooled the one extra Tahoe line as `other`; η² moved by less than 0.0003).
- `cilantro_encodings_20261012`: `hpc/run.sh prior` with `configs/context_prior/cilantro_encodings.yaml` at `97a3066`
  ([table](cilantro_encodings_20261012.results.md)).
- Row by row: what each row of both tables does and what its number is ([table guide](table_guide.md)).
- Inputs built for it: per-line cell draws and per-cell program scores under `data/geneeffect_joint/v2/noise/` (434
  lines), the 99 planned second deposits under `data/sl_dependency_v0/processed/basal_raw_umi_second_source/` (all built,
  none missing).

**Question.** Does technical noise in a line's single cells limit the default prior: sampling of cells, the deposit the
cells come from, too few training lines, unmodelled cell states? And do Cilantro-SL's
input reductions (rank-value encoding, the top 2,048 genes) clean the input?

## Per-cell sampling does not bind

Two halves of a line's cells give pseudo-bulks with a per-gene split-half reliability (Spearman-Brown, 434 lines) of
0.939 [0.886, 0.967], 0.947 on the selective genes. The prior's predictions from the two halves agree at 0.990 per
selective gene. Queried with draws of each validation and test line's cells:

| Cells per line | Val | Val gain | Test | Test gain |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.2015 | −0.0030 [−0.0092, +0.0036] | 0.1965 | −0.0060 [−0.0132, +0.0016] |
| 25 | 0.2033 | −0.0012 [−0.0058, +0.0036] | 0.2013 | −0.0012 [−0.0059, +0.0040] |
| 50 | 0.2044 | −0.0001 [−0.0034, +0.0030] | 0.2016 | −0.0008 [−0.0039, +0.0025] |
| half A | 0.2049 | +0.0004 [−0.0013, +0.0019] | 0.2025 | +0.0001 [−0.0023, +0.0018] |
| all (the prior) | 0.2045 | | 0.2025 | |

Ten cells per line read as well as all of them within the intervals. The bridge's out-of-fold agreement with bulk RNA
(0.453 per gene, 0.400 on selective genes) is half of what the split-half reliability allows (0.484 of the ceiling):
the pseudo-bulk to bulk gap is systematic, not sampling. Bulk RNA itself as the validation input (the oracle) scores
0.2216, so the whole bridge costs about 0.017.

## Deposit batch moves the input and the prediction, not the accuracy

Ninety-nine labelled training lines have a second deposit (36 Kinker to MIX-seq, 13 Tahoe-100M to MIX-seq, the rest
GEO), predicted out of fold from four readings:

| Reading | Selective Spearman (99 lines) | Gain over primary |
| --- | ---: | ---: |
| primary deposit | 0.1828 | |
| second deposit | 0.1894 | +0.0066 [−0.0052, +0.0184] |
| half A of primary | 0.1813 | −0.0015 [−0.0030, −0.0002] |

Per gene across these lines, the second deposit's input agrees with the primary's at 0.375 [0.228, 0.530] against 0.865
for the two halves, and its predictions at 0.764 [0.671, 0.830] against 0.982. The deposit explains 0.0365 of the
out-of-fold error's variance per selective gene against a within-lineage null of 0.0315 (95% at 0.0332, p 0.001);
lineage explains 0.0927 against 0.0640. By the design's reading (second-deposit predictions agreeing as well as the
halves, deposit η² at its null) deposit batch binds; by score it does not: the excess η² is 0.005 and a line read
from another deposit is predicted no worse. Per-deposit error mean squares run from 0.73 to 1.29 of the overall.

## More lines helped, up to a point

The reference row refitted on nested patient-grouped fractions of the training side:

| Training-side lines | Labelled | Paired | Val | Test |
| ---: | ---: | ---: | ---: | ---: |
| 173 | 126 | 41 | 0.1132 | 0.1331 |
| 354 | 251 | 86 | 0.1617 | 0.1412 |
| 708 | 486 | 165 | 0.1778 | 0.1753 |
| 1,060 | 721 | 246 | 0.1930 | 0.1903 |
| 1,399 | 947 | 343 | 0.2045 | 0.2025 |

The last quarter of lines adds 0.0115 validation / 0.0122 test (intervals above zero). The 2026-10-14 rerun with a 7/8
row ([record](../noise_lines_seed0/README.md)) puts all of that gain between 3/4 and 7/8: from 7/8 to all lines the
curve is flat (−0.0006 [−0.0062, +0.0050] validation).

## Cell states and Cilantro's encodings add nothing

Per-line features from the training lines' cells (Tirosh S and G2M plus the 21 Gavish meta-programs without a cell-line
study; K-means at 8 and 16 clusters on line-centred cells), stacked on the prior's export: the best row at any penalty
is +0.0010 on validation (`program_means` at 100: [+0.0001, +0.0018]; test +0.0009 [−0.0007, +0.0024]), and program
fractions never beat program means (+0.0000 [−0.0009, +0.0009] at the same penalty). Each source lists a median 0.911 of
the scoring set (unlisted genes score as unexpressed); 81 lines list under 90%.

| Cilantro encoding | Val | Val gain | Test | Test gain |
| --- | ---: | ---: | ---: | ---: |
| rank value, every gene | 0.2023 | −0.0022 [−0.0076, +0.0033] | 0.2037 | +0.0013 [−0.0036, +0.0067] |
| rank value, top 2,048 | 0.1853 | −0.0192 [−0.0300, −0.0092] | 0.1948 | −0.0077 [−0.0212, +0.0063] |
| expression, top 2,048 | 0.1904 | −0.0142 [−0.0201, −0.0067] | 0.1996 | −0.0029 [−0.0141, +0.0080] |

Rank-value encoding is a tie; truncating to 2,048 genes loses on validation (bridge agreement falls from 0.453 to
0.302 and 0.252 per gene).

**Reading.** The technical noise of single cells does not limit the default prior: it reads a line from ten cells as
well as from all of them, and neither state programs, clusters nor Cilantro's reductions improve it. Deposit batch
changes a line's input more than anything else measured here but not its accuracy.
