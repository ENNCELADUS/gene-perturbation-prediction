# Single-cell statistics stacked on the default prior, seed 0

Two runs on 2026-10-11 on the H20 container on port 30838, CPU only, branch `feat/single-cell-stack`
(GeneEffect protocol §13): `single_cell_stack_20261011` (`configs/context_prior/single_cell_stack.yaml`, at `7ab8521`)
and, because every arm's best validation row sat at the edge of its selected-penalty grid,
`single_cell_stack_grid_20261011` (`configs/context_prior/single_cell_stack_penalty_grid.yaml`, at `1e21d92`). Both stack
on the default prior's export `default_prior_export` (affine bridge, components penalty 1, data-selected genes at gene
penalty 10). Tables: [first_grid_results.md](first_grid_results.md), [penalty_grid_results.md](penalty_grid_results.md)
and their `facts.json`; per-arm row files stay in the run directories on the host. Nothing here is
synthetic-lethality evidence.

**Question.** Do per-gene statistics of a line's basal single cells (detected fraction and variance across cells)
predict what the default prior misses, beyond the same stack built on the log-space mean (close to pseudo-bulk)? This is
the first step of Task 2 in the [research plan](../../docs/01-blueprint.md#5-research-plan).

**Setup.** 170 single-cell training lines, 27 validation, 27 test; 17,787 target genes, 3,057 selective; statistics over
the 9,585 panel genes every validation and test line measures, quantile-normalised per line to the training profile.
Each arm fits own statistics (pooled weights plus a per-gene deviation) and then 50 data-selected statistic columns per
gene, on what the prior's out-of-fold export leaves on the training lines. The prior alone scores 0.2290 validation /
0.2338 test, its recorded value. The penalty-100 rows are identical in both runs.

## Single-cell statistics add nothing over the mean

Each arm's best validation row (own shrinkage 10, selected penalty 100 in every arm) and the stack's best row against
the mean arm at matched penalties:

| Arm | Statistics | Val | Gain over prior (val) | Gain over `mean` (val) | Test | Gain over prior (test) | Gain over `mean` (test) |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| `mean` | log-space mean | 0.2329 | +0.0040 [−0.0042, +0.0140] | — | 0.2409 | +0.0072 [−0.0024, +0.0156] | — |
| `distribution` | detected fraction, variance | **0.2340** | +0.0050 [−0.0036, +0.0155] | +0.0011 [−0.0005, +0.0026] | 0.2404 | +0.0066 [−0.0039, +0.0160] | −0.0006 [−0.0025, +0.0015] |
| `single_cell` | all three | 0.2329 | +0.0039 [−0.0052, +0.0145] | −0.0000 [−0.0020, +0.0020] | 0.2397 | +0.0059 [−0.0042, +0.0155] | −0.0013 [−0.0031, +0.0008] |

- **Against the mean arm:** no row of either single-cell arm, at any of the 13 penalty pairs, beats `mean` on
  validation by an interval above zero; the largest validation gain is +0.0011. At weak penalties both single-cell
  arms are below `mean` (to −0.0061 [−0.0090, −0.0031] for `single_cell` at own 1, selected 10).
- **Against the prior:** no arm's best validation row clears the prior by an interval above zero. The one validation
  interval above zero in either run is the `mean` arm at own 10, selected 1,000: +0.0018 [+0.0001, +0.0037]
  (+0.0028 [+0.0004, +0.0051] on test), a refit of mean-expression features on the prior's out-of-fold residual.
- **Penalties:** validation peaks at selected penalty 100 in every arm and falls at 1,000 and 10,000; weak penalties
  overfit the 170 lines (selected penalty 1: −0.055 to −0.061 against the prior). The own shrinkage barely matters.

**Reading.** In this linear form, the detected fraction and variance of a gene across a line's cells carry no
dependency information that the gene's mean expression does not, and a second stage of any of them adds at most about
0.005 selective Spearman to the default prior. Per-line cell-state proportions and partner fractions were not part of
this stack.
