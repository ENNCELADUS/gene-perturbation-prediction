# The default prior and the single-cell statistics stack on the final split, seed 0

Runs of 2026-10-12 on the H20 container on port 30838 (branch `feat/final-split` at `aa49676`), CPU only, on the
[final split](../../docs/data/cell-line-geneeffect-split.md) (`configs/benchmarks/cell_line_geneeffect_split.json`):
379 single-cell training lines (377 labelled: the 226's 169 and the 208 single-cell extras), 27 validation, 28 test,
713 labelled and 576 unlabelled bulk-only lines. Every definition (gene means, gene sets, residual SD) fits on the 377
labelled training lines, so the metric differs from earlier records: 4,323 selective genes against 3,071 on the 226
split. Nothing here is synthetic-lethality evidence.

- `default_prior_final_split_20261012`: `hpc/run.sh prior` then `prior-export` with
  `configs/context_prior/default_prior.yaml` ([table](default_prior_final_split_20261012.results.md)).
- `single_cell_stack_final_split_20261012` and `single_cell_stack_grid_final_split_20261012`: `hpc/run.sh prior-stack`
  with `configs/context_prior/single_cell_stack.yaml` and `single_cell_stack_penalty_grid.yaml` on that prior run.

## The default prior

| Prior | Fit lines | Bridge pairs | Val selective Spearman | Test selective Spearman | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final split | 947 | 343 | **0.2045** | **0.2025** | 0.1963 | 0.1827 | 0.2220 | 0.2221 |
| Final split, lymphoid and myeloid single-cell lines fitted (superseded) | 975 | 371 | 0.1998 | 0.2018 | 0.1953 | 0.1836 | 0.2183 | 0.2217 |

The fit lines are the labelled training-side lines with bulk RNA (of 1,399 training-side lines), outside the dropped
lymphoid and myeloid lineages; the bridge pairs the single-cell training lines with bulk RNA among them. The export
predicts all 377 labelled single-cell training lines out of fold (selective Spearman 0.1835 on them). The oracle
(validation lines' bulk RNA) scores 0.2216. Shared space 8,737 genes, as on the 226 split with A549 in test.

**The drop from 0.2302 is the metric's.** Per-gene Spearman across lines does not depend on a gene's mean or scale, so
the priors can be scored on one gene set. On the 4,323 final-split selective genes (95% paired line bootstrap, 1,000
resamples), the final-split prior minus:

| Prior | Validation | Test |
| --- | --- | --- |
| The 226 split's default prior (`default_prior_resplit_20261012`) | −0.0014 [−0.0030, 0.0003] | −0.0012 [−0.0047, 0.0019] |
| The 226 split's prior with the single-cell extras (`default_prior_single_cell_extras_20261012`) | −0.0000 [−0.0017, 0.0021] | +0.0004 [−0.0010, 0.0019] |
| The final split's prior with lymphoid and myeloid single-cell lines fitted | +0.0047 [+0.0009, +0.0084] | +0.0007 [−0.0027, 0.0043] |

The 226 split's prior scores 0.2302 on its own 3,071 selective genes, 0.2059 on the final split's 4,323 and 0.1486 on
the 1,275 added ones (validation), so the added genes are the harder ones.

**Lymphoid and myeloid lines.** A first run (`default_prior_final_split_haematopoietic_fitted_20261012`, at `a90891b`)
applied the config's dropped lineages to bulk-only lines only, so 24 lymphoid and myeloid single-cell extras entered the
fit and the bridge; it scored 0.1998 / 0.2018. Refitting its base without them gained +0.0047 [+0.0011, +0.0083] on
validation (without all 28 haematopoietic single-cell training lines, +0.0048 [+0.0011, +0.0084]); test −0.0000 and
+0.0007, intervals spanning zero. The code now drops those lineages from every training-side line (`aa49676`); such a
single-cell training line still has an out-of-fold export row.

The correction configs stack on this export: `load_inputs` with `configs/correction/standardized_mse_no_state.yaml`
opens 377 training lines and 6,690,108 training rows, every prior offset finite (no training was run).

## The single-cell statistics stack

Stacked on `default_prior_final_split_20261012` (prior alone 0.2045 validation / 0.2025 test). Tables:
[first grid](single_cell_stack_final_split_20261012.results.md) (selected penalties 1 to 100),
[penalty grid](single_cell_stack_grid_final_split_20261012.results.md) (100 to 10,000). Gains are 95% paired line
bootstraps.

| Arm | Own shrinkage | Selected penalty | Val | Gain over prior (val) | Gain over `mean` (val) | Test | Gain over prior (test) | Gain over `mean` (test) |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `mean` | 10 | 100 | **0.2089** | +0.0044 [−0.0004, +0.0098] | | 0.2102 | +0.0077 [+0.0000, +0.0148] | |
| `distribution` | 10 | 100 | **0.2089** | +0.0044 [−0.0003, +0.0098] | +0.0000 [−0.0014, +0.0017] | 0.2107 | +0.0082 [−0.0002, +0.0158] | +0.0005 [−0.0010, +0.0019] |
| `single_cell` | 10 | 100 | **0.2089** | +0.0044 [−0.0005, +0.0101] | −0.0000 [−0.0015, +0.0015] | 0.2111 | +0.0087 [+0.0002, +0.0163] | +0.0010 [−0.0005, +0.0022] |
| `mean` | 10 | 1,000 | 0.2059 | +0.0014 [+0.0002, +0.0026] | | 0.2048 | +0.0024 [+0.0007, +0.0037] | |
| `distribution` | 10 | 1,000 | 0.2064 | +0.0019 [+0.0006, +0.0032] | +0.0005 [−0.0003, +0.0013] | 0.2051 | +0.0026 [+0.0007, +0.0042] | +0.0003 [−0.0005, +0.0010] |
| `single_cell` | 10 | 1,000 | 0.2068 | +0.0022 [+0.0011, +0.0034] | **+0.0009 [+0.0001, +0.0017]** | 0.2051 | +0.0026 [+0.0008, +0.0043] | +0.0003 [−0.0004, +0.0010] |

- Every arm's best validation row is at own shrinkage 10 and selected penalty 100, all three at 0.2089, and none clears
  the prior by an interval above zero on validation.
- At selected penalty 1,000 every arm clears the prior on validation and test with intervals above zero, by +0.0014 to
  +0.0026: the stack's own-statistics and data-selected stages add a little to the prior once shrunk hard, the `mean`
  arm (close to pseudo-bulk) included.
- One of the 26 distinct single-cell-against-`mean` comparisons on validation has an interval above zero: `single_cell`
  at own shrinkage 10 and selected penalty 1,000, +0.0009 [+0.0001, +0.0017]; on test it is +0.0003 [−0.0004, +0.0010].
  Nine fall below zero, all at selected penalties 1 and 10, where every arm is under the prior; the other 16 span zero.
  On the 226 split (`single_cell_stack_grid_resplit_20261012`, [record](../default_prior_rerun_seed0/README.md)) the
  same row was +0.0010 [−0.0006, +0.0023].
