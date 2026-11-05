# The default linear model on the current split, with and without the single-cell extras, seed 0

Superseded the same day by the [final split](../final_split_seed0/README.md), which makes the single-cell extras
split members and fits the metric on 377 lines; the `single_cell_extras` switches these runs used are gone.

Runs of 2026-10-12 on the H20 container on port 30838 (branch `feat/extras-training-lines` at `f4e1e02`), CPU only,
each `hpc/run.sh linear-model CONFIG --run-id ID` then `hpc/run.sh linear-model-export CONFIG --run-id ID`:

- `default_linear_model_resplit_20261012`: `configs/linear_model/default_linear_model.yaml`, the default linear model unchanged
  (`training_side.single_cell_extras: false`), on the split rebuilt on 2026-10-11 with A549 in test (171/27/28);
- `default_linear_model_single_cell_extras_20261012`: `configs/linear_model/default_linear_model_single_cell_extras.yaml` (since removed), the same
  linear model with the 208 single-cell extras as single-cell training lines (GeneEffect protocol §10).

Tables: [unchanged](default_linear_model_resplit_20261012.results.md), [with the extras](default_linear_model_single_cell_extras_20261012.results.md);
the exports stay in `outputs/linear_model/<run id>/export/` on the host. Nothing here is synthetic-lethality evidence.

## Scores

Both runs fit the linear model on the same 951 labelled training-side lines with bulk RNA (of 1,403 training-side lines),
in a shared space of 8,737 genes, scored on 3,071 selective genes; only the single-cell training lines differ.

| Run | Single-cell training lines | Bridge pairs | Val selective Spearman | Test selective Spearman | Val AUPR lift | Test AUPR lift | Val residual Pearson | Test residual Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 169 | 166 | **0.2302** | **0.2318** | 0.1987 | 0.1858 | 0.2242 | 0.2266 |
| With the single-cell extras | 377 | 347 | 0.2290 | 0.2301 | 0.1987 | 0.1861 | 0.2231 | 0.2236 |

With the extras minus unchanged (95% paired line bootstrap, 1,000 resamples, seed 0): **−0.0012 [−0.0036, 0.0010]**
on validation, **−0.0017 [−0.0052, 0.0011]** on test. The extras add 181 bridge pairs; the 24 haematopoietic extras
(dropped lineages) and the three without bulk RNA are predicted but neither fitted nor paired. The out-of-fold bridge
correlation rises (median per gene 0.452 against 0.424; genes at or above 0.5: 3,534 against 3,165 of 8,737), but over
different paired lines, so that comparison is not like for like. The oracle (validation lines' bulk RNA) scores 0.2481
in both.

The exports: the unchanged run predicts its 169 single-cell training lines out of fold (selective Spearman 0.2109), the
extras run its 377 (the 208 extras among them; 0.2121). The second, read by
`configs/correction/standardized_mse_no_state.yaml` on the host (no training): 377 training lines, 6,690,108 training
rows, every linear model offset finite.

## Against the old split

On the 172/27/27 split the default linear model scored 0.2290 validation / 0.2338 test over 953 fit lines, 167 bridge pairs,
a 9,711-gene space and 3,057 selective genes ([record](../default_linear_model_followups_seed0/README.md)). The space shrinks
to 8,737 genes because A549, now a test line, comes from the Tahoe-100M DMSO source, which measures 11,480 of the
19,215 bulk genes the pseudo-bulk covers; without A549 the held-out lines share the old 9,711. The fit lines lose A549
and ACH-003012, which shares A549's patient.

## Check

An earlier run of `default_linear_model.yaml` with the code before the switch (`0db59ad`, run `default_linear_model_20261012`, its
directory since removed) gave an identical export (largest absolute difference 0) and identical row files.

## The single-cell statistics stack on this split

Runs of 2026-10-12 (branch `chore/stack-current-split` at `9ab012e`), stacked on `default_linear_model_resplit_20261012`:
`single_cell_stack_resplit_20261012` (`configs/linear_model/single_cell_stack.yaml`) and
`single_cell_stack_grid_resplit_20261012` (`single_cell_stack_penalty_grid.yaml`, selected penalties 100 to 10,000).
Tables: [first grid](single_cell_stack_resplit_20261012.results.md), [penalty grid](single_cell_stack_grid_resplit_20261012.results.md).
They read the 226's 169 labelled training lines and their definitions (3,071 selective genes), so the final split's
runs supersede them ([record](../final_split_seed0/README.md)).

The 2026-10-11 reading holds. No single-cell arm beats `mean` on validation by an interval above zero at any penalty
pair (largest +0.0010 [−0.0006, +0.0023], `single_cell` at own shrinkage 10 and selected penalty 1,000), and no arm's
best validation row clears the linear model: `distribution` 0.2344 (+0.0042 [−0.0050, +0.0147]), `mean` 0.2342, `single_cell`
0.2334 (own shrinkage 10, selected penalty 100), against 0.2302 for the linear model alone (test 0.2415, 0.2411, 0.2409 against
0.2318).
