# Follow-ups on the default linear context prior, seed 0

Run `default_prior_followups_20261007`, 2026-10-07, on the H20 container on port 30838 (worktree
`/2023533015/VCC_Project_default_prior`, branch `feat/default-prior` at `77949cf`), CPU only, config
`configs/context_prior/default_prior_followups.yaml`. Two processes on one run directory:
`hpc/run.sh prior CONFIG --run-id default_prior_followups_20261007` (every block set and penalty pair, scored on
validation and test) and `hpc/run.sh prior-selected CONFIG --run-id default_prior_followups_20261007` (the data-selected
stage of the reference row). Tables: [results.md](results.md), [data_selected_summary.md](data_selected_summary.md),
[data_selected_ablation.json](data_selected_ablation.json); the per-feature and per-target tables (`features.csv`,
`targets.csv`) stay in the run directory on the host. Nothing here is synthetic-lethality evidence.

The reference row is the default prior chosen from the [bridge remedies penalty grid](../bridge_remedies_seed0/README.md#rerun-over-the-full-penalty-grid):
the affine bridge, 128 expression components at penalty 1, then own expression, partners and 50 data-selected genes at
gene penalty 10 (block set `all`). Training side, space and folds as in that run: 953 labelled lines with bulk RNA,
167 paired lines for the bridge, 9,711 space genes, 3,057 selective genes, five patient-grouped folds, 1,000 bootstrap
resamples. Penalties (× n): components 0.1, 1, 10; gene 1, 10, 100. A gain is selective Spearman minus the reference
row's, with a 95% paired line-bootstrap interval.

## Do own expression and partners help once data-selected genes are in?

Barely. Each block set's best validation row is at components penalty 1 and gene penalty 10:

| Block set | Val | Test | Val gain | Test gain |
| --- | ---: | ---: | --- | --- |
| Components alone (penalty 10) | 0.2232 | 0.2227 | −0.0068 [−0.0129, −0.0011] | −0.0125 [−0.0187, −0.0049] |
| Components + data-selected genes (`selected`) | 0.2290 | 0.2338 | −0.0010 [−0.0022, 0.0002] | −0.0014 [−0.0027, −0.0002] |
| + own expression (`own_and_selected`) | 0.2292 | 0.2347 | −0.0008 [−0.0017, 0.0001] | −0.0005 [−0.0012, 0.0003] |
| + partners (`partners_and_selected`) | 0.2296 | 0.2343 | −0.0004 [−0.0017, 0.0010] | −0.0009 [−0.0019, 0.0001] |
| All (reference) | **0.2300** | **0.2352** | — | — |

Own expression and partners together add +0.0010 on validation (interval spans zero) and +0.0014 on test once the
data-selected genes are in. The data-selected genes carry the gain over components alone.

## Which data-selected genes carry the gain?

None in particular: the gain is spread over thousands of features and is a small, broad lift across targets.
Masking the stage's weights on the training side (the prior is not refitted):

| Variant | Val | Test | Val gain | Test gain |
| --- | ---: | ---: | --- | --- |
| Without the stage | 0.2234 | 0.2237 | −0.0065 [−0.0133, −0.0005] | −0.0115 [−0.0184, −0.0029] |
| Top 10 features only | 0.2231 | 0.2245 | −0.0069 [−0.0138, −0.0009] | −0.0106 [−0.0171, −0.0028] |
| Without top 10 | 0.2307 | 0.2348 | +0.0008 [−0.0003, 0.0018] | −0.0003 [−0.0012, 0.0005] |
| Top 200 only | 0.2243 | 0.2285 | −0.0057 [−0.0102, −0.0015] | −0.0067 [−0.0111, −0.0013] |
| Without top 200 | 0.2306 | 0.2327 | +0.0006 [−0.0024, 0.0031] | −0.0024 [−0.0048, 0.0001] |
| Top 1000 only | 0.2262 | 0.2303 | −0.0038 [−0.0062, −0.0013] | −0.0049 [−0.0073, −0.0018] |

Features are ranked by summed absolute weight over the selective targets; the 50 and the remaining rows are in the
summary.

- **Breadth.** The stage reads 9,636 of the 9,711 space genes; 367 of its 152,850 selections are a paralog or complex
  partner of their target.
- **Per target.** Mean gain +0.0065 validation, +0.0115 test; positive for 1,657 and 1,766 of 3,057 targets. Validation
  and test gains are uncorrelated across targets (Spearman 0.041): the top validation decile gains +0.119 on validation
  and +0.018 on test, the bottom −0.107 and +0.005. On test the lift is about +0.01 in every decile.
- **What the heaviest features are.** A p53-activity signature (CDKN1A, ZMAT3, BAX, RPS27L) serving the MDM2, PPM1D,
  USP7 and TP53BP1 dependencies; a mesenchymal and extracellular-matrix programme (FBN1, COL1A1, FN1, CALD1, ACTA2)
  serving a mixed set (JUN, KEAP1, NF2, FERMT2, ITGB3, purine synthesis); PSAT1 and SLC25A5 serving mitochondrial
  translation and respiratory-chain targets.
- **Relation to the components.** Over the fit lines the expression components explain a median 0.74 of a top-50
  feature's variance (0.75 over every space gene) and lineage 0.20 (0.14): the stage reads what is left of an
  expression programme after the components, not lineage.

## Decision (2026-10-07, read on validation)

The single-cell correction stacks on **the affine bridge with expression components at penalty 1 and data-selected
genes at gene penalty 10** (`selected`; 0.2290 validation, 0.2338 test), which becomes the default prior. It is within
the bootstrap interval of the all-block reference on validation, and it leaves partner information to the correction's
single-cell partner features. Recorded in the [single-cell correction plan](../../docs/specs/2026-10-07-single-cell-correction-plan.md).

## Caveats

- **Marginal masking.** The ablation zeroes weights in the fitted stage; it does not refit. Correlated features share
  credit, so "top K only" understates and "without top K" overstates what a refitted stage on those features would do.
- **Per-target noise.** With 27 lines a target's Spearman gain is noisy; the decile table shows regression to the mean,
  not a set of reliably improved targets.
- **Seed 0, one run.** Test numbers are reported for every row and used for none of the choices.
- **Claim boundary.** Single-gene dependency evidence only, with a training-data change (extra DepMap bulk-RNA lines)
  scored on the unchanged validation and test lines; validation lines' bulk RNA entered only the oracle column.
