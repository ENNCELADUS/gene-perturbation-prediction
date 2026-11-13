# SL backbone, two single-cell inputs (seed 0)

SL protocol §13. Run on the H20 container on port 30838 on 2026-11-13, CPU only, code at `eea6efb` (branch
`sl-single-cell-inputs`), from the host worktree `/2023533015/VCC_wt/sl-single-cell-inputs`:

| Step | Command | Wall time (UTC) |
| --- | --- | --- |
| Co-detection cache, 86 lines | `hpc/run.sh sl-codetection configs/revision/frozen_huber.yaml data/sl_studies/benchmark_v2/labels.parquet configs/benchmarks/cell_line_sl_split.json` (16 processes, 2 threads) | 05:23:31–05:27:00 |
| Backbone, narrow space (rerun) | `hpc/run.sh sl-conditional configs/sl/conditional.yaml --run-id single_cell_narrow_seed0` | 05:23:31–05:28:43 |
| Backbone, each line in its own genes | `hpc/run.sh sl-conditional configs/sl/conditional_wide.yaml --run-id conditional_wide_seed0` | 05:23:31–05:34:22 |
| Head with co-detection | `hpc/run.sh sl-conditional configs/sl/conditional_codetection.yaml --run-id conditional_codetection_seed0` | 05:27:38–05:32:53 |
| Head with shuffled co-detection | `hpc/run.sh sl-conditional configs/sl/conditional_codetection_shuffled.yaml --run-id conditional_codetection_shuffled_seed0` | 05:27:38–05:33:24 |
| Harness | `hpc/run.sh sl-baselines configs/sl/benchmark_single_cell_inputs.yaml --run-id benchmark_single_cell_inputs_seed0` | 05:34:35–05:36:10 |

Outputs: `outputs/sl/conditional/<run id>/`, `outputs/sl/benchmark_baselines/benchmark_single_cell_inputs_seed0/` and
the cache `data/geneeffect_joint/v2/codetection/` ([card](../../docs/data/sl-codetection.md)) on the host. Every run
reads the default linear expression model fitted on the SL split's training side (979 fit lines, no SL held-out
line).

- **The reader in each line's own genes** (the wide space): the bulk genes that at least half of the 415 labelled
  single-cell training lines measure in pseudo-bulk, 18,294 genes, against 6,523 (the genes that every held-out line
  measures, the narrow space). The threshold is the point where the count falls: 18,839 genes at 1 line, 18,302 at
  207 lines (half), 17,142 at 311 lines (75%) and 4,316 at all 415. At half or more, a gene's bridge slope is fitted
  mostly on measured values.
- **Co-detection of A and B in the line's cells**: two inputs of the backbone's head per row, the share of the line's
  cells that detect both genes and `log((k + 0.5) / (e + 0.5))`, with `k` and `e` the observed and the expected counts
  of such cells under independence. The control is the same inputs after each gene's counts were permuted across the
  cells. This part uses the narrow-space reader.

## SL benchmark

**Table 1.** Recall in the top 10% (AUROC) on the multi-study SL benchmark, all held-out pairs, mean over lines
(17 validation, 16 test; the training SL fraction 15 / 13). Higher is better. Best model in bold; the training SL
fraction (the share of the study's training lines that called the pair SL), the labels and random are reference rows.

| Model | Validation | Test |
| --- | ---: | ---: |
| Pan-essentiality (gene-mean GeneEffect) | 0.369 (0.73) | 0.276 (0.68) |
| Predicted GeneEffect, narrow space (held out) | 0.371 (0.73) | 0.276 (0.68) |
| Conditional score, narrow space | 0.295 (0.62) | 0.288 (0.63) |
| Conditional score, narrow space, context-ablated | 0.296 (0.63) | 0.295 (0.64) |
| Backbone head, narrow space | 0.374 (0.74) | 0.322 (0.72) |
| Backbone head, narrow space, context-ablated | 0.358 (0.74) | 0.314 (0.72) |
| Conditional score, each line in its own genes | 0.292 (0.62) | 0.272 (0.64) |
| Conditional score, each line in its own genes, context-ablated | 0.289 (0.63) | 0.262 (0.64) |
| Backbone head, each line in its own genes | **0.381** (0.75) | 0.304 (0.72) |
| Backbone head, each line in its own genes, context-ablated | 0.376 (0.75) | 0.306 (0.71) |
| Backbone head with co-detection | 0.374 (0.75) | **0.331** (0.72) |
| Backbone head with co-detection, context-ablated | 0.355 (0.74) | 0.312 (0.72) |
| Backbone head with shuffled co-detection (control) | 0.369 (0.74) | 0.330 (0.72) |
| Backbone head with shuffled co-detection, context-ablated | 0.355 (0.74) | 0.312 (0.72) |
| Training SL fraction (reference) | 0.524 (0.77) | 0.495 (0.77) |
| Labels (ceiling) / random | 0.964 / 0.093 | 0.920 / 0.106 |

**Table 2.** Recall in the top 10% on two subsets of held-out rows, ranked inside the subset in each screen, mean over
lines (validation / test). Partner pairs: 81,657 rows (15.0%). No gene a dependency alone: 286,650 rows (52.7%). Best
model per column and split in bold.

| Model | Partner pairs | No gene a dependency alone |
| --- | ---: | ---: |
| Conditional score, narrow space | 0.311 / 0.318 | 0.327 / 0.301 |
| Conditional score, narrow space, context-ablated | 0.323 / 0.322 | 0.324 / 0.287 |
| Backbone head, narrow space | 0.332 / 0.279 | 0.408 / 0.329 |
| Backbone head, narrow space, context-ablated | **0.344** / 0.288 | 0.381 / **0.338** |
| Conditional score, each line in its own genes | 0.287 / **0.340** | 0.310 / 0.295 |
| Conditional score, each line in its own genes, context-ablated | 0.295 / 0.269 | 0.316 / 0.297 |
| Backbone head, each line in its own genes | 0.320 / 0.262 | **0.417** / 0.335 |
| Backbone head, each line in its own genes, context-ablated | 0.330 / 0.277 | 0.392 / 0.316 |
| Backbone head with co-detection | 0.330 / 0.284 | 0.414 / 0.321 |
| Backbone head with co-detection, context-ablated | 0.343 / 0.290 | 0.382 / 0.333 |
| Backbone head with shuffled co-detection (control) | 0.330 / 0.281 | 0.412 / 0.321 |
| Training SL fraction (reference) | 0.548 / 0.478 | 0.587 / 0.511 |

**Table 3.** Per-line difference in recall in the top 10% (model minus control), mean over lines with a 95% line
bootstrap (1,000 draws), line count in parentheses. Higher is better.

| Contrast | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Head, own genes − head, narrow space | All pairs | +0.007 [−0.013, +0.026] (17) | −0.017 [−0.038, +0.002] (16) |
| | Partner pairs | −0.012 [−0.030, +0.008] (16) | −0.017 [−0.046, +0.010] (15) |
| | No gene a dependency alone | +0.009 [−0.026, +0.042] (12) | +0.006 [−0.028, +0.046] (11) |
| Conditional, own genes − conditional, narrow space | All pairs | −0.003 [−0.029, +0.033] (17) | −0.015 [−0.032, −0.000] (16) |
| | Partner pairs | −0.024 [−0.046, +0.000] (16) | +0.022 [−0.086, +0.179] (15) |
| | No gene a dependency alone | −0.017 [−0.070, +0.043] (12) | −0.006 [−0.026, +0.016] (11) |
| Conditional, own genes − its context-ablated score | All pairs | +0.004 [−0.003, +0.011] (17) | +0.011 [−0.005, +0.030] (16) |
| | Partner pairs | −0.007 [−0.028, +0.009] (16) | +0.070 [−0.005, +0.212] (15) |
| | No gene a dependency alone | −0.006 [−0.015, +0.002] (12) | −0.003 [−0.012, +0.007] (11) |
| Head, own genes − its context-ablated head | All pairs | +0.005 [−0.012, +0.021] (17) | −0.002 [−0.030, +0.025] (16) |
| | Partner pairs | −0.010 [−0.035, +0.013] (16) | −0.015 [−0.045, +0.012] (15) |
| | No gene a dependency alone | +0.025 [+0.006, +0.047] (12) | +0.019 [+0.004, +0.033] (11) |
| Head with co-detection − head | All pairs | −0.000 [−0.012, +0.011] (17) | +0.010 [−0.006, +0.025] (16) |
| | Partner pairs | −0.002 [−0.011, +0.007] (16) | +0.005 [−0.001, +0.012] (15) |
| | No gene a dependency alone | +0.006 [−0.005, +0.019] (12) | −0.008 [−0.015, −0.002] (11) |
| Head with co-detection − head with shuffled co-detection | All pairs | +0.004 [+0.001, +0.008] (17) | +0.001 [−0.001, +0.003] (16) |
| | Partner pairs | +0.000 [−0.002, +0.003] (16) | +0.003 [−0.002, +0.011] (15) |
| | No gene a dependency alone | +0.003 [−0.006, +0.014] (12) | −0.000 [−0.004, +0.003] (11) |
| Head with co-detection − its context-ablated head | All pairs | +0.019 [−0.005, +0.041] (17) | +0.019 [−0.006, +0.049] (16) |
| | Partner pairs | −0.012 [−0.035, +0.011] (16) | −0.006 [−0.033, +0.020] (15) |
| | No gene a dependency alone | +0.032 [+0.010, +0.063] (12) | −0.012 [−0.035, +0.005] (11) |
| Head, narrow space − its context-ablated head | All pairs | +0.016 [−0.006, +0.035] (17) | +0.007 [−0.011, +0.028] (16) |
| Conditional, narrow space − its context-ablated score | All pairs | −0.001 [−0.011, +0.008] (17) | −0.007 [−0.027, +0.007] (16) |

The strata (seen pair, label agrees or differs; new pair) are in the harness's `results.md` Table 5.

**Table 4.** Recall in the top 10% (AUROC) on the rows that each space covers (both genes edited, and each lost gene
measured by the row's own line), beside the training SL fraction on the same rows; mean over lines.

| Model | Rows covered | Validation | Validation, training SL fraction | Test | Test, training SL fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional score, narrow space | 238,048 (43.8%) | 0.219 (0.62) | 0.392 (0.77) | 0.238 (0.65) | 0.382 (0.77) |
| Backbone head, narrow space | 238,048 (43.8%) | 0.246 (0.68) | 0.392 (0.77) | 0.223 (0.67) | 0.382 (0.77) |
| Conditional score, each line in its own genes | 429,476 (79.0%) | 0.274 (0.62) | 0.516 (0.77) | 0.283 (0.65) | 0.488 (0.77) |
| Backbone head, each line in its own genes | 429,476 (79.0%) | 0.360 (0.75) | 0.516 (0.77) | 0.298 (0.71) | 0.488 (0.77) |
| Backbone head with co-detection | 238,048 (43.8%) | 0.254 (0.68) | 0.392 (0.77) | 0.225 (0.67) | 0.382 (0.77) |
| Backbone head with shuffled co-detection | 238,048 (43.8%) | 0.252 (0.68) | 0.392 (0.77) | 0.223 (0.67) | 0.382 (0.77) |

The two spaces cover different rows, so Table 4 does not compare them.

**Table 5.** Share of the rows that the reader covers, per split and design (the run's `coverage_by_design`).

| Space | Train, anchor | Train, combinatorial | Validation, anchor | Validation, combinatorial | Test, anchor | Test, combinatorial | Train / validation / test, all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Narrow (6,523 genes) | 0.669 | 0.356 | 0.680 | 0.286 | 0.182 | 0.380 | 0.492 / 0.520 / 0.365 |
| Each line in its own genes (18,294 genes) | 0.781 | 0.813 | 0.786 | 0.915 | 0.191 | 0.792 | 0.799 / 0.838 / 0.747 |

## Stage one (the reader)

**Table 6.** Stage-one metrics of the reader on the SL lines with GeneEffect: training lines out of fold (415),
validation (12) and test (11). Higher is better, except the calibration slope (1 is ideal). Best per column in bold.

| Model | Per-line residual Spearman | Per-gene residual Spearman | Residual skill score | Calibration slope | Top-20 recall | Genes with per-gene Spearman ≥ 0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Narrow space, training (out of fold) | 0.178 | 0.188 | 0.030 | 1.84 | 0.070 | 993 |
| Each line in its own genes, training (out of fold) | **0.194** | **0.197** | **0.038** | 1.68 | **0.084** | **1,093** |
| Narrow space, validation | 0.183 | 0.170 | 0.029 | 1.60 | 0.079 | 1,606 |
| Each line in its own genes, validation | **0.207** | **0.182** | **0.040** | 1.78 | **0.108** | **1,706** |
| Narrow space, test | 0.193 | 0.157 | 0.027 | 1.70 | 0.055 | 1,649 |
| Each line in its own genes, test | **0.197** | **0.165** | **0.031** | 1.74 | 0.055 | **1,669** |
| Each line in its own genes, validation, lines permuted | 0.037 | 0.001 | −0.010 | −0.05 | 0.025 | 801 |
| Each line in its own genes, test, lines permuted | 0.056 | 0.032 | −0.002 | 0.27 | 0.005 | 905 |

## Co-detection

- **Rows with a value.** 1,503,228 of the 1,963,757 benchmark rows (76.6%): 75.1% of the training rows, 82.2% of the
  validation rows, 78.9% of the test rows. A row without a value (a drug anchor, or a gene that the line's source
  does not list) reads 0. Decision: an anchor list's pair keeps its co-detection when the anchor is a gene the line's
  source lists, because the value belongs to the pair, not to gene A alone.
- **Cells.** 18 to 58,324 cells per line (median 292), so a pair's count of cells with both genes is small in most
  lines.
- **Log ratio on the benchmark rows with a value** (quantiles 5%, 25%, 50%, 75%, 95%): real −0.524, −0.040, 0.000,
  +0.029, +0.333 (mean −0.028); shuffled −0.542, −0.046, 0.000, +0.020, +0.342 (mean −0.036). The shuffled median is
  0.000. The real and the shuffled log ratios of the same row correlate at 0.047; the shares correlate at 0.997. The
  median log ratio is 0.000 for SL and for non-SL rows, real and shuffled.
- **The expected share needs each gene's rate matched to its detection share.** With the rate from the line's mean
  profile (the formula of `detected_expected`), the shuffled log ratio has median −0.124 on the cache's pairs
  (real −0.036): a Poisson rate at each cell's depth predicts more detection than the overdispersed counts give.
  With each gene's rate matched to its detection share, the shuffled median is 0.000. The head reads the matched
  version; both are in the cache.
- **Head coefficients** (standardised inputs, penalty chosen on validation; every penalty gave the same validation
  recall): co-detection share +0.138, log ratio −0.032; with shuffled cells +0.138 and −0.068. The other inputs move
  little (expression of A 0.417 → 0.380, of B 0.365 → 0.314).

## Output checks

- The narrow-space rerun reproduces the backbone record: conditional 0.295 / 0.288, ablation 0.296 / 0.295, head
  0.374 / 0.322, head ablated 0.358 / 0.314, the training SL fraction 0.524 / 0.495. Its four score tables equal
  those of `conditional_seed0` on all 1,759,975 rows (largest difference 3.4e−13). The coverage flag changes on 9,762
  training rows only, where a training line does not measure a lost gene: training coverage 49.9% → 49.2%,
  validation and test unchanged (52.0% and 36.5%): passed.
- Wide space: 18,294 genes. Values of a held-out line that take the training mean: minimum 9, median 990, maximum
  10,027 of 18,294.
- No SL validation or test line is among the 979 fit lines in any run (the run stops otherwise): passed.
- The shuffled log ratio has median 0.000: passed. A gene that a source lists twice is summed into one column, so its
  shuffled detection share can differ from the real one (largest difference 0.008 of the cells).

Known side effects of the wide space, not removed: a filled value takes a rank slot in the per-line quantile
normalisation; a gene that half of the training lines lack has a bridge slope fitted on up to half filled values;
the expression components are fitted on rows that hold filled values. The edit of a gene that the line does not
measure moves a filled value, so its term is computed but the row is not covered.

## Reading

- **Reading each line in its own genes improves stage one.** Per-line residual Spearman 0.207 / 0.197 against 0.183 /
  0.193 (validation / test), and 0.194 against 0.178 out of fold on 415 training lines, with the other metrics up
  or equal, except the calibration slope. The gain is larger on validation than on test.
- **It does not improve the SL ranking.** The head in the wide space is +0.007 on validation and −0.017 on test
  against the narrow-space head, both intervals across zero; the conditional score is −0.003 / −0.015. Coverage of
  the held-out rows rises from 43.8% to 79.0%, but on the partner pairs neither score gains, and the conditional score
  minus its context-ablated score stays at zero on all pairs (+0.004 / +0.011). The gene space did not hide a line
  effect.
- **Co-detection adds nothing beyond the expression of each gene.** The head with co-detection equals the head
  (−0.000 / +0.010) and the head with shuffled co-detection (+0.004 / +0.001; validation interval [+0.001, +0.008]
  above zero, test across zero). The weight sits on the co-detection share, which shuffling keeps (correlation
  0.997); the log ratio, which shuffling destroys, carries almost no signal (real against shuffled 0.047) at a
  median of 292 cells per line. This is a null result for the hypothesis that a buffer shows as A and B in the same
  cells, at this cell count.
- **The training SL fraction stays far ahead** (0.524 / 0.495); the best model is 0.381 on validation (head, own
  genes) and 0.331 on test (head with co-detection).
- **Decision (plan, row C).** Neither the wide space nor co-detection helps the SL ranking, so neither joins the
  backbone.
