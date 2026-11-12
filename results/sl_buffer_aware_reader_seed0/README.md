# SL backbone, buffer-aware reader (seed 0)

SL protocol §12 ([spec](../../docs/specs/2026-11-12-sl-buffer-aware-reader-design.md)). Run on the H20 container on
port 30838 on 2026-11-12, CPU only (16 threads), code at `5f35ade`:
`hpc/run.sh sl-conditional configs/sl/conditional_buffer.yaml --run-id conditional_buffer_seed0` (14.9 min), then
`hpc/run.sh sl-baselines configs/sl/benchmark_baselines.yaml --run-id benchmark_buffer_seed0` (1.6 min). Outputs are
in `outputs/sl/conditional/conditional_buffer_seed0/` and `outputs/sl/benchmark_baselines/benchmark_buffer_seed0/` on
the host. The harness also reads the backbone run's tables (`conditional_seed0`), so both readers are in one report.

The buffer-aware reader is the default linear expression model with the partner block (each gene's paralogs and
complex members in the line), fitted on the SL split's training side (979 fit lines, no SL held-out line). The
reader can edit both genes of 52% of the validation rows and 37% of the test rows, as in the backbone run.

## SL benchmark

**Table 1.** Recall in the top 10% (AUROC) on the multi-study SL benchmark, all held-out pairs, mean over lines.
Higher is better. Best model in bold. The training SL fraction (the share of the study's training lines that called
the pair SL) and the checks are reference rows.

| Model | Validation | Test |
| --- | ---: | ---: |
| Pan-essentiality (gene-mean GeneEffect) | 0.369 (0.73) | 0.276 (0.68) |
| Predicted GeneEffect, default linear expression model (held out) | 0.371 (0.73) | 0.276 (0.68) |
| Conditional score, default reader | 0.295 (0.62) | 0.288 (0.63) |
| Conditional score, default reader, context-ablated | 0.296 (0.63) | 0.295 (0.64) |
| Backbone head, default reader | 0.374 (0.74) | 0.322 (0.72) |
| Backbone head, default reader, context-ablated | 0.358 (0.74) | 0.314 (0.72) |
| Conditional score, buffer-aware reader | 0.293 (0.64) | 0.287 (0.64) |
| Conditional score, buffer-aware reader, context-ablated | 0.300 (0.64) | 0.291 (0.65) |
| Backbone head, buffer-aware reader | **0.383** (0.74) | **0.328** (0.72) |
| Backbone head, buffer-aware reader, context-ablated | 0.373 (0.74) | 0.321 (0.72) |
| KG4SL | 0.352 (0.69) | 0.306 (0.65) |
| Training SL fraction (reference) | 0.524 (0.77) | 0.495 (0.77) |
| Measured GeneEffect (reference) | 0.412 (0.74) | 0.269 (0.64) |
| Labels (ceiling) / random | 0.964 / 0.093 | 0.920 / 0.106 |

**Table 2.** Recall in the top 10% on the two subsets of held-out rows, ranked inside the subset in each screen,
mean over lines (validation / test). Partner pairs (A is a paralog or complex member of B, or B of A): 81,657 rows,
15.0% of the held-out rows, 16 / 15 lines. No gene a dependency alone (measured GeneEffect of B, and of A for a pair,
at or above −0.5; lines with GeneEffect): 286,650 rows, 52.7%, 12 / 11 lines. Labels (ceiling): 0.923 / 0.864 and
0.953 / 0.890; random: 0.167 / 0.073 and 0.094 / 0.094. Best model per split in bold.

| Model | Partner pairs | No gene a dependency alone |
| --- | ---: | ---: |
| Pan-essentiality | 0.273 / 0.267 | 0.359 / 0.254 |
| Conditional score, default reader | 0.311 / 0.318 | 0.327 / 0.301 |
| Conditional score, default reader, context-ablated | 0.323 / 0.322 | 0.324 / 0.287 |
| Backbone head, default reader | 0.332 / 0.279 | 0.408 / 0.329 |
| Backbone head, default reader, context-ablated | 0.344 / 0.288 | 0.381 / **0.338** |
| Conditional score, buffer-aware reader | 0.322 / 0.330 | 0.315 / 0.308 |
| Conditional score, buffer-aware reader, context-ablated | 0.326 / **0.334** | 0.321 / 0.292 |
| Backbone head, buffer-aware reader | 0.335 / 0.295 | **0.418** / 0.332 |
| Backbone head, buffer-aware reader, context-ablated | **0.402** / 0.309 | 0.401 / 0.333 |
| KG4SL | 0.346 / 0.297 | 0.359 / 0.325 |
| Training SL fraction (reference) | 0.548 / 0.478 | 0.587 / 0.511 |

**Table 3.** Per-line difference in recall in the top 10%, score minus its context-ablated score, mean over lines with
a 95% line bootstrap (1,000 draws), validation | test, line count in parentheses.

| Contrast | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Conditional, buffer-aware reader | All pairs | −0.006 [−0.022, +0.008] (17) | −0.003 [−0.025, +0.016] (16) |
| | Partner pairs | −0.004 [−0.018, +0.009] (16) | −0.004 [−0.029, +0.020] (15) |
| | No gene a dependency alone | −0.007 [−0.029, +0.014] (12) | +0.016 [+0.001, +0.031] (11) |
| | Seen, label agrees | +0.034 [−0.097, +0.196] (15) | −0.010 [−0.031, +0.009] (12) |
| | Seen, label differs | −0.004 [−0.010, +0.002] (15) | +0.004 [−0.002, +0.009] (13) |
| | New pair | −0.001 [−0.015, +0.010] (8) | −0.002 [−0.047, +0.051] (10) |
| Head, buffer-aware reader | All pairs | +0.010 [−0.008, +0.028] (17) | +0.006 [−0.013, +0.028] (16) |
| | Partner pairs | −0.067 [−0.194, +0.012] (16) | −0.014 [−0.044, +0.015] (15) |
| | No gene a dependency alone | +0.017 [+0.001, +0.035] (12) | −0.001 [−0.018, +0.014] (11) |
| | Seen, label agrees | +0.085 [+0.003, +0.227] (15) | +0.007 [−0.011, +0.024] (12) |
| | Seen, label differs | +0.001 [−0.004, +0.005] (15) | +0.004 [−0.005, +0.013] (13) |
| | New pair | +0.033 [−0.004, +0.095] (8) | +0.029 [−0.015, +0.093] (10) |
| Conditional, default reader | All pairs | −0.001 [−0.011, +0.008] (17) | −0.007 [−0.029, +0.007] (16) |
| | Partner pairs | −0.012 [−0.027, +0.003] (16) | −0.005 [−0.021, +0.013] (15) |
| | No gene a dependency alone | +0.002 [−0.010, +0.014] (12) | +0.013 [+0.001, +0.029] (11) |
| | Seen, label agrees | +0.033 [−0.092, +0.202] (15) | +0.005 [−0.010, +0.023] (12) |
| | Seen, label differs | −0.000 [−0.004, +0.004] (15) | +0.006 [+0.001, +0.011] (13) |
| | New pair | −0.005 [−0.012, +0.000] (8) | −0.002 [−0.044, +0.047] (10) |
| Head, default reader | All pairs | +0.016 [−0.004, +0.035] (17) | +0.007 [−0.011, +0.027] (16) |
| | Partner pairs | −0.012 [−0.033, +0.010] (16) | −0.009 [−0.033, +0.009] (15) |
| | No gene a dependency alone | +0.027 [+0.009, +0.053] (12) | −0.010 [−0.040, +0.010] (11) |
| | Seen, label agrees | +0.071 [−0.023, +0.216] (15) | −0.003 [−0.027, +0.018] (12) |
| | Seen, label differs | +0.036 [−0.003, +0.106] (15) | +0.001 [−0.006, +0.009] (13) |
| | New pair | +0.031 [−0.013, +0.098] (8) | +0.016 [−0.022, +0.068] (10) |

On the covered rows only (43.8% of the held-out rows; the harness's Table 4): buffer-aware conditional score 0.210 /
0.231, its ablation 0.215 / 0.225; buffer-aware head 0.257 / 0.244, its ablation 0.262 / 0.238.

## Stage one (the reader) and its ceiling

**Table 4.** Stage-one metrics on the SL held-out lines with GeneEffect (validation / test). The ceiling is one
DepMap screen's residual as the prediction of another screen's residual of the same line, on the lines with two
screens: 4 validation and 8 test lines (124 training lines: per-line residual Spearman 0.246). The ceiling is on fewer
lines than the readers. Best reader in bold; the ceiling is a reference row.

| Model | Per-line residual Spearman | Per-gene residual Spearman | Residual skill score | Calibration slope | Top-20 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default linear expression model | 0.183 / 0.193 | 0.170 / **0.157** | 0.029 / 0.027 | 1.60 / 1.70 | 0.079 / 0.055 |
| Buffer-aware reader | **0.183** / **0.194** | **0.171** / 0.157 | **0.030** / **0.027** | 1.60 / 1.70 | 0.079 / 0.055 |
| Two screens of one line (ceiling) | 0.265 / 0.367 | 0.194 / 0.345 | −0.073 / 0.028 | 0.43 / 0.52 | 0.100 / 0.163 |

## Output checks

- No SL validation or test line is among the 979 fit lines (the run stops otherwise): passed.
- The default-reader rows in this report reproduce the backbone record (conditional 0.295 / 0.288, ablation 0.296 /
  0.295, head 0.374 / 0.322, head ablated 0.358 / 0.314, training SL fraction 0.524 / 0.495): passed.
- The context-ablated buffer-aware score has zero spread across held-out lines for each pair and design (396,436
  pairs and designs, largest spread 0.0): passed.
- Outside the partner tables, the buffer-aware conditional score follows the default reader's score (covered held-out
  rows, 203,301: Pearson 0.994, Spearman 0.9995). On the partner pairs it differs (13,806 rows: Pearson 0.866,
  Spearman 0.587).

## Reading

- **The buffer-aware reader does not make the line matter.** On the partner pairs, where the module can act, the
  conditional score minus its context-ablated score is −0.004 on validation and −0.004 on test, both intervals
  around zero. On all pairs it is −0.006 / −0.003. This is a null result. The reader can edit both genes in only
  19.5% of the partner rows of the genetic screens (15,848 of 81,362), but the result is the same on those rows:
  +0.004 [−0.008, +0.015] / +0.003 [−0.024, +0.028]. The gene space does not explain the null.
- **The reader changes the scores of partner pairs but not their ranking quality.** Its conditional score differs from
  the default reader's score on the partner pairs (Spearman 0.587) and is almost the same elsewhere (0.9995), so the
  partner block acts where it should. On the partner pairs it gains +0.011 / +0.012 over the default reader's
  conditional score, and its context-ablated version gains +0.003 / +0.012: the gain does not come from the line.
- **The head on the buffer-aware terms is the best model** (0.383 / 0.328), +0.009 / +0.006 over the head on the
  default terms, and its gain over its ablation is inside the interval (+0.010 / +0.006). On the partner pairs its
  ablation is higher (0.402 / 0.309 against 0.335 / 0.295). The training SL fraction stays far ahead (0.524 / 0.495).
- **No gene a dependency alone.** Without the rows where a gene is a dependency alone, the differences from the
  ablation are small and do not agree between splits (conditional −0.007 / +0.016; head +0.017 / −0.001).
- **No model ranks the label-differs pairs above random** (all near 0.06–0.09 against random 0.08 / 0.09).
- **Stage one.** The partner block does not change the predicted GeneEffect (per-line residual Spearman 0.183 / 0.194
  against 0.183 / 0.193). Two DepMap screens of one line agree at 0.246 on the 124 training lines with two screens
  (0.265 / 0.367 on 4 and 8 held-out lines). This number is the agreement of two noisy screens, not the top for a
  reader. If the noise of the two screens is independent, a prediction of the true dependency agrees with one screen
  at about the square root of it, about 0.5. The reader (0.18–0.19) is thus far below what the data permit.
- **Decision (spec §4).** The conditional score is equal to its context-ablated score on the partner pairs, so the
  next module is the context editor (PIE's profile after A's loss).
