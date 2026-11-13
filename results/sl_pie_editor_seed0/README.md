# SL backbone, PIE editor (seed 0)

SL protocol §13 (plan [round A](../../docs/specs/2026-11-12-pie-and-single-cell-plan.md)). Run on the H20 container on
port 30838 on 2026-11-13. PIE's side in `.venv-pie`, code at `a58946b` (pass) and `1bb062b` (own-task scores):
`hpc/run.sh sl-pie-pass configs/sl/pie_editor.yaml --steps tables`, then `--steps pass --shard I/4` on four H20 GPUs
(22 min, 0.35 s per lost gene for 86 lines per process), then `--steps measured,rescore,own_task` (one GPU, 13 min).
Then on the CPU (16 threads), code at `a58946b`: `hpc/run.sh sl-pie-editor configs/sl/pie_editor.yaml --run-id
pie_editor_seed0` (35 min) and, at `8168c3f`, `hpc/run.sh sl-baselines configs/sl/benchmark_pie_editor.yaml --run-id
benchmark_pie_editor_seed0` (2 min). Outputs: `data/sl_pie/v1/` (tables, `pie/change/`, `measured/`, `pie/run.json`,
`pie/own_task/pairs.csv`), `outputs/sl/pie_editor/pie_editor_seed0/` and
`outputs/sl/benchmark_baselines/benchmark_pie_editor_seed0/`. The harness reads the simple edit's rows from the backbone
run (`conditional_seed0`).

The editor is the only change. The reader is the default linear expression model of the backbone run. PIE (the frozen
cross-dataset checkpoint, DepMap source masked) predicts the profile change of each of the 86 screened lines after
each of 13,950 lost genes, on the reader's 6,523 expression genes. The change is moved to the raw pseudo-bulk's unit by
a per-gene slope fitted on 415 SL training lines (quartiles 0.86, 2.66, 4.19, 6.61, 34.3), added to the line's raw
pseudo-bulk, and then goes through the reader's quantile normalisation, the line's bridge and the reader. PIE's profile
of another line uses one seeded permutation of the 86 lines (no line keeps its own).

**Context text.** The OpenAI account had no credits, so the Cellosaurus text could be embedded only for the 18
screened lines whose text PIE's release already holds (Cellosaurus 57.0). The other 66 lines, Jurkat among them, take
PIE's no-source path: for them PIE sees the line only through its expression input.

## SL benchmark

**Table 1.** Recall in the top 10% (AUROC) on the multi-study SL benchmark, all held-out pairs, mean over lines. Higher
is better. Best model in bold. The training SL fraction, the measured Perturb-seq rows and the checks are reference
rows. The PIE rows on "rows the simple edit covers" are 0 elsewhere, as the simple edit's rows are.

| Model | Validation | Test |
| --- | ---: | ---: |
| Predicted GeneEffect, default linear expression model (held out) | 0.371 (0.73) | 0.276 (0.68) |
| Conditional score, simple edit | 0.295 (0.62) | 0.288 (0.63) |
| Conditional score, simple edit, context-ablated | 0.296 (0.63) | 0.295 (0.64) |
| Backbone head, simple edit | **0.374** (0.74) | **0.322** (0.72) |
| Backbone head, simple edit, context-ablated | 0.358 (0.74) | 0.314 (0.72) |
| Conditional score, PIE's profile (rows the simple edit covers) | 0.252 (0.54) | 0.200 (0.52) |
| Conditional score, PIE's profile of another line (same rows) | 0.230 (0.53) | 0.196 (0.51) |
| Backbone head, PIE's profile (same rows) | 0.317 (0.72) | 0.248 (0.69) |
| Backbone head, PIE's profile of another line (same rows) | 0.317 (0.72) | 0.244 (0.69) |
| Conditional score, PIE's profile (every row PIE can edit) | 0.158 (0.54) | 0.149 (0.53) |
| Conditional score, PIE's profile of another line (every row PIE can edit) | 0.144 (0.52) | 0.138 (0.52) |
| Backbone head, PIE's profile (every row PIE can edit) | 0.317 (0.72) | 0.248 (0.69) |
| Backbone head, PIE's profile of another line (every row PIE can edit) | 0.316 (0.72) | 0.243 (0.69) |
| Training SL fraction (reference) | 0.524 (0.77) | 0.495 (0.77) |
| Conditional score, measured Perturb-seq profile of the line (reference, 2 test lines) | — | 0.114 (0.51) |
| Conditional score, measured Perturb-seq profile of another line (reference, 2 test lines) | — | 0.074 (0.52) |
| Labels (ceiling) / random | 0.964 / 0.093 | 0.920 / 0.106 |

**Table 2.** Per-line difference in recall in the top 10%, mean over lines with a 95% line bootstrap (1,000 draws),
validation | test, line count in parentheses. All held-out pairs, then the partner pairs.

| Contrast | Pairs | Validation | Test |
| --- | --- | ---: | ---: |
| Conditional: PIE's profile − simple edit | All | −0.043 [−0.094, +0.004] (17) | −0.088 [−0.137, −0.044] (16) |
| | Partner | −0.069 [−0.106, −0.030] (16) | −0.105 [−0.163, −0.043] (15) |
| Conditional: PIE's profile − PIE's profile of another line | All | +0.022 [+0.001, +0.046] (17) | +0.004 [−0.017, +0.023] (16) |
| | Partner | +0.030 [+0.015, +0.047] (16) | −0.006 [−0.022, +0.008] (15) |
| Head: PIE's profile − simple edit | All | −0.057 [−0.102, −0.022] (17) | −0.073 [−0.120, −0.031] (16) |
| Head: PIE's profile − PIE's profile of another line | All | +0.000 [−0.011, +0.014] (17) | +0.004 [−0.002, +0.010] (16) |
| Conditional, every row PIE can edit: PIE's profile − another line's | All | +0.013 [−0.003, +0.031] (17) | +0.012 [−0.006, +0.029] (16) |
| Head, every row PIE can edit: PIE's profile − another line's | All | +0.001 [−0.006, +0.008] (17) | +0.005 [+0.001, +0.010] (16) |
| Measured profile of the line − of another line (reference) | All | — | +0.040 [+0.000, +0.080] (2) |

The measured reference rows: Jurkat (ACH-000995, Replogle, against K562) 14,535 held-out rows whose lost genes both
lines perturbed (1,068 perturbations in both); HCT116 (ACH-000971, X-Atlas/Orion, against HEK293T) 200 rows; HEK293T
(CVCL_0063) 0 rows (all its rows are pairs, and no pair has both genes perturbed in both lines). On these 14,735 rows
the training SL fraction scores 0.305 and the labels 0.843 (test).

On the rows each editor can edit (the harness's Table 4): the simple edit's conditional score 0.219 / 0.238 on 43.8% of
the held-out rows; PIE's conditional score 0.131 / 0.139 on the same rows and 0.157 / 0.152 on the 93.6% it can edit.

## PIE on its own task

**Table 3.** PIE's four Replogle folds (one line held out each; 867–983 perturbations per fold), DepMap source
masked: correlation over genes of each (line, perturbation) between a prediction and the measured response
(`delta_p`), mean over pairs, then over folds. Higher is better. The two deviation rows subtract the mean measured
response of the other lines from both sides. Best on each target in bold.

| Prediction | Measured response, Pearson | Measured response, Spearman | Deviation from the other lines' mean, Pearson | Deviation, Spearman |
| --- | ---: | ---: | ---: | ---: |
| PIE's predicted response of the held-out line | 0.303 | **0.231** | 0.264 | 0.268 |
| Mean measured response of the other lines | **0.331** | **0.231** | — (zero) | — (zero) |
| One other line's measured response (control) | — | — | 0.003 | −0.014 |
| The other lines' mean shrunk toward zero (control) | — | — | **0.436** | **0.451** |

With the DepMap source unmasked (the released setting): PIE 0.320 / 0.246, deviation 0.255 / 0.253. The masked
re-score of the cross-dataset checkpoint reproduces the earlier record: Spearman LFC 0.450 on seen perturbations and
0.114 on unseen ones (binary AUPRC 0.359 / 0.231).

## Output checks

- A zero change reproduces each line's row (largest difference 0.0 over all 454 SL lines) and gives zero terms
  (largest |term| 4.2e−17 over 38,657 terms of 20,000 held-out rows). A test checks the same on synthetic lines.
- The existing rows reproduce the buffer-aware record: conditional 0.295 / 0.288, its ablation 0.296 / 0.295, head
  0.374 / 0.322, training SL fraction 0.524 / 0.495: passed.
- Coverage: PIE can edit 93.6% of the held-out rows (93.8% validation, 93.5% test) against the simple edit's 43.8%
  (52.0% / 36.5%).
- The edit survives normalisation. On 2,000 held-out edits the genes whose raw value moved change rank by a median of
  61 of 6,523; 0.83 of the raw change's norm is left after normalisation, and the raw and normalised changes correlate
  at 0.66 (medians over edits). The terms are small: median |term of A's loss on B| 0.0097 residual SD for PIE's
  profile and 0.013 for another line's, on covered held-out rows.
- PIE's batches agree with PIE's own data path in float32 (largest delta-p logit difference 0.0); our bf16 decode is
  within PIE's own bf16 error (0.30–0.44 against 0.32–0.49).

## Reading

- **PIE's profile is below the simple edit.** On the rows the simple edit covers, the conditional score loses 0.043 /
  0.088 and the head 0.057 / 0.073 (test intervals below zero). Its conditional score is near random on the rows it
  covers (AUROC 0.54 / 0.53).
- **PIE's profile does not carry the line.** Against PIE's profile of another line, the conditional score gains
  +0.022 [+0.001, +0.046] on validation and +0.004 on test; the head gains 0.000 / +0.004. The validation gain does
  not repeat on test. This is a null result.
- **The normalisation does not erase the edit.** The change moves the edited genes by 61 rank places and keeps 83% of
  its norm, so the null is not an artefact of the quantile normalisation.
- **On its own task PIE holds no line-specific response.** The mean measured response of the other lines predicts the
  held-out line as well as PIE (Pearson 0.331 against 0.303; Spearman 0.231 both). PIE's deviation from that mean
  correlates with the measured deviation (0.26), but the other lines' mean shrunk toward zero does better (0.44), so
  the deviation score does not show line knowledge.
- **The measured own-line profile is above another line's, on little data.** +0.040 [+0.000, +0.080] on two test
  lines (mostly Jurkat, 14,535 rows); both rows are near random (0.114 and 0.074 against random 0.10), below the
  training SL fraction on the same rows (0.305).
- **Limit.** 66 of 86 lines had no context text (no OpenAI credits), so for them PIE read the line only from its
  expression input. A rerun of `sl-pie-pass --steps pass` with full context text (22 min on four GPUs) and of the
  editor would remove this limit.
- **Decision (plan, after round A).** PIE's profile is neither above the simple edit nor above its permuted version,
  and the measured own-line profile is above another line's only at the edge of its interval on two lines, so the
  second row applies with weak support: round D (a line-specific PIE), with the third row (repeat with the per-gene
  reader) not excluded.
