# SL backbone, round zero (seed 0)

SL protocol §11. Run on the H20 container on port 30838 on 2026-11-12, CPU only, code at `16455e1`:
`hpc/run.sh sl-conditional configs/sl/conditional.yaml --run-id conditional_seed0`, then
`hpc/run.sh sl-baselines configs/sl/benchmark_baselines.yaml --run-id benchmark_backbone_seed0`. Outputs are in
`outputs/sl/conditional/conditional_seed0/` and `outputs/sl/benchmark_baselines/benchmark_backbone_seed0/` on the
host. The first run (`*_nan_rows`, code `7eed720`) left the rows that the reader cannot edit unranked; it is kept on
the host and is not used here.

The reader is the default linear expression model fitted on the SL split's training side (979 fit lines, no SL
held-out line). The reader can edit both genes of 52% of the validation rows and 37% of the test rows; the other
rows tie (term 0).

## SL benchmark

**Table 1.** Recall in the top 10% (AUROC) on the multi-study SL benchmark, all held-out pairs, mean over lines
(validation 15, test 13). Higher is better. Best model in bold. The training SL fraction (the share of the study's
training lines that called the pair SL; 0.524 / 0.495) is left out of the tables; it is in the harness's report.

| Model | Validation | Test |
| --- | ---: | ---: |
| Pan-essentiality (gene-mean GeneEffect) | 0.369 (0.73) | 0.276 (0.68) |
| Predicted GeneEffect, linear expression model (held out) | 0.371 (0.73) | 0.276 (0.68) |
| Conditional score | 0.295 (0.63) | 0.288 (0.63) |
| Conditional score, context-ablated | 0.296 (0.63) | 0.295 (0.64) |
| Backbone head | **0.374** (0.74) | **0.322** (0.72) |
| Backbone head, context-ablated | 0.358 (0.74) | 0.314 (0.72) |
| KG4SL | 0.352 (0.69) | 0.306 (0.65) |
| Measured GeneEffect (reference) | 0.412 (0.74) | 0.269 (0.64) |
| Labels (ceiling) / random | 0.964 / 0.093 | 0.920 / 0.106 |

**Table 2.** Recall in the top 10% per stratum of held-out pairs (validation / test).

| Model | Seen, label agrees | Seen, label differs | New pair |
| --- | ---: | ---: | ---: |
| Pan-essentiality | 0.403 / 0.317 | **0.100 / 0.125** | 0.311 / 0.250 |
| Conditional score | 0.422 / 0.472 | 0.082 / 0.072 | 0.234 / 0.306 |
| Conditional score, context-ablated | 0.389 / 0.468 | 0.082 / 0.067 | 0.239 / 0.308 |
| Backbone head | 0.502 / 0.496 | 0.098 / 0.066 | **0.400 / 0.322** |
| Backbone head, context-ablated | 0.432 / 0.499 | 0.062 / 0.065 | 0.368 / 0.306 |
| KG4SL | **0.521 / 0.542** | 0.079 / 0.081 | 0.278 / 0.228 |
| Random | 0.119 / 0.103 | 0.081 / 0.087 | 0.103 / 0.106 |

On the covered rows only (the harness's Table 4): conditional score 0.219 / 0.238, its ablation 0.220 / 0.239;
head 0.246 / 0.223, its ablation 0.250 / 0.237.

**Table 3.** Per-line difference in recall in the top 10%, mean over lines with a 95% paired line bootstrap
(2,000 draws), validation | test. The line counts are the lines with both rows defined.

| Contrast | All pairs | Seen, label differs | New pair |
| --- | --- | --- | --- |
| Head − head, context-ablated | +0.016 [−0.005, +0.035] \| +0.007 [−0.011, +0.026] | +0.036 [−0.003, +0.104] \| +0.001 [−0.006, +0.008] | +0.031 [−0.011, +0.100] \| +0.016 [−0.024, +0.063] |
| Conditional − conditional, context-ablated | −0.001 [−0.011, +0.008] \| −0.007 [−0.027, +0.008] | −0.000 [−0.005, +0.004] \| +0.006 [+0.001, +0.011] | −0.005 [−0.012, +0.000] \| −0.002 [−0.045, +0.053] |
| Head − pan-essentiality | +0.005 [−0.076, +0.090] \| +0.046 [−0.030, +0.124] | −0.002 [−0.076, +0.088] \| −0.059 [−0.109, −0.019] | +0.088 [−0.083, +0.254] \| +0.072 [−0.019, +0.163] |
| Head − KG4SL | +0.022 [−0.032, +0.070] \| +0.016 [−0.025, +0.055] | +0.018 [−0.042, +0.104] \| −0.015 [−0.055, +0.017] | +0.122 [+0.042, +0.197] \| +0.094 [+0.017, +0.185] |

**Label agreement (the stage-two ceiling).** Two labels of one pair agree 98.3% of the time within a line and 96.1%
within a study across lines, but a second label is SL when the first is SL only 25% and 30% of the time.

## Stage one (the reader)

| Metric | Validation | Validation, lines permuted | Test | Test, lines permuted |
| --- | ---: | ---: | ---: | ---: |
| Per-line residual Spearman | 0.183 | 0.040 | 0.193 | 0.038 |
| Per-gene residual Spearman | 0.170 | 0.022 | 0.157 | 0.017 |
| Residual skill score | 0.029 | −0.011 | 0.027 | −0.005 |
| Calibration slope | 1.60 | 0.11 | 1.70 | 0.05 |
| Top-20 recall of the strongest dependencies | 0.079 | 0.008 | 0.055 | 0.005 |
| Genes with per-gene Spearman ≥ 0.3 | 1,606 | 892 | 1,649 | 876 |
| Whole-line Pearson (sanity) | 0.927 | 0.924 | 0.919 | 0.917 |

## Reading

- **The line adds nothing in linear form.** The conditional score equals its context-ablated score on every stratum
  and on the covered rows. In this reader, the response of B to A is the same in every line; the line enters only
  through how far A falls, and that does not move the ranking. This equality is by construction (in an anchor list
  the fall of A is one number for the whole list), so it is not evidence about the reader. The next module is the
  buffer-aware reader ([spec](../../docs/specs/2026-11-12-sl-buffer-aware-reader-design.md)).
- **Coverage.** The reader's gene space is the genes that every held-out line measures in pseudo-bulk (6,523). Each
  held-out line measures 8,295 to 18,768 genes; read in its own genes, 75% of the held-out rows have both genes.
- **The head has the best mean among these models** on all pairs (0.374 / 0.322) and on new pairs (0.400 / 0.322).
  Its gain over pan-essentiality is inside the interval on both splits; on new pairs it beats KG4SL on both splits.
- **The line context in the head is not shown to help.** The head minus its context-ablated version is +0.016 /
  +0.007 on all pairs, with every interval including zero (Table 3). Its weight is mostly on the expression of A and
  B and the design, which is gene-level information. On the label-differs pairs (the context test) it falls below
  pan-essentiality on test (−0.059 [−0.109, −0.019]).
- **No model ranks the label-differs pairs above random.** That stratum is the context test, and it stays open.
- **The named pairs rank near the top in almost every line** (percentile 0.93–1.00), SL or not: the reader knows the
  paralog pairs, not which line needs them. In two lines (ACH-000510, ACH-000757) SMARCA2–SMARCA4 ties at 0 (no fall).
- **Stage one is under-dispersed** (calibration slope 1.6–1.7) and finds 6–8% of a line's 20 strongest
  dependencies. The predictable-gene count is noisy on 13–15 lines (892 genes pass with permuted lines).
- **Not measured:** the stage-one ceiling (two DepMap screens of one line). The per-screen GeneEffect is on the H20
  (`ScreenGeneEffect.csv`, `CRISPRScreenMap.csv`; 136 lines of the SL split have two or more screens); the command
  comes with the buffer-aware reader.
