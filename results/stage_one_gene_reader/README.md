# Stage one: the per-gene reader

Round B of the [PIE and single-cell plan](../../docs/specs/2026-11-12-pie-and-single-cell-plan.md) (protocol §18). The
question: does a reader with a weight vector for each gene on a shared nonlinear encoder of the line do better than the
default linear model refitted on the frame? The reader predicts the linear model's prediction plus `u_g · h(line)`,
with `u_g` zero at the start, so its epoch 0 is the linear model. Each row is a main run plus five fold runs, one config
each under `configs/stage_one/gene_reader/`; the report is `configs/stage_one/gene_reader/report_gene_reader.yaml`.

Run on the H20 container on port 30757, with outputs in `outputs/stage_one/gene_reader/` there. The code is at
`a35254e` (branch `stage-one-gene-reader`, later rebased onto `main` as `fd7bc79`; the training queue ran at
`502db67`, which differs from it only in the report's ridge fix). The full report is
[report_gene_reader_results.md](report_gene_reader_results.md), a copy of the host's `report_gene_reader/results.md`.
Seed 0. Nothing here is SL evidence.

## Benchmark

The stage-one inputs were rebuilt on 2026-11-13 for the final split: a frame of 987 training lines (384 single-cell,
603 bulk-only) in five folds (195–199 lines each), 27 validation and 28 test lines. The scored genes are the 4,344
selective genes (54 of them outside the decode panel); the mean channel has 9,929 genes.

## What ran

| Step | Finished (UTC) | Notes |
| --- | --- | --- |
| stage-one inputs (lines, cells, mean) | 2026-11-13 05:21 | 21 new cell files; 455 lines with cells |
| linear model refitted on the frame | 05:47 | `linear_model_reference_20261113`, with each fold's own predictions; 12 min on CPU |
| queue | 06:00 | 48 runs, two per GPU, none failed; 12 min |
| report | 07:09 | |

## Validation and test

Per-line residual Spearman over the selective genes (the selector, higher is better) and within-gene R² pooled over
the selective genes (higher is better). The best value in each column is in bold. Every metric is in the full report.

| Row | Val per-line | Test per-line | Val within-gene R² | Test within-gene R² |
| --- | ---: | ---: | ---: | ---: |
| base | 0.0177 | 0.0544 | −0.0015 | 0.0025 |
| default linear model refitted on the frame | 0.2261 | 0.2266 | 0.0495 | 0.0482 |
| per-gene reader on components | 0.2261 | 0.2266 | 0.0495 | 0.0482 |
| per-gene reader on expression | 0.2261 | 0.2266 | 0.0495 | 0.0482 |
| per-gene reader on components, weight decay 1.0 | **0.2278** | **0.2311** | **0.0548** | **0.0535** |
| per-gene reader on expression, weight decay 1.0 | 0.2261 | 0.2266 | 0.0495 | 0.0482 |
| each of the four controls (line input permuted) | 0.2261 | 0.2266 | 0.0495 | 0.0482 |

Seven of the eight main runs chose epoch 0, which is the linear model on the selective genes. Their residual Pearson,
per-line Pearson and RMSE differ from the linear model row only because the reader predicts the base outside the
selective genes.

## Contrasts

Left minus right, per-line residual Spearman, 95% paired line-bootstrap interval.

| Contrast | Validation | Test |
| --- | --- | --- |
| reader on components, weight decay 1.0 − linear model | +0.0018 [−0.0157, +0.0221] | +0.0045 [−0.0106, +0.0222] |
| reader on components, weight decay 1.0 − its control | +0.0018 [−0.0157, +0.0221] | +0.0045 [−0.0106, +0.0222] |
| every other arm − linear model, and − its control | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

## Training

| Row | Chosen epoch / epochs | Train per-line | Val per-line | Val per-line at epoch 1 |
| --- | --- | ---: | ---: | ---: |
| reader on components | 0 / 5 | 0.1911 | 0.2261 | 0.2249 |
| reader on expression | 0 / 5 | 0.1911 | 0.2261 | 0.2004 |
| reader on components, weight decay 1.0 | 2 / 7 | 0.2446 | 0.2278 | 0.2250 |
| reader on expression, weight decay 1.0 | 0 / 5 | 0.1911 | 0.2261 | 0.2113 |
| control of the reader on expression | 0 / 5 | 0.1911 | 0.2261 | 0.1420 |

Train per-line is in sample for the chosen epoch (at epoch 0 the frame lines read the linear model out of fold). The
epoch-1 column is from each main run's `row.json` history. The training loss falls each epoch while the validation
score falls from epoch 1 on.

## Reading

- **Null result: no arm beats the linear model.** Seven of eight main runs select epoch 0 and return the linear model
  exactly; the one that moves (components, weight decay 1.0) gains +0.002 / +0.005 with intervals across zero.
- **Every arm overfits within one or two epochs.** Training fit rises (train per-line 0.24, within-gene R² 0.068 for
  the moving arm) while validation falls; the expression input falls fastest (0.226 to 0.200 after one epoch).
- **The line input carries little beyond the linear model.** The permuted controls fall more after one epoch (0.142 on
  expression) than the real inputs, so the real input is not noise, but no real input lifts validation above epoch 0.
- **Decision (plan, row B):** no arm is above the linear model, so the per-gene reader does not become the backbone's
  reader.
