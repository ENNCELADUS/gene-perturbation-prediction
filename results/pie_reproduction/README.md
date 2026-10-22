# PIE reproduction

Task 1 of the [stage-one design](../../docs/specs/2026-10-16-pie-geneeffect-ladder-design.md): PIE's released
checkpoints, run through the vendored PIE on PIE's own data, splits and sources, against the metric files released
with them. Run `pie_reproduction_20261016`, finished 2026-10-16 10:33 UTC on the H20 container on port 30838
(one H20, driver 550.144.03), `hpc/run.sh pie-reproduce --run-id pie_reproduction_20261016` at `84c4d8f`, vendored
PIE `21a526a` in `.venv-pie` (Python 3.12.13, torch 2.10.0+cu128). Tables: [results_run.md](results_run.md) (the
run's `results.md`), [comparison.csv](comparison.csv) (every context and metric), [granular_comparison.csv](granular_comparison.csv)
(per pair metric), [run.json](run.json) (pins, environment, step times).

## What ran

- `arcinstitute/PIE_replogle_wdataset@055c7a2`: the four folds (HepG2, Jurkat, K562, RPE1 held out), each on its
  `unseen_ctx` test split.
- `arcinstitute/PIE_replogle_xdataset@f6da960` (trained on Tahoe, Jiang, ARC VCC 25 and X-Atlas/Orion): Replogle
  `test_seen` and `test_unseen`, zero-shot.
- Splits `PIE_splits@396ab95`; datasets and sources at the revisions each checkpoint pins (27 assets in
  `data/pie/hf/`). Evaluation rebuilt PIE's
  same-perturbation evidence from each checkpoint's training split; `best_auprc.ckpt` loaded strictly in all five.

## Results

Reproduced minus released, `all` row (mean over cell lines):

| Checkpoint, test split | Rows | AUPRC released / reproduced | Spearman LFC released / reproduced | Largest \|Δ\| over the six metrics |
| --- | ---: | --- | --- | ---: |
| wdataset HepG2, `unseen_ctx` | 867 | 0.443248 / 0.443248 | 0.711455 / 0.711454 | 5.6e-06 |
| wdataset Jurkat, `unseen_ctx` | 978 | 0.375403 / 0.375403 | 0.615647 / 0.615647 | 4.7e-06 |
| wdataset K562, `unseen_ctx` | 884 | 0.422211 / 0.422211 | 0.662161 / 0.662161 | 2.6e-06 |
| wdataset RPE1, `unseen_ctx` | 983 | 0.421119 / 0.421119 | 0.637869 / 0.637870 | 5.4e-05 |
| xdataset, `test_seen` | 2,286 | 0.380246 / 0.380246 | 0.512979 / 0.512971 | 8.5e-06 |
| xdataset, `test_unseen` | 3,173 | 0.246348 / 0.246348 | 0.269830 / 0.269794 | 3.6e-05 |

The wdataset model card's `unseen_ctx` summary (mean ± sample SD over the four folds) is regenerated exactly:
AUPRC 0.4155 ± 0.0286 and Spearman LFC 0.6568 ± 0.0411, released and reproduced alike. Over every context the
largest difference is 1.4e-04 (xdataset `test_unseen`, RPE1, Spearman LFC). Row and pair sets are identical, and no
pair metric is defined on one side only.

Per pair, the rank-based metrics move in whole steps when two near-tied predictions swap order (bf16 inference on an
H20 against PIE's H100s): the discrimination L1 score by multiples of one rank over a line's perturbations (up to
0.011; 591 of 983 RPE1 fold pairs differ by more than 1e-3, the `all` row by 5e-5), and a Spearman over few
significant genes by more (ZNF706 in RPE1, `test_unseen`: 0.119; 3 of 2,903 pairs above 1e-3). AUPRC differs by at
most 4.3e-04 on any pair.

## Reading

The vendored copy, `.venv-pie`, the data path and the sources reproduce PIE's released results to rounding on this
hardware, so Task 2 can read the frozen cross-dataset checkpoint through this copy.

The four wdataset folds were scored at 09:41–09:43 UTC by an earlier `evaluate` process; `run.json` records the step
times of the process that finished the xdataset.
