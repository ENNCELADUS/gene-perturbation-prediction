# Tx1 GMM64-ridge CPU baseline

Completed 2026-09-07 on 30030, host `apg3op6hp3v99-0`, revision
`7d82afd0fdcf08b407515bc38a0803cec72a852d`. Both GMM and reference commands exited
with code 0; supervisor 72951 and workers 72952/73029 exited. CPU-only execution
used eight threads with GPUs hidden; unrelated GPU jobs were preserved.

**Result: the fixed GMM64 / alpha=1 candidate does not improve over Tx1 PCA8-ridge.**
This is one seed and a validation comparison. Test was not evaluated.

## Matched validation comparison

GMM and Tx1 PCA-ridge have exactly the same 479,084 observed (ModelID, gene_symbol)
keys over 27 validation lines, identical absolute/residual targets (maximum difference
0), and the same 4,447 train-defined variable genes. All fitting uses the 170 labeled
training lines and the same prepared inputs. The reference command also exported the
existing HVG, nearest-line, copy-prior and gene-mean controls.

| Method | Huber ↓ | RMSE ↓ | Residual Pearson ↑ | Residual Spearman ↑ | Macro per-gene SD ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tx1 GMM64-ridge, alpha=1 | 0.054734 | 0.344255 | 0.078027 | 0.081147 | 1.312647 |
| Tx1 PCA8-ridge, alpha=1 | 0.016281 | 0.181303 | 0.132973 | 0.123286 | 0.272707 |
| HVG PCA8-ridge, alpha=1 | 0.016451 | 0.182288 | 0.080383 | 0.087934 | 0.203722 |
| Gene mean | 0.016317 | 0.181596 | undefined | undefined | 0 |

The GMM train residual Pearson/Spearman are 0.662724/0.615166; train Huber is
0.008793 over 3,015,332 observed pairs. The large train/validation gap and worse
validation amplitude/error demonstrate poor transfer for this configuration, despite
numerical convergence. They do not identify a unique mechanism or rule out all GMM
representations.

## Distribution diagnostics

The diagonal GMM converged in 22 iterations on 21,760 cached cell rows, 128 from
each training line. These are cached rows, not a claim of distinct biological cells.
Mean training assignment confidence is 0.999969. Median effective component count
per bag is 1.663 in train and 2.962 in validation; median NLL rises from 2836.52 to
3414.63. These observations suggest strongly concentrated assignments and a shift in
the feature distribution; no ablation establishing the cause was run. SD ratios
are averages of per-gene prediction SD / target SD, not a global variance ratio
and not a calibration coefficient.

## Artifacts and reproduction

Remote root:
`/2023533015/VCC_Project/outputs/baselines/tx1_gmm_vs_pca_20260907T111054Z`.

- `gmm/model.joblib`, `gmm/diagnostics.json`, `gmm/run.json`
- `gmm/evaluation/train/` and `gmm/evaluation/val/`: P0 metrics, predictions,
  per-gene/per-line details and context features
- `reference_val/`: all six reference methods and their P0 metrics/details
- `run.log`, `launch.json`, `gmm.exit`, `reference.exit`, `comparison.json`

Model SHA256: `9165f640ec04774169127a33d4edc8c810d74dafb420f50c9a505879fa673c35`.
Compact metrics, exact alignment checks and process/launch evidence:
[evidence.json](evidence.json).

From that revision at the repository root, use the installed interpreter with
`CUDA_VISIBLE_DEVICES=` and `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8`:

```bash
.venv-tx1/bin/python -m src.experiments.tx1_gmm_ridge fit --config configs/geneeffect_joint.yaml --out-dir <new-root>/gmm
.venv-tx1/bin/python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split val --out-dir <new-root>/reference_val
```

No hyperparameters were selected in this run. The archived observed-response K562
GMM results are a different input and generalization setting. These basal-only
results are GeneEffect evidence, not perturbation-response or SL interaction evidence;
the benchmark's Tx1 pretraining-exposure qualification remains.
