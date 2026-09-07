# Joint GeneEffect execution

Run from the repository root on H20. The launcher uses `.venv-tx1/bin/python`;
set `PYTHON_BIN` to select another installed environment. Preparation runs once
in one process; training uses Torch's visible GPU count and respects
`CUDA_VISIBLE_DEVICES`. Synchronize code with Git before using the remote checkout.

```bash
hpc/run.sh prepare configs/geneeffect_joint.yaml
hpc/run.sh train configs/geneeffect_joint.yaml --run-id joint_seed0
hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/joint_seed0/last.pt
hpc/run.sh test outputs/geneeffect_joint/joint_seed0/best.pt
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/joint_seed0/best.pt --split val
uv run python -m src.evaluate --checkpoint outputs/geneeffect_joint/joint_seed0/best.pt --split train
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split test --out-dir outputs/geneeffect_joint/baselines_seed0
```

All configuration fields are explicit in `configs/geneeffect_joint.yaml`. Input
paths are relative to the repository root. Preparation requires the raw source
registry, GeneEffect CSV, supplied ESM2 table, STATE gene order and response
sources. Missing Tx1 caches additionally require the configured local Tx1 model
and a GPU. Existing Tx1 cache seed provenance is preserved. Newly encoded cells
use collation seed 0. q_sc uses raw UMI counts; response sampling seed 42 and
the fixed 10%/seed-13 holdout are preparation settings, distinct from runtime
seeds 0/0/0. The response cache header records gene order established during
raw target alignment; old headers without gene order require preparation.

Training only opens prepared caches and never rebuilds raw inputs. A fresh run
requires a new run ID. Resume uses the checkpoint's configuration and rejects
any conflicting supplied configuration. `last.pt` supports epoch-boundary resume;
`best.pt` strictly minimizes validation GeneEffect Huber loss. `metrics.jsonl`
contains every update and one validation record per completed epoch.

That epoch record includes fixed-model `train_eval_*` GeneEffect diagnostics over
all labeled training rows, evaluated before validation without refitting or changing
training RNG. It also records actual epoch updates, replay updates, dependency/response
row exposures, dropped dependency rows and effective global batch sizes. The train
diagnostic does not evaluate response targets or select checkpoints. Its extra full
training-set inference cost should be included in runtime estimates.

For a response-readout ablation, set both `model.head_blocks.use_delta_proj` and
`model.head_blocks.use_s` to `false`; all five boolean flags are explicit in the
default config and saved in checkpoint architecture. Disabled blocks and their masks
do not enter normalization or the head. Response replay remains independently active.
Keep prepared inputs, world size, batches and the complete schedule fixed between
arms; early-stopped runs need comparison at matched updates and exposure counts.

The current dependency batch is 1024 per rank (2048 across two H20s); response
replay remains 64 per rank every four updates. A requested batch change uses a
new run directory and explicitly documented derived checkpoint with original
hash/configuration, preserving optimizer/preprocessing/RNG state. Ordinary resume
still rejects configuration conflicts; historical checkpoint metadata is retained.

`run.json` records separate training and evaluation states. Testing is explicit
and does not control training completion. Checkpoint evaluation restores fitted
preprocessing, weights and actual ESM2 vectors, then exports
`evaluation/<checkpoint-name>/<split>/predictions.parquet`, `metrics.json`,
`per_line.csv`, `per_gene.csv` and `response.csv`. An export failure can be retried
with the same evaluation command without optimizer steps. Scalar test names have
`test_` prefixes; `--split train` uses `train_eval_` and an empty response table.
Per-gene details include residual target/prediction SD, SD ratio, RMSE and MAE on
the same finite rows and train-derived variable genes. Undefined quantities retain
explicit counts and null scalar values. These commands produce GeneEffect evidence,
not SL interaction evidence; held-out lines retain the documented Tx1 pretraining
exposure boundary.

## P1-A fixed-backbone head diagnostics

The [approved design](../docs/specs/2026-09-07-p1a-fixed-backbone-head-diagnostics-design.md)
uses A0-A3, head seed 0, FP32 AdamW at 1e-4, global batch 1024 with the tail
retained, and minimum-validation-Huber early stopping (patience 5, cap 50 epochs).
PCA8 scores have unit training population SD; the explicit branch uses lambda
0.01 averaged over all training-covered genes, with no additional weight decay.
The production joint-training configuration is not changed.

```bash
hpc/run.sh p1a extract --checkpoint outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024/best.pt --out-dir outputs/p1a/features
hpc/run.sh p1a train --cache outputs/p1a/features --out-dir outputs/p1a/heads
hpc/run.sh p1a compare --cache outputs/p1a/features --runs outputs/p1a/heads --out-dir outputs/p1a/comparison
```

Extraction restores the supplied checkpoint and its fixed prepared inputs, records
its SHA256, and computes train/validation features in its saved inference precision.
It never runs the original head or refits target preprocessing. Raw pair features
are streamed to memory-mapped arrays; z_c and e_g are stored once per identity.
The new standardizer and PCA fit only on training data. A failed/incomplete cache
cannot be opened; retry extraction into a new directory after diagnosing the error.

This diagnostic uses **one process and one device per arm**. The default train
command runs the four arms sequentially on cuda:0; use `--arms A0 A2` to select
arms. To run independent arms on multiple GPUs, launch separate commands with
disjoint arm lists and explicit `CUDA_VISIBLE_DEVICES` masks. Do not use Accelerate
or torchrun around this entry point: the global batch stays 1024. CPU inspection
and small fixtures use `--device cpu`. Head training opens only the extracted cache
and runs no backbone forward or response replay. Extraction's `--batch-size` is
independent of the fixed head-training batch.

Each arm directory contains `run.json`, epoch `metrics.jsonl`, `best.pt`, `last.pt`,
and selected train/val predictions, metrics and per-line/per-gene tables under
`evaluation/best/`. The log distinguishes optimizer-time loss from fixed-model
train and validation metrics. Resume and export retry are explicit:

```bash
hpc/run.sh p1a train --cache outputs/p1a/features --out-dir outputs/p1a/heads --arms A2 --resume outputs/p1a/heads/A2/last.pt
hpc/run.sh p1a evaluate --cache outputs/p1a/features --checkpoint outputs/p1a/heads/A2/best.pt
```

Resume restores optimizer, fitted scaler and epoch/update counters, retaining the
deterministic epoch-specific order. It requires the same arm, settings and cache.
An evaluation/export failure preserves training completion and saved checkpoints.
The optional old-scaler contrast is an explicit `train --arms A1 --old-scaler`
invocation, written to `A1-old-scaler/`; it is not run by default.

Comparison writes `selected.csv`, four paired per-gene tables, shared-update curve
tables, and `paired.json` with 1,000 seed-0 cluster-bootstrap replicates. The default
context map is the checked-in benchmark split CSV; patients stay together, with
ModelID grouping for missing PatientID. Bootstrap recomputes per-gene correlations
and reports common defined-gene support. Intervals are conditional on the selected
checkpoints and single head seed; they do not estimate initialization variability.
Use repeatable `--reference-val PATH` arguments with existing single-method P0 or
PCA-ridge prediction exports to verify matching row keys and targets without rerunning
them. Comparison uses a new output directory. No test-set evaluation is exposed.

## Tx1 GMM-ridge baseline

This CPU/scikit-learn baseline uses the existing Tx1 basal bags directly, without
PCA or ST responses. It fits a shared diagonal GMM64 on equal numbers of cells from
each labeled training line (seed 0), then a standardized 68-feature context vector
and independent Ridge(alpha=1) per gene. Both standardizers and the GMM are train-only.
The frozen settings and comparison boundaries are in the
[design](../docs/specs/2026-09-07-tx1-gmm-ridge-design.md).

```bash
uv run python -m src.experiments.tx1_gmm_ridge fit --config configs/geneeffect_joint.yaml --out-dir outputs/baselines/tx1_gmm_seed0
uv run python -m src.experiments.tx1_gmm_ridge evaluate --model outputs/baselines/tx1_gmm_seed0/model.joblib --split val
```

Fit requires a new output directory and automatically exports train and val, never
test. `model.joblib` stores the fitted model, preprocessing and split; `run.json`
records source/configuration and separate fitting/evaluation status. `diagnostics.json`
reports convergence, component weights, training occupancies and per-line fit-cell
positions/counts. Check convergence before interpreting scores; the code does not
silently lower K or change the embedding when fitting is difficult.

`evaluation/<split>/` contains P0 predictions/metrics, per-line/per-gene tables and
`context_features.csv` (64 occupancies plus entropy, effective component count,
assignment confidence and negative log likelihood). Train scalars use `train_eval_`;
the response table is empty. A failed export can be retried with `evaluate`, restoring
the saved transforms and readout without refitting. Use only trusted local joblib
artifacts with their recorded sklearn version. This is a candidate baseline; no
performance improvement is established by the implementation tests.
