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

## P1-B response adaptation

P1-B uses one process/GPU per arm, seed 0, 192 conditions/update (64 per source
anchor), 257 updates/epoch, and internal response-loss early stopping (patience 5,
maximum 50 epochs). It does not train a GeneEffect head. The full protocol is in
[the P1-B design](../docs/specs/2026-09-07-p1b-response-adaptation-design.md).

Run from the repository root in `.venv-tx1`. Set `P0_CHECKPOINT` to the exact P0
joint checkpoint used for P1-A. Preparation opens existing caches, reads only raw
gene metadata to recover the 1,957 measured coordinates, fits source-training
baselines, and records immutable input/model identities. It does not score Jurkat.
The output directory must be new; failures are recorded in `status.json`.

```bash
P1B_PREPARED=outputs/p1b/prepared_seed0
P1B_RUNS=outputs/p1b/response_seed0
hpc/run.sh p1b prepare --checkpoint "$P0_CHECKPOINT" --out-dir "$P1B_PREPARED"
hpc/run.sh p1b evaluate --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --state B-native
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b evaluate --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --state B-init
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b evaluate --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --state B-joint
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b train-interface --prepared "$P1B_PREPARED" --runs "$P1B_RUNS"
```

`B-native` currently writes an explicit unavailable record: original numerical
preprocessing/batch semantics are not verified. It never substitutes Tx1 inputs
or fabricated native predictions. Model reconstruction for B-init uses the original
P0 config, released checkpoint, and seed-0 construction order. No head scaler is
required by P1-B.

Read `stage2.json` after B-interface completes. Only if `eligible` is true (at least
1% internal validation loss reduction from B-init), run both commands below. They
may run concurrently on separate GPUs. Both start from the same best interface
checkpoint with fresh optimizers; no Jurkat result determines this decision.

```bash
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b train-stage2 --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --arm B-continue
CUDA_VISIBLE_DEVICES=1 hpc/run.sh p1b train-stage2 --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --arm B-unfreeze
```

Add `--resume` to the same training command for epoch-boundary recovery. Completed
runs are not overwritten without that flag. No automatic batch reduction occurs
on OOM. The first full training batch exercises the fixed production allocation;
finite gradients and clipping are checked on every update.

After required training finishes, evaluate each completed state internally and
externally. External evaluation fixes selected checkpoint hashes in
`external_evaluation.json` and prevents further adaptation in that run directory.
If stage two is ineligible, omit B-continue/B-unfreeze from this list.

```bash
for P1B_STATE in B-interface B-continue B-unfreeze; do
  CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b evaluate --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --state "$P1B_STATE"
done
for P1B_STATE in B-init B-joint B-interface B-continue B-unfreeze; do
  CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1b evaluate --prepared "$P1B_PREPARED" --runs "$P1B_RUNS" --state "$P1B_STATE" --external
done
hpc/run.sh p1b compare --prepared "$P1B_PREPARED" --runs "$P1B_RUNS"
```

Training and export have separate status records. Re-run the exact `evaluate`
command after an export failure; it does not optimize or rewrite weights. Results
include condition metrics, mean effects (no full predicted cell matrices), measured
coordinate and identity provenance, true-gene-coordinate removal, ten fixed
held-out identity derangements, baseline comparisons, cross-context metrics,
1,000 paired gene-bootstrap intervals, actual exposure counts, and stage-specific
curves/common-update contrasts. Identical panels reuse inference; training exports
use correct identities only. Only held-out diagnostics receive identity shuffles.

Jurkat reporting distinguishes 2,373 training-seen perturbations, four unseen,
2,006 native-and-seen, and 2,009 native-vocabulary-covered conditions. Jurkat is
held out from interface adaptation, not verified absent from ST/Tx1 pretraining.

## P1-C interface isolation

P1-C repeats the response-adaptation contrast across four held-out anchors
(`jurkat k562 hepg2 hct116`) and five input/readout variants (`V0 V1 V2-null V2 V3`),
so an adaptation gain is attributed to the interface rather than to one held-out
context. The pre-registered keep predicate and the V3 precondition are in
[the P1-C design](../docs/specs/2026-09-08-p1c-interface-isolation-design.md).
`hpc/p1c_pipeline.sh` runs one complete round; every stage is also available as an
ordinary `hpc/run.sh p1c` command.

```bash
RUN=outputs/p1c/p1c_seed0 \
P0_CHECKPOINT=outputs/geneeffect_joint/joint_seed0_20260906T174818Z_b1024/best.pt \
P1B_PREPARED=outputs/p1b/p1b_seed0_20260907T164813Z/prepared \
P1B_RUNS=outputs/p1b/p1b_seed0_20260907T164813Z/runs \
P1A_FEATURES=outputs/p1a/features \
P1A_REFERENCE_P0=outputs/p1a/reference/p0/predictions.parquet \
P1A_REFERENCE_PCA=outputs/p1a/reference/pca/predictions.parquet \
nohup hpc/p1c_pipeline.sh > outputs/p1c/p1c_seed0.out 2>&1 &
```

All seven variables are required and their paths must exist; `P1B_RUNS` is the
P1-B *runs* directory containing `evaluation/`. `GPUS` (default `"0 1"`) lists the
visible devices, `PIPELINE_SKIP_TIER4=1` omits the P1-A head seeds, `PYTHON_BIN`
selects the environment, and `PIPELINE_POLL_SECONDS` (default 30) sets the queue
poll interval. The script refuses to start when `$RUN/phase.txt` already exists.

Waves run in order: fold preparation (four folds sequentially on CPU); Tier 0
plus native Jurkat evaluation plus two Tier-4 head seeds; one wave per label in
`V0 V1 V2-null V2`, each training and evaluating the four folds; `compare-1`;
the two learning-rate arms (`V0-lr1e-6`, `V0-lr1e-5`, Jurkat) together with the
four V3 folds when `$RUN/comparison/kept.json` reports `V3_eligible` true
(otherwise `$RUN/v3_skipped.txt` records the skip); and `compare-final`.

Every GPU job goes through a queue that holds at most one job per listed GPU and
starts the next queued job on a device as soon as its predecessor exits, so the
wave order does not assume a GPU count. Each queued job writes `$RUN/<job>.log`,
`$RUN/<job>.pid` and `$RUN/<job>.exit` — including a job killed before it could
record its own status. Foreground steps (`prepare-<fold>`, `compare-1`,
`compare-final`) write only a `.log`; `tier0` is a background job and writes all
three. A nonzero exit fails the wave only after the other queued jobs finish, and
the pipeline then exits nonzero with `failed` in `phase.txt` and the status in
`$RUN/exit_code`. Phase names are written to `$RUN/phase.txt` as the round
progresses, ending in `completed`. SIGTERM or Ctrl-C terminates every running
queued job, writes `interrupted` to `phase.txt` and `143` to `exit_code`, and
exits 143; an unreadable `kept.json` still runs the learning-rate arms and the
final comparison, records the reason in `$RUN/v3_skipped.txt`, and fails the
round at the end.

Fold bundles live in `$RUN/prepared/<fold>`, Tier 0 in `$RUN/tier0`, Tier-4 heads
in `$RUN/heads/seed<n>`, comparisons in `$RUN/comparison`, and every trained arm in
`$RUN/runs/<label>/<fold>` (the native evaluation in `$RUN/runs/N-native/jurkat`);
`compare` reads exactly that layout. To recover one arm, resume it and re-export
both evaluations by hand — the pipeline is not restartable in place:

```bash
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1c train --prepared "$RUN/prepared/k562" \
  --runs "$RUN/runs/V1/k562" --variant V1 --lr 1e-4 --resume
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1c evaluate --prepared "$RUN/prepared/k562" --runs "$RUN/runs/V1/k562"
CUDA_VISIBLE_DEVICES=0 hpc/run.sh p1c evaluate --prepared "$RUN/prepared/k562" --runs "$RUN/runs/V1/k562" --external
hpc/run.sh p1c compare --root "$RUN" --out-dir "$RUN/comparison"
```

External evaluation fixes the selected checkpoint and blocks further training in
that run directory. Comparison re-derives its verdict from existing exports only
and can be rerun into the same directory.
