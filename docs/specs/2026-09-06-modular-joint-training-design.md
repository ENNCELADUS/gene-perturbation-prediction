# Modular GeneEffect training with recurring response supervision

Date: 2026-09-06. Status: approved direction, including the owner's validation and
seed corrections below; implemented locally, with no new experiment run.

The approved direction is a topology-style repository layout and one joint training
run that keeps revisiting the four Perturb-seq lines. This document specifies that
direction, including proposed starting settings. The existing Exp13 reports describe
the old protocol and remain historical evidence.

## 1. Purpose and scope

Replace the response-only training → artifact seal → frozen-feature build → head
warmup → gradient calibration → joint training sequence with one model and trainer.
Make preparation, model computation, optimization, evaluation and reporting readable
independently. Training must not depend on a collection of completion certificates.

This is a new training protocol as well as a code refactor: Huber-only dependency
training and periodic response supervision differ from historical Exp13. Numerical
equivalence to the old trained model is not a requirement or a claimed result.
Retain the current five-block GeneEffect head and response computation initially;
changing the model architecture, fine-tuning Tx1, adding SL training, launching GPU
jobs and moving remote artifacts are outside this implementation's scope.

## 2. Model and supervision

Keep Tx1 frozen. Its cached basal embeddings feed the trainable STATE transition
model and ESM2 perturbation adapter. Predicted responses feed the existing five-block
residual head; both the transition model and head learn from GeneEffect regression.
Use the current STATE checkpoint initialization path, with a newly initialized gene
adapter and head, rather than requiring a previously sealed Stage 1 run. Report loaded
and newly initialized layers once. Unexpected checkpoint incompatibilities are errors;
intentional input/adapter changes are explicit in the model constructor.

Use the checked-in `cell_line_geneeffect_226_split.json` unchanged:

| Cohort | Supervision during training |
| --- | --- |
| K562, Jurkat, HepG2, HCT116 | GeneEffect regression and available Perturb-seq responses |
| Other 166 labeled training lines | GeneEffect regression |
| PC9 and HeLa, the two unlabeled training members | No supervised GeneEffect loss |
| 27 validation lines | Evaluation and checkpoint selection only |
| 27 test lines | Final evaluation only |

The four response anchors are all labeled training members. Join everything by
DepMap ModelID and gene identifiers. A response condition need not have a GeneEffect
observation to contribute response supervision: the loaders are separate, not restricted
to the intersection of their measured conditions.

Fit one gene mean on the 170 labeled training lines. The head predicts
`delta_hat(g,c)` against `GeneEffect(g,c) - train_gene_mean(g)` using mean Huber loss
with delta 1. Absolute predictions add that same fixed mean. Keep train-derived
variable-gene membership for evaluation, but remove the Pearson training loss and its
special gene-major batches and cross-rank objective reductions.

Response supervision retains mean-delta MSE plus energy distance, each with weight 1.
It compares observed and predicted cell distributions; no paired-cell reconstruction
is introduced. Preserve the fixed per-anchor response-condition holdout. Those
conditions contribute no response targets to optimization, although their GeneEffect
labels can remain eligible: response holdout is not an unseen-gene claim.

## 3. One training loop

One shuffled dependency loader supplies finite labeled `(gene, ModelID)` rows from
all 170 training lines. One independently shuffled, cycling response loader supplies
equal numbers of conditions from each of the four anchors. Their shuffle streams
derive from the recorded seed; neither loader reads validation/test response targets.

Every optimizer update contains a dependency batch. Updates 0, 4, 8, … also contain
a response batch. On a replay update, one top-level model call returns dependency
predictions and response predictions; take one backward pass and optimizer step.

```text
ordinary update: loss = dependency_huber
replay update:   loss = dependency_huber + response_weight * response_loss
```

Each task loss is averaged within its own batch, so response weight does not depend
on the relative batch sizes. `response_weight` is the coefficient on replay updates;
it is not multiplied automatically by the replay interval. The interval and coefficient
together determine response influence. Log the two losses separately and identify
updates without a response loss as missing, not as a measured zero.

Current settings, to be assessed on validation rather than treated as tuned:

| Setting | Value |
| --- | --- |
| Response interval / weight | Every 4 optimizer updates / 1.0 |
| Dependency conditions per rank | 1024 |
| Response conditions per replay per rank | 64, 16 from each anchor |
| STATE / gene-adapter / head learning rates | `1e-6` / `1e-5` / `1e-4` |
| Optimizer / weight decay / gradient clipping | AdamW / 0.01 / 1.0 |
| Maximum epochs / validation patience | 50 / 5 |
| Cell bag / STATE window | 128 cells / 64 cells |
| Training / cell-collation / projection seed | 0 / 0 / 0 |

Dependency batch 1024/rank replaces 256/rank at the user's explicit request after
H20 throughput/memory probes. This changes updates per epoch and replay exposure
per dependency row; it is not a validation-selected hyperparameter.

These are the three runtime base seeds. Epoch/rank-specific sampling streams derive
from base seed 0. Do not change frozen benchmark membership or relabel historical
input/checkpoint metadata to make old artifacts appear to have been generated with 0.

An epoch is one pass through the dependency loader. Response batches cycle within
that epoch; both loaders restart from deterministic epoch/rank-specific seeds at the
next epoch. This keeps epoch-boundary resume independent of an unsaved iterator cursor.
A too-large batch is corrected explicitly after a throughput
probe; there is no silent OOM retry that changes batch size or the objective.

Reuse Accelerate, BF16 autocast and ordinary DDP. Loss reductions and metric
accumulation use FP32. All ranks use the same replay decision from the optimizer-step
counter. Use equal full training batches with standard distributed sampling and
`drop_last`; log the number of processed rows. Disable the current static-graph
optimization because response computation is conditional. Both batch types contain
dependency predictions, so the head participates in every update. Never bypass the
DDP wrapper for a training forward.

Retain the existing paired basal/HVG cell sampling convention. There is no frozen
head-only warmup or trainable-feature cache. To initialize feature scales, select up
to 32 finite dependency conditions per labeled training line with the recorded seed,
run the initialized model on rank zero in evaluation mode without gradients, and stream the
existing block standardizer statistics. Apply the existing coverage-mask semantics.
Broadcast the resulting statistics, restore training RNG state and start optimization.
This bounded pass fits on training data only, writes no feature-store artifact and is
skipped on resume because the statistics are already in the checkpoint.

## 4. Validation, testing and checkpoints

Run validation exactly once at the end of every completed training epoch, over all
27 validation lines and the fixed held-out response conditions from the four anchors.
Use evaluation mode and no gradients, with one inference/evaluation implementation
shared by in-memory validation and checkpoint-based final testing. Neither path refits
preprocessing. There is no validation-skipping interval or eligibility gate.

Every epoch reports the following in the console summary and `metrics.jsonl`:

| Field | Definition |
| --- | --- |
| `val_geneeffect_loss` | Mean Huber loss, delta 1, over all finite labeled validation gene/line pairs |
| `val_response_mean_delta_mse` | Mean-delta MSE averaged within each response anchor, then equally across the four anchors |
| `val_response_energy_distance` | Energy distance with the same per-anchor averaging |
| `val_response_loss` | Sum of the two response loss terms, each with weight 1 |
| `val_total_loss` | `val_geneeffect_loss + response_weight * val_response_loss` |
| `val_geneeffect_pearson_macro_per_line` / `val_geneeffect_spearman_macro_per_line` | Correlation across genes on absolute GeneEffect, calculated separately for each validation line and then macro-averaged |
| `val_residual_pearson_macro_per_gene` / `val_residual_spearman_macro_per_gene` | Correlation across validation lines on residuals, macro-averaged over the train-derived variable-gene set |
| `val_geneeffect_rmse` / `val_geneeffect_mae` | Errors over the same finite labeled validation pairs |
| Coverage and per-line/per-gene details | Valid pair counts, scored/undefined correlation counts and the corresponding detailed scores |

`val_total_loss` evaluates both tasks' held-out means together. It is not diluted by
the response replay interval or presented as an average of the sparse training-step
totals. Aggregate GeneEffect loss by the total loss sum divided by valid pair count,
not an unweighted mean of batch/rank means. Remove distributed-sampler padding from
evaluation counts. Keep response aggregation equally weighted across the four anchors.
Console/JSONL epoch summaries contain scalar metrics and counts. The evaluator returns
the detailed tables for exports; those exports are not required completion artifacts.

Early stopping and `best.pt` selection both **minimize `val_geneeffect_loss`**, because
the downstream goal is accurate GeneEffect prediction. Huber loss on the residual is
identical to Huber loss on absolute GeneEffect when the same fixed training gene mean
is added to prediction and target. Neither total loss, response loss nor correlation
metrics select the checkpoint or gate stopping. A strict decrease saves the new best
and resets patience; a tie or increase keeps the earlier best and increments patience.
There is no improvement tolerance or composite selector. A non-finite GeneEffect loss
or an empty validation set is an evaluation error. Individual undefined correlations
remain undefined with counts; they do not block selection when GeneEffect loss is valid.

A response-only baseline evaluation before training may establish the initial response
quality; it is outside the epoch loop and must not cause a second validation pass in
an epoch. Final testing reports the same loss terms and metrics with `test_` prefixes.

The explicit test command evaluates the chosen checkpoint after selection. It
produces test predictions, metrics and the selected model's held-out response metrics.
The baseline command fits the retained gene-mean, copy-prior, nearest-line and
context-PCA-ridge predictors on training data and uses the same evaluation row/gene
universe. Baseline fitting/evaluation is independent of training completion. A missing
baseline artifact cannot invalidate a completed training run.

The common evaluation gene panel is prepared once using the retained Exp13 coverage
definition and named directly by the new configuration. Training does not reconstruct
it from a chain of prior-run artifacts. Missing observations remain masked; no method
silently changes the comparison panel. Test values do not enter preparation or fitting;
the retained panel's pre-existing coverage policy uses availability only.

Write `last.pt` and `best.pt` atomically on rank zero. Store model weights, architecture
configuration, gene order/ESM2 buffers, gene means, projection, normalization, epoch,
optimizer step, best score, patience, optimizer and relevant AMP state. Collect each
rank's RNG state for the checkpoint, rather than saving rank zero's RNG alone. Do not copy
the frozen Tx1 weights into each checkpoint. Resume at epoch boundaries, retaining
sampler seeds and the global replay counter; do not promise exact mid-epoch recovery.
The same-world-size resume test must reproduce the next update. A requested world-size
change requires a fresh run, avoiding an unimplemented reproducibility promise.

An explicitly requested batch-size change may continue from a derived checkpoint
in a new run directory. Record the original checkpoint hash/configuration and
the changed batch field, preserving weights, optimizer, fitted preprocessing,
training counters and rank RNG states. Preserve the original run and checkpoints.
The derived checkpoint declares its new continuation configuration; this is a
documented batch transition, not an exact-next-update resume claim.

```text
outputs/geneeffect_joint/<run_id>/
  config.yaml
  run.json             # source revision, inputs, environment, seed, execution status
  metrics.jsonl
  last.pt
  best.pt
  evaluation/
    best/test/          # checkpoint name and requested split
      predictions.parquet
      metrics.json
      per_line.csv
      per_gene.csv
      response.csv
```

`run.json` has separate training and evaluation statuses. A normal trainer exit after
saving its checkpoint completes training. Testing writes its own outcome. Exceptions
record the failing operation and preserve checkpoints. A plotting/export failure can
be retried without retraining. There is no seal, publish step, required feature dump,
duplicate model package or post-training reconstruction of the entire artifact graph.

## 5. Module and command structure

Adopt the reference repository's `src.<area>` imports and root-level module execution.
Update Hatch's package inclusion accordingly; do not retain an `aivc_model` shim.

```text
src/
  data/                # split/batch types, basal/response inputs, embeddings, caches
    prepare/           # reusable dataset and fixed-cache builders
  model/               # STATE adapters, forward functions, features, head, losses
  training/            # trainer, replay sampling, distributed setup, checkpoint I/O
  eval/                # inference, response/GeneEffect metrics and scoring
  baselines/           # residual baseline fitting/prediction
  experiments/         # concrete GeneEffect wiring and thin preparation commands
    historical/        # retained probes and the separate context-screen builder
  train.py             # joint-training CLI
  evaluate.py          # checkpoint evaluation CLI
scripts/               # thin operational utilities, including the Tahoe download shell
hpc/                   # one run.sh launcher and README
configs/geneeffect_joint.yaml
outputs/               # ignored generated runs
docs/results/          # tracked result notes and small supporting evidence
```

The `historical` placement of the context-screen builder preserves the preceding
Exp13-only preparation organization; it does not retire the active SL research question.
Do not add an SL training path in this change.

| Existing responsibility | New owner |
| --- | --- |
| `FixedSplit`, split guards, residual targets, shared batch records | `src/data/` |
| Basal/response assembly, ESM2 loading, fixed input caches | `src/data/` |
| `GeneBags` in `state_core.py` | `src/data/` |
| STATE/perturbation adapters and forward prediction in mixed modules | `src/model/` |
| `predict_bags`, response loss, pooling and head loss primitives | `src/model/` |
| Optimization and retained DDP helpers | `src/training/` |
| Metric computation and held-out scoring | `src/eval/` |
| `residual_ladder.py` fitting/prediction | `src/baselines/` |
| Current Stage 2 runner's useful assembly | `src/experiments/` |
| Python script algorithms and model-loading helpers | Appropriate `src/` owner |
| Current stage-specific configs, seals and feature-store orchestration | Removed from the active path |

Data/model primitives do not import trainers, evaluators or command modules. Trainer
and evaluator share model/data interfaces. Experiment wiring composes those modules;
it does not implement optimization, tensor math or artifact auditing. Reusable code
never imports a script. Split large files at these responsibilities, with no line-count
quota, generic pipeline framework, registry or empty future package.

Proposed operator interface:

```bash
hpc/run.sh prepare configs/geneeffect_joint.yaml
hpc/run.sh train configs/geneeffect_joint.yaml --run-id <run_id>
hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/<run_id>/last.pt
hpc/run.sh test outputs/geneeffect_joint/<run_id>/best.pt
uv run python -m src.evaluate --checkpoint <best.pt> --split val
uv run python -m src.experiments.baselines --config configs/geneeffect_joint.yaml --split test --out-dir <baseline_dir>
```

`prepare` runs the experiment's fixed-input preparation once in a single process;
training opens the resulting caches instead of making every DDP worker rebuild them.
It consumes supplied source data/models and does not silently download them. Source
downloads and optional full Tx1 cache regeneration remain explicit preparation tools.
The launcher selects the existing project environment and visible GPUs, calls the
Python entry points and propagates exit codes. Training does not automatically run test.

## 6. Checks and artifact migration

At data preparation/loading, check required files, raw-UMI versus transformed input
meaning, ModelID/gene order, shapes, masks and finite values in the data being used.
Persist small cache metadata identifying source/configuration and feature order. Read
it once when opening a cache; do not rehash all source shards on every rank or before
and after loading. Rebuilding a changed dataset is an explicit preparation operation.
Keep ordinary configuration errors, checkpoint compatibility, train/test exclusion,
non-finite computation, DDP correctness and I/O errors as failures.

Remove hard-coded historical digest gates from active training, recursive provenance
requirements, duplicate per-batch validation at every call layer, gradient-magnitude
quality gates, gradient-ratio calibration and post-run equality checks between redundant
artifacts. Log diagnostic values without letting them veto completion. Do not replace
these checks with another certification system under a different name.

| Current files | Destination and treatment |
| --- | --- |
| Four tracked files in `results/phase_a_tx1_20260724/` | `configs/benchmarks/provenance/phase_a_tx1_20260724/`; preserve bytes and update active consumers |
| Two tracked `results/stage0/*.json` files and the Stage 0 result note | `docs/results/exp13_stage0/`, with the note as `README.md` |
| `docs/results/exp13-stage2-full.md` | Existing `docs/results/exp13_stage2_full/README.md`; retain its figures/tables and repair links |
| New training outputs | `outputs/geneeffect_joint/<run_id>/` |
| Existing ignored datasets, caches, archived runs and remote outputs | Keep their current locations; accept explicit input paths |

The Phase-A registration and cell-line manifest are still consumed by retained input
preparation, so age alone is not grounds for deletion. Move only the listed tracked
files in this refactor. Preserve embedded historical paths and result values as evidence;
update active code/config paths and document the old command snapshot at `e6341d2`.
Do not mutate historical manifests to make them claim the new protocol.

Update `.gitignore`, package metadata, active config/CLI references, retained tests,
README navigation, project `AGENTS.md` and the execution runbook/skill together during
implementation. Put the new training protocol in the current blueprint and experiment
documentation while labeling the old Exp13 protocol historical. Leave `CLAUDE.md` alone
unless separately requested. Other historical result notes need only affected-link fixes.

## 7. Verification and delivery

First obtain the retained-suite baseline. Keep tests of data semantics, residual
targets, model computation and metrics, updating their imports. Remove tests whose sole
purpose is enforcing deleted seals or the retired stage lifecycle. Add focused tests for:

- Regression-only and replay updates reaching the intended parameters; regression
  must reach STATE and the adapter through live features.
- Replay frequency, equal anchor sampling and no response-holdout target consumption.
- Train-only gene means/normalization and exclusion of validation/test from loaders.
- A small two-process CPU update agreeing with the corresponding single-process
  effective batch for both update types, and correct evaluation row counts.
- Exactly one validation per completed epoch, all loss/metric fields and correct
  count-weighted aggregation with uneven evaluation batches.
- Checkpoint selection and patience following only minimum `val_geneeffect_loss`,
  including ties and epochs where total loss or correlations improve in the other direction.
- Saving/reloading predictions and epoch-boundary resume preserving the next update;
  all three configured base seeds are 0, including the generated projection.
- Training completion surviving an independent evaluation/export failure, while the
  failed evaluation reports its exception and remains retryable.

Run the relevant retained suite, Ruff, affected CLI help, package-build/import checks
and final diff/link inspection. Asset-dependent skips must be stated; a mocked model
does not demonstrate the real Tx1/STATE pipeline. The first later GPU execution should
measure production throughput/memory and exercise the real training/checkpoint/evaluation
path before claiming it works at scale. This design does not authorize a launch or new
scientific conclusion.

The implementation plan sequences module extraction, the joint trainer and evaluator,
then command/artifact migration and removal of obsolete paths. Keep each increment
runnable, but finish with one active training route. The owner's approval and explicit
validation/seed corrections above govern the plan.
