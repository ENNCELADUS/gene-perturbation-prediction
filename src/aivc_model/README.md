# AIVC Model Implementation Notes

This directory is the self-contained implementation surface for the AIVC model
integration path. Keep it narrow and easy to review.

## File Roles

- `prepare.py`: configuration parsing, data loading, split construction, local
  artifact loading, and data-contract helpers.
- `model.py`: torch modules, external model adapters, feature layers, and loss
  assembly.
- `train.py`: the single training entrypoint and minimal CSV/artifact writing.

Do not add extra modules unless one of these files becomes unreviewable.

## Edit Rules

- Keep changes local to this directory plus the matching experiment config and
  tests.
- Do not change `prepare.py` unless the data, split, checkpoint, or metric input
  contract is intentionally changing.
- Do not add a separate package, environment, CLI family, or artifact system.
- Do not hardcode data paths, checkpoint paths, dimensions, thresholds, or run
  settings in Python; put them in YAML config.
- Keep output writing simple: per-epoch training CSV, final test metrics CSV, and
  analysis-only files under `artifacts/`.

## Local Assets

Large local model files belong under gitignored checkpoint directories, not in
Git. Download commands should be explicit and user-triggered; training code must
not silently fetch remote weights.

Expected local checkpoint paths should be configured in YAML. If a checkpoint
path is missing or incompatible, fail early instead of falling back to random
initialization.

## Runtime

Run through the project `uv` environment:

```bash
uv run python src/aivc_model/train.py --config <config.yaml>
```

For multi-GPU DDP training, launch the same entrypoint with Accelerate:

```bash
uv run accelerate launch src/aivc_model/train.py --config <config.yaml>
```

On Slurm, use `srun` around the Accelerate launch so the job allocation and the
Python process topology agree:

```bash
srun uv run --locked --no-sync --offline accelerate launch --num_processes 4 src/aivc_model/train.py --config <config.yaml>
```

The training loop uses `Accelerator` for rank/device setup, model wrapping,
gradient synchronization, distributed gene-index sharding, and rank0-only CSV,
artifact, and checkpoint writes.

When `projector.teacher: scvi`, the rank0 teacher fit configures Lightning
quietly for this internal preprocessing step: `scvi_num_workers: null` auto-uses
CPU workers, `scvi_disable_lightning_logger: true` disables Lightning experiment
logger setup, and `scvi_suppress_slurm_warning: true` filters Lightning's
irrelevant `srun` warning when the surrounding job is launched by Accelerate.
`train.float32_matmul_precision: high` sets PyTorch matmul precision before
CUDA/scVI training to avoid Tensor Core performance warnings.

scVI teacher latents are materialized before the AIVC epoch loop. Rank0 writes a
validated run-local cache under `artifacts/scvi_teacher_latents/`; other ranks
wait and then read that cache. The cache is reused only when its metadata matches
the current primary genes, external-test identity, feature names, latent
dimension, seed, and teacher settings. During a cache miss, rank0 loads the saved
teacher once and logs projection progress every 50 genes.

After dependency or lockfile changes, run:

```bash
uv sync
```

## Verification

For focused changes, run:

```bash
uv run ruff check src/aivc_model tests/test_aivc_model.py
uv run ruff format --check src/aivc_model tests/test_aivc_model.py
uv run python -m pytest tests/test_aivc_model.py
```

Full data/checkpoint runs are local/remote experiment jobs and should not be
required for small implementation edits unless the changed contract cannot be
validated synthetically.
