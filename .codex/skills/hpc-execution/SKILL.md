---
name: hpc-execution
description: Use before running the `all` pipeline (preparation, response-model comparison, joint GeneEffect training, validation evaluation) or a checkpoint test for this repository, or when a task needs the Replogle GWPS h5ad, ESM2 embeddings, STATE checkpoint or Tx1-3B weights that the local Mac lacks. Covers the H20 container ports, Git-only code sync to the shared checkout, hpc/run.sh, GPU inspection and process reporting.
---

# HPC execution

Treat `hpc/README.md` and `hpc/run.sh` as the live execution sources. The repository has
one launcher, `hpc/run.sh`, and no scheduler or qualification ladder. Run experiments
directly; do not add preregistration, hash, eligibility or qualification preflights.

The local Mac has no GPU, Tx1 weights, STATE checkpoint, ESM2 table or raw data. It runs
synthetic-fixture tests only; real preparation, training and evaluation run on the H20 host.

## Connect to an H20 container

Containers on the same host share the `/2023533015` filesystem, so one checkout, one
`data/`, and one `outputs/` tree serve all of them. Pick a container by its SSH port:

| Port | GPUs | Use |
|---|---|---|
| 30838 | 4 × H20 | default and, as of 2026-09-29, the only reachable one; no GitHub access |
| 30030, 30670, 30846 | 4 × H20 | extra lanes when they accept connections (all refused on 2026-09-29) |

```bash
ssh -p 30838 root@10.15.171.204
cd /2023533015/VCC_Project
```

Ports change when a container is recreated. If the listed one refuses, ask the user
rather than scanning. Historical ports in `results/` are not a connection authority.
Do not store the SSH password in the repository.

Each container sees only its own GPUs, so jobs on different containers never contend for
a GPU, but they share the host's 224 CPU cores. `hpc/run.sh` does not cap threads; set
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` on torch processes you launch.

## Sync code through Git only

Never rsync, scp or tar a working tree. The host has no GitHub access, so push from the
laptop straight into the checkout over SSH, then move the checkout onto that branch:

```bash
git push ssh://root@10.15.171.204:30838/2023533015/VCC_Project main
ssh -p 30838 root@10.15.171.204 'cd /2023533015/VCC_Project && git checkout main && git log -1 --oneline'
```

The checkout keeps its own branch and can drift from local `main`. Git refuses a push to
the checked-out branch, so push a branch it is not on, or check `main` out first. Look at
`git branch --show-current`, `git log -1` and `git status --short` there before assuming
your code is present, and never discard the host's uncommitted changes.

The host holds the only copies of the prepared caches, checkpoints and `outputs/`.
Git-ignored assets (`data/`, `outputs/`, `*.pt`) never travel through Git.

## Environment

`hpc/run.sh` uses `.venv-tx1/bin/python`, or an explicit `PYTHON_BIN`; plain `.venv` is
not the training environment. Environment names do not establish contents: before
consequential work check that the interpreter imports torch, accelerate, `state` and the
input libraries, and that `nvidia-smi` shows the expected devices. Run module commands
from the repository root (`src.*` imports; `uv` is at `/2023533015/.uv/bin/uv`).

## GPUs — query, never assume

Hardware changes with the container.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps aux | grep '[s]rc\.'
```

`hpc/run.sh all` uses every visible GPU (respecting `CUDA_VISIBLE_DEVICES`) unless
`--gpus` lists some. Do not hard-code a GPU count. Resume needs the same world size, and a batch-size change needs a new run id (`hpc/README.md`). Choose
batch sizes from measured throughput; never shrink one silently after an OOM.

## Commands

```bash
hpc/run.sh all  configs/geneeffect_joint.yaml [--run-id <id>] [--gpus 0,1,2,3]   # prepare → untrained comparison → train → trained comparison → val eval, baselines, readout → summary.md
hpc/run.sh test outputs/geneeffect_joint/<id>/train/best.pt     # explicit; `all` never runs test
```

Every GPU step of `all` uses all chosen GPUs: the untrained comparison arms on the first,
joint training on all of them (`accelerate`, one process per GPU), then the trained
comparison jobs one per GPU at a time. Rerunning with the same run id resumes and skips
finished steps, and refuses a changed config; unfinished training resumes only on the GPU
count it started on. Subprocess logs are under `outputs/geneeffect_joint/<id>/logs/`.
SIGINT or SIGTERM to `all` terminates its subprocesses; after a SIGKILL, check for orphaned
`src.train`/`response_comparison` processes before relaunching.

Preparation is the only step that reads raw data; training opens caches and never
rebuilds them. Testing is explicit, restores fitted preprocessing from the checkpoint and
does not refit. The test split is spent, so model decisions use validation.

For a disconnect-safe launch, add only shell-level logging around the same command, then
exit the session and confirm the process from a fresh one (a `nohup` launch inside a
single `ssh` command can hang that session):

```bash
mkdir -p outputs/launches
nohup hpc/run.sh all configs/geneeffect_joint.yaml --run-id <new_id> > outputs/launches/<new_id>.log 2>&1 &
```

## Reporting

Distinguish launch, running state, completed training and scientific evaluation. A
launcher PID or GPU utilization is execution evidence only. Confirm each step of an `all`
run from its output in the run directory: `comparison/verdicts.json` for the comparison,
`train/done.json` (with `train/metrics.jsonl`, `best.pt` and `last.pt`) for training,
`evaluation/val/metrics.json`, `baselines/val/metrics.json` and `readout/metrics.json` for
validation, and `summary.md` last; read `logs/<step>.log` for a failure. A failed
evaluation step is retried by rerunning `all` with the same run id, without retraining.
Historical runs keep their historical records; do not fabricate status files or relabel
their protocols.
Nothing produced here is SL interaction evidence.
