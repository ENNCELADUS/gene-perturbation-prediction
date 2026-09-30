---
name: hpc-execution
description: Use for authorized GPU preparation, joint GeneEffect training or checkpoint evaluation on the VCC H20 checkout. Inspect the current endpoint and environment; sync code through Git and use hpc/run.sh.
---

# Running VCC jobs on H20

The local Mac supports synthetic CPU tests. Real Tx1/STATE work needs the supplied
model checkpoints, ESM2 table, basal caches and Perturb-seq sources on H20.
The operator interface and configuration live in `hpc/README.md` and
`configs/geneeffect_joint.yaml`; the training contract is
`docs/specs/2026-09-06-modular-joint-training-design.md`.

## Connection and environment

Containers on host `10.15.171.204` share the `/2023533015` filesystem, so one checkout,
`data/` and `outputs/` serve all of them. Ports (`ssh -p <port> root@10.15.171.204`):

| Port | GPUs | Use |
|---|---|---|
| 30838 | 4 × H20 | default and, as of 2026-09-29, the only reachable one; no GitHub access |
| 30030, 30670, 30846 | 4 × H20 | extra lanes when they accept connections (all refused on 2026-09-29) |

Ports change when a container is recreated; if the listed one refuses, ask the user
rather than scanning. Historical endpoints in result notes are not a connection
authority. Before consequential work, inspect hostname, checkout, branch/HEAD, visible
GPUs, existing processes and the requested input paths. The established checkout is
`/2023533015/VCC_Project`; verify it on the connected container. Preserve unrelated jobs.

Sync working code through Git only, within the user's authorization. The host has no
GitHub access, so push over SSH into the checkout from the laptop
(`git push ssh://root@10.15.171.204:30838/2023533015/VCC_Project main`), then check the
branch out there; Git refuses a push to the checkout's current branch. Do not
rsync/scp/tar working trees.

`hpc/run.sh` uses `.venv-tx1/bin/python`, or explicit `PYTHON_BIN`. Check that the
selected interpreter imports Torch, Accelerate, STATE and required input libraries;
environment names alone do not establish their contents. Run module commands from
the repository root with the root on Python's import path (`src.*` imports).

## Commands

```bash
hpc/run.sh prepare configs/geneeffect_joint.yaml
hpc/run.sh train configs/geneeffect_joint.yaml --run-id joint_seed0
hpc/run.sh train configs/geneeffect_joint.yaml --resume outputs/geneeffect_joint/joint_seed0/last.pt
hpc/run.sh test outputs/geneeffect_joint/joint_seed0/best.pt
```

Preparation runs once in a single process and consumes supplied data/models.
Training opens fixed caches without rebuilding raw sources on each rank. Training
uses visible GPUs and respects `CUDA_VISIBLE_DEVICES`; choose batch sizes from
measured throughput and stability. Do not silently shrink a batch after OOM.

GeneEffect regression runs every update and balanced four-anchor reconstruction
runs every fourth update. Validate once per epoch, report all loss terms and
metrics, and select/early-stop on minimum `val_geneeffect_loss`. Training,
cell-collation and projection base seeds are 0. Resume is at epoch boundaries
with the same world size. Testing is explicit and does not refit preprocessing.

## Reporting

Distinguish launch, running state, completed training and scientific evaluation.
Inspect the exact requested workers and logs; a launcher PID or GPU utilization
alone does not prove completion. The joint run uses ordinary `best.pt`, `last.pt`,
`metrics.jsonl` and separate training/evaluation outcomes in `run.json`.
An export failure is retryable without retraining and does not erase successful
training. Historical runs use their historical records; do not manufacture new
status files or relabel their protocols. No seals or qualification ladder govern
the new training route.
