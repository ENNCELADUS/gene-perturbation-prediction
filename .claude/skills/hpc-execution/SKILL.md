---
name: hpc-execution
description: Use before running the `all` pipeline or a checkpoint test, or when a task needs the raw data, ESM2 embeddings, STATE checkpoint or Tx1-3B weights the local Mac lacks. Covers H20 container ports, pushing code to the shared checkout, environment and GPU checks, disconnect-safe launches and run reporting.
---

# HPC execution
`hpc/README.md` and `hpc/run.sh` are the live sources; CLAUDE.md covers commands, resume and sync rules.

## Connect

Containers on one host share `/2023533015` (one checkout, `data/`, `outputs/`) but each sees only its own GPUs, and all
share 224 CPU cores: set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` on torch processes you launch (`hpc/run.sh` does not).

| Port | GPUs | Use |
|---|---|---|
| 30838 | 4 × H20 | default; the only reachable one on 2026-09-29; no GitHub access |
| 30030, 30670, 30846 | 4 × H20 | extra lanes when they accept connections |

`ssh -p 30838 root@10.15.171.204`, then `cd /2023533015/VCC_Project`. Ports change when a container is recreated; if one
refuses, ask the user rather than scanning. Ports in `docs/results/` are historical. Never store the SSH password in the repo.

## Push code (the host has no GitHub access; push from the laptop over SSH)

```bash
git push ssh://root@10.15.171.204:30838/2023533015/VCC_Project main
ssh -p 30838 root@10.15.171.204 'cd /2023533015/VCC_Project && git checkout main && git log -1 --oneline'
```

Git refuses a push to the checked-out branch, so check out another branch there first. The checkout drifts: read
`git branch --show-current`, `git log -1` and `git status --short` before assuming your code is present, and never discard
the host's uncommitted changes. The host holds the only copies of caches, checkpoints and `outputs/`.

## Environment and GPUs

`hpc/run.sh` uses `.venv-tx1/bin/python` (or `PYTHON_BIN`), not `.venv`; `uv` is `/2023533015/.uv/bin/uv`. Before
consequential work confirm the interpreter imports torch, accelerate and `state`, and query hardware, never assume it:
`nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader` and `ps aux | grep '[s]rc\.'`.
`all` runs the response-model comparison on the last visible GPU and training on the other N−1 (`accelerate`). Size batches
from measured throughput; never shrink one silently after an OOM. After killing `all`, check for orphaned `src.train` /
`response_comparison` processes before relaunching.

## Launch and report

Disconnect-safe launch (then exit and confirm from a fresh session; `nohup` inside a single `ssh` command can hang it):
`mkdir -p outputs/launches && nohup hpc/run.sh all configs/geneeffect_joint.yaml --run-id <id> > outputs/launches/<id>.log 2>&1 &`
A PID or GPU utilization is launch evidence only. Confirm each step from `outputs/geneeffect_joint/<id>/`:
`comparison/verdicts.json`; `train/done.json` (+ `metrics.jsonl`, `best.pt`, `last.pt`); `evaluation/val/metrics.json`,
`baselines/val/metrics.json`, `readout/metrics.json`; `summary.md` last. Failures are in `logs/<step>.log`; rerun `all` with
the same id to retry evaluation without retraining. Never fabricate status files or relabel historical runs.
