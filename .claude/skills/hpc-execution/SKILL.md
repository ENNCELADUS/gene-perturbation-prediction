---
name: hpc-execution
description: Use before running the `all` pipeline or a checkpoint test, or when a task needs the raw data, ESM2 embeddings, STATE checkpoint or Tx1-3B weights the local Mac lacks. Covers H20 container ports, pushing code to the shared checkout, environment and GPU checks, disconnect-safe launches and run reporting.
---

# HPC execution
`hpc/README.md` and `hpc/run.sh` are the live sources; CLAUDE.md covers commands, resume and sync rules.

## Connect

Containers on one host share `/2023533015` (one checkout, `data/`, `outputs/`) but each sees only its own GPUs, and all
share 224 CPU cores: set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` on torch processes you launch (`hpc/run.sh` does not).

| Available port | GPUs |
|---|---|
| 30838 | **4 × H20** |
| 30757 | **4 × H20** |

`ssh -p 30838 root@10.15.171.204`, then `cd /2023533015/VCC_Project`. Both ports accept password-free login with the
laptop's existing SSH key; use `ssh -p 30757 root@10.15.171.204` for the second container. Other ports (30030, 30734,
30986) are out of date; do not use them. The checkout fetches from GitHub over HTTPS (the repository is public), so
no agent forwarding and no GitHub key is necessary. Ports change when a container is recreated; if one
refuses, ask the user rather than scanning. Ports in `results/` are historical. Never store the SSH password in the repo.

## Sync code through Git only

Never rsync, scp or tar a working tree. Commit and push authorized changes to GitHub from the laptop, then pull on
either port with `hpc/pull.sh`:

```bash
git push origin main
ssh -p 30838 root@10.15.171.204 'cd /2023533015/VCC_Project && hpc/pull.sh'
```

`hpc/pull.sh [BRANCH]` sets the fetch address to HTTPS and sends each try to one of GitHub's own addresses, because
the containers' DNS sometimes answers `github.com` with an address that does not connect (one plain pull fails about
one time in two). It tries up to 12 times and prints the checked-out commit; confirm that it is the commit you
pushed. Do not use a plain `git pull` there, and chain the pull to a launch with `&&`.

Git on the H20 is 2.56.1, built from source (without the Rust parts) into `/2023533015/.local/git-2.56.1` on the
shared disk; `/usr/local/bin/git` in each container is a link to it, and `hpc/pull.sh` calls it by its full path. A
recreated container has only its own git 2.25.1 until the link is made again:
`ln -sfn /2023533015/.local/git-2.56.1/bin/git /usr/local/bin/git`.

The checkout drifts: read
`git branch --show-current`, `git log -1` and `git status --short` before assuming your code is present, and never discard
the host's uncommitted changes. The host holds the only copies of caches, checkpoints and `outputs/`.

## Environment and GPUs

`hpc/run.sh` uses `.venv-tx1/bin/python` (or `PYTHON_BIN`), not `.venv`; `uv` is `/2023533015/.uv/bin/uv`. Before
consequential work confirm the interpreter imports torch, accelerate and `state`, and query hardware, never assume it:
`nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader` and `ps aux | grep '[s]rc\.'`.
Every GPU step of `all` uses every visible GPU, or the ids given by `--gpus 0,1,2,3`: untrained comparison arms on the
first, training on all (`accelerate`, one process each), then one trained comparison job per GPU at a time. Unfinished
training resumes only on its starting GPU count. Size batches from measured throughput; never shrink one silently after an
OOM. Stop `all` with SIGINT or SIGTERM (it terminates its subprocess groups); after a SIGKILL, check for orphaned
`src.experiments.joint.train` / `src.experiments.joint.response_comparison` processes before relaunching.

## Launch and report

Disconnect-safe launch (then exit and confirm from a fresh session; `nohup` inside a single `ssh` command can hang it):
`mkdir -p outputs/launches && nohup hpc/run.sh all configs/geneeffect_joint.yaml --run-id <id> > outputs/launches/<id>.log 2>&1 &`
A PID or GPU utilization is launch evidence only. Confirm each step from `outputs/geneeffect_joint/<id>/`:
`comparison/verdicts.json`; `train/done.json` (+ `metrics.jsonl`, `best.pt`, `last.pt`); `evaluation/val/metrics.json`,
`baselines/val/metrics.json`, `readout/metrics.json`; `summary.md` last. Failures are in `logs/<step>.log`; rerun `all` with
the same id to retry evaluation without retraining. Never fabricate status files or relabel historical runs.
