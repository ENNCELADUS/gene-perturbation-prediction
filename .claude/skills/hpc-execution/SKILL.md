---
name: hpc-execution
description: Use when a task needs GPU, the Replogle GWPS h5ad, ESM2 embeddings, Tx1-3B weights, or the frozen exp05 checkpoint — none of which exist on the local Mac. Covers the SSH host and its changing port, which venv to use, PYTHONPATH, GPU selection, which datasets are absent remotely, and the shard/verify protocol.
---

# Running jobs on the HPC

Data-heavy and GPU work does **not** run on the local Mac — it lacks the Replogle
GWPS h5ad, the ESM2 npz, the Tx1-3B weights, and the frozen exp05 checkpoint.
Author code locally so it can be reviewed, then rsync and run remotely.

## Connection

```bash
ssh root@10.15.171.204 -p 30735      # key-based, non-interactive
```

**The port changes whenever the container is recreated** — 30310, then 30838, now
30735 (container `fqa28o3dqluat-0`, verified 2026-08-15); if it fails, ask the user
rather than scanning. Repo + data live at `/2023533015/VCC_Project`; the sandboxed
Bash tool reaches it, rsync may need `dangerouslyDisableSandbox: true`. The remote
clone keeps its own branch and drifts from local `main` — it was on
`feat/tx1-integrated` @ `e369260` — so `git log -1` there before assuming your code
is present.

## What is NOT on the HPC

The **SL benchmark label trees are Mac-only.** Neither `data/SL_benchmark/` (11 GB)
nor `data/SL_Benchmark_Formal/` (1.0 GB, holding the 946 MB `sl_integrated_pairs.csv`
and the v1 `context_screen_v1/` build) exists remotely, so anything touching SL pair
labels runs locally or needs an explicit transfer first. Verified present:
`data/models/tahoe_x1_3b`, `data/esm2/*.npz`, the Replogle and Adamson h5ads under
`data/sl_dependency_v0/raw/`, and the frozen exp05 checkpoint. Disk is not a
constraint — 953 TB free.

## Pick the right venv — there are three, and they are not interchangeable

| venv | Contents | Use for |
|---|---|---|
| `.venv-tx1` | torch 2.6.0+cu124, `tahoe_x1` | **all Tx1 work** |
| `.venv-esm2` | torch 2.8+cu128, `state`, anndata, scanpy, lightning | STATE / ESM2 work |
| `.venv` | no torch | never |

The repo is **not** pip-installed into `.venv-tx1`, and the scripts import
`scripts.*`, so Tx1 invocations need `PYTHONPATH=src:.` — plain `PYTHONPATH=src`
fails on `import scripts.…`.

```bash
PYTHONPATH=src:. .venv-tx1/bin/python scripts/build_tx1_basal_embeddings.py ...
```

## GPUs — query, never assume

The hardware changes with the container: it has been 4× H20 (97 GB) shared with the
user's `tciep` project at ~90 GB used and 100% util; as of 2026-08-15 it is **2×
H20-3e (140 GB), both idle**. Check first, then pin:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES=<freest>
```

Keep the footprint small and **do not disturb other processes**. Bridge A runs in
~750 MiB; if a job wants far more than expected, stop and re-check the config.

## Standard pass

1. Commit and push locally; check the branch out on the HPC, or rsync to the same
   relative path.
2. Run the test suite remotely under the chosen venv — local green does not imply
   remote green, since torch and extras differ.
3. Benchmark **one small cell line** before the full run.
4. Shard the real run with `--only-line`, monitoring long jobs with a poller.
5. Finish with an **unrestricted** `--verify-only` pass requiring
   `"status": "verified"`.

**Sharded verification is not full verification.** `verify_cache(only_lines=...)`
skips the completeness and untracked-directory checks, so a shard exiting 0 says
nothing about whether the cache as a whole is complete.

## Known gotchas

- Shards built before commit `62eae43` exit nonzero benignly — check the status
  payload, not just the exit code.
- The frozen exp05 checkpoint is
  `results/experiments/05_aivc_a_to_b_to_c/runs/exp05_fixed_k562_pool_v1/models/best/pytorch_model.bin`
  (SHA-256 `48097722…`). Verify the hash before trusting a run built on it.
- Full command lines and what has actually been run live in `.superpowers/sdd/`
  (gitignored, local only): `phase-b-plan.md` and `progress.md`.
