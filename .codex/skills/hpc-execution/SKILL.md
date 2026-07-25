---
name: hpc-execution
description: Use when a task needs GPU, the Replogle GWPS h5ad, ESM2 embeddings, Tx1-3B weights, or the frozen exp05 checkpoint — none of which exist on the local Mac. Covers the SSH host, which of the two venvs to use, PYTHONPATH, GPU selection, and the shard/verify protocol.
---

# Running jobs on the HPC

Data-heavy and GPU work does **not** run on the local Mac — it lacks the Replogle
GWPS h5ad, the ESM2 npz, the Tx1-3B weights, and the frozen exp05 checkpoint.
Author code locally so it can be reviewed, then rsync and run remotely.

## Connection

```bash
ssh root@10.15.171.204 -p 30838      # key-based, non-interactive
```
Repo + data live at `/2023533015/VCC_Project` (full clone). The sandboxed Bash
tool reaches it; rsync/ssh may need `dangerouslyDisableSandbox: true`.

## Pick the right venv — there are two, and they are not interchangeable

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

## GPUs are shared

4× H20 (97 GB), shared with the user's other `tciep` project — often ~90 GB used
at 100% util. Before launching, pick the freest device and pin it:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES=<freest>
```

Keep the footprint small and **do not disturb other processes**. Bridge A runs in
~750 MiB; if a job wants far more than expected, stop and re-check the config.

## Standard pass

1. Commit and push the branch locally; check it out on the HPC (or rsync to the
   same relative path).
2. Run the test suite remotely under the chosen venv to confirm imports resolve
   there — local green does not imply remote green (different torch, different
   extras).
3. Benchmark **one small cell line** before the full run.
4. Shard the real run with `--only-line`, monitoring long jobs with a poller.
5. Finish with an **unrestricted** `--verify-only` pass and require
   `"status": "verified"`.

**Sharded verification is not full verification.** `verify_cache(only_lines=...)`
skips the completeness and untracked-directory checks, so a shard exiting 0 says
nothing about whether the cache as a whole is complete. Always end on one
unrestricted pass.

## Known gotchas

- Shards built before commit `62eae43` exit nonzero benignly — check the status
  payload, not just the exit code.
- The frozen exp05 checkpoint is
  `results/experiments/05_aivc_a_to_b_to_c/runs/exp05_fixed_k562_pool_v1/models/best/pytorch_model.bin`
  (SHA-256 `48097722…`). Verify the hash before trusting a run built on it.

The live execution ledger and full command lines are in `.superpowers/sdd/`
(gitignored, local only) — `phase-b-plan.md` for the host/venv facts,
`progress.md` for what has actually been run.
