# exp08 fold-orchestration fix: replace NCCL gather with a filesystem work-queue

**Date:** 2026-06-20
**Status:** Approved (design)
**Scope:** `src/sl_dl_model/evaluate.py`, `scripts/sl_dl_model.sh`, tests. Surgical
fix — keeps `accelerate launch` + `PartialState`; touches no model, loss, scoring,
or metric code.

## 1. Problem

The first multi-rank exp08 run (`results/experiments/08_k562_sl_pair_state_dl/slurm_927818.err`)
died with a distributed-collective timeout:

```
[rank3]: torch.distributed.DistBackendError: [3] is setting up NCCL communicator and
retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but
store->get('0') got error: wait timeout after 600000ms
```

### Verified root cause

- The **only** distributed collective in the whole `sl_dl_model` package is
  `gather_object([local_rows])` at `evaluate.py:171` (confirmed by grep: no
  `all_reduce` / `all_gather` / `broadcast` / `barrier` / `wait_for_everyone`
  elsewhere). `gather_object` internally calls `torch.distributed.all_gather_object`.
- There is **no** `Accelerator(...)`, `InitProcessGroupKwargs`, or `timeout=`
  anywhere — so the c10d store/PG timeout is the **default 600000 ms**, which is
  exactly the value in the log.
- `PartialState.__init__` (accelerate 1.13.0) *does* eagerly call
  `torch.distributed.init_process_group` when `LOCAL_RANK != -1` (set by
  `accelerate launch`), but that is only the **TCPStore rendezvous**, which all 4
  ranks complete in milliseconds at launch. The **NCCL communicator** (the
  `ncclUniqueId` broadcast through the store — the thing that timed out) is created
  **lazily on the first collective**. The first (and only) collective is the
  end-of-run `gather_object`.
- `_shard_jobs` (`evaluate.py:396`) is static round-robin `jobs[rank::num_processes]`.
  For the submitted CV2-only run with folds `[0,1,2,3,4]`: rank0=`[0,4]`,
  rank1=`[1]`, rank2=`[2]`, rank3=`[3]`. Ranks owning a single fold reach
  `gather_object` and trigger the lazy NCCL setup, then block on `store->get('0')`
  waiting for rank0 — which is still training (the `.err` shows the progress bar at
  `epoch 8: 89% ... 39223/44046` *after* rank3's timeout). 600 s of skew → timeout →
  elastic SIGTERMs the other ranks → job fails non-zero.

The crash is an **orchestration/timing** problem (uneven fold runtimes vs a
collective with a 10-minute default timeout), not an HPC or model failure.

### Non-causes (verified, documented to prevent rabbit-holing)

- `pert_dim=328 vs checkpoint 2024` warnings — the encoder intentionally uses the
  checkpoint width; not the crash.
- "More than one GPU was found" — expected from `--num_processes 4`.

### Stale-runtime caveat (verified, affects operations only)

The `.err` progress bar shows `44046` steps == CV2 fold-0 **train-pair** count, i.e.
**one tqdm tick per pair**. Current `main` advances the bar **once per optimizer
step** (`68ab6f4`) and batches by `batch_pairs=1024` (`df5b917`), which would show
~43 steps, not 44046. So job 927818 ran a **cluster checkout behind HEAD**. The new
run must be launched from an up-to-date checkout. This is an operational step, not a
code fix, but the spec records it because it changes how we read that log.

## 2. Goals (locked with user)

1. **Stop the timeout crash** — the new run completes all of **CV1, CV2, CV3 × 5
   folds = 15 jobs**.
2. **Use the 4 L40 GPUs well** — no long idle tail where 3 GPUs wait on 1.
3. **Robust + resumable** — a mid-run crash must not throw away completed folds.
4. **Surgical** — keep `accelerate launch` + `PartialState`; do not rewrite the
   launcher into hand-rolled bash multi-process.

## 3. Locked design decisions

| # | Decision | Choice |
|---|----------|--------|
| Scope | C: robust + well-utilized | filesystem work-queue, per-fold result files, resume |
| Fold-exception policy | B: quarantine & continue | write `.failed` marker + traceback, keep going; run exits non-zero if any fold failed |
| Cross-run claim model | C: per-run claims | claims are intra-run only; `.result.json` is the sole cross-run (resume) state |
| Launch model | A: keep accelerate | keep `accelerate launch --num_processes 4` + `PartialState`; only remove the collective |

Two guard requirements attached to launch-model A:
- **Guard G1 — no collectives:** the package must call **no** `gather_object`,
  `broadcast`, `barrier`, `wait_for_everyone`, or any `torch.distributed` collective.
  `PartialState` is used only for `process_index`, `num_processes`, `device`,
  `is_main_process`.
- **Guard G2 — verify no lazy NCCL init:** a cluster smoke test must confirm that
  `accelerate launch --num_processes 4` with **no** collective never triggers an
  NCCL/timeout (each rank only prints rank/device and writes a file).

## 4. Architecture

`run_cv` changes from "static shard → run slice → `gather_object` → rank0 writes"
to "**filesystem work-queue → per-fold result files → rank0 assembly**", with zero
collectives. Everything else (`run_fold_with_producer`, `make_fold_producer`, the
exp07 metric harness, per-epoch CSV flush, manifest/summary writers) is unchanged.

All coordination state lives under a new `<output_dir>/_fold_results/` directory:

- **Job list:** `jobs = [(s, f) for s in split_types for f in config.folds]`
  (ordered, deterministic). 15 jobs for the full run.
- **Result file:** `<output_dir>/_fold_results/<split>_fold<k>.result.json` —
  JSON list of the metric-row dicts returned by `run_fold_with_producer`. Atomic
  write (temp file + `os.replace`). **Only cross-run state** → resume.
- **Failure marker:** `<output_dir>/_fold_results/<split>_fold<k>.failed` — JSON
  `{traceback, rank, timestamp}`. Written when a fold raises (quarantine).
- **Claim marker:** `<output_dir>/_fold_results/.claims/<run_token>/<split>_fold<k>`
  directory, created with atomic `os.mkdir`. Intra-run only; scoped by
  `run_token` (`SLURM_JOB_ID` | `SL_DL_RUN_ID` | `local-<ppid>`) so a prior run's
  orphan claim never blocks a resume.

### 4.1 Worker loop (every rank runs the identical loop)

```text
state = PartialState()
for (split, fold) in jobs:                     # same ordered list on every rank
    if result_path(split, fold).exists():      # resume / done by someone
        continue
    if failed_path(split, fold).exists():       # already quarantined this run
        continue
    try:
        os.mkdir(claim_path(split, fold))        # ATOMIC claim
    except FileExistsError:
        continue                                 # another rank owns it
    # double-check after winning the claim (handles resume race)
    if result_path(split, fold).exists():
        continue
    try:
        rows = run_fold_with_producer(frame, split, fold, config, producer_for(split, fold))
        atomic_write_json(result_path(split, fold), rows)
    except Exception as exc:                     # QUARANTINE (decision B)
        atomic_write_json(failed_path(split, fold),
                          {"traceback": format_exc(), "rank": state.process_index, ...})
        logger.error("fold %s/%d failed on rank %d; quarantined", split, fold, rank)
        continue
```

Because every rank walks the **same ordered job list** and the claim is atomic, each
job is executed by exactly one rank; a rank that loses a claim immediately tries the
next job → **GPUs stay busy until the queue drains** (decision C / goal 2). The only
idle tail is the unavoidable final 1–3 jobs.

`producer_for(split, fold)` is the existing branch: `make_fold_producer(config,
shared, frame, split, fold)` for `"state_dl"`, else the reusable producer instance.
`shared` caches are still loaded once per process before the loop, unchanged.

### 4.2 Assembly (rank0 only, no collective)

After its worker loop returns, **rank0** assembles the combined artifacts; non-main
ranks simply return after their loop (they write nothing global).

```text
if not state.is_main_process:
    return summary_placeholder            # non-main ranks done

# rank0: wait until every job is terminal (result OR failed), bounded by a deadline
poll until all (split,fold) have result_path | failed_path, or assembly_timeout hit
collect rows from every existing *.result.json
if no rows at all -> RuntimeError (same empty-guard as today)
fold_metrics = concat(rows); sort canonical; _summarize
write fold_metrics.csv / summary.csv / manifest.json / per-split dirs / official_metrics_summary.csv
   (identical writer code to today)
failed = [jobs without a result.json]
if failed: log them and exit non-zero (decision B: surface failure, keep good results)
```

The rank0 **poll** replaces the `gather_object` barrier. It is a filesystem poll
(e.g. every few seconds), **not** a collective — no NCCL, no fixed 600 s PG timeout.
A configurable `assembly_timeout` (generous, e.g. ≥ longest expected single fold)
bounds it so a hard-crashed worker that left a claim but no result can't hang rank0
forever; on timeout rank0 assembles whatever results exist and treats the missing
folds as failed (non-zero exit). Per-run claims (decision C) mean a hard crash leaves
an orphan claim dir, which is fine: it is ignored on the next resubmit (only
`.result.json` is consulted across runs), and the missing fold is simply
re-claimable next run.

### 4.3 Output equivalence

Rows are concatenated from all `*.result.json` and sorted by the existing canonical
key `["split_type", "fold_id", "model", "slice", "metric"]` before `_summarize`, so
the assembled `fold_metrics.csv` / `summary.csv` are **byte-identical** to a
single-process run (same invariant the old `gather_object` path guaranteed). This is
an explicit test (§6).

## 5. Components & interfaces

New small helpers, local to `evaluate.py` (each one job, independently testable):

- `_fold_results_dir(config) -> Path` — `<output_dir>/_fold_results`.
- `_result_path / _failed_path / _claim_path(dir, split, fold) -> Path`.
- `_atomic_write_json(path, obj)` — temp + `os.replace`.
- `_try_claim(dir, split, fold) -> bool` — `os.mkdir`, returns False on
  `FileExistsError`.
- `_is_done(dir, split, fold) -> bool` — `result_path.exists()`.
- `_run_worker_queue(config, shared, frame, jobs, producer, state) -> None` — the
  §4.1 loop. Replaces `_shard_jobs` + `_run_local_jobs`.
- `_assemble(config, jobs, split_types, ...) -> pd.DataFrame` — the §4.2 rank0
  collect/poll/write. Reuses every existing writer + `_build_manifest` verbatim.

`_shard_jobs` and `_run_local_jobs` and the `gather_object` import are **removed**.
`run_cv`'s signature, the `EmbeddingProducer` protocol, `ZeroEmbeddingProducer`,
`StateDlCaches`, `_load_state_dl_caches`, `_gwps_coverage_count`, `_build_manifest`
are **unchanged**.

New config fields on `SLDLConfig` (with conservative defaults so existing configs
keep working): `assembly_poll_seconds` (default ~5), `assembly_timeout_seconds`
(default generous, e.g. ≥ one fold's wall-clock). Optional `fold_results_subdir`
name override (default `_fold_results`).

## 6. Testing (TDD)

Unit tests (`tests/`, no GPU, no cluster — must run on the macOS box per
`[[macos-omp-pytest-segfault]]` constraints, `OMP_NUM_THREADS=1` already in
conftest):

1. **Atomic claim is exclusive** — two simulated "ranks" call `_try_claim` on the
   same job; exactly one returns True.
2. **Resume skips done folds** — pre-write a `.result.json`; worker loop does not
   re-run it (use a stub producer that records calls).
3. **Quarantine on fold exception** — stub producer raises; loop writes `.failed`
   with traceback, continues to next job, run exits non-zero at assembly.
4. **Assembly output-equivalence** — run the queue with `num_processes=1` and a
   deterministic stub producer; assert `fold_metrics.csv` byte-identical to a
   reference produced by the old serial path (or a hand-built expected frame).
5. **Empty-rows guard preserved** — no matching splits → `RuntimeError` (existing
   behavior).
6. **No collective symbols** — static assertion / grep-style test that
   `evaluate.py` imports/calls none of `gather_object`, `broadcast`, `barrier`,
   `wait_for_everyone` (encodes Guard G1 so it can't regress).
7. **Assembly deadline** — simulate a job that never produces a result; assembler
   stops at `assembly_timeout_seconds`, writes partial results, exits non-zero.

Cluster smoke test (Guard G2, manual, documented in `scripts/`): a tiny
`accelerate launch --num_processes 4` job where each rank prints
`process_index`/`device` and writes a per-rank file, calling **no** collective —
confirm it exits 0 with 4 files and **no** NCCL/timeout. This is the empirical proof
that keeping `PartialState` under launch-model A is safe before burning the 4-day
run.

## 7. Operational checklist (not code)

1. `git pull` the cluster checkout to current `main` (+ this fix branch) so the
   stale per-pair runtime is gone (stale-runtime caveat §1).
2. Run the Guard G2 smoke test once; confirm exit 0, no timeout.
3. Launch the full run: `phase3_bag_supervision.yaml` (CV1/CV2/CV3, folds 0–4),
   `producer=state_dl`, `--num_processes 4`.
4. On any crash, resubmit the same command: completed folds' `.result.json` are
   skipped, only unfinished folds re-run.
5. Assembly writes `official_metrics_summary.csv`; compare CV2/CV3 NDCG@k & MAP@k
   against in-harness exp06 (`[[exp06-sl-baseline-facts]]`). CV1 is degree-gameable
   and does not count toward success.

## 8. Out of scope

- No change to the model, losses, pooling, pair head, scoring, or metric harness.
- No change to the NaN-guard work already on `fix/exp08-phase3-nan-guards`
  (`[[exp08-task-definition]]`); this fix composes with it (a fold that hits the
  zero-step / non-finite guards now raises → gets quarantined instead of killing the
  run).
- No multi-node support (run is `-N 1`, 4 GPUs).
