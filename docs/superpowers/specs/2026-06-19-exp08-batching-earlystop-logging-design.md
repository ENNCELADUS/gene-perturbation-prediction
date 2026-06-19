# exp08 Training-Loop Fixes: Batching, Test-Fold Early Stopping, Per-Rank Logging

- **Date**: 2026-06-19
- **Experiment**: 08 — K562 SL-pair STATE-adapter DL ranking
- **Status**: Design approved (Q&A 2026-06-19); pending written-spec review
- **Scope**: `src/sl_dl_model/` training loop, logging, and manifest only.
  The official metric path (`scoring.py:_metric_rows`, `_build_*_score_matrix`)
  is NOT touched.

## Problem

A code review of the exp08 pipeline surfaced four verified defects in the
training loop and logging. All four are confirmed against the code at commit
`1f78e13`:

1. **`batch_pairs` is dead.** Defined at `config.py:63`, set to `1024` in both
   phase configs, written nowhere-consumed. `train.py:535` iterates
   `self.train_pairs` one tuple at a time; `optimizer.step()` at `train.py:606`
   fires once per pair. Effective batch size is 1.
2. **tqdm spam.** `train.py:527` wraps `self.train_pairs` per epoch. With one
   step per pair that is ~44,046 steps/epoch, producing an unreadable stdout.
3. **Logging is not rank/fold-aware.** `__main__.py:68` calls `basicConfig`
   once; `--log-file` is optional and the SLURM wrapper does not pass it. Under
   fold-parallel multi-rank execution all ranks write the same file, producing
   the duplicated warnings seen in the untracked `slurm_*.err`.
4. **No epoch-level validation or best-epoch selection.** `train.py:524` runs a
   fixed `max_epochs` and uses final-epoch weights directly.

`lambda_rank` (also dead, `config.py:60`) is **explicitly out of scope** this
round — it stays as-is, disclosed in the config README.

## Decisions (locked via Q&A)

| Topic | Decision |
|---|---|
| Validation source | The fold's **own test split** is the validation set (user override; matches SynLethDB `valid_rat=0` protocol). Leakage accepted by the user. |
| Reported metric | **Best-epoch only** (no separate fixed-final-epoch metric). |
| Best-epoch selector | **Cheap proxy**: pair AUROC over the test-fold pairs each epoch. |
| Stop criterion | **Patience-based** early stop on the proxy. |
| Official metric | Computed **once**, on the restored best-epoch weights, for reporting. |
| Batching | **Gradient accumulation** over `batch_pairs` pairs, **mean** loss reduction. |
| `batch_pairs` value | Keep `1024` in both phase configs (now effective). |
| Log file | **On by default** at `output_dir/train.log` (rank0); `--log-file` overrides. |
| Per-fold curves | **Per-rank metric logs** `train_rank{N}.log`; tqdm + infra warnings rank0-only. |
| Epoch line content | `split_type, fold_id, epoch, mean train loss + sl/bag/distill components, val pair AUROC, peak GPU mem`. |
| Curve data | Structured `output_dir/<split>/epoch_metrics_fold{N}.csv` per fold. |
| tqdm | Per-batch bar (~43/epoch), rank0-only. |
| `lambda_rank` | Out of scope; unchanged. |

## Design

### 1. Effective batching (`train.py` `_train`)

Replace the per-pair optimizer step with gradient accumulation:

- Iterate `self.train_pairs` in contiguous chunks of `config.batch_pairs`.
- For each pair in the chunk, compute its combined loss exactly as today
  (`sl` + optional `bag` + optional `distill`, weighted by the epoch's warmup
  `_epoch_weights`). Pairs with missing ESM2 vectors are skipped as today.
- Accumulate the per-pair combined losses, take the **mean over the pairs
  actually trained in the chunk**, then a single
  `optimizer.zero_grad()/backward()/step()` per chunk.
- The per-pair embedding path (frozen STATE forwarded per gene via
  `embed_gene`) is unchanged; only the gradient step is batched.
- `lr` stays `1e-3`. Mean reduction keeps gradient magnitude invariant to
  `batch_pairs`, so no lr retuning.

Result: ~44k pairs / 1024 ≈ 43 optimizer steps per epoch.

Edge cases: a trailing partial chunk is stepped on its actual count; a chunk
where every pair is skipped contributes no step. The existing
`trained == 0` → `RuntimeError` guard (`train.py:615`) is preserved at the
epoch level.

### 2. Test-fold early stopping

**Data threading.** `make_fold_producer` (`scoring.py:265`) currently discards
`test_df` (`train_df, _ = fold_split(...)`). It must build `val_pairs` from
`test_df` (same 5-tuple shape as `train_pairs`) and pass them to
`StateDlProducer`. `StateDlProducer.__init__` gains a `val_pairs` parameter.

**Per-epoch proxy.** After each epoch's training pass, with `model.eval()` and
`torch.no_grad()`, score every `val_pair` (forward through `embed_gene` +
`score_pairs`, sigmoid) and compute **pair AUROC** against the val labels.
Val pairs with missing ESM2 vectors are skipped (consistent with training).

**Best-epoch tracking + patience.** Track the best val AUROC and the
`state_dict` of the model at that epoch (kept in memory). A new config field
`early_stop_patience: int` controls stopping: if val AUROC has not improved for
`patience` consecutive (tracked) epochs, stop and restore the best weights.

**Warmup interaction (proposed default — flagged for review).** Epochs
`< warmup_epochs` train with `lambda_sl = 0` (distill+bag only, per
`_epoch_weights`), so the SL head receives no direct signal and val AUROC is
uninformative there. Proposed: **do not track best-epoch or count patience
until `epoch >= warmup_epochs`**. This prevents selecting a warmup epoch or
tripping patience before SL training begins. If no epoch `>= warmup_epochs`
improves (or `max_epochs <= warmup_epochs`), fall back to the final epoch's
weights. **This default needs the user's explicit confirmation.**

**Official metric.** After training stops, `produce`/`score_matrix` run on the
restored best-epoch weights exactly as today. The official per-anchor metric is
computed once, downstream, unchanged.

**Determinism.** Best-epoch tracking uses strict improvement (`>`) so ties
resolve to the earliest epoch, keeping selection deterministic under fixed
seed.

### 3. Logging & tqdm

- **Default log file.** `__main__.py` defaults `--log-file` to
  `output_dir/train.log` when not provided; an explicit `--log-file` overrides.
  The file handler is attached **only on the main process** (rank0). The SLURM
  wrapper needs no change.
- **Per-rank metric logs.** Each rank writes its own folds' per-epoch metric
  lines to `output_dir/train_rank{process_index}.log`, every line tagged
  `[rank N][split/fold]`. This captures all folds' curves regardless of which
  rank owns them.
- **rank0-only channels.** The tqdm per-batch progress bar (one bar per epoch,
  ~43 steps) and infrastructure-level warnings/errors stay gated on
  `state.is_main_process` (as the existing `disable=` pattern does).
- **Epoch line content.** `split_type, fold_id, epoch, mean_train_loss,
  loss_sl, loss_bag, loss_distill, val_pair_auroc, peak_gpu_mem_mb`
  (`torch.cuda.max_memory_allocated`, reset per epoch; emit `nan`/`0` on CPU).
- **Curve CSV.** Each fold also writes
  `output_dir/<split_type>/epoch_metrics_fold{fold_id}.csv` with one row per
  epoch and the same columns, for pandas plotting. Written by the rank that
  owns the fold.

### 4. Manifest / reproducibility

Add to `_build_manifest` (`evaluate.py:273`):

- `batch_pairs` (now meaningful)
- `early_stop_patience`
- `early_stop_metric = "val_pair_auroc"`
- `val_source = "test_fold"`
- per-fold `stopped_epoch` (the epoch whose weights were restored)

The manifest must carry an honesty note: best-epoch selection uses the test
fold (SynLethDB-style `valid_rat=0`), so exp08-vs-exp06 is **selection-matched
to the benchmark**, not a strict embedding-only ablation. This note also
belongs in the experiment doc and config README.

## New / changed config fields (`config.py`)

- `early_stop_patience: int` — new. Default proposed `5` (confirm in plan).
- `batch_pairs: int = 1024` — unchanged value, now consumed.
- No change to `lambda_rank`, `max_epochs`, `warmup_epochs`, `lr`.

## Out of scope

- `lambda_rank` / RankNet wiring (stays dead, README-disclosed).
- Any change to the official metric, `_metric_rows`, score-matrix construction,
  or the exp06 comparison harness.
- True vectorized batching of the STATE forward path (gradient accumulation
  only).
- Changing `max_epochs`, `warmup_epochs`, loss weights, or lr.
- Inner train/val carving or any leakage-avoidance split (explicitly rejected
  by the user in favor of test-fold validation).

## Testing strategy (TDD)

Unit tests (CPU, `linear_mock`/tiny-fixture backend where possible):

1. **Batching steps.** With a stub optimizer counting `.step()` calls and a
   fixture of N pairs, assert step count ≈ `ceil(N / batch_pairs)` per epoch,
   not N.
2. **Mean reduction.** A two-pair batch yields a gradient equal to the mean of
   the two per-pair gradients (within tolerance).
3. **Val AUROC selection.** With a deterministic fixture where a known epoch is
   best, assert the restored `stopped_epoch` matches and that final weights
   equal the best-epoch weights.
4. **Patience stop.** Construct a monotonically-worsening proxy after a peak;
   assert training stops `patience` epochs after the peak and restores the
   peak weights.
5. **Warmup guard.** Assert no epoch `< warmup_epochs` is ever selected as
   best and patience does not count during warmup.
6. **Per-rank log routing.** Assert epoch lines are tagged with rank/split/fold
   and that the per-fold epoch CSV exists with the expected columns and one row
   per trained epoch.
7. **Default log file.** Assert that with no `--log-file`, a handler targets
   `output_dir/train.log` on the main process only.
8. **Manifest fields.** Assert `batch_pairs`, `early_stop_patience`,
   `early_stop_metric`, `val_source`, and `stopped_epoch` appear.

Verification: `uv run ruff check`, `uv run ruff format`, and the targeted
pytest module. Per the macOS note, set `OMP_NUM_THREADS=1` (already handled in
`conftest`).
