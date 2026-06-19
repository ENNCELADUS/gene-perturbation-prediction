# exp08 Per-Epoch Metrics Live Flush Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write each epoch's training/validation metrics row to the per-fold CSV as soon as that epoch's train+val completes, so the run can be inspected (and judged stable / worth continuing) while training is still in progress.

**Architecture:** Today `_train` appends each epoch's metrics to the in-memory `self.epoch_metrics` list, and the CSV + per-rank log line are only emitted in `scoring.run_fold_with_producer` AFTER training and the expensive `score_matrix` finish. This plan makes `_train` append one row to `<output_dir>/<split_type>/epoch_metrics_fold{fold_id}.csv` at the end of every epoch (incrementally, header written once), and emit the per-rank `logger.info` line per epoch. To do that, the producer must know where to write: a new optional `epoch_log_target` is threaded from `make_fold_producer` into `StateDlProducer`. The end-of-fold bulk write in `scoring.py` is removed to avoid double-writing.

**Tech Stack:** Python 3.11+, PyTorch, pandas, Accelerate `PartialState`, pytest, ruff, uv.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings. No `print` in library code; use `logging`. No hardcoded paths/thresholds; use config. Handle specific exceptions, no bare `except`. (CLAUDE.md)
- Prefix all Python/pytest/ruff invocations with `uv run`. (CLAUDE.md)
- CPU unit tests use `state_backend="linear_mock"`; set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch. (tests/sl_dl_model/test_train.py)
- Conventional Commits; attribution disabled globally (no Co-Authored-By trailer).
- Behavior to PRESERVE exactly: batching/optimizer-step count, loss math, `_validate_auroc`, best-epoch selection + patience early stop, the per-batch tqdm bar, and the final CSV's columns/content/order (`split_type, fold_id, epoch, mean_train_loss, val_pair_auroc, peak_gpu_mem_mb`). The only change is WHEN rows are written (incrementally per epoch instead of once at fold end) and that `_train` does the writing.
- `epoch_metrics` (in-memory list) stays populated — other code and tests read it. Live flushing is ADDITIVE, not a replacement of the attribute.
- Writing must stay rank-aware: a fold runs entirely on one rank (`run_cv` shards folds across ranks), so the rank that owns the fold writes its own CSV. Do not add cross-rank coordination.

---

### Task 1: Flush epoch metrics to the per-fold CSV at end of each epoch

**Files:**
- Modify: `src/sl_dl_model/scoring.py` — refactor `write_epoch_metrics` to append a single epoch row (header once); thread an `epoch_log_target` into `make_fold_producer`'s `StateDlProducer(...)`; remove the bulk `write_epoch_metrics` call in `run_fold_with_producer` (keep the per-epoch logging there? No — moved into `_train`; see Step 5).
- Modify: `src/sl_dl_model/train.py` — `StateDlProducer.__init__` accepts `epoch_log_target`; `_train` writes + logs each epoch row as it completes.
- Test: `tests/sl_dl_model/test_logging.py` (append-row behavior) and `tests/sl_dl_model/test_train_earlystop.py` (live-flush during `_train`).

**Interfaces:**
- Consumes: `SLDLConfig.output_dir`, `PartialState().process_index`, existing per-epoch metric dict.
- Produces:
  - `append_epoch_metric_row(output_dir: Path, split_type: str, fold_id: int, row: dict[str, float]) -> Path` in `scoring.py` — appends ONE row to `<output_dir>/<split_type>/epoch_metrics_fold{fold_id}.csv`, writing the header only when the file does not yet exist. Returns the CSV path.
  - `StateDlProducer.__init__(..., epoch_log_target: tuple[Path, str, int] | None = None)` where the tuple is `(output_dir, split_type, fold_id)`. When `None` (unit tests, zero-producer), `_train` skips disk writes but still appends to `self.epoch_metrics`.

**Design note — why a tuple, not three params:** `_train` already has many locals; one optional bundle keeps the signature change minimal and makes "logging disabled" a single `None` check. `make_fold_producer` is the only production caller and has all three values.

- [ ] **Step 1: Write the failing test for incremental append**

Add to `tests/sl_dl_model/test_logging.py`:

```python
def test_append_epoch_metric_row_writes_header_once(tmp_path: Path):
    from sl_dl_model.scoring import append_epoch_metric_row

    r0 = {"epoch": 0.0, "mean_train_loss": 0.7, "val_pair_auroc": 0.5, "peak_gpu_mem_mb": 0.0}
    r1 = {"epoch": 1.0, "mean_train_loss": 0.6, "val_pair_auroc": 0.55, "peak_gpu_mem_mb": 0.0}

    out = append_epoch_metric_row(tmp_path, "CV2", 3, r0)
    assert out == tmp_path / "CV2" / "epoch_metrics_fold3.csv"
    # After the first row the file is readable with exactly one data row.
    df0 = pd.read_csv(out)
    assert list(df0.columns) == [
        "split_type", "fold_id", "epoch",
        "mean_train_loss", "val_pair_auroc", "peak_gpu_mem_mb",
    ]
    assert len(df0) == 1

    append_epoch_metric_row(tmp_path, "CV2", 3, r1)
    df1 = pd.read_csv(out)
    # Second call appends (not overwrites) and does NOT add a second header.
    assert len(df1) == 2
    assert df1["epoch"].tolist() == [0.0, 1.0]
    assert (df1["split_type"] == "CV2").all()
    assert (df1["fold_id"] == 3).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_logging.py::test_append_epoch_metric_row_writes_header_once -v`
Expected: FAIL with `ImportError: cannot import name 'append_epoch_metric_row'`.

- [ ] **Step 3: Commit the reproducer (RED checkpoint)**

```bash
git add tests/sl_dl_model/test_logging.py
git commit -m "test: add reproducer for incremental per-epoch metric append (exp08)"
```

- [ ] **Step 4: Implement `append_epoch_metric_row` in `scoring.py`**

Add this function next to `write_epoch_metrics` (keep `write_epoch_metrics` for the existing `test_write_epoch_metrics_csv` test — it is unchanged and still valid as a batch writer):

```python
def append_epoch_metric_row(
    output_dir: Path,
    split_type: str,
    fold_id: int,
    row: dict[str, float],
) -> Path:
    """Append one epoch's metrics row to the per-fold CSV, header written once.

    Designed for live flushing during training: the file is created with a
    header on the first call and appended to (no repeated header) on every
    subsequent call, so the CSV is readable mid-run.

    Args:
        output_dir: Run output directory.
        split_type: CV split type (e.g. ``"CV2"``).
        fold_id: Fold id.
        row: One epoch's metrics with keys ``epoch``, ``mean_train_loss``,
            ``val_pair_auroc``, ``peak_gpu_mem_mb``.

    Returns:
        Path to the per-fold CSV.
    """
    split_dir = output_dir / split_type
    split_dir.mkdir(parents=True, exist_ok=True)
    out = split_dir / f"epoch_metrics_fold{fold_id}.csv"
    record = {"split_type": split_type, "fold_id": fold_id, **row}
    frame = pd.DataFrame([record])
    write_header = not out.exists()
    frame.to_csv(out, mode="a", header=write_header, index=False)
    return out
```

- [ ] **Step 5: Run the append test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_logging.py -v`
Expected: PASS (new test + existing `test_write_epoch_metrics_csv` + `test_default_log_file_path`).

- [ ] **Step 6: Write the failing test for live flush inside `_train`**

Add to `tests/sl_dl_model/test_train_earlystop.py` (it already has the `_producer` helper, torch, numpy, `PartialState`, `os`/MPS-fallback):

```python
def test_epoch_metrics_flushed_to_disk_during_training(tmp_path):
    """Each epoch's row is on disk before training ends (live flush)."""
    import pandas as pd

    producer = _producer(max_epochs=3, patience=5)
    producer.epoch_log_target = (tmp_path, "CV2", 0)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )

    csv_path = tmp_path / "CV2" / "epoch_metrics_fold0.csv"
    rows_seen: list[int] = []

    # Patch the per-epoch validation hook to observe the CSV mid-training:
    # after epoch e completes, the file must already contain e+1 rows.
    real_validate = producer._validate_auroc

    def spy_validate(model_, device_, control_):
        result = real_validate(model_, device_, control_)
        if csv_path.exists():
            rows_seen.append(len(pd.read_csv(csv_path)))
        return result

    producer._validate_auroc = spy_validate
    producer._train(model, opt, state, {"G0", "G1", "G2", "G3"})

    # Final file has one row per trained epoch.
    final = pd.read_csv(csv_path)
    assert len(final) == len(producer.epoch_metrics)
    assert final["epoch"].tolist() == [
        m["epoch"] for m in producer.epoch_metrics
    ]
```

Note: `_validate_auroc` runs BEFORE the row for the current epoch is appended, so on entry for epoch `e` the file holds rows for epochs `0..e-1`. The assertion above checks the final state (rows == epochs trained); `rows_seen` documents intent. Keep the final-state assertion as the hard check.

- [ ] **Step 7: Run it to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_earlystop.py::test_epoch_metrics_flushed_to_disk_during_training -v`
Expected: FAIL — `epoch_log_target` is not an attribute / no CSV is written by `_train`, so `csv_path` never exists (`FileNotFoundError` from `pd.read_csv`).

- [ ] **Step 8: Add `epoch_log_target` to `StateDlProducer.__init__`**

In `src/sl_dl_model/train.py`, add the parameter and store it (after `val_pairs` at line 115):

```python
        val_pairs: list[tuple[str, str, int, float, float]] | None = None,
        epoch_log_target: tuple[Path, str, int] | None = None,
    ) -> None:
        self.config = config
        self.esm = esm
        self.bags = bags
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.epoch_log_target = epoch_log_target
        self.input_dim = input_dim
        self.output_dim = output_dim
```

(`Path` is already imported at train.py:19.)

- [ ] **Step 9: Flush + log each epoch row inside `_train`**

In `src/sl_dl_model/train.py`, replace the current `self.epoch_metrics.append({...})` block (lines 729-736) so it builds the row, appends to the in-memory list, AND flushes to disk + logs when a target is set. Add `from sl_dl_model.scoring import append_epoch_metric_row` as a LOCAL import inside `_train` (module-level would create a circular import: `scoring` imports `train`).

```python
            row = {
                "epoch": float(epoch),
                "mean_train_loss": mean_loss,
                "val_pair_auroc": float("nan") if val_auroc is None else val_auroc,
                "peak_gpu_mem_mb": peak_mb,
            }
            self.epoch_metrics.append(row)
            if self.epoch_log_target is not None:
                from sl_dl_model.scoring import append_epoch_metric_row

                out_dir, split_type, fold_id = self.epoch_log_target
                append_epoch_metric_row(out_dir, split_type, fold_id, row)
                logger.info(
                    "[rank %d][%s/fold%d] epoch %d: loss=%.4f val_auroc=%.4f "
                    "peak_gpu_mb=%.1f",
                    state.process_index,
                    split_type,
                    fold_id,
                    epoch,
                    row["mean_train_loss"],
                    row["val_pair_auroc"],
                    row["peak_gpu_mem_mb"],
                )
```

- [ ] **Step 10: Thread `epoch_log_target` in `make_fold_producer`**

In `src/sl_dl_model/scoring.py`, pass the target into the producer (after `val_pairs=val_pairs`):

```python
    return StateDlProducer(
        config,
        esm=caches.esm,
        bags=caches.bags,
        train_pairs=train_pairs,
        input_dim=caches.input_dim,
        output_dim=caches.output_dim,
        val_pairs=val_pairs,
        epoch_log_target=(Path(config.output_dir), split_type, fold_id),
    )
```

- [ ] **Step 11: Remove the now-duplicate bulk write/log in `run_fold_with_producer`**

In `src/sl_dl_model/scoring.py`, the DL branch currently calls `write_epoch_metrics(...)` and loops `logger.info` over all epochs after `score_matrix`. With live flushing in `_train`, this would write the CSV a SECOND time (overwriting the incrementally-built one) and double-log. Replace the whole `epoch_metrics = getattr(...)` / `if epoch_metrics:` block (scoring.py ~194-211, up to and including the `stopped_epoch` log line) with a single stopped-epoch summary log (the per-epoch rows are already on disk and logged):

```python
        sm = producer.score_matrix(universe.symbols, universe.gene_effects)
        if getattr(producer, "epoch_metrics", None):
            logger.info(
                "[rank %d][%s/fold%d] training complete: %d epochs, stopped_epoch=%s",
                PartialState().process_index,
                split_type,
                fold_id,
                len(producer.epoch_metrics),
                getattr(producer, "stopped_epoch", None),
            )
```

Keep the `_metric_rows(...)` calls and `return rows` below unchanged. `write_epoch_metrics` remains defined (still covered by `test_write_epoch_metrics_csv`) but is no longer called in this path.

- [ ] **Step 12: Run the live-flush test + all affected suites**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_earlystop.py tests/sl_dl_model/test_logging.py tests/sl_dl_model/test_scoring.py tests/sl_dl_model/test_train.py tests/sl_dl_model/test_train_batching.py -v`
Expected: PASS. If `test_scoring.py` asserted on the removed bulk-write/log block, update it to the new stopped-epoch summary log (only if it fails; do not pre-edit).

- [ ] **Step 13: Full exp08 suite + lint/format**

Run: `uv run python -m pytest tests/sl_dl_model/ -q`
Expected: all pass.
Run: `uv run ruff check src/sl_dl_model/ tests/sl_dl_model/ && uv run ruff format src/sl_dl_model/ tests/sl_dl_model/`
Expected: clean. If format changed files, re-run the suite before committing.

- [ ] **Step 14: Commit (GREEN checkpoint)**

```bash
git add src/sl_dl_model/train.py src/sl_dl_model/scoring.py tests/sl_dl_model/test_logging.py tests/sl_dl_model/test_train_earlystop.py
git commit -m "feat: flush exp08 epoch metrics to disk per epoch for live monitoring"
```

---

## Self-Review Notes

- **Spec coverage:** the defect (metrics only written after training+scoring finish) is fixed by writing each row at epoch end inside `_train` (Step 9), fed by `epoch_log_target` (Steps 8, 10), with the end-of-fold bulk write removed to prevent double-writing (Step 11).
- **No behavior drift:** batching, loss, `_validate_auroc`, best-epoch/patience, and the tqdm bar are untouched; `epoch_metrics` list still populated; final CSV columns/content identical to before. Steps 12-13 regression-guard this.
- **Circular import:** `scoring` imports `train` (via `make_fold_producer`), so `train` must import `append_epoch_metric_row` locally inside `_train`, not at module level (Step 9).
- **Crash resilience:** because rows are appended per epoch, a fold that crashes mid-run leaves all completed epochs on disk — the original goal.
- **Rank-awareness:** each fold runs on a single rank; that rank appends to its own per-fold CSV. No cross-rank writes, matching the existing sharding model.
- **Type consistency:** `append_epoch_metric_row` row keys match the dict built in `_train`; `epoch_log_target` tuple order `(output_dir, split_type, fold_id)` is identical at construction (Step 10) and unpacking (Step 9).
