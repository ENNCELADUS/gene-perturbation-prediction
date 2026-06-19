# exp08 tqdm Per-Batch Progress Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-epoch tqdm bar in `StateDlProducer._train` advance once per optimizer step (batch), not once per training pair.

**Architecture:** Task 2 introduced gradient accumulation (`_flush()` does one optimizer step per `batch_pairs` pairs) but the tqdm bar at `train.py:599-603` still wraps `self.train_pairs`, so it ticks ~44k times/epoch — the original log-spam defect. Replace the iterable-style bar with a fixed-`total` bar (`total = ceil(len(train_pairs)/batch_pairs)`) and call `pbar.update(1)` inside `_flush()`. The bar stays rank0-only via the existing `disable=not state.is_main_process`.

**Tech Stack:** Python 3.11+, PyTorch, `tqdm.auto`, Accelerate `PartialState`, pytest, ruff, uv.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings. No `print` in library code; use `logging`. (CLAUDE.md)
- Prefix all Python/pytest/ruff invocations with `uv run`. (CLAUDE.md)
- CPU unit tests use `state_backend="linear_mock"`; set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch. (tests/sl_dl_model/test_train.py)
- Conventional Commits; attribution disabled globally (no Co-Authored-By trailer).
- Edit ONLY the training loop's progress-bar wiring. Do NOT change batching math, loss, validation, early-stopping, or logging behavior. The optimizer-step count and `epoch_metrics` must be unchanged.
- The bar must remain disabled on non-main processes (rank0-only), matching the current `disable=not state.is_main_process`.

---

### Task 1: Drive the per-epoch tqdm bar by optimizer step

**Files:**
- Modify: `src/sl_dl_model/train.py` (`StateDlProducer._train`, the per-epoch block ~lines 596-697)
- Test: `tests/sl_dl_model/test_train_batching.py` (add one test; reuse existing `_producer` helper)

**Interfaces:**
- Consumes: `SLDLConfig.batch_pairs`, the existing nested `_flush()` and `self.train_pairs`.
- Produces: no signature change. The module-level `tqdm` (`from tqdm.auto import tqdm`, train.py:25) is now constructed with `total=` and advanced via `.update(1)` per flush.

**Behavior contract:** For one epoch with `N = len(self.train_pairs)` pairs and `B = config.batch_pairs`, the bar is constructed once with `total = max(1, (N + B - 1) // B)` and `.update(1)` is called exactly once per non-empty `_flush()`. With no skipped pairs that is `ceil(N/B)` updates. The optimizer-step count is unchanged from Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_train_batching.py` (it already defines `_producer(n_pairs, batch_pairs, max_epochs)` and imports `os`/`PYTORCH_ENABLE_MPS_FALLBACK`, numpy, torch, `PartialState`):

```python
def test_tqdm_advances_once_per_batch(monkeypatch):
    """The epoch progress bar ticks once per optimizer step, not per pair.

    10 pairs, batch_pairs=4, 1 epoch -> ceil(10/4)=3 batches -> total=3 and
    exactly 3 .update(1) calls (not 10).
    """
    import sl_dl_model.train as train_mod

    constructed: list[dict] = []

    class _FakeBar:
        def __init__(self, *args, total=None, **kwargs):
            self.updates = 0
            constructed.append({"total": total, "bar": self})

        def update(self, n: int = 1) -> None:
            self.updates += n

        def close(self) -> None:
            pass

    monkeypatch.setattr(train_mod, "tqdm", _FakeBar)

    producer = _producer(n_pairs=10, batch_pairs=4, max_epochs=1)
    producer._fit_ge_standardizer()
    model = producer._build_model()
    state = PartialState()
    model = model.to(state.device)
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    producer._train(model, opt, state, {"G0", "G1"})

    # Exactly one bar constructed for the single training epoch.
    assert len(constructed) == 1, f"expected 1 bar, got {len(constructed)}"
    assert constructed[0]["total"] == 3, f"expected total=3, got {constructed[0]['total']}"
    assert constructed[0]["bar"].updates == 3, (
        f"expected 3 per-batch updates, got {constructed[0]['bar'].updates}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_batching.py::test_tqdm_advances_once_per_batch -v`

Expected: FAIL for the intended reason — the current code constructs the bar as `tqdm(self.train_pairs, ...)` (no `total` kwarg, so `total=None`) and never calls `.update()`, so `constructed[0]["total"]` is `None` (assert mismatch `None == 3`) and `.updates` is `0`. This is the RED gate; capture the output.

- [ ] **Step 3: Commit the reproducer (RED checkpoint)**

```bash
git add tests/sl_dl_model/test_train_batching.py
git commit -m "test: add reproducer for per-batch tqdm progress (exp08)"
```

- [ ] **Step 4: Rewire the bar to advance per flush**

In `src/sl_dl_model/train.py`, inside `_train`'s epoch loop, replace the bar construction (currently lines 599-603):

```python
            pbar = tqdm(
                self.train_pairs,
                desc=f"epoch {epoch}",
                disable=not state.is_main_process,
            )
```

with a fixed-`total` bar driven by optimizer steps:

```python
            n_batches = max(
                1,
                (len(self.train_pairs) + self.config.batch_pairs - 1)
                // self.config.batch_pairs,
            )
            pbar = tqdm(
                total=n_batches,
                desc=f"epoch {epoch}",
                disable=not state.is_main_process,
            )
```

In the nested `_flush()` (lines 610-621), advance the bar after a successful step. Change the body so that after `optimizer.step()` and the loss bookkeeping it calls `pbar.update(1)`:

```python
            def _flush() -> None:
                nonlocal batch_loss_count, batch_loss_sum, batch_losses
                if not batch_losses:
                    return
                batch_total = torch.stack(batch_losses).mean()
                batch_size = len(batch_losses)
                optimizer.zero_grad()
                batch_total.backward()
                optimizer.step()
                batch_loss_sum += float(batch_total.detach().cpu()) * batch_size
                batch_loss_count += batch_size
                batch_losses = []
                pbar.update(1)
```

Change the training iteration from `for a, b, label, ea, eb in pbar:` (line 623) to iterate the pairs directly:

```python
            for a, b, label, ea, eb in self.train_pairs:
```

After the trailing `_flush()` (line 697), close the bar:

```python
            _flush()
            pbar.close()
```

(`pbar` is created before `_flush` is defined, so the closure reads it correctly; only methods are called, no rebinding, so no `nonlocal pbar` is needed.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_train_batching.py -v`
Expected: PASS (both `test_one_step_per_batch_not_per_pair` and the new test).

- [ ] **Step 6: Run regression suites**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py tests/sl_dl_model/test_train_earlystop.py -v`
Expected: PASS — optimizer-step count, `epoch_metrics`, and early-stopping behavior are unchanged.

- [ ] **Step 7: Lint and format**

Run: `uv run ruff check src/sl_dl_model/train.py tests/sl_dl_model/test_train_batching.py && uv run ruff format src/sl_dl_model/train.py tests/sl_dl_model/test_train_batching.py`
Expected: clean. If format changed either file, re-run Step 5 to confirm green before committing.

- [ ] **Step 8: Commit (GREEN checkpoint)**

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_train_batching.py
git commit -m "fix: advance epoch tqdm bar once per optimizer step, not per pair (exp08)"
```

---

## Self-Review Notes

- **Spec coverage:** the single defect (tqdm wraps `self.train_pairs` → per-pair ticks) is fixed by Task 1; the new test pins update-per-batch granularity and the `total` value.
- **No behavior drift:** batching math, `_flush()` step semantics, `batch_loss_sum/count`, `epoch_metrics`, and early stopping are untouched; Step 6 regressions guard this.
- **Rank-awareness preserved:** the bar keeps `disable=not state.is_main_process`.
- **Type consistency:** `n_batches` integer ceil matches the optimizer-step count asserted by the pre-existing `test_one_step_per_batch_not_per_pair`.
- **Edge case:** `max(1, ...)` guards an empty `train_pairs` (bar total never 0); a fully-skipped epoch still raises the existing `trained == 0` RuntimeError before any metric use.
