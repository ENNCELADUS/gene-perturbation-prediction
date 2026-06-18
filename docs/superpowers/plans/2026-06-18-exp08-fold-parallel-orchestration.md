# exp08 Fold-Parallel Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `accelerate launch --num_processes 4` do distinct work by sharding the (split × fold) CV jobs across ranks, instead of every rank re-running the identical CV loop and discarding ranks 1–3.

**Architecture:** Fold-level task-parallelism with **no gradient all-reduce**. Each rank trains + embeds + scores its assigned `(split_type, fold_id)` jobs on its own GPU using a plain local model (the DDP wrap is removed), then `gather_object` collects every rank's metric rows onto the main process, which writes the existing per-split `cvN/` + combined artifacts. Because each fold is computed exactly once by exactly one rank through the unchanged per-fold code path, 1-process and N-process runs produce byte-identical metrics.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Accelerate (`PartialState`, `accelerate.utils.gather_object`), pandas, pytest, `uv`.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings. (CLAUDE.md Code Style)
- No `print` in library code; use `logging`. No bare `except`; handle specific exceptions. (CLAUDE.md)
- Prefix every Python/pytest/ruff invocation with `uv run`. (CLAUDE.md Environment)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `tests/conftest.py` before torch import — do not remove it; tests rely on it for `bag_loss` (`torch.cdist` backward). (tests/conftest.py)
- Conventional Commits: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`. (CLAUDE.md Commit Guidelines)
- **LOCKED — must not change (out of scope):** model graph (frozen STATE + ESM2→PertAdapter(1280→328) + MeanStdPool + SymmetricPairHead); training paradigm (3-part loss, warmup schedule, λ weights `sl=1.0, distill=0.5→0.1, bag=1.0, rank=0.0`, Adam-on-trainable-only, 20 epochs, **1 pair / optimizer step**, no `batch_pairs` activation); evaluation protocol (`official_ranking_metrics`/`official_classification_metrics` verbatim, −999999 seen-mask both directions, diagonal zero, CV1/CV2/CV3 semantics, per-split `cvN/` + `official_metrics_summary.csv`); leakage rules §5 (train-fold genes only); `sl_benchmark_baseline/*` reused untouched.
- **Parity gate (acceptance):** 1-process run == 4-process run, exact equality on `official_metrics_summary.csv`. Guaranteed because each fold is computed once by one rank via the unchanged serial code path.
- **Failure semantics:** fail-fast. Any fold crash aborts the whole run non-zero with traceback; no partial `official_metrics_summary.csv`. Do not wrap folds in try/except to survive a dead rank.

## File Map

| File | Responsibility | Change |
|---|---|---|
| `src/sl_dl_model/train.py` | `StateDlProducer.produce` / `_train` | Remove the DDP wrap: `Accelerator`→`PartialState`, plain `.backward()`, drop `prepare`/`unwrap_model`. Loss math, epoch loop, 1-pair/step **unchanged**. Update stale module docstring. (Task 1) |
| `src/sl_dl_model/evaluate.py` | `run_cv` orchestration | Add `_shard_jobs` round-robin helper (Task 2). Refactor `run_cv` to shard `(split,fold)` jobs by rank, run only this rank's shard via new `_run_local_jobs`, `gather_object` rows to main, keep the existing main-process write guard (Task 3). |
| `tests/sl_dl_model/test_fold_parallel.py` | Offline 1-vs-N parity gate | NEW test file (Task 4). |
| `scripts/sl_dl_model.sh` | Slurm launcher | Update the stale "redundant compute" NOTE comment to describe fold-parallelism; launch flags unchanged (Task 5). |
| `configs/experiments/08_k562_sl_pair_state_dl/README.md` | Run docs | Document the fold-parallel behavior (Task 5). |

**Explicitly NOT modified:** `src/sl_dl_model/scoring.py` (per-fold, already rank-agnostic — the shard boundary lives in `run_cv`, so `make_fold_producer` / `run_fold_with_producer` need no change), `src/sl_dl_model/__main__.py` (CLI unchanged), `model.py`/`encoder.py`/`pooling.py`/`pair_head.py`/`losses.py`/`bags.py`/`config.py`/`gene_embeddings.py`, and all of `src/sl_benchmark_baseline/`.

---

### Task 1: Remove the DDP wrap from `StateDlProducer` (Accelerator → PartialState)

The current `produce()` calls `accelerator.prepare(model, optimizer)` which, under `accelerate launch --num_processes N`, wraps the model in `DistributedDataParallel` and broadcasts rank-0 params to all ranks. Once ranks run *different* folds (Task 3), that broadcast/all-reduce **deadlocks** (ranks on different graphs with different step counts). Training already runs every forward/backward through the *unwrapped* model (`inner`), so the DDP wrapper does nothing useful today anyway. This task replaces `Accelerator` with `PartialState` (device + rank info only) and uses a plain local model + plain `.backward()`. This is numerically identical to today's single-process path, so it is a safe standalone change before any sharding.

**Files:**
- Modify: `src/sl_dl_model/train.py` (imports line 22; `produce` lines 357–367; `_train` signature line 490–496 and body lines 513–602; module docstring lines 1–13)
- Test: `tests/sl_dl_model/test_train.py` (existing `test_producer_emits_universe_table` already exercises `produce()` end-to-end — it is the regression guard for this refactor)

**Interfaces:**
- Consumes: `from accelerate import PartialState` (already proven available; used in `evaluate.py:22`).
- Produces: `StateDlProducer.produce(symbols, train_symbols) -> tuple[np.ndarray, np.ndarray]` (signature **unchanged**). Internal `_train(self, model, optimizer, state, train_symbols) -> None` where `state: PartialState` replaces the old `accelerator: Accelerator` parameter.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/sl_dl_model/test_train.py` (it asserts the new internal contract — that `produce` no longer constructs an `Accelerator` and that training still works through `PartialState`):

```python
def test_produce_uses_partialstate_not_accelerator(tmp_path, monkeypatch) -> None:
    """produce() must not instantiate Accelerator (DDP wrap removed)."""
    import sl_dl_model.train as train_mod

    # Fail loudly if any code path constructs an Accelerator.
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Accelerator must not be constructed in produce()")

    monkeypatch.setattr(train_mod, "Accelerator", _boom, raising=False)

    rng = np.random.default_rng(1)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = train_mod.Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B", "C", "D"]
        },
    )
    bags = train_mod.GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={"A": rng.standard_normal((8, 6)).astype("float32")},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=1,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
    )
    producer = train_mod.StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=[("A", "B", 1, -1.0, -0.5), ("C", "D", 0, 0.1, 0.2)],
        input_dim=6,
        output_dim=6,
    )
    emb, mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert emb.shape == (4, producer._model.emb_dim)
    assert mask.shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py::test_produce_uses_partialstate_not_accelerator -v`
Expected: FAIL with `AssertionError: Accelerator must not be constructed in produce()` (because `produce` currently calls `Accelerator()` at line 357).

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/train.py`, change the import (line 22):

```python
from accelerate import PartialState
```

In `produce()`, replace the Accelerator block (current lines 357–367) with:

```python
        state = PartialState()
        model = self._build_model()
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.config.lr,
        )
        model = model.to(state.device)
        self._train(model, optimizer, state, train_symbols)
        self._model = model

        device = state.device
```

Update the `_train` signature (current lines 490–496) and the lines inside it that reference `accelerator`:

```python
    def _train(
        self,
        model: SlDlModel,
        optimizer: optim.Optimizer,
        state: PartialState,
        train_symbols: set[str],
    ) -> None:
```

Inside `_train`, replace these three spots:
- `device = accelerator.device` (line 513) → `device = state.device`
- `inner = accelerator.unwrap_model(model)` (line 518) → delete this line; replace the following `self._model = inner` with `self._model = model`, and replace every later `inner.` in the loop body (`inner.embed_gene`, `inner.score_pairs`, and the `_bag_part(inner, ...)` call) with `model.` — the model is no longer wrapped, so `model` IS the trainable graph.
- `disable=not accelerator.is_main_process` (lines 382 and 527) → `disable=not state.is_main_process`
- `accelerator.backward(total)` (line 602) → `total.backward()`

In `produce()`, the embed-universe tqdm `disable=not accelerator.is_main_process` (line 382) → `disable=not state.is_main_process`.

Update the module docstring (lines 1–13) to remove the stale "DDP gradient sync is not engaged / single process" wording. Replace with:

```python
"""Per-fold training loop and the StateDlProducer.

StateDlProducer trains the model on one fold's pairs and then embeds every
gene in the universe through the frozen STATE backbone to produce a per-gene
embedding table.

Training runs on a single device per fold: each gene is forwarded one at a
time through the frozen STATE backbone, and gradients update only the trainable
adapter/pooling/pair-head. There is no DDP gradient all-reduce — fold-level
parallelism (one fold per rank) is orchestrated in
:func:`sl_dl_model.evaluate.run_cv`, which assigns disjoint folds to each rank
and gathers metric rows on the main process. ``PartialState`` supplies the
device and rank info; it does not wrap the model.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_train.py -v`
Expected: PASS (all existing tests + the new one). The `test_producer_emits_universe_table`, `test_dl_score_matrix`, and distill tests must still pass — they prove the loss math and embedding output are unchanged.

- [ ] **Step 5: Run lint**

Run: `uv run ruff check src/sl_dl_model/train.py tests/sl_dl_model/test_train.py && uv run ruff format src/sl_dl_model/train.py`
Expected: no errors. (The `Accelerator` import is now unused — `ruff` will flag F401; removing it is part of the import change in Step 3, so confirm it is gone.)

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/train.py tests/sl_dl_model/test_train.py
git commit -m "refactor: drop DDP wrap from StateDlProducer (Accelerator -> PartialState)"
```

---

### Task 2: Add the `_shard_jobs` round-robin helper to `evaluate.py`

A pure function that splits the ordered list of `(split_type, fold_id)` jobs across ranks by round-robin (`jobs[rank::num_processes]`). Round-robin (not contiguous chunks) keeps load balanced when fold costs vary. This is a standalone, side-effect-free helper with its own test cycle.

**Files:**
- Modify: `src/sl_dl_model/evaluate.py` (add helper near the other module-level helpers, after `_load_state_dl_caches`)
- Test: `tests/sl_dl_model/test_fold_parallel.py` (CREATE)

**Interfaces:**
- Produces: `_shard_jobs(jobs: list[tuple[str, int]], rank: int, num_processes: int) -> list[tuple[str, int]]` — returns the sublist of jobs owned by `rank`. Every job is owned by exactly one rank; the union across ranks equals `jobs`; order within a rank is preserved.

- [ ] **Step 1: Write the failing test**

Create `tests/sl_dl_model/test_fold_parallel.py`:

```python
"""Tests for fold-level task-parallel orchestration in run_cv (no all-reduce)."""

from __future__ import annotations

from sl_dl_model.evaluate import _shard_jobs


def test_shard_jobs_partitions_disjointly_and_covers_all() -> None:
    jobs = [("CV2", f) for f in range(5)] + [("CV3", f) for f in range(5)]
    num = 4
    shards = [_shard_jobs(jobs, r, num) for r in range(num)]

    # Disjoint: no job appears on two ranks.
    flat = [j for s in shards for j in s]
    assert len(flat) == len(jobs), "a job was duplicated or dropped"
    # Covers all: union equals the input set.
    assert set(flat) == set(jobs)
    # Balanced: 10 jobs / 4 ranks -> sizes 3,3,2,2.
    assert sorted(len(s) for s in shards) == [2, 2, 3, 3]


def test_shard_jobs_single_process_owns_everything() -> None:
    jobs = [("CV2", f) for f in range(5)]
    assert _shard_jobs(jobs, 0, 1) == jobs


def test_shard_jobs_more_ranks_than_jobs() -> None:
    jobs = [("CV2", 0), ("CV2", 1)]
    # Ranks 2 and 3 get nothing; no crash.
    assert _shard_jobs(jobs, 0, 4) == [("CV2", 0)]
    assert _shard_jobs(jobs, 1, 4) == [("CV2", 1)]
    assert _shard_jobs(jobs, 2, 4) == []
    assert _shard_jobs(jobs, 3, 4) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_fold_parallel.py -v`
Expected: FAIL with `ImportError: cannot import name '_shard_jobs' from 'sl_dl_model.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/evaluate.py`, add after `_load_state_dl_caches` (after line 310):

```python
def _shard_jobs(
    jobs: list[tuple[str, int]],
    rank: int,
    num_processes: int,
) -> list[tuple[str, int]]:
    """Return the round-robin slice of CV jobs owned by ``rank``.

    Round-robin (``jobs[rank::num_processes]``) keeps load balanced across
    ranks when per-fold cost varies. Every job is owned by exactly one rank and
    the union across all ranks reconstructs ``jobs`` in order.

    Args:
        jobs: Ordered ``(split_type, fold_id)`` pairs to distribute.
        rank: Zero-based process index of the calling rank.
        num_processes: Total number of ranks.

    Returns:
        The sublist of ``jobs`` this rank should run (possibly empty).
    """
    return jobs[rank::num_processes]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/sl_dl_model/test_fold_parallel.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Run lint**

Run: `uv run ruff check src/sl_dl_model/evaluate.py tests/sl_dl_model/test_fold_parallel.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/evaluate.py tests/sl_dl_model/test_fold_parallel.py
git commit -m "feat: add _shard_jobs round-robin helper for fold-parallel CV"
```

---

### Task 3: Refactor `run_cv` to shard folds across ranks and gather metrics

Rewire `run_cv` so each rank runs only its `_shard_jobs` slice of `(split_type, fold_id)` jobs, then all ranks call `gather_object` (a collective) to send their metric rows to the main process, which dedups, sorts canonically, and writes the existing artifacts. The empty-rows guard and all artifact-writing logic move *after* the gather. Because shards are disjoint + complete and each fold runs through the unchanged `run_fold_with_producer`, the gathered row set equals the serial row set; canonical sorting before write makes the output byte-identical to a 1-process run.

**Files:**
- Modify: `src/sl_dl_model/evaluate.py` (`run_cv` lines 90–200; add import + `_run_local_jobs` helper)
- Test: `tests/sl_dl_model/test_fold_parallel.py` (extend; plus existing `test_evaluate_manifest.py` / `test_evaluate_parity.py` remain the single-process regression guard)

**Interfaces:**
- Consumes: `_shard_jobs` (Task 2); `make_fold_producer`, `run_fold_with_producer` (scoring.py, unchanged); `PartialState` (already imported line 22); `from accelerate.utils import gather_object` (proven available).
- Produces: `_run_local_jobs(config, shared, frame, jobs, producer) -> list[dict[str, object]]` — runs this rank's jobs and returns its metric rows. `run_cv` signature **unchanged**: `run_cv(config, producer) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_fold_parallel.py`:

```python
def test_run_cv_single_process_matches_serial_baseline(tmp_path) -> None:
    """run_cv under 1 process must produce the same rows as a direct serial loop.

    PartialState reports num_processes=1 in pytest, so this pins the refactored
    run_cv against a hand-rolled serial loop over the same jobs — the N-process
    parity gate (Task 4) relies on this 1-process path being correct first.
    """
    import pandas as pd

    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv
    from sl_dl_model.scoring import run_fold_with_producer

    frame = _toy_cv_frame()  # defined below
    csv = tmp_path / "bench.csv"
    frame.to_csv(csv, index=False)
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "out",
        split_types=("CV2", "CV3"),
        folds=(0, 1),
        ranking_k=(10,),
        include_coverage_flag=False,
    )

    summary = run_cv(cfg, ZeroEmbeddingProducer())

    # Serial reference: same jobs, same producer, no sharding.
    ref_rows: list[dict[str, object]] = []
    for split in ("CV2", "CV3"):
        for fold in (0, 1):
            ref_rows.extend(
                run_fold_with_producer(frame, split, fold, cfg, ZeroEmbeddingProducer())
            )
    ref = pd.DataFrame(ref_rows)

    # The written official summary must exist and be non-empty.
    written = pd.read_csv(cfg.output_dir / "official_metrics_summary.csv")
    assert not written.empty
    # Same set of (split_type, model, slice, metric) keys as the serial baseline.
    from sl_benchmark_baseline.evaluate import _summarize

    ref_summary = _summarize(ref)
    key_cols = ["split_type", "model", "slice", "metric"]
    got_keys = written[key_cols].apply(tuple, axis=1).tolist()
    exp_keys = ref_summary[key_cols].apply(tuple, axis=1).tolist()
    assert sorted(got_keys) == sorted(exp_keys)
```

Add this fixture helper at the top of the file (after the imports):

```python
def _toy_cv_frame():
    """Two splits (CV2, CV3) x two folds, deterministic labels over G0..G5."""
    import pandas as pd

    genes = [f"G{i}" for i in range(6)]
    eff = {g: float(i) - 2.5 for i, g in enumerate(genes)}
    rows = []
    pid = 0
    for split in ("CV2", "CV3"):
        for fold in (0, 1):
            for role in ("train", "test"):
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        rows.append(
                            {
                                "pair_id": f"p{pid}",
                                "fold_id": fold,
                                "split_type": split,
                                "split_role": role,
                                "sl_label": (i + j + fold) % 2,
                                "gene_a_symbol": genes[i],
                                "gene_b_symbol": genes[j],
                                "gene_a_k562_gene_effect": eff[genes[i]],
                                "gene_b_k562_gene_effect": eff[genes[j]],
                            }
                        )
                        pid += 1
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/sl_dl_model/test_fold_parallel.py::test_run_cv_single_process_matches_serial_baseline -v`
Expected: FAIL — initially because `run_cv` does not yet write a canonical/deduped summary via the gather path. (If it happens to pass pre-refactor because num_processes=1, that is acceptable; the test still pins behavior. The N-process gate is Task 4.)

- [ ] **Step 3: Write minimal implementation**

In `src/sl_dl_model/evaluate.py`, add the import near line 22:

```python
from accelerate import PartialState
from accelerate.utils import gather_object
```

Add a `_run_local_jobs` helper (place it just above `run_cv`):

```python
def _run_local_jobs(
    config: SLDLConfig,
    shared: "StateDlCaches | None",
    frame: pd.DataFrame,
    jobs: list[tuple[str, int]],
    producer: "EmbeddingProducer | str",
) -> list[dict[str, object]]:
    """Run this rank's assigned ``(split_type, fold_id)`` jobs; return metric rows.

    Each job is run through the unchanged per-fold path
    (:func:`~sl_dl_model.scoring.run_fold_with_producer`), so the rows are
    identical to a serial run of the same jobs. A failure in any job propagates
    (fail-fast); callers must not swallow it.

    Args:
        config: Run configuration.
        shared: Shared caches for the ``state_dl`` path, else ``None``.
        frame: Full benchmark DataFrame.
        jobs: This rank's ``(split_type, fold_id)`` slice from :func:`_shard_jobs`.
        producer: ``"state_dl"`` or a reusable :class:`EmbeddingProducer`.

    Returns:
        Metric row dicts for this rank's jobs (possibly empty if no jobs).
    """
    from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

    rows: list[dict[str, object]] = []
    for split_type, fold_id in jobs:
        fold_producer = (
            make_fold_producer(config, shared, frame, split_type, fold_id)
            if producer == "state_dl"
            else producer
        )
        rows.extend(
            run_fold_with_producer(frame, split_type, fold_id, config, fold_producer)
        )
    return rows
```

Replace the body of `run_cv` from the job loop through the empty-rows guard (current lines 125–152) with:

```python
    # Build the full ordered job list, shard it across ranks, run only ours.
    state = PartialState()
    jobs = [(s, f) for s in split_types for f in config.folds]
    local_jobs = _shard_jobs(jobs, state.process_index, state.num_processes)
    local_rows = _run_local_jobs(config, shared, frame, local_jobs, producer)

    # Collective: every rank contributes its rows; all ranks receive the union.
    # gather_object preserves rank order, so rank r's rows land contiguously.
    gathered: list[list[dict[str, object]]] = gather_object([local_rows])
    all_rows = [row for rank_rows in gathered for row in rank_rows]

    # FIX 2: guard empty metric rows — indicates a config/data mismatch
    if not all_rows:
        logger.error(
            "no metric rows produced — split_types=%s not found in frame "
            "(available: %s); check split_types and training data",
            list(config.split_types or ("CV1", "CV2", "CV3")),
            sorted(available),
        )
        raise RuntimeError(
            "no metric rows produced; check split_types and training data"
        )

    fold_metrics = pd.DataFrame(all_rows)
    # Canonical ordering so 1-process and N-process runs are byte-identical.
    sort_cols = ["split_type", "fold_id", "model", "slice", "metric"]
    fold_metrics = fold_metrics.sort_values(sort_cols).reset_index(drop=True)
    summary = _summarize(fold_metrics)
```

Everything below (the `if not PartialState().is_main_process: return summary` guard at line 155 and all artifact writes) stays as-is. The existing guard now correctly fires after the collective, so non-main ranks return the in-memory summary without writing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/sl_dl_model/test_fold_parallel.py tests/sl_dl_model/test_evaluate_manifest.py tests/sl_dl_model/test_evaluate_parity.py -v`
Expected: PASS. `test_evaluate_manifest.py` (which monkeypatches `PartialState` to simulate a non-main rank) must still pass — confirm the non-main-rank early-return is intact.

- [ ] **Step 5: Run lint**

Run: `uv run ruff check src/sl_dl_model/evaluate.py tests/sl_dl_model/test_fold_parallel.py && uv run ruff format src/sl_dl_model/evaluate.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/sl_dl_model/evaluate.py tests/sl_dl_model/test_fold_parallel.py
git commit -m "feat: shard CV folds across ranks + gather metrics in run_cv"
```

---

### Task 4: Multi-process parity gate (1-process == N-process, byte-identical)

The acceptance test. Launch the real CLI under `accelerate launch` with `--num_processes 1` and `--num_processes 2` against the same toy CSV using the `zero` producer (which drives the identical `run_cv` shard/gather/write path — only the producer body differs, so this validates the orchestration without needing the gitignored STATE/ESM2/bags caches). Assert the two `official_metrics_summary.csv` files are byte-identical. Runs on CPU; no GPU required.

**Files:**
- Test: `tests/sl_dl_model/test_fold_parallel.py` (extend)
- No source changes (this task only adds the gate; it passes once Tasks 1–3 are correct).

**Interfaces:**
- Consumes: the `sl_dl_model run-cv --producer zero` CLI (unchanged); `accelerate launch`.

- [ ] **Step 1: Write the failing test**

Add to `tests/sl_dl_model/test_fold_parallel.py`:

```python
def _write_toy_config(tmp_path, out_subdir: str):
    """Write a toy CSV + SLDLConfig YAML; return (config_path, output_dir)."""
    import yaml

    frame = _toy_cv_frame()
    csv = tmp_path / "bench.csv"
    frame.to_csv(csv, index=False)
    output_dir = tmp_path / out_subdir
    cfg = {
        "input_csv": str(csv),
        "output_dir": str(output_dir),
        "split_types": ["CV2", "CV3"],
        "folds": [0, 1],
        "ranking_k": [10],
        "include_coverage_flag": False,
    }
    cfg_path = tmp_path / f"{out_subdir}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, output_dir


def _run_cli(cfg_path, num_processes: int) -> None:
    """Invoke `accelerate launch --num_processes N -m sl_dl_model run-cv` on CPU."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU so the test runs anywhere
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(num_processes),
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        "--cpu",
        "-m", "sl_dl_model", "run-cv",
        "--config", str(cfg_path),
        "--producer", "zero",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"accelerate launch (np={num_processes}) failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_one_vs_two_process_official_summary_byte_identical(tmp_path) -> None:
    """The parity gate: 1-process and 2-process runs write identical summaries."""
    cfg1, out1 = _write_toy_config(tmp_path, "np1")
    cfg2, out2 = _write_toy_config(tmp_path, "np2")

    _run_cli(cfg1, 1)
    _run_cli(cfg2, 2)

    f1 = (out1 / "official_metrics_summary.csv").read_bytes()
    f2 = (out2 / "official_metrics_summary.csv").read_bytes()
    assert f1 == f2, "1-process and 2-process official_metrics_summary.csv differ"
```

- [ ] **Step 2: Run test to verify it fails (or is correctly gated)**

Run: `uv run python -m pytest tests/sl_dl_model/test_fold_parallel.py::test_one_vs_two_process_official_summary_byte_identical -v`
Expected before Tasks 1–3: FAIL (2-process run either deadlocks on the old DDP `prepare`, or writes a non-canonical/duplicated summary). After Tasks 1–3: PASS.

- [ ] **Step 3: No implementation needed**

This task adds only the gate. If it fails after Tasks 1–3, the bug is in those tasks — fix there, do not weaken this test. If `accelerate` CPU multi-process launch is unavailable in the runner, mark the test with `@pytest.mark.skipif` keyed on a probe of `accelerate.commands.launch` importability — but never skip silently on assertion failure.

- [ ] **Step 4: Run the full sl_dl_model suite**

Run: `uv run python -m pytest tests/sl_dl_model/ -v`
Expected: PASS (all tests, including the parity gate).

- [ ] **Step 5: Run lint**

Run: `uv run ruff check tests/sl_dl_model/test_fold_parallel.py`
Expected: no errors. (`yaml` is already a project dependency via the config loader.)

- [ ] **Step 6: Commit**

```bash
git add tests/sl_dl_model/test_fold_parallel.py
git commit -m "test: add 1-vs-N process parity gate for fold-parallel run_cv"
```

---

### Task 5: Update the Slurm wrapper NOTE and README to describe fold-parallelism

The `scripts/sl_dl_model.sh` NOTE comment currently claims "DDP gradient sync is not yet engaged ... every rank re-runs the full CV loop ... compute is redundant." That is now false. Replace it with an accurate description of fold-parallelism. Launch flags (`--num_processes 4`, etc.) are unchanged. Add a short "Parallelism" subsection to the README.

**Files:**
- Modify: `scripts/sl_dl_model.sh` (the NOTE comment block, currently lines ~37–43)
- Modify: `configs/experiments/08_k562_sl_pair_state_dl/README.md` (add a Parallelism subsection)

**Interfaces:** none (docs/comments only).

- [ ] **Step 1: Replace the NOTE comment in `scripts/sl_dl_model.sh`**

Replace the comment block that begins `# NOTE: StateDlProducer trains per fold on the unwrapped model` (through `...true data-parallel run.`) with:

```bash
# NOTE: Fold-level task parallelism (no gradient all-reduce). run_cv shards the
# (split_type, fold_id) jobs round-robin across the 4 ranks; each rank trains +
# embeds + scores its own folds on one GPU, then gather_object collects every
# rank's metric rows onto the main process, which writes the cvN/ + combined
# artifacts. Each fold is computed exactly once by exactly one rank, so the
# 4-process output is byte-identical to a 1-process run. If a fold crashes, the
# gather is a collective and the whole run aborts non-zero (fail-fast).
```

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n scripts/sl_dl_model.sh`
Expected: no output, exit 0 (syntax OK).

- [ ] **Step 3: Add a Parallelism subsection to the README**

Append to `configs/experiments/08_k562_sl_pair_state_dl/README.md`:

```markdown
## Parallelism

`scripts/sl_dl_model.sh` launches `accelerate launch --num_processes 4`. The 4
ranks split the `(split_type, fold_id)` CV jobs round-robin — each rank trains,
embeds, and scores its own folds on one L40, with no DDP gradient all-reduce
(the per-fold trainable head is tiny and folds are independent). Metric rows are
gathered onto the main process, which writes the per-split `cvN/` directories
and the combined `official_metrics_summary.csv`. Each fold runs on exactly one
rank, so an N-process run is byte-identical to a 1-process run; a crash in any
fold aborts the whole run (fail-fast). For a config with F folds across S
splits, useful rank counts are any divisor up to `S * F` (10 for the default
CV2+CV3 × 5; 15 with CV1 added).
```

- [ ] **Step 4: Commit**

```bash
git add scripts/sl_dl_model.sh configs/experiments/08_k562_sl_pair_state_dl/README.md
git commit -m "docs: describe fold-parallel orchestration in slurm wrapper + README"
```

---

## Self-Review

- **Spec coverage:** Task boundary's four locked decisions are all honored — fold task-parallel/no all-reduce (Tasks 2–3), orchestration-only/training math untouched (Task 1 changes only the wrapper, not the loss/loop), exact parity (Task 4 gate), fail-fast (gather is a collective; no try/except added — documented in Tasks 3 and 5). LOCKED out-of-scope contracts are untouched: `scoring.py`, `model.py`, losses, `sl_benchmark_baseline/*`, CLI.
- **Type consistency:** `_shard_jobs(jobs, rank, num_processes) -> list[tuple[str,int]]`, `_run_local_jobs(config, shared, frame, jobs, producer) -> list[dict]`, `run_cv(config, producer) -> pd.DataFrame` (unchanged), `_train(self, model, optimizer, state, train_symbols) -> None` — consistent across tasks. `PartialState().process_index` / `.num_processes` / `.is_main_process` and `gather_object` confirmed available in the env.
- **Placeholder scan:** every code step shows complete code; no TBD/TODO.
- **Parity mechanism note:** today's loop already runs forward/backward on the *unwrapped* fp32 model, so `Accelerator`→`PartialState` + `.backward()` is numerically identical (not merely "within noise"); combined with disjoint+complete sharding and canonical row sorting, the N-process `official_metrics_summary.csv` is byte-identical to 1-process.

## Risks / Notes for the Executor

- `gather_object` is a **collective** — all ranks must reach it. Tasks must not add an early `return`/`raise` between `_shard_jobs` and `gather_object` on a subset of ranks, or the run hangs. Fail-fast on a fold crash is the intended behavior (one rank raises → its process dies → others error out of the collective with a non-zero run), not a hang to engineer around.
- A rank with an empty shard (more ranks than jobs) still calls `gather_object([[]])` and contributes no rows — handled, see Task 2's `test_shard_jobs_more_ranks_than_jobs`.
- The DDP wrap removal (Task 1) is independently shippable and safe even before Task 3; it is numerically inert under 1 process.

