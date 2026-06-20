# exp08 Fold-Orchestration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single end-of-run NCCL `gather_object` collective in `run_cv` with a filesystem work-queue + per-fold result files, so the 4-GPU exp08 run completes all CV1/CV2/CV3 × 5 folds without a collective timeout, stays GPU-busy, and resumes after a crash.

**Architecture:** Every rank (still launched by `accelerate launch --num_processes 4` + `PartialState`) walks the same ordered job list, atomically claims each unfinished `(split, fold)` job via `os.mkdir` under a per-run `.claims/<run_token>/` directory, runs the unchanged `run_fold_with_producer`, and writes fingerprinted metric rows to a per-fold `.result.json` (or a fingerprinted `.failed` marker on exception). Rank 0 then polls the filesystem until every job is terminal (or a deadline), collects same-fingerprint result files, and writes the combined artifacts with the existing writer code. No `gather_object`, `broadcast`, `barrier`, `wait_for_everyone`, or any `torch.distributed` collective remains.

**Tech Stack:** Python 3.11, accelerate 1.13.0 (`PartialState` only), pandas, pytest. `uv run` for all invocations.

## Global Constraints

- Python 3.11+, strict type hints, absolute imports, Google-style docstrings (copied verbatim from CLAUDE.md Code Style).
- No `print` in library code; use `logging`.
- No hardcoded paths or thresholds; use config.
- Handle specific exceptions. No bare `except` — except the deliberate quarantine catch in the worker loop, which catches `Exception` by design (decision B) and re-logs + records the traceback.
- Target <50 lines per function, <600 lines per file.
- Prefix all Python/pytest/ruff invocations with `uv run`.
- Tests must run on the macOS box: `OMP_NUM_THREADS=1` is already set in `tests/conftest.py` (per the macos-omp-pytest-segfault constraint). Do not remove it.
- **Guard G1 (no collectives):** the `sl_dl_model` package must call none of `gather_object`, `broadcast`, `barrier`, `wait_for_everyone`, or any `torch.distributed` collective. `PartialState` is used only for `process_index`, `num_processes`, `device`, `is_main_process`.
- Byte-identical output invariant: assembled `fold_metrics.csv` / `summary.csv` must equal a single-process run, via canonical sort key `["split_type", "fold_id", "model", "slice", "metric"]`.

---

## File Structure

- `src/sl_dl_model/config.py` — add 3 config fields (`fold_results_subdir`, `assembly_poll_seconds`, `assembly_timeout_seconds`). Modify.
- `src/sl_dl_model/fold_queue.py` — **new** module: path helpers, result fingerprinting, atomic JSON write, run-scoped atomic claim, done/failed checks. One responsibility: filesystem coordination primitives. Keeps `evaluate.py` small.
- `src/sl_dl_model/evaluate.py` — rewrite `run_cv` body; add `_run_worker_queue` and `_assemble`; remove `_shard_jobs`, `_run_local_jobs`, and the `gather_object` import. Modify.
- `tests/test_fold_queue.py` — **new**: unit tests for the coordination primitives (claim exclusivity, per-run claim scoping, atomic write, fingerprinted done/failed checks, sidecar/cache fingerprint coverage).
- `tests/test_run_cv_queue.py` — **new**: worker-loop + assembly behavior (resume skip, quarantine, output-equivalence, empty guard, deadline).
- `tests/test_no_collectives.py` — **new**: AST static guard that the recursive `sl_dl_model` package imports/calls no collective APIs, including the old `accelerate.utils.gather_object` shape (encodes Guard G1).
- `scripts/smoke_accelerate_no_collective.py` — **new**: cluster smoke test for Guard G2 (manual, documented).
- `scripts/sl_dl_model.sh` — update the orchestration comment block (no behavior change to the launch line). Modify.

---

## Task 1: Config fields for queue + assembly

**Files:**
- Modify: `src/sl_dl_model/config.py:74` (add fields before the `embedding_method` line / `augmented` property)
- Test: `tests/test_config_queue_fields.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `SLDLConfig.fold_results_subdir: str` (default `"_fold_results"`), `SLDLConfig.assembly_poll_seconds: float` (default `5.0`), `SLDLConfig.assembly_timeout_seconds: float` (default `21600.0` = 6 h, comfortably above one fold's wall-clock). All three are plain non-path, non-tuple scalars, so `load_config` handles them via the `else` branch with no change to `_PATH_FIELDS` / `_TUPLE_FIELDS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_queue_fields.py
"""Queue/assembly config fields exist with safe defaults and load from YAML."""

from __future__ import annotations

from pathlib import Path

from sl_dl_model.config import SLDLConfig, load_config


def test_queue_fields_have_defaults():
    cfg = SLDLConfig()
    assert cfg.fold_results_subdir == "_fold_results"
    assert cfg.assembly_poll_seconds == 5.0
    assert cfg.assembly_timeout_seconds == 21600.0


def test_queue_fields_load_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "assembly_poll_seconds: 1.0\n"
        "assembly_timeout_seconds: 120.0\n"
        "fold_results_subdir: _fr\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.assembly_poll_seconds == 1.0
    assert cfg.assembly_timeout_seconds == 120.0
    assert cfg.fold_results_subdir == "_fr"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_config_queue_fields.py -v`
Expected: FAIL — `AttributeError: 'SLDLConfig' object has no attribute 'fold_results_subdir'`.

- [ ] **Step 3: Add the fields**

In `src/sl_dl_model/config.py`, insert immediately before line 74 (`embedding_method: str = ...`):

```python
    # Fold-orchestration (filesystem work-queue; replaces the NCCL gather).
    # Per-fold results + claims live under output_dir / fold_results_subdir.
    fold_results_subdir: str = "_fold_results"
    # Rank-0 assembly polls the filesystem this often (seconds) for terminal
    # markers, and gives up after assembly_timeout_seconds (seconds) — a bound
    # so a hard-crashed worker that left a claim but no result cannot hang it.
    assembly_poll_seconds: float = 5.0
    assembly_timeout_seconds: float = 21600.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_config_queue_fields.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sl_dl_model/config.py tests/test_config_queue_fields.py
git commit -m "feat: add fold-queue + assembly config fields (exp08)"
```

---

## Task 2: Filesystem coordination primitives (`fold_queue.py`)

**Files:**
- Create: `src/sl_dl_model/fold_queue.py`
- Test: `tests/test_fold_queue.py`

**Interfaces:**
- Consumes: `SLDLConfig` (for `output_dir`, `fold_results_subdir`).
- Produces:
  - `fold_results_dir(config: SLDLConfig) -> Path`
  - `result_path(results_dir: Path, split: str, fold: int) -> Path` → `<dir>/<split>_fold<fold>.result.json`
  - `failed_path(results_dir: Path, split: str, fold: int) -> Path` → `<dir>/<split>_fold<fold>.failed`
  - `fingerprint(config: SLDLConfig) -> str` → short run fingerprint over input, result-affecting config, caches, and STATE sidecars (`var_dims.pkl`, `pert_onehot_map.pt`)
  - `run_token() -> str` → `SLURM_JOB_ID` | `SL_DL_RUN_ID` | `local-<ppid>`
  - `claim_path(results_dir: Path, split: str, fold: int, run_token: str | None = None) -> Path` → `<dir>/.claims/<run_token>/<split>_fold<fold>`
  - `atomic_write_json(path: Path, obj: object) -> None` (temp file + `os.replace`)
  - `try_claim(results_dir: Path, split: str, fold: int, run_token: str | None = None) -> bool` (atomic `os.mkdir`; `False` on `FileExistsError`)
  - `is_done(results_dir: Path, split: str, fold: int, fingerprint: str | None = None) -> bool`
  - `is_failed(results_dir: Path, split: str, fold: int, fingerprint: str | None = None) -> bool`
  - `write_result(results_dir: Path, split: str, fold: int, rows: object, fingerprint: str) -> None`
  - `read_result_rows(results_dir: Path, split: str, fold: int, fingerprint: str) -> list | None`
  - `write_failed(results_dir: Path, split: str, fold: int, marker: dict, fingerprint: str) -> None`
  - `read_json(path: Path) -> object`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fold_queue.py
"""Filesystem coordination primitives for the exp08 fold work-queue."""

from __future__ import annotations

import json
from pathlib import Path

from sl_dl_model import fold_queue as fq
from sl_dl_model.config import SLDLConfig


def _results_dir(tmp_path: Path) -> Path:
    cfg = SLDLConfig(output_dir=tmp_path / "run")
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / ".claims").mkdir(parents=True, exist_ok=True)
    return d


def test_path_shapes(tmp_path: Path):
    d = _results_dir(tmp_path)
    assert fq.result_path(d, "CV2", 4).name == "CV2_fold4.result.json"
    assert fq.failed_path(d, "CV2", 4).name == "CV2_fold4.failed"
    claim = fq.claim_path(d, "CV2", 4, run_token="tok")
    assert claim.name == "CV2_fold4"
    assert claim.parent.name == "tok"
    assert claim.parent.parent.name == ".claims"


def test_atomic_write_and_read_json(tmp_path: Path):
    d = _results_dir(tmp_path)
    p = fq.result_path(d, "CV1", 0)
    fq.atomic_write_json(p, [{"metric": "ndcg", "value": 0.5}])
    assert fq.read_json(p) == [{"metric": "ndcg", "value": 0.5}]


def test_try_claim_is_exclusive(tmp_path: Path):
    d = _results_dir(tmp_path)
    assert fq.try_claim(d, "CV3", 2) is True
    # Second claim on the same job loses.
    assert fq.try_claim(d, "CV3", 2) is False
    # A different job is independently claimable.
    assert fq.try_claim(d, "CV3", 3) is True


def test_claim_is_scoped_by_run_token(tmp_path: Path):
    d = _results_dir(tmp_path)
    assert fq.try_claim(d, "CV2", 0, run_token="runA") is True
    assert fq.try_claim(d, "CV2", 0, run_token="runB") is True
    assert fq.try_claim(d, "CV2", 0, run_token="runA") is False


def test_is_done_tracks_result_file(tmp_path: Path):
    d = _results_dir(tmp_path)
    assert fq.is_done(d, "CV1", 1, fingerprint="fp1") is False
    fq.write_result(d, "CV1", 1, [], fingerprint="fp1")
    assert fq.is_done(d, "CV1", 1, fingerprint="fp1") is True
    assert fq.is_done(d, "CV1", 1, fingerprint="fp2") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_fold_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sl_dl_model.fold_queue'`.

- [ ] **Step 3: Implement `fold_queue.py`**

```python
# src/sl_dl_model/fold_queue.py
"""Filesystem coordination primitives for the exp08 fold work-queue.

Replaces the single end-of-run NCCL ``gather_object`` collective. Every rank
walks the same job list and uses these primitives to claim, run, and record
``(split_type, fold_id)`` jobs through the filesystem only — no
``torch.distributed`` collective is involved (Guard G1).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sl_dl_model.config import SLDLConfig


_FINGERPRINT_FIELDS = (
    "split_types",
    "folds",
    "ranking_k",
    "seed",
    "fallback_strategy",
    "include_coverage_flag",
    "esm2_model",
    "state_backend",
    "pooling",
    "pair_hidden",
    "adapter_hidden",
    "pert_dim",
    "control_template_size",
    "cells_per_bag",
    "lambda_sl",
    "lambda_distill",
    "lambda_distill_after_warmup",
    "lambda_bag",
    "lambda_rank",
    "warmup_epochs",
    "max_epochs",
    "batch_pairs",
    "lr",
    "early_stop_patience",
    "max_grad_norm",
    "embedding_method",
)
_FINGERPRINT_PATH_FIELDS = (
    "esm2_npz",
    "bags_npz",
    "gwps_h5ad",
    "gwps_overlap_csv",
    "state_checkpoint",
)
_STATE_SIDECAR_NAMES = ("var_dims.pkl", "pert_onehot_map.pt")


def _path_signature(value: object) -> str:
    if value is None:
        return "<none>"
    path = Path(value)
    try:
        st = path.stat()
    except OSError:
        return f"{path}:<absent>"
    return f"{path}:{st.st_size}:{st.st_mtime_ns}"


def fingerprint(config: SLDLConfig) -> str:
    """Return a short hash of result-affecting config fields + cache signatures."""
    h = hashlib.sha256()
    input_path = Path(config.input_csv)
    if input_path.exists():
        h.update(b"input_csv=")
        h.update(input_path.read_bytes())
    else:
        h.update(f"input_csv=<absent:{input_path}>".encode())
    for name in _FINGERPRINT_FIELDS:
        h.update(f"{name}={getattr(config, name, None)!r}".encode())
    for name in _FINGERPRINT_PATH_FIELDS:
        h.update(f"{name}={_path_signature(getattr(config, name, None))}".encode())
    if getattr(config, "state_backend", None) != "linear_mock":
        ckpt = getattr(config, "state_checkpoint", None)
        sidecar_root = Path(ckpt).parent.parent if ckpt is not None else None
        for sidecar in _STATE_SIDECAR_NAMES:
            value = sidecar_root / sidecar if sidecar_root is not None else None
            h.update(f"{sidecar}={_path_signature(value)}".encode())
    return h.hexdigest()[:16]


def run_token() -> str:
    """Return a per-run token for intra-run claim markers."""
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        return slurm
    explicit = os.environ.get("SL_DL_RUN_ID")
    if explicit:
        return explicit
    return f"local-{os.getppid()}"


def fold_results_dir(config: SLDLConfig) -> Path:
    """Return the per-run fold-results directory under ``output_dir``."""
    return Path(config.output_dir) / config.fold_results_subdir


def result_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the success-result JSON path for one job."""
    return results_dir / f"{split}_fold{fold}.result.json"


def failed_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the quarantine-marker path for one job."""
    return results_dir / f"{split}_fold{fold}.failed"


def claim_path(
    results_dir: Path,
    split: str,
    fold: int,
    run_token: str | None = None,
    *,
    _default_token=run_token,
) -> Path:
    """Return the atomic-claim path for one job, scoped by run token."""
    token = run_token if run_token is not None else _default_token()
    return results_dir / ".claims" / token / f"{split}_fold{fold}"


def atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON atomically (temp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    """Read and parse a JSON file written by :func:`atomic_write_json`."""
    return json.loads(Path(path).read_text())


def try_claim(
    results_dir: Path, split: str, fold: int, run_token: str | None = None
) -> bool:
    """Atomically claim one job. Return ``True`` if this caller won the claim.

    Uses ``os.mkdir`` (POSIX/Lustre-atomic). A returned ``False`` means another
    rank already owns the job in this run. Claims are scoped by run token, so
    a prior Slurm run's orphan claim never blocks a resume.
    """
    claim = claim_path(results_dir, split, fold, run_token)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
        return True
    except FileExistsError:
        return False


def is_done(
    results_dir: Path, split: str, fold: int, fingerprint: str | None = None
) -> bool:
    """Return ``True`` if this job has a same-fingerprint success result."""
    path = result_path(results_dir, split, fold)
    if not path.exists():
        return False
    if fingerprint is None:
        return True
    payload = read_json(path)
    return isinstance(payload, dict) and payload.get("fingerprint") == fingerprint


def is_failed(
    results_dir: Path, split: str, fold: int, fingerprint: str | None = None
) -> bool:
    """Return ``True`` if this job has a same-fingerprint failure marker."""
    path = failed_path(results_dir, split, fold)
    if not path.exists():
        return False
    if fingerprint is None:
        return True
    payload = read_json(path)
    return isinstance(payload, dict) and payload.get("fingerprint") == fingerprint


def write_result(
    results_dir: Path, split: str, fold: int, rows: object, fingerprint: str
) -> None:
    """Atomically write a fold's success result with its fingerprint."""
    atomic_write_json(
        result_path(results_dir, split, fold),
        {"fingerprint": fingerprint, "rows": rows},
    )


def write_failed(
    results_dir: Path, split: str, fold: int, marker: dict, fingerprint: str
) -> None:
    """Atomically write a fold's quarantine marker with its fingerprint."""
    atomic_write_json(
        failed_path(results_dir, split, fold),
        {"fingerprint": fingerprint, **marker},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_fold_queue.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/sl_dl_model/fold_queue.py tests/test_fold_queue.py
uv run ruff format src/sl_dl_model/fold_queue.py tests/test_fold_queue.py
git add src/sl_dl_model/fold_queue.py tests/test_fold_queue.py
git commit -m "feat: filesystem coordination primitives for exp08 fold queue"
```

---

## Task 3: Worker queue loop in `evaluate.py`

**Files:**
- Modify: `src/sl_dl_model/evaluate.py` — add `_run_worker_queue`; keep `_run_local_jobs` for now (removed in Task 5).
- Test: `tests/test_run_cv_queue.py` (create; extended in Tasks 4–6)

**Interfaces:**
- Consumes: `fold_queue` primitives (Task 2); existing `run_fold_with_producer`, `make_fold_producer`.
- Produces: `_run_worker_queue(config: SLDLConfig, shared, frame, jobs: list[tuple[str, int]], producer, state) -> None`. Walks `jobs` in order; for each, skips if `is_done` or `failed_path` exists, else `try_claim`; on a won claim re-checks `is_done` (resume race), runs the fold, writes `.result.json`, or on `Exception` writes `.failed` (with `traceback`, `rank`, `timestamp`) and continues.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_cv_queue.py
"""Worker-queue + assembly behavior for exp08 run_cv (no collectives)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sl_dl_model import evaluate, fold_queue as fq
from sl_dl_model.config import SLDLConfig


class _StubProducer:
    """Deterministic producer: records produce() calls, emits one metric row."""

    def __init__(self, calls: list[tuple[str, int]]):
        self._calls = calls

    def for_fold(self, split: str, fold: int):
        outer = self

        class _P:
            def produce(self_inner, symbols, train_symbols):  # noqa: N805
                outer._calls.append((split, fold))
                import numpy as np

                n = len(symbols)
                return np.zeros((n, 1)), np.zeros(n, dtype=int)

        return _P()


def _frame_two_jobs() -> pd.DataFrame:
    # Minimal frame: 2 splits, fold 0 only, enough columns for fold_split.
    rows = []
    for split in ("CV1", "CV2"):
        for role, label in (("train", 1), ("train", 0), ("test", 1), ("test", 0)):
            rows.append(
                {
                    "split_type": split,
                    "fold_id": 0,
                    "cv_split": role,
                    "gene_a_symbol": "AAA",
                    "gene_b_symbol": "BBB",
                    "sl_label": label,
                    "gene_a_k562_gene_effect": -0.5,
                    "gene_b_k562_gene_effect": -0.3,
                }
            )
    return pd.DataFrame(rows)


def test_worker_runs_unclaimed_jobs(tmp_path: Path, monkeypatch):
    cfg = SLDLConfig(output_dir=tmp_path / "run")
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fp = fq.fingerprint(cfg)
    calls: list[tuple[str, int]] = []
    stub = _StubProducer(calls)

    # Patch run_fold_with_producer to a deterministic 1-row emitter that
    # exercises the claim/result path without the full scoring harness.
    def fake_run(frame, split, fold, config, producer):
        producer.produce(["AAA", "BBB"], {"AAA"})
        return [{"split_type": split, "fold_id": fold, "model": "state_dl",
                 "slice": "full_universe", "metric": "ndcg", "value": 1.0}]

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)

    state = evaluate.PartialState()
    jobs = [("CV1", 0), ("CV2", 0)]
    evaluate._run_worker_queue(
        cfg, None, _frame_two_jobs(), jobs,
        lambda s, f: stub.for_fold(s, f), state,
    )
    assert fq.is_done(d, "CV1", 0, fingerprint=fp)
    assert fq.is_done(d, "CV2", 0, fingerprint=fp)
    assert set(calls) == {("CV1", 0), ("CV2", 0)}


def test_worker_skips_already_done(tmp_path: Path, monkeypatch):
    cfg = SLDLConfig(output_dir=tmp_path / "run")
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fp = fq.fingerprint(cfg)
    fq.write_result(d, "CV1", 0, [{"pre": "existing"}], fingerprint=fp)
    calls: list[tuple[str, int]] = []

    def fake_run(frame, split, fold, config, producer):
        calls.append((split, fold))
        return [{"split_type": split, "fold_id": fold, "model": "m",
                 "slice": "s", "metric": "x", "value": 0.0}]

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)
    state = evaluate.PartialState()
    evaluate._run_worker_queue(
        cfg, None, _frame_two_jobs(), [("CV1", 0)], lambda s, f: object(), state,
    )
    # The done fold was not re-run.
    assert calls == []
    # Existing result preserved.
    assert fq.read_result_rows(d, "CV1", 0, fingerprint=fp) == [{"pre": "existing"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_run_cv_queue.py -v`
Expected: FAIL — `AttributeError: module 'sl_dl_model.evaluate' has no attribute '_run_worker_queue'`.

- [ ] **Step 3: Add `_run_worker_queue` to `evaluate.py`**

Add this import near the top of `evaluate.py` (after the existing imports):

```python
import time
import traceback as _tb
from sl_dl_model import fold_queue as fq
```

Add the function (place it just below `_run_local_jobs`):

```python
def _run_worker_queue(
    config: SLDLConfig,
    shared: "StateDlCaches | None",
    frame: pd.DataFrame,
    jobs: list[tuple[str, int]],
    producer: "EmbeddingProducer | str",
    state: PartialState,
) -> None:
    """Walk ``jobs`` in order, atomically claiming and running each unfinished one.

    Every rank runs the identical loop. Atomic ``mkdir`` claims guarantee each
    job runs on exactly one rank; a lost claim or an already-done job is skipped
    immediately, so a rank never idles while work remains (decision C). A fold
    that raises is quarantined: a ``.failed`` marker with the traceback is
    written and the loop continues (decision B). No collective is used.

    Args:
        config: Run configuration.
        shared: Shared caches for the ``state_dl`` path, else ``None``.
        frame: Full benchmark DataFrame.
        jobs: Ordered ``(split_type, fold_id)`` pairs (the full job list).
        producer: ``"state_dl"`` or a reusable :class:`EmbeddingProducer`.
        state: Active :class:`PartialState` (for ``process_index`` in logs).
    """
    from sl_dl_model.scoring import make_fold_producer, run_fold_with_producer

    results_dir = fq.fold_results_dir(config)
    token = fq.run_token()
    fp = fq.fingerprint(config)
    (results_dir / ".claims" / token).mkdir(parents=True, exist_ok=True)

    for split_type, fold_id in jobs:
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if fq.is_failed(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        if not fq.try_claim(results_dir, split_type, fold_id, run_token=token):
            continue
        # Re-check after winning the claim: a prior run may have produced a
        # result between our is_done check and the claim.
        if fq.is_done(results_dir, split_type, fold_id, fingerprint=fp):
            continue
        try:
            fold_producer = (
                make_fold_producer(config, shared, frame, split_type, fold_id)
                if producer == "state_dl"
                else producer
            )
            rows = run_fold_with_producer(
                frame, split_type, fold_id, config, fold_producer
            )
            fq.write_result(results_dir, split_type, fold_id, rows, fingerprint=fp)
            logger.info(
                "[rank %d] fold %s/%d done (%d rows)",
                state.process_index, split_type, fold_id, len(rows),
            )
        except Exception:  # noqa: BLE001 — deliberate quarantine (decision B)
            fq.write_failed(
                results_dir,
                split_type,
                fold_id,
                {
                    "split_type": split_type,
                    "fold_id": fold_id,
                    "rank": state.process_index,
                    "timestamp": time.time(),
                    "traceback": _tb.format_exc(),
                },
                fingerprint=fp,
            )
            logger.error(
                "[rank %d] fold %s/%d FAILED; quarantined and continuing",
                state.process_index, split_type, fold_id,
            )
```

Note for the test: `_run_worker_queue` accepts `producer` as either the string `"state_dl"`, a reusable `EmbeddingProducer`, or (in tests) a callable `lambda s, f: ...`. To keep the production contract unchanged, the tests monkeypatch `run_fold_with_producer` and pass a callable that the patched function invokes — the production path only ever receives `"state_dl"` or an `EmbeddingProducer`. Make the test's `fake_run` call `producer(split, fold).produce(...)` when `producer` is callable; the real `run_fold_with_producer` calls `producer.produce(...)`. To avoid branching in production code, update the two tests above so `fake_run` does: `prod = producer(split, fold) if callable(producer) else producer` then `prod.produce(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_run_cv_queue.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
uv run ruff format src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git add src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git commit -m "feat: add filesystem worker-queue loop to exp08 run_cv"
```

---

## Task 4: Rank-0 assembly (poll + collect + write)

**Files:**
- Modify: `src/sl_dl_model/evaluate.py` — add `_assemble`.
- Test: `tests/test_run_cv_queue.py` (extend)

**Interfaces:**
- Consumes: `fold_queue` primitives; the existing `_summarize`, `_build_manifest`, `_gwps_coverage_count`, writer code.
- Produces: `_assemble(config: SLDLConfig, jobs: list[tuple[str, int]], split_types: tuple[str, ...], frame: pd.DataFrame, shared) -> pd.DataFrame`. Polls until every job has a `.result.json` or `.failed` (or `assembly_timeout_seconds` elapses), collects rows from all `.result.json`, sorts canonically, summarizes, **writes all artifacts for the succeeded folds**, then — per decision B — **raises `RuntimeError` if any job lacks a result** (quarantined or deadline-missed) so the run exits non-zero. Returns the summary frame only when every fold succeeded.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_run_cv_queue.py

def test_assemble_collects_results_and_writes(tmp_path: Path):
    cfg = SLDLConfig(output_dir=tmp_path / "run", split_types=("CV1", "CV2"),
                     folds=(0,), assembly_poll_seconds=0.01,
                     assembly_timeout_seconds=2.0)
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fp = fq.fingerprint(cfg)
    for split in ("CV1", "CV2"):
        fq.write_result(
            d,
            split,
            0,
            [{"split_type": split, "fold_id": 0, "model": "state_dl",
              "slice": "full_universe", "metric": "ndcg@10", "value": 0.5}],
            fingerprint=fp,
        )
    jobs = [("CV1", 0), ("CV2", 0)]
    summary = evaluate._assemble(cfg, jobs, ("CV1", "CV2"), _frame_two_jobs(), None)
    assert (cfg.output_dir / "fold_metrics.csv").exists()
    assert (cfg.output_dir / "official_metrics_summary.csv").exists()
    fm = pd.read_csv(cfg.output_dir / "fold_metrics.csv")
    # Canonical sort: CV1 before CV2.
    assert list(fm["split_type"]) == ["CV1", "CV2"]
    assert not summary.empty


def test_assemble_deadline_with_partial_results(tmp_path: Path):
    cfg = SLDLConfig(output_dir=tmp_path / "run", split_types=("CV1", "CV2"),
                     folds=(0,), assembly_poll_seconds=0.01,
                     assembly_timeout_seconds=0.05)
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fp = fq.fingerprint(cfg)
    # Only CV1 finished; CV2 never produces a terminal marker.
    fq.write_result(
        d,
        "CV1",
        0,
        [{"split_type": "CV1", "fold_id": 0, "model": "state_dl",
          "slice": "full_universe", "metric": "ndcg@10", "value": 0.5}],
        fingerprint=fp,
    )
    jobs = [("CV1", 0), ("CV2", 0)]
    import pytest
    # Decision B: artifacts for the succeeded fold are still written, but the
    # run raises (→ non-zero exit) because CV2 has no result.
    with pytest.raises(RuntimeError):
        evaluate._assemble(cfg, jobs, ("CV1", "CV2"), _frame_two_jobs(), None)
    fm = pd.read_csv(cfg.output_dir / "fold_metrics.csv")
    assert set(fm["split_type"]) == {"CV1"}  # partial results persisted


def test_assemble_failed_fold_raises_after_writing(tmp_path: Path):
    cfg = SLDLConfig(output_dir=tmp_path / "run", split_types=("CV1", "CV2"),
                     folds=(0,), assembly_poll_seconds=0.01,
                     assembly_timeout_seconds=2.0)
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fp = fq.fingerprint(cfg)
    fq.write_result(
        d,
        "CV1",
        0,
        [{"split_type": "CV1", "fold_id": 0, "model": "state_dl",
          "slice": "full_universe", "metric": "ndcg@10", "value": 0.5}],
        fingerprint=fp,
    )
    fq.write_failed(d, "CV2", 0, {"traceback": "boom"}, fingerprint=fp)
    import pytest
    with pytest.raises(RuntimeError):
        evaluate._assemble(cfg, [("CV1", 0), ("CV2", 0)], ("CV1", "CV2"),
                           _frame_two_jobs(), None)
    # Succeeded fold's artifacts were written before raising.
    assert (cfg.output_dir / "official_metrics_summary.csv").exists()


def test_assemble_empty_raises(tmp_path: Path):
    cfg = SLDLConfig(output_dir=tmp_path / "run", split_types=("CV1",),
                     folds=(0,), assembly_poll_seconds=0.01,
                     assembly_timeout_seconds=0.05)
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fq.write_failed(
        d, "CV1", 0, {"traceback": "boom"}, fingerprint=fq.fingerprint(cfg)
    )
    import pytest
    with pytest.raises(RuntimeError):
        evaluate._assemble(cfg, [("CV1", 0)], ("CV1",), _frame_two_jobs(), None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_run_cv_queue.py -k assemble -v`
Expected: FAIL — `AttributeError: ... has no attribute '_assemble'`.

- [ ] **Step 3: Add `_assemble` to `evaluate.py`**

```python
def _assemble(
    config: SLDLConfig,
    jobs: list[tuple[str, int]],
    split_types: tuple[str, ...],
    frame: pd.DataFrame,
    shared: "StateDlCaches | None",
) -> pd.DataFrame:
    """Poll for terminal markers, collect results, write combined artifacts.

    Rank-0 only. Replaces the ``gather_object`` barrier with a bounded
    filesystem poll (no collective). Stops when every job is terminal (result
    or failed) or ``assembly_timeout_seconds`` elapses, then assembles whatever
    results exist. Logs and counts folds with no result.

    Args:
        config: Run configuration.
        jobs: Full ordered ``(split_type, fold_id)`` list.
        split_types: Split types covered by this run.
        frame: Full benchmark DataFrame (for candidate-gene count).
        shared: Shared caches (for gwps coverage count) or ``None``.

    Returns:
        Summary :class:`pandas.DataFrame` (only when every fold succeeded).

    Raises:
        RuntimeError: If no result rows were produced by any fold, or if any
            job lacks a result (quarantined or deadline-missed) — after the
            succeeded folds' artifacts have been written (decision B).
    """
    results_dir = fq.fold_results_dir(config)
    fp = fq.fingerprint(config)
    deadline = time.monotonic() + float(config.assembly_timeout_seconds)
    while time.monotonic() < deadline:
        terminal = all(
            fq.is_done(results_dir, s, f, fingerprint=fp)
            or fq.is_failed(results_dir, s, f, fingerprint=fp)
            for s, f in jobs
        )
        if terminal:
            break
        time.sleep(float(config.assembly_poll_seconds))

    all_rows: list[dict[str, object]] = []
    produced: set[tuple[str, int]] = set()
    for split_type, fold_id in jobs:
        rows = fq.read_result_rows(results_dir, split_type, fold_id, fingerprint=fp)
        if rows is not None:
            all_rows.extend(rows)
            produced.add((split_type, fold_id))

    if not all_rows:
        logger.error(
            "no metric rows produced — split_types=%s; check splits and data",
            list(split_types),
        )
        raise RuntimeError(
            "no metric rows produced; check split_types and training data"
        )

    fold_metrics = pd.DataFrame(all_rows)
    sort_cols = ["split_type", "fold_id", "model", "slice", "metric"]
    fold_metrics = fold_metrics.sort_values(sort_cols).reset_index(drop=True)
    summary = _summarize(fold_metrics)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_gene_count = len(
        set(frame["gene_a_symbol"]) | set(frame["gene_b_symbol"])
    )
    gwps_coverage_gene_count = _gwps_coverage_count(shared)
    manifest = _build_manifest(
        config,
        split_types=split_types,
        candidate_gene_count=candidate_gene_count,
        gwps_coverage_gene_count=gwps_coverage_gene_count,
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for split_type in split_types:
        split_rows = fold_metrics[fold_metrics["split_type"] == split_type]
        if split_rows.empty:
            continue
        split_dir = output_dir / split_type
        split_dir.mkdir(parents=True, exist_ok=True)
        split_rows.to_csv(split_dir / "fold_metrics.csv", index=False)
        _summarize(split_rows).to_csv(split_dir / "summary.csv", index=False)
        split_manifest = _build_manifest(
            config,
            split_types=(split_type,),
            candidate_gene_count=candidate_gene_count,
            gwps_coverage_gene_count=gwps_coverage_gene_count,
        )
        (split_dir / "manifest.json").write_text(
            json.dumps(split_manifest, indent=2)
        )

    summary.to_csv(output_dir / "official_metrics_summary.csv", index=False)

    # Decision B: artifacts for succeeded folds are now safely on disk; fail the
    # run if any fold did not produce a result. Same-fingerprint .failed markers
    # are quarantined and will NOT re-run on resubmit unless the marker is
    # removed or the fingerprint changes; merely missing folds resume.
    missing = [job for job in jobs if job not in produced]
    if missing:
        quarantined = [
            (s, f) for s, f in missing if fq.is_failed(results_dir, s, f, fingerprint=fp)
        ]
        incomplete = [job for job in missing if job not in quarantined]
        lines = [f"{len(missing)} fold(s) did not produce results."]
        if quarantined:
            lines.append(
                "  Quarantined (failed; will NOT re-run on resubmit unless you act): "
                f"{quarantined}"
            )
            lines.append(
                "  To retry a quarantined fold: delete its .failed marker above, "
                "or change the config/input (which bumps the fingerprint)."
            )
        if incomplete:
            lines.append(
                f"  Incomplete (crashed/deadline; will resume automatically on "
                f"resubmit): {incomplete}"
            )
        lines.append("  Succeeded folds' artifacts were written.")
        message = "\n".join(lines)
        logger.error("assembly: %s", message)
        raise RuntimeError(message)
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_run_cv_queue.py -k assemble -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
uv run ruff format src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git add src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git commit -m "feat: rank-0 filesystem assembly for exp08 run_cv (no gather)"
```

---

## Task 5: Rewire `run_cv`; remove `gather_object`, `_shard_jobs`, `_run_local_jobs`

**Files:**
- Modify: `src/sl_dl_model/evaluate.py` — rewrite `run_cv` body (lines ~159–238); delete `_shard_jobs` (lines ~377–396) and `_run_local_jobs` (lines ~91–127); remove `from accelerate.utils import gather_object` (line 23).
- Test: `tests/test_run_cv_queue.py` (extend with an end-to-end `run_cv` test using a stub producer)

**Interfaces:**
- Consumes: `_run_worker_queue` (Task 3), `_assemble` (Task 4).
- Produces: unchanged `run_cv(config, producer) -> pd.DataFrame` signature; now collective-free. Non-main ranks return the summary from `_assemble`'s output only on rank 0; non-main ranks return an empty/placeholder summary after their worker loop.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# append to tests/test_run_cv_queue.py

def test_run_cv_end_to_end_single_process(tmp_path: Path, monkeypatch):
    cfg = SLDLConfig(output_dir=tmp_path / "run", split_types=("CV1", "CV2"),
                     folds=(0,), assembly_poll_seconds=0.01,
                     assembly_timeout_seconds=2.0)

    def fake_run(frame, split, fold, config, producer):
        return [{"split_type": split, "fold_id": fold, "model": "state_dl",
                 "slice": "full_universe", "metric": "ndcg@10", "value": 0.7}]

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)
    monkeypatch.setattr(evaluate, "load_benchmark", lambda _p: _frame_two_jobs())

    summary = evaluate.run_cv(cfg, evaluate.ZeroEmbeddingProducer())
    assert (cfg.output_dir / "official_metrics_summary.csv").exists()
    fm = pd.read_csv(cfg.output_dir / "fold_metrics.csv")
    assert set(fm["split_type"]) == {"CV1", "CV2"}
    assert not summary.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_run_cv_queue.py::test_run_cv_end_to_end_single_process -v`
Expected: FAIL — current `run_cv` calls `gather_object`, which under a single non-launched process either errors or does not route through the new files; the assertion on `official_metrics_summary.csv` path / split set will not match the new layout. (If it happens to pass against the old path, the subsequent removal in Step 3 is still required.)

- [ ] **Step 3: Rewrite `run_cv` and delete dead code**

Replace the body of `run_cv` from the `state = PartialState()` line through the final `return summary` (current lines ~164–238) with:

```python
    # Build the full ordered job list; every rank walks the same list and
    # claims jobs atomically through the filesystem (no collective).
    state = PartialState()
    jobs = [(s, f) for s in split_types for f in config.folds]
    _run_worker_queue(config, shared, frame, jobs, producer, state)

    # Rank-0 assembles from the per-fold result files. Non-main ranks are done.
    if not state.is_main_process:
        return pd.DataFrame()
    return _assemble(config, jobs, split_types, frame, shared)
```

Delete:
- `from accelerate.utils import gather_object` (line 23).
- the entire `_run_local_jobs` function (lines ~91–127).
- the entire `_shard_jobs` function (lines ~377–396).

Keep `from accelerate import PartialState` (line 22).

- [ ] **Step 4: Run the full sl_dl_model test suite**

Run: `uv run python -m pytest tests/test_run_cv_queue.py tests/test_fold_queue.py tests/test_config_queue_fields.py -v`
Expected: PASS (all). Then run the existing exp08 suite to confirm no regression: `uv run python -m pytest tests/ -k "sl_dl or evaluate or scoring" -v` — Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
uv run ruff format src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git add src/sl_dl_model/evaluate.py tests/test_run_cv_queue.py
git commit -m "refactor: rewire exp08 run_cv to filesystem queue; drop gather_object"
```

---

## Task 6: Guard G1 — static no-collectives test

**Files:**
- Create: `tests/test_no_collectives.py`

**Interfaces:**
- Consumes: AST-parsed source of the recursive `src/sl_dl_model/**/*.py` package.
- Produces: a regression guard that fails if any forbidden collective import or call reappears, including the exact old `from accelerate.utils import gather_object; gather_object(...)` shape.

- [ ] **Step 1: Write the failing-then-passing test**

```python
# tests/test_no_collectives.py
"""Guard G1: the sl_dl_model package uses no distributed collective."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UNAMBIGUOUS_COLLECTIVES = frozenset(
    {
        "all_gather",
        "all_gather_object",
        "gather_object",
        "scatter_object_list",
        "all_gather_into_tensor",
        "all_reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "all_to_all",
        "all_to_all_single",
    }
)
AMBIGUOUS_COLLECTIVES = frozenset(
    {"gather", "scatter", "reduce", "broadcast", "barrier"}
)
ACCELERATE_COLLECTIVES = frozenset({"gather_object", "wait_for_everyone"})
_DIST_RECEIVERS = frozenset({"dist", "distributed", "torch_dist"})


class _CollectiveVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.offenders: list[str] = []
        self.direct_collective_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch.distributed" or alias.name.startswith(
                "torch.distributed."
            ):
                self.offenders.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module == "torch.distributed" or module.startswith("torch.distributed."):
            self.offenders.append(f"from {module} import ...")
            for alias in node.names:
                self.direct_collective_names.add(alias.asname or alias.name)
        if module == "accelerate.utils":
            for alias in node.names:
                if alias.name in ACCELERATE_COLLECTIVES:
                    self.offenders.append(f"from {module} import {alias.name}")
                    self.direct_collective_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.direct_collective_names:
            self.offenders.append(f"call {func.id}(")
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            receiver = (
                func.value.attr
                if isinstance(func.value, ast.Attribute)
                else getattr(func.value, "id", None)
            )
            if attr in UNAMBIGUOUS_COLLECTIVES or attr in ACCELERATE_COLLECTIVES:
                self.offenders.append(f"call .{attr}(")
            elif attr in AMBIGUOUS_COLLECTIVES and receiver in _DIST_RECEIVERS:
                self.offenders.append(f"call {receiver}.{attr}(")
        self.generic_visit(node)


def _scan(source: str) -> list[str]:
    visitor = _CollectiveVisitor()
    visitor.visit(ast.parse(source))
    return visitor.offenders


def test_sl_dl_model_uses_no_distributed_collective():
    src_dir = Path(__file__).resolve().parents[1] / "src" / "sl_dl_model"
    offenders: list[str] = []
    for py in sorted(src_dir.rglob("*.py")):
        for hit in _scan(py.read_text()):
            offenders.append(f"{py.name}: {hit}")
    assert offenders == [], f"forbidden collective(s) found: {offenders}"


@pytest.mark.parametrize(
    "snippet",
    [
        "from accelerate.utils import gather_object\ngather_object([rows])\n",
        "import torch.distributed as dist\ndist.all_reduce(x)\n",
        "torch.distributed.reduce_scatter(out, ins)\n",
        "accelerator.wait_for_everyone()\n",
    ],
)
def test_guard_rejects_distributed_constructs(snippet: str):
    assert _scan(snippet), f"guard failed to flag: {snippet!r}"
```

- [ ] **Step 2: Run test**

Run: `uv run python -m pytest tests/test_no_collectives.py -v`
Expected: PASS (after Task 5 removed `gather_object`). If it FAILS, Task 5's deletions are incomplete — remove the named import/call. The parametrized guard must fail on `from accelerate.utils import gather_object; gather_object(...)`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_no_collectives.py
git commit -m "test: guard exp08 sl_dl_model against torch.distributed collectives"
```

---

## Task 7: Cluster smoke test for Guard G2

**Files:**
- Create: `scripts/smoke_accelerate_no_collective.py`

**Interfaces:**
- Consumes: `accelerate` `PartialState`.
- Produces: a manual cluster check — each rank prints `process_index`/`device`, writes a per-rank file, calls **no** collective; the script exits 0 with N files and no NCCL timeout.

- [ ] **Step 1: Write the script**

```python
# scripts/smoke_accelerate_no_collective.py
"""Guard G2 smoke test: PartialState under accelerate launch with NO collective.

Run on the cluster:

    accelerate launch --num_processes 4 \\
        scripts/smoke_accelerate_no_collective.py --out-dir /tmp/g2_smoke

Confirms that keeping accelerate launch + PartialState (launch-model A) never
triggers a lazy NCCL setup / 600s store timeout when no collective is called.
Expected: exit 0, one file per rank, no timeout traceback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import PartialState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = PartialState()
    # No gather/broadcast/barrier — only local attributes.
    marker = args.out_dir / f"rank_{state.process_index}.ok"
    marker.write_text(
        f"rank={state.process_index} "
        f"num_processes={state.num_processes} "
        f"device={state.device}\n"
    )
    print(  # noqa: T201 — smoke script, stdout is the signal
        f"[rank {state.process_index}/{state.num_processes}] "
        f"device={state.device} wrote {marker}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Local sanity (single process, no GPU needed)**

Run: `uv run python scripts/smoke_accelerate_no_collective.py --out-dir /tmp/g2_smoke_local`
Expected: prints one rank line, creates `/tmp/g2_smoke_local/rank_0.ok`, exits 0.

- [ ] **Step 3: Commit (cluster run is a manual operational step)**

```bash
git add scripts/smoke_accelerate_no_collective.py
git commit -m "test: cluster smoke for accelerate-no-collective (exp08 Guard G2)"
```

Operational note (not automated): before the 4-day run, on the cluster run
`accelerate launch --num_processes 4 scripts/smoke_accelerate_no_collective.py --out-dir /tmp/g2_smoke`
and confirm exit 0, 4 `.ok` files, and **no** NCCL/timeout traceback.

---

## Task 8: Update `sl_dl_model.sh` orchestration comment

**Files:**
- Modify: `scripts/sl_dl_model.sh:37-43` (comment block only; launch line unchanged)

**Interfaces:**
- Consumes: nothing. Produces: nothing (doc-only).

- [ ] **Step 1: Replace the comment block**

Replace lines 37–43 (the `# NOTE: Fold-level task parallelism ...` block) with:

```bash
# NOTE: Fold-level task parallelism via a FILESYSTEM WORK-QUEUE (no collective,
# no gradient all-reduce). run_cv builds the full (split_type, fold_id) job list;
# every rank walks it and atomically claims (os.mkdir) each unfinished job under
# <output_dir>/_fold_results/.claims/<run_token>/, trains + embeds + scores it on
# its GPU, and writes a same-fingerprint <split>_fold<k>.result.json. A fold that
# raises is quarantined with a same-fingerprint .failed marker and the run
# continues. Rank 0 then polls the filesystem until every job is terminal (or
# assembly_timeout_seconds) and writes the cvN/ + combined artifacts; output is
# byte-identical to a 1-process run. Resume: re-submitting skips same-fingerprint
# .result.json files; missing folds resume automatically; quarantined folds need
# their .failed marker removed or a fingerprint-changing input/config update.
# There is NO torch.distributed collective, so uneven fold runtimes can no longer
# cause a gather/NCCL timeout.
```

- [ ] **Step 2: Verify the launch line is unchanged**

Run: `uv run --offline python -c "print('shell comment-only change')"` and visually confirm `scripts/sl_dl_model.sh` lines 45–52 (`srun uv run ... accelerate launch --num_processes 4 ... run-cv`) are untouched.

- [ ] **Step 3: Commit**

```bash
git add scripts/sl_dl_model.sh
git commit -m "docs: update exp08 slurm orchestration comment for work-queue"
```

---

## Task 9: Full suite + lint gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run python -m pytest`
Expected: PASS — all `sl_dl_model` tests (the prior 96 + new queue tests), `aivc_model` untouched. `OMP_NUM_THREADS=1` conftest keeps macOS from segfaulting.

- [ ] **Step 2: Ruff gate**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 3: Commit any formatting fixups (if needed)**

```bash
git add -A
git commit -m "style: ruff format exp08 fold-queue changes"
```

---

## Self-Review

**Spec coverage:**
- §1 root cause / non-causes / stale caveat → diagnosis is fixed by Tasks 3–5 (queue replaces gather); stale-runtime is an operational note carried in Task 7/§7 of spec (no code).
- §2 goals: complete 15 folds (Tasks 3–5), use 4 GPUs well (queue claim, Task 3), robust+resume (same-fingerprint `.result.json` skip, run-token claims, Task 3 + Task 4 partial-deadline), surgical (kept `accelerate launch`/`PartialState`, Task 5).
- §3 decisions: scope C (Tasks 2–5), policy B quarantine (Task 3 + Task 4 quarantined/incomplete reporting + non-zero handled by §7 op note / assembly logging), claim model C per-run `.claims/<run_token>/` (Task 2), launch A (Task 5 keeps `PartialState`).
- §3 Guard G1 → Task 6, including direct `accelerate.utils.gather_object`; Guard G2 → Task 7.
- §4 architecture → Tasks 2/3/4; §4.3 output equivalence → Task 4 canonical sort + Task 5 e2e test.
- §5 interfaces → match Tasks 2–4 signatures. §6 tests → Tasks 2,3,4,6,7. §7 operational → Task 7 note + Task 8 comment.

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. The one "if it happens to pass" note in Task 5 Step 2 is an honest acknowledgment that the old path may not error in-process; the deletion in Step 3 is unconditional, so coverage holds.

**Type consistency:** `fold_queue` function names (`fold_results_dir`, `fingerprint`, `run_token`, `result_path`, `failed_path`, `claim_path`, `atomic_write_json`, `read_json`, `try_claim`, `is_done`, `is_failed`, `write_result`, `read_result_rows`, `write_failed`) are used identically in Tasks 2–4. `_run_worker_queue(config, shared, frame, jobs, producer, state)` and `_assemble(config, jobs, split_types, frame, shared)` signatures match between definition (Tasks 3/4) and call site (Task 5). Config fields `fold_results_subdir` / `assembly_poll_seconds` / `assembly_timeout_seconds` consistent across Tasks 1/2/4.

**Exit-code (decision B):** `_assemble` writes all succeeded folds' artifacts first, then raises `RuntimeError` if any job lacks a same-fingerprint result. It distinguishes quarantined folds (same-fingerprint `.failed`, not retried unless marker is removed or fingerprint changes) from incomplete folds (missing/deadline, resumes automatically). `run_cv` propagates the exception and `__main__.main` lets it surface, so the process exits non-zero — no CLI change needed. Good results persist for resume; the failure is loud. Verified by `test_assemble_failed_fold_raises_after_writing` and `test_assemble_deadline_with_partial_results` (Task 4).
