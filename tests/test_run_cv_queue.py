"""Worker-queue + assembly behavior for exp08 run_cv (no collectives)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sl_dl_model import evaluate
from sl_dl_model import fold_queue as fq
from sl_dl_model.config import SLDLConfig


class _StubProducer:
    """Deterministic producer: records produce() calls, emits zero embeddings."""

    def __init__(self, calls: list[tuple[str, int]]):
        self._calls = calls

    def for_fold(self, split: str, fold: int):
        outer = self

        class _P:
            def produce(self_inner, symbols, train_symbols):  # noqa: N805
                outer._calls.append((split, fold))
                n = len(symbols)
                return np.zeros((n, 1)), np.zeros(n, dtype=int)

        return _P()


def _frame_two_jobs() -> pd.DataFrame:
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
    calls: list[tuple[str, int]] = []
    stub = _StubProducer(calls)

    def fake_run(frame, split, fold, config, producer):
        prod = producer(split, fold) if callable(producer) else producer
        prod.produce(["AAA", "BBB"], {"AAA"})
        return [
            {
                "split_type": split,
                "fold_id": fold,
                "model": "state_dl",
                "slice": "full_universe",
                "metric": "ndcg",
                "value": 1.0,
            }
        ]

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)

    state = evaluate.PartialState()
    jobs = [("CV1", 0), ("CV2", 0)]
    evaluate._run_worker_queue(
        cfg,
        None,
        _frame_two_jobs(),
        jobs,
        lambda s, f: stub.for_fold(s, f),
        state,
    )
    assert fq.is_done(d, "CV1", 0)
    assert fq.is_done(d, "CV2", 0)
    assert set(calls) == {("CV1", 0), ("CV2", 0)}


def test_worker_skips_already_done(tmp_path: Path, monkeypatch):
    cfg = SLDLConfig(output_dir=tmp_path / "run")
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fq.atomic_write_json(fq.result_path(d, "CV1", 0), [{"pre": "existing"}])
    calls: list[tuple[str, int]] = []

    def fake_run(frame, split, fold, config, producer):
        calls.append((split, fold))
        return [
            {
                "split_type": split,
                "fold_id": fold,
                "model": "m",
                "slice": "s",
                "metric": "x",
                "value": 0.0,
            }
        ]

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)
    state = evaluate.PartialState()
    evaluate._run_worker_queue(
        cfg,
        None,
        _frame_two_jobs(),
        [("CV1", 0)],
        lambda s, f: object(),
        state,
    )
    # The done fold was not re-run.
    assert calls == []
    # Existing result preserved.
    assert fq.read_json(fq.result_path(d, "CV1", 0)) == [{"pre": "existing"}]


def test_worker_quarantines_failed_fold(tmp_path: Path, monkeypatch):
    cfg = SLDLConfig(output_dir=tmp_path / "run")
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)

    def fake_run(frame, split, fold, config, producer):
        raise ValueError("diverged")

    monkeypatch.setattr(evaluate, "run_fold_with_producer", fake_run)
    state = evaluate.PartialState()
    evaluate._run_worker_queue(
        cfg,
        None,
        _frame_two_jobs(),
        [("CV1", 0), ("CV2", 0)],
        lambda s, f: object(),
        state,
    )
    # No result files, both quarantined with a traceback.
    assert not fq.is_done(d, "CV1", 0)
    assert fq.failed_path(d, "CV1", 0).exists()
    assert fq.failed_path(d, "CV2", 0).exists()
    marker = fq.read_json(fq.failed_path(d, "CV1", 0))
    assert "diverged" in marker["traceback"]
