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


def test_assemble_collects_results_and_writes(tmp_path: Path):
    cfg = SLDLConfig(
        output_dir=tmp_path / "run",
        split_types=("CV1", "CV2"),
        folds=(0,),
        assembly_poll_seconds=0.01,
        assembly_timeout_seconds=2.0,
    )
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for split in ("CV1", "CV2"):
        fq.atomic_write_json(
            fq.result_path(d, split, 0),
            [
                {
                    "split_type": split,
                    "fold_id": 0,
                    "model": "state_dl",
                    "slice": "full_universe",
                    "metric": "ndcg@10",
                    "value": 0.5,
                }
            ],
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
    cfg = SLDLConfig(
        output_dir=tmp_path / "run",
        split_types=("CV1", "CV2"),
        folds=(0,),
        assembly_poll_seconds=0.01,
        assembly_timeout_seconds=0.05,
    )
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    # Only CV1 finished; CV2 never produces a terminal marker.
    fq.atomic_write_json(
        fq.result_path(d, "CV1", 0),
        [
            {
                "split_type": "CV1",
                "fold_id": 0,
                "model": "state_dl",
                "slice": "full_universe",
                "metric": "ndcg@10",
                "value": 0.5,
            }
        ],
    )
    jobs = [("CV1", 0), ("CV2", 0)]
    import pytest

    # Decision B: artifacts for the succeeded fold are still written, but the
    # run raises (-> non-zero exit) because CV2 has no result.
    with pytest.raises(RuntimeError):
        evaluate._assemble(cfg, jobs, ("CV1", "CV2"), _frame_two_jobs(), None)
    fm = pd.read_csv(cfg.output_dir / "fold_metrics.csv")
    assert set(fm["split_type"]) == {"CV1"}  # partial results persisted


def test_assemble_failed_fold_raises_after_writing(tmp_path: Path):
    cfg = SLDLConfig(
        output_dir=tmp_path / "run",
        split_types=("CV1", "CV2"),
        folds=(0,),
        assembly_poll_seconds=0.01,
        assembly_timeout_seconds=2.0,
    )
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fq.atomic_write_json(
        fq.result_path(d, "CV1", 0),
        [
            {
                "split_type": "CV1",
                "fold_id": 0,
                "model": "state_dl",
                "slice": "full_universe",
                "metric": "ndcg@10",
                "value": 0.5,
            }
        ],
    )
    fq.atomic_write_json(fq.failed_path(d, "CV2", 0), {"traceback": "boom"})
    import pytest

    with pytest.raises(RuntimeError):
        evaluate._assemble(
            cfg, [("CV1", 0), ("CV2", 0)], ("CV1", "CV2"), _frame_two_jobs(), None
        )
    # Succeeded fold's artifacts were written before raising.
    assert (cfg.output_dir / "official_metrics_summary.csv").exists()


def test_assemble_empty_raises(tmp_path: Path):
    cfg = SLDLConfig(
        output_dir=tmp_path / "run",
        split_types=("CV1",),
        folds=(0,),
        assembly_poll_seconds=0.01,
        assembly_timeout_seconds=0.05,
    )
    d = fq.fold_results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fq.atomic_write_json(fq.failed_path(d, "CV1", 0), {"traceback": "boom"})
    import pytest

    with pytest.raises(RuntimeError):
        evaluate._assemble(cfg, [("CV1", 0)], ("CV1",), _frame_two_jobs(), None)
