"""Filesystem coordination primitives for the exp08 fold work-queue."""

from __future__ import annotations

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
    assert fq.claim_path(d, "CV2", 4).parent.name == ".claims"


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


def test_is_done_tracks_result_file(tmp_path: Path):
    d = _results_dir(tmp_path)
    assert fq.is_done(d, "CV1", 1) is False
    fq.atomic_write_json(fq.result_path(d, "CV1", 1), [])
    assert fq.is_done(d, "CV1", 1) is True
