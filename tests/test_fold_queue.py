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


def test_run_token_changes_with_slurm_job_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "111111")
    t1 = fq.run_token()
    monkeypatch.setenv("SLURM_JOB_ID", "222222")
    t2 = fq.run_token()
    assert t1 == "111111"
    assert t2 == "222222"
    assert t1 != t2


def test_run_token_prefers_explicit_env_then_falls_back(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SL_DL_RUN_ID", "abc123")
    assert fq.run_token() == "abc123"
    monkeypatch.delenv("SL_DL_RUN_ID", raising=False)
    # No env at all: a stable, non-empty fallback (process-tree scoped).
    fallback = fq.run_token()
    assert isinstance(fallback, str) and fallback != ""


def test_claim_is_scoped_by_run_token(tmp_path: Path):
    d = _results_dir(tmp_path)
    # Same job claimed under two different run tokens => both win (independent).
    assert fq.try_claim(d, "CV2", 0, run_token="runA") is True
    assert fq.try_claim(d, "CV2", 0, run_token="runB") is True
    # Same token claiming twice still loses the second time.
    assert fq.try_claim(d, "CV2", 0, run_token="runA") is False


def test_stale_claim_from_prior_run_does_not_block_resume(tmp_path: Path):
    """Crash-resume: a claim left by a crashed prior run must not block a new run.

    Reproduces the reviewer's Critical finding. Run A claims CV2/fold0 then
    "crashes" (claim dir exists, no result). Run B (new token) must still be
    able to claim and run that fold.
    """
    d = _results_dir(tmp_path)
    # Run A claims, then crashes before writing a result.
    assert fq.try_claim(d, "CV2", 0, run_token="jobA") is True
    assert not fq.is_done(d, "CV2", 0)
    # Run B, fresh token, must NOT be blocked by run A's orphan claim.
    assert fq.try_claim(d, "CV2", 0, run_token="jobB") is True

