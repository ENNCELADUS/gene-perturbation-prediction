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
    # Claims are scoped under .claims/<run_token>/, so the leaf's grandparent
    # is the .claims dir.
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


def test_fingerprint_changes_with_input_and_config(tmp_path: Path):
    csv_a = tmp_path / "a.csv"
    csv_a.write_text("gene_a_symbol,gene_b_symbol\nAAA,BBB\n")
    csv_b = tmp_path / "b.csv"
    csv_b.write_text("gene_a_symbol,gene_b_symbol\nCCC,DDD\n")

    cfg1 = SLDLConfig(input_csv=csv_a, output_dir=tmp_path / "o", seed=17)
    cfg1_same = SLDLConfig(input_csv=csv_a, output_dir=tmp_path / "other", seed=17)
    cfg2_seed = SLDLConfig(input_csv=csv_a, output_dir=tmp_path / "o", seed=18)
    cfg3_input = SLDLConfig(input_csv=csv_b, output_dir=tmp_path / "o", seed=17)

    # output_dir does not affect the fingerprint; input + result-affecting config do.
    assert fq.fingerprint(cfg1) == fq.fingerprint(cfg1_same)
    assert fq.fingerprint(cfg1) != fq.fingerprint(cfg2_seed)
    assert fq.fingerprint(cfg1) != fq.fingerprint(cfg3_input)


def test_fingerprint_changes_with_each_result_affecting_scalar(tmp_path: Path):
    csv = tmp_path / "a.csv"
    csv.write_text("gene_a_symbol,gene_b_symbol\nAAA,BBB\n")
    base = SLDLConfig(input_csv=csv, output_dir=tmp_path / "o")
    base_fp = fq.fingerprint(base)
    # Each of these changes the computed metrics and must bust the fingerprint.
    variants = {
        "state_backend": "linear_mock",
        "pert_dim": base.pert_dim + 1,
        "control_template_size": base.control_template_size + 1,
        "cells_per_bag": base.cells_per_bag + 1,
    }
    from dataclasses import replace

    for field, value in variants.items():
        changed = replace(base, **{field: value})
        assert fq.fingerprint(changed) != base_fp, f"{field} did not bust fingerprint"


def test_fingerprint_changes_when_cache_file_rebuilt_at_same_path(tmp_path: Path):
    """A cache regenerated at the SAME path (new size/mtime) must bust reuse."""
    csv = tmp_path / "a.csv"
    csv.write_text("gene_a_symbol,gene_b_symbol\nAAA,BBB\n")
    esm = tmp_path / "esm2.npz"
    esm.write_bytes(b"v1-contents")
    cfg = SLDLConfig(input_csv=csv, output_dir=tmp_path / "o", esm2_npz=esm)
    fp_v1 = fq.fingerprint(cfg)

    # Rebuild the cache at the same path with different contents/size + bump mtime.
    esm.write_bytes(b"v2-different-and-longer-contents")
    import os

    st = esm.stat()
    os.utime(esm, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    fp_v2 = fq.fingerprint(cfg)
    assert fp_v1 != fp_v2


def test_fingerprint_covers_all_path_caches(tmp_path: Path):
    """gwps_h5ad / bags_npz / gwps_overlap_csv / state_checkpoint each matter."""
    from dataclasses import replace

    csv = tmp_path / "a.csv"
    csv.write_text("g\n")
    cache_a = tmp_path / "cache_a.bin"
    cache_a.write_bytes(b"a")
    cache_b = tmp_path / "cache_b.bin"
    cache_b.write_bytes(b"bb")
    base = SLDLConfig(input_csv=csv, output_dir=tmp_path / "o", bags_npz=cache_a)
    base_fp = fq.fingerprint(base)
    for field in ("bags_npz", "gwps_h5ad", "gwps_overlap_csv", "state_checkpoint"):
        changed = replace(base, **{field: cache_b})
        assert fq.fingerprint(changed) != base_fp, f"{field} ignored by fingerprint"


def test_fingerprint_changes_when_state_sidecar_rebuilt(tmp_path: Path):
    """STATE sidecars (var_dims.pkl, pert_onehot_map.pt) affect results.

    They live at ``state_checkpoint.parent.parent`` and can change while the
    checkpoint file itself does not (e.g. a cache rebuild touches the sidecar).
    A rebuilt sidecar must bust stale fold reuse.
    """
    import os

    csv = tmp_path / "a.csv"
    csv.write_text("gene_a_symbol,gene_b_symbol\nAAA,BBB\n")
    ckpt_root = tmp_path / "state"
    ckpt = ckpt_root / "checkpoints" / "final.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    var_dims = ckpt_root / "var_dims.pkl"
    var_dims.write_bytes(b"v1")
    pert_map = ckpt_root / "pert_onehot_map.pt"
    pert_map.write_bytes(b"p1")
    cfg = SLDLConfig(
        input_csv=csv,
        output_dir=tmp_path / "o",
        state_checkpoint=ckpt,
        state_backend="state_checkpoint",
    )
    fp_v1 = fq.fingerprint(cfg)

    # Rebuild var_dims.pkl at the same path with new contents + bumped mtime.
    var_dims.write_bytes(b"v2-longer")
    st = var_dims.stat()
    os.utime(var_dims, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    fp_v2 = fq.fingerprint(cfg)
    assert fp_v1 != fp_v2, "var_dims.pkl rebuild ignored by fingerprint"

    # Likewise for pert_onehot_map.pt.
    pert_map.write_bytes(b"p2-longer")
    st = pert_map.stat()
    os.utime(pert_map, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    fp_v3 = fq.fingerprint(cfg)
    assert fp_v2 != fp_v3, "pert_onehot_map.pt rebuild ignored by fingerprint"


def test_fingerprint_input_csv_content_hashed_even_at_same_size(tmp_path: Path):
    """input_csv is content-hashed, so a same-size in-place edit still busts it."""
    import os

    csv = tmp_path / "a.csv"
    csv.write_text("AAAA")
    cfg = SLDLConfig(input_csv=csv, output_dir=tmp_path / "o")
    fp1 = fq.fingerprint(cfg)
    st = csv.stat()
    csv.write_text("BBBB")  # identical size
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns))  # identical mtime
    assert fq.fingerprint(cfg) != fp1


def test_is_done_requires_matching_fingerprint(tmp_path: Path):
    d = _results_dir(tmp_path)
    fq.write_result(d, "CV1", 0, [{"metric": "x", "value": 1.0}], fingerprint="fp1")
    assert fq.is_done(d, "CV1", 0, fingerprint="fp1") is True
    # A result written under a different fingerprint is treated as not-done.
    assert fq.is_done(d, "CV1", 0, fingerprint="fp2") is False


def test_read_result_rows_only_on_match(tmp_path: Path):
    d = _results_dir(tmp_path)
    rows = [{"metric": "ndcg", "value": 0.5}]
    fq.write_result(d, "CV1", 0, rows, fingerprint="fp1")
    assert fq.read_result_rows(d, "CV1", 0, fingerprint="fp1") == rows
    assert fq.read_result_rows(d, "CV1", 0, fingerprint="fp2") is None
    assert fq.read_result_rows(d, "CV9", 9, fingerprint="fp1") is None


def test_is_failed_requires_matching_fingerprint(tmp_path: Path):
    d = _results_dir(tmp_path)
    fq.write_failed(d, "CV1", 0, {"traceback": "boom"}, fingerprint="fp1")
    assert fq.is_failed(d, "CV1", 0, fingerprint="fp1") is True
    # A failure recorded under an old fingerprint must not block a new config.
    assert fq.is_failed(d, "CV1", 0, fingerprint="fp2") is False
