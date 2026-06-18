"""Tests for fold-level task-parallel orchestration in run_cv (no all-reduce)."""

from __future__ import annotations

import pytest

from sl_dl_model.evaluate import _shard_jobs


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

    frame = _toy_cv_frame()  # defined above
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
    _ = summary  # return value not directly asserted; artifacts checked on disk

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


# ---------------------------------------------------------------------------
# Task 4: Multi-process parity gate (1-process == N-process, byte-identical)
# ---------------------------------------------------------------------------

_ACCELERATE_LAUNCH_AVAILABLE: bool
try:
    import accelerate.commands.launch as _accel_launch  # noqa: F401

    _ACCELERATE_LAUNCH_AVAILABLE = True
except ImportError:
    _ACCELERATE_LAUNCH_AVAILABLE = False


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
    # Prevent fork+BLAS crashes on macOS (vecLib/OpenBLAS + multiprocessing).
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--cpu",
        "-m",
        "sl_dl_model",
        "run-cv",
        "--config",
        str(cfg_path),
        "--producer",
        "zero",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"accelerate launch (np={num_processes}) failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not _ACCELERATE_LAUNCH_AVAILABLE,
    reason="accelerate.commands.launch not importable in this environment",
)
def test_cli_launch_is_deterministic(tmp_path) -> None:
    """CLI smoke + determinism gate via ``accelerate launch``.

    NOTE: on a CPU-only host the local launcher collapses
    ``--num_processes 2`` to a single process (``PartialState`` reports
    ``num_processes == 1``), so this does NOT exercise true multi-rank
    execution here — it is an end-to-end CLI smoke test plus a run-to-run
    determinism check (two launches must write byte-identical summaries). The
    genuine shard -> gather -> sort parity logic is pinned, backend-free, by
    :func:`test_run_cv_simulated_multirank_matches_single_process`. On the
    cluster (real GPUs + NCCL) the same launch becomes a true N-rank gate.
    """
    cfg1, out1 = _write_toy_config(tmp_path, "np1")
    cfg2, out2 = _write_toy_config(tmp_path, "np2")

    _run_cli(cfg1, 1)
    _run_cli(cfg2, 2)

    f1 = (out1 / "official_metrics_summary.csv").read_bytes()
    f2 = (out2 / "official_metrics_summary.csv").read_bytes()
    assert f1 == f2, "two CLI launches wrote differing official_metrics_summary.csv"


def test_run_cv_simulated_multirank_matches_single_process(
    tmp_path, monkeypatch
) -> None:
    """True parity gate: a simulated 3-rank gather must equal the 1-process run.

    The local CPU launcher cannot spawn real ranks, so this pins the only logic
    that differs between 1 and N processes — shard -> gather -> flatten ->
    canonical sort -> write — by running the *real* ``run_cv`` with a simulated
    3-rank topology. ``gather_object`` is patched to return the genuine per-rank
    contributions (each computed via the real ``_shard_jobs`` + ``_run_local_jobs``
    over a disjoint job shard), exactly as a real collective would on the cluster.
    The written summary must be byte-identical to a genuine 1-process ``run_cv``.
    """
    import sl_dl_model.evaluate as ev
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.evaluate import (
        ZeroEmbeddingProducer,
        _run_local_jobs,
        _shard_jobs,
        run_cv,
    )

    frame = _toy_cv_frame()
    csv = tmp_path / "bench.csv"
    frame.to_csv(csv, index=False)
    split_types = ("CV2", "CV3")
    folds = (0, 1)

    def _make_cfg(out_dir):
        return SLDLConfig(
            input_csv=csv,
            output_dir=out_dir,
            split_types=split_types,
            folds=folds,
            ranking_k=(10,),
            include_coverage_flag=False,
        )

    # 1-process reference: real run_cv, real PartialState (num_processes == 1).
    run_cv(_make_cfg(tmp_path / "single"), ZeroEmbeddingProducer())
    single = (tmp_path / "single" / "official_metrics_summary.csv").read_bytes()

    # Simulate N=3 ranks. 4 jobs / 3 ranks -> uneven shards (2,1,1), which
    # stresses the flatten/sort reassembly under imbalance.
    n_ranks = 3
    cfg_multi = _make_cfg(tmp_path / "multi")
    jobs = [(s, f) for s in split_types for f in folds]
    contributions = [
        _run_local_jobs(
            cfg_multi,
            None,
            frame,
            _shard_jobs(jobs, r, n_ranks),
            ZeroEmbeddingProducer(),
        )
        for r in range(n_ranks)
    ]

    class _Rank0Of3:
        process_index = 0
        num_processes = n_ranks
        is_main_process = True

    # run_cv computes rank 0's shard itself, then hits the (patched) collective,
    # which returns every rank's contribution in rank order — no double-count
    # because contributions[0] is exactly rank 0's shard.
    monkeypatch.setattr(ev, "PartialState", lambda: _Rank0Of3())
    monkeypatch.setattr(ev, "gather_object", lambda obj: contributions)

    run_cv(cfg_multi, ZeroEmbeddingProducer())
    multi = (tmp_path / "multi" / "official_metrics_summary.csv").read_bytes()

    assert multi == single, (
        "simulated 3-rank gather+sort differs from the 1-process run"
    )
