"""Tests for fold-level task-parallel orchestration in run_cv (filesystem queue).

The old static-shard + ``gather_object`` design was replaced by a filesystem
work-queue (see ``tests/test_run_cv_queue.py`` for the unit-level coverage). The
behavioral guarantees that still matter here are end-to-end:

* a 1-process ``run_cv`` matches a hand-rolled serial loop, and
* the assembled output is byte-identical whether folds were produced in this run
  or pre-existing from other ranks/prior runs (the resume + multi-rank parity
  invariant that the deleted ``gather_object`` simulation used to pin).
"""

from __future__ import annotations

import pytest


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


def test_run_cv_single_process_matches_serial_baseline(tmp_path) -> None:
    """run_cv under 1 process must produce the same rows as a direct serial loop.

    PartialState reports num_processes=1 in pytest, so this pins run_cv against a
    hand-rolled serial loop over the same jobs.
    """
    import pandas as pd

    from sl_benchmark_baseline.evaluate import _summarize
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv
    from sl_dl_model.scoring import run_fold_with_producer

    frame = _toy_cv_frame()
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

    run_cv(cfg, ZeroEmbeddingProducer())

    # Serial reference: same jobs, same producer, no queue.
    ref_rows: list[dict[str, object]] = []
    for split in ("CV2", "CV3"):
        for fold in (0, 1):
            ref_rows.extend(
                run_fold_with_producer(frame, split, fold, cfg, ZeroEmbeddingProducer())
            )
    ref = pd.DataFrame(ref_rows)

    written = pd.read_csv(cfg.output_dir / "official_metrics_summary.csv")
    assert not written.empty
    ref_summary = _summarize(ref)
    key_cols = ["split_type", "model", "slice", "metric"]
    got_keys = written[key_cols].apply(tuple, axis=1).tolist()
    exp_keys = ref_summary[key_cols].apply(tuple, axis=1).tolist()
    assert sorted(got_keys) == sorted(exp_keys)


def test_run_cv_resume_assembly_matches_single_process(tmp_path, monkeypatch) -> None:
    """Parity gate: pre-existing per-fold results assemble identically.

    Replaces the deleted ``gather_object`` simulation. A genuine 1-process run
    produces the reference. A second run is given two of its four folds as
    pre-existing ``.result.json`` files (as though other ranks / a prior run had
    written them); ``run_cv`` must skip those, produce the remaining two, and
    assemble a byte-identical ``official_metrics_summary.csv``. This pins both
    the resume path and the cross-rank reassembly invariant.
    """
    import sl_dl_model.evaluate as ev
    from sl_dl_model import fold_queue as fq
    from sl_dl_model.config import SLDLConfig
    from sl_dl_model.evaluate import ZeroEmbeddingProducer, run_cv

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
            assembly_poll_seconds=0.01,
            assembly_timeout_seconds=5.0,
        )

    # 1-process reference: real run_cv.
    run_cv(_make_cfg(tmp_path / "single"), ZeroEmbeddingProducer())
    single = (tmp_path / "single" / "official_metrics_summary.csv").read_bytes()

    # Resume run: pre-seed two folds' result files (CV2/0 and CV3/1), copied
    # verbatim from the reference's per-fold result files, then run_cv computes
    # only the remaining two and assembles.
    cfg_resume = _make_cfg(tmp_path / "resume")
    ref_dir = fq.fold_results_dir(_make_cfg(tmp_path / "single"))
    resume_dir = fq.fold_results_dir(cfg_resume)
    resume_dir.mkdir(parents=True, exist_ok=True)
    for split, fold in (("CV2", 0), ("CV3", 1)):
        fq.atomic_write_json(
            fq.result_path(resume_dir, split, fold),
            fq.read_json(fq.result_path(ref_dir, split, fold)),
        )

    # Count which folds actually get computed this run.
    computed: list[tuple[str, int]] = []
    real_run_fold = ev.run_fold_with_producer

    def _tracking_run_fold(frame_, split_, fold_, config_, producer_):
        computed.append((split_, fold_))
        return real_run_fold(frame_, split_, fold_, config_, producer_)

    monkeypatch.setattr(ev, "run_fold_with_producer", _tracking_run_fold)

    run_cv(cfg_resume, ZeroEmbeddingProducer())
    resume = (tmp_path / "resume" / "official_metrics_summary.csv").read_bytes()

    # The two pre-seeded folds were skipped; only the other two were computed.
    assert sorted(computed) == [("CV2", 1), ("CV3", 0)]
    assert resume == single, (
        "resume assembly differs from the 1-process run (parity broken)"
    )


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

    On a CPU-only host the local launcher collapses ``--num_processes 2`` to a
    single process, so this is an end-to-end CLI smoke test plus a run-to-run
    determinism check (two launches must write byte-identical summaries). On the
    cluster the same launch becomes a true N-rank run over the filesystem queue.
    """
    cfg1, out1 = _write_toy_config(tmp_path, "np1")
    cfg2, out2 = _write_toy_config(tmp_path, "np2")

    _run_cli(cfg1, 1)
    _run_cli(cfg2, 2)

    f1 = (out1 / "official_metrics_summary.csv").read_bytes()
    f2 = (out2 / "official_metrics_summary.csv").read_bytes()
    assert f1 == f2, "two CLI launches wrote differing official_metrics_summary.csv"
