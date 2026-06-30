from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_manifest_path,
    save_embedding_cache,
    write_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_queue import (
    read_step2_failed_cache_fp,
    read_step2_result_cache_fp,
    step2_fold_fingerprint,
    step2_metric_config,
    step2_metric_model_name,
    step_failed_path,
    step_result_path,
    step_results_dir,
)


def _benchmark_csv(tmp_path: Path) -> Path:
    rows = []
    effects = {"A": -1.0, "B": -0.5, "C": 0.2, "D": 0.7}
    specs = [
        ("train", 1, "A", "B"),
        ("train", 0, "A", "C"),
        ("test", 1, "A", "D"),
        ("test", 0, "B", "D"),
    ]
    for idx, (role, label, gene_a, gene_b) in enumerate(specs):
        rows.append(
            {
                "pair_id": f"p{idx}",
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": role,
                "sl_label": label,
                "gene_a_symbol": gene_a,
                "gene_b_symbol": gene_b,
                "gene_a_k562_gene_effect": effects[gene_a],
                "gene_b_k562_gene_effect": effects[gene_b],
            }
        )
    path = tmp_path / "benchmark.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _config(tmp_path: Path, **kwargs: object) -> Exp08bConfig:
    defaults: dict[str, object] = {
        "input_csv": _benchmark_csv(tmp_path),
        "output_dir": tmp_path / "run",
        "split_types": ("CV2",),
        "folds": (0,),
        "ranking_k": (10,),
        "max_epochs": 1,
        "batch_pairs": 2,
        "pair_hidden": (8,),
        "assembly_poll_seconds": 0.01,
        "assembly_timeout_seconds": 0.01,
    }
    defaults.update(kwargs)
    return Exp08bConfig(**defaults)


def _write_config(path: Path, cfg: Exp08bConfig) -> None:
    payload = {
        "input_csv": str(cfg.input_csv),
        "output_dir": str(cfg.output_dir),
        "split_types": list(cfg.split_types or ()),
        "folds": list(cfg.folds),
        "ranking_k": list(cfg.ranking_k),
        "max_epochs": cfg.max_epochs,
        "batch_pairs": cfg.batch_pairs,
        "pair_hidden": list(cfg.pair_hidden),
        "assembly_poll_seconds": cfg.assembly_poll_seconds,
        "assembly_timeout_seconds": cfg.assembly_timeout_seconds,
    }
    path.write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in payload.items())
    )


def _write_step1_artifacts(
    cfg: Exp08bConfig,
    *,
    fold_id: int = 0,
    generator_kind: str = "state_adapter",
    payload: bytes = b"cache-v1",
) -> None:
    cache = embedding_cache_path(cfg, "CV2", fold_id)
    save_embedding_cache(
        cache,
        symbols=np.array(["A", "B", "C", "D"], dtype=object),
        embeddings=np.arange(8, dtype=np.float32).reshape(4, 2),
        coverage_mask=np.array([1, 1, 1, 1], dtype=np.int64),
        embedding_method="test",
    )
    cache.write_bytes(payload)
    write_generator_manifest(
        generator_manifest_path(cfg, "CV2", fold_id),
        {"generator_kind": generator_kind},
    )


def test_step_queue_paths_include_step_name(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    assert step_result_path(cfg, "generator", "CV2", 0).parent.name == "generator"
    assert step_failed_path(cfg, "sl_head", "CV2", 0).parent.name == "sl_head"


def test_cli_help_lists_exp08b_entrypoints(capsys: pytest.CaptureFixture[str]) -> None:
    from sl_dl_model.__main__ import _build_parser

    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "train-generator" in help_text
    assert "train-sl-head" in help_text


def test_train_sl_head_missing_cache_fails_fast(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sl_dl_model",
            "train-sl-head",
            "--config",
            str(cfg_path),
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing Step 1 cache" in result.stderr


def test_runner_exception_handlers_quarantine_and_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, folds=(0, 1))
    _write_step1_artifacts(cfg, fold_id=0)
    _write_step1_artifacts(cfg, fold_id=1, payload=b"cache-v1-fold1")

    import sl_dl_model.exp08b_step2_runner as runner

    calls: list[tuple[str, int]] = []

    def fake_run_fold(frame, split_type, fold_id, config, producer):
        calls.append((split_type, fold_id))
        if fold_id == 0:
            raise ValueError("boom")
        return [
            {
                "split_type": split_type,
                "fold_id": fold_id,
                "model": "exp08b",
                "slice": "full_universe",
                "metric": "ndcg@10",
                "value": 1.0,
            }
        ]

    class DummyProducer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(runner, "run_fold_with_producer", fake_run_fold)
    monkeypatch.setattr(runner, "CachedEmbeddingPairHeadProducer", DummyProducer)

    with pytest.raises(RuntimeError):
        runner.run_train_sl_head(cfg)

    assert calls == [("CV2", 0), ("CV2", 1)]
    results_dir = fq.fold_results_dir(step2_metric_config(cfg))
    assert fq.failed_path(results_dir, "CV2", 0).exists()
    assert fq.result_path(results_dir, "CV2", 1).exists()


def test_step1_runner_does_not_import_pair_head_or_sl_label() -> None:
    source = Path("src/sl_dl_model/exp08b_step1_runner.py").read_text()
    forbidden = {
        "SymmetricPairHead",
        "CachedEmbeddingPairHeadProducer",
        "SlHeadConfig",
        "run_fold_with_producer",
        "exp08b_sl_head",
        "sl_label",
        "sl_dl_model.scoring",
        "train_symbols_for_fold",
        "load_benchmark",
        "fold_split",
    }

    for value in forbidden:
        assert value not in source


def test_step1_runner_import_does_not_load_scoring() -> None:
    snippet = (
        "import sys\n"
        "import sl_dl_model.exp08b_step1_runner\n"
        "print('loaded=' + str('sl_dl_model.scoring' in sys.modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "loaded=False" in result.stdout


def test_step1_generator_input_loader_accepts_symbol_only_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "generator_pairs.csv"
    pd.DataFrame(
        [
            {
                "pair_id": "p0",
                "fold_id": 0,
                "split_role": "train",
                "gene_a_symbol": "a",
                "gene_b_symbol": "b",
            },
            {
                "pair_id": "p1",
                "fold_id": 0,
                "split_role": "test",
                "gene_a_symbol": "c",
                "gene_b_symbol": "d",
            },
        ]
    ).to_csv(csv_path, index=False)

    from sl_dl_model.exp08b_runner import jobs
    from sl_dl_model.exp08b_step1_runner import (
        _fold_rows,
        _load_pairs_for_generator,
        _train_symbols,
    )

    frame = _load_pairs_for_generator(csv_path)
    cfg = Exp08bConfig(
        input_csv=csv_path,
        output_dir=tmp_path / "run",
        split_types=("CV1",),
        folds=(0,),
    )

    assert frame["split_type"].tolist() == ["CV1", "CV1"]
    train_df, test_df = _fold_rows(frame, "CV1", 0)
    assert _train_symbols(train_df) == {"A", "B"}
    assert test_df["gene_a_symbol"].tolist() == ["c"]
    assert jobs(frame, cfg) == [("CV1", 0)]

    cv2_path = tmp_path / "generator_pairs_cv2.csv"
    pd.DataFrame(
        [
            {
                "fold_id": 0,
                "split_type": "CV2",
                "split_role": "train",
                "gene_a_symbol": "x",
                "gene_b_symbol": "y",
            }
        ]
    ).to_csv(cv2_path, index=False)
    assert _load_pairs_for_generator(cv2_path)["split_type"].tolist() == ["CV2"]


def test_step1_generator_loader_reads_only_generator_columns(tmp_path: Path) -> None:
    from sl_dl_model.exp08b_step1_runner import _load_pairs_for_generator

    frame = _load_pairs_for_generator(_benchmark_csv(tmp_path))

    assert set(frame.columns) == {
        "fold_id",
        "split_type",
        "split_role",
        "gene_a_symbol",
        "gene_b_symbol",
    }


def test_step2_runner_does_not_import_generator_or_state() -> None:
    source = Path("src/sl_dl_model/exp08b_step2_runner.py").read_text()
    forbidden = {
        "Step1GeneratorTrainer",
        "StateEncoder",
        "PertAdapter",
        "SlDlModel",
        "exp08b_generator",
        "state_checkpoint",
        "_load_state_dl_caches",
    }

    for value in forbidden:
        assert value not in source


def test_step2_fold_fingerprint_changes_when_step1_cache_rewritten(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _write_step1_artifacts(cfg, payload=b"first")
    first = step2_fold_fingerprint(cfg, "CV2", 0)

    cache = embedding_cache_path(cfg, "CV2", 0)
    cache.write_bytes(b"second-longer")
    stat = cache.stat()
    os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = step2_fold_fingerprint(cfg, "CV2", 0)

    assert first != second


def test_step2_result_cache_fp_round_trips_inside_result_json(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    metric_cfg = step2_metric_config(cfg)
    results_dir = fq.fold_results_dir(metric_cfg)
    fp = fq.fingerprint(metric_cfg)

    fq.write_result(
        results_dir,
        "CV2",
        0,
        [{"metric": "x"}],
        fingerprint=fp,
        extra={"cache_fp": "abc"},
    )
    assert read_step2_result_cache_fp(results_dir, "CV2", 0) == "abc"

    fq.write_result(results_dir, "CV2", 1, [{"metric": "x"}], fingerprint=fp)
    assert read_step2_result_cache_fp(results_dir, "CV2", 1) is None


def test_step2_result_cache_fp_ignores_malformed_result_json(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    results_dir = fq.fold_results_dir(step2_metric_config(cfg))
    path = fq.result_path(results_dir, "CV2", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")

    assert read_step2_result_cache_fp(results_dir, "CV2", 0) is None


def test_step2_failed_cache_fp_ignores_malformed_failed_marker(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    results_dir = fq.fold_results_dir(step2_metric_config(cfg))
    path = fq.failed_path(results_dir, "CV2", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")

    assert read_step2_failed_cache_fp(results_dir, "CV2", 0) is None


@pytest.mark.parametrize("marker_kind", ["result", "failed"])
def test_step2_runner_ignores_malformed_marker_before_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    cfg = _config(tmp_path)
    _write_step1_artifacts(cfg)
    results_dir = fq.fold_results_dir(step2_metric_config(cfg))
    marker_path = (
        fq.result_path(results_dir, "CV2", 0)
        if marker_kind == "result"
        else fq.failed_path(results_dir, "CV2", 0)
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("{not-json")

    import sl_dl_model.exp08b_step2_runner as runner

    calls: list[tuple[str, int]] = []

    def fake_run_fold(frame, split_type, fold_id, config, producer):
        calls.append((split_type, fold_id))
        return [
            {
                "split_type": split_type,
                "fold_id": fold_id,
                "model": "exp08b",
                "slice": "full_universe",
                "metric": "ndcg@10",
                "value": 1.0,
            }
        ]

    class DummyProducer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(runner, "run_fold_with_producer", fake_run_fold)
    monkeypatch.setattr(runner, "CachedEmbeddingPairHeadProducer", DummyProducer)

    runner.run_train_sl_head(cfg)

    assert calls == [("CV2", 0)]


def test_wait_for_step_complete_polls_until_missing_job_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sl_dl_model.exp08b_runner import wait_for_step_complete

    cfg = _config(tmp_path, assembly_timeout_seconds=1.0)
    results_dir = step_results_dir(cfg, "generator")
    fp = fq.fingerprint(cfg)
    jobs = [("CV2", 0), ("CV2", 1)]
    fq.write_result(results_dir, "CV2", 0, [{"fold_id": 0}], fingerprint=fp)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fq.write_result(results_dir, "CV2", 1, [{"fold_id": 1}], fingerprint=fp)

    monkeypatch.setattr("sl_dl_model.exp08b_runner.time.sleep", fake_sleep)

    wait_for_step_complete(
        results_dir,
        jobs,
        fp,
        "generator",
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert sleeps == [0.01]


def test_step2_completion_waits_for_current_result_after_stale_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path, assembly_timeout_seconds=1.0)
    _write_step1_artifacts(cfg)
    metric_cfg = step2_metric_config(cfg)
    fp = fq.fingerprint(metric_cfg)
    results_dir = fq.fold_results_dir(metric_cfg)
    fq.write_failed(
        results_dir,
        "CV2",
        0,
        {"traceback": "old failure", "cache_fp": "stale-cache"},
        fingerprint=fp,
    )
    cache_fp = step2_fold_fingerprint(cfg, "CV2", 0)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fq.write_result(
            results_dir,
            "CV2",
            0,
            [{"metric": "current"}],
            fingerprint=fp,
            extra={"cache_fp": cache_fp},
        )

    monkeypatch.setattr("sl_dl_model.exp08b_step2_runner.time.sleep", fake_sleep)

    from sl_dl_model.exp08b_step2_runner import _raise_if_step2_incomplete

    _raise_if_step2_incomplete(
        cfg,
        metric_cfg,
        results_dir,
        [("CV2", 0)],
        fp,
    )

    assert sleeps == [0.01]


def test_step2_completion_rejects_stale_result_cache_fp(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_step1_artifacts(cfg, payload=b"old-cache")
    metric_cfg = step2_metric_config(cfg)
    fp = fq.fingerprint(metric_cfg)
    results_dir = fq.fold_results_dir(metric_cfg)
    old_cache_fp = step2_fold_fingerprint(cfg, "CV2", 0)
    fq.write_result(
        results_dir,
        "CV2",
        0,
        [{"metric": "stale"}],
        fingerprint=fp,
        extra={"cache_fp": old_cache_fp},
    )

    cache = embedding_cache_path(cfg, "CV2", 0)
    cache.write_bytes(b"new-cache-longer")
    stat = cache.stat()
    os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert step2_fold_fingerprint(cfg, "CV2", 0) != old_cache_fp

    from sl_dl_model.exp08b_step2_runner import _raise_if_step2_incomplete

    with pytest.raises(RuntimeError, match="stale"):
        _raise_if_step2_incomplete(
            cfg,
            metric_cfg,
            results_dir,
            [("CV2", 0)],
            fp,
        )


def test_step2_completion_rejects_stale_failed_cache_fp(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _write_step1_artifacts(cfg, payload=b"old-cache")
    metric_cfg = step2_metric_config(cfg)
    fp = fq.fingerprint(metric_cfg)
    results_dir = fq.fold_results_dir(metric_cfg)
    old_cache_fp = step2_fold_fingerprint(cfg, "CV2", 0)
    fq.write_failed(
        results_dir,
        "CV2",
        0,
        {"traceback": "old failure", "cache_fp": old_cache_fp},
        fingerprint=fp,
    )

    cache = embedding_cache_path(cfg, "CV2", 0)
    cache.write_bytes(b"new-cache-longer")
    stat = cache.stat()
    os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert step2_fold_fingerprint(cfg, "CV2", 0) != old_cache_fp

    from sl_dl_model.exp08b_step2_runner import _raise_if_step2_incomplete

    with pytest.raises(RuntimeError, match="stale"):
        _raise_if_step2_incomplete(
            cfg,
            metric_cfg,
            results_dir,
            [("CV2", 0)],
            fp,
        )


def test_step2_metric_model_name_reads_generator_kind_from_manifest(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    _write_step1_artifacts(cfg, generator_kind="direct_mlp")

    assert step2_metric_model_name(cfg, "CV2", 0) == "direct_esm2_mlp"


def test_step2_metric_config_is_state_neutral(tmp_path: Path) -> None:
    base = _config(tmp_path)
    other = replace(
        base,
        generator_kind="direct_mlp",
        esm2_npz=tmp_path / "other_esm.npz",
        bags_npz=tmp_path / "other_bags.npz",
        gwps_h5ad=tmp_path / "other.h5ad",
    )

    base_metric = step2_metric_config(base)
    other_metric = step2_metric_config(other)

    assert base_metric.state_backend == "linear_mock"
    assert base_metric.output_dir == base.output_dir / base.step2_results_subdir
    assert fq.fingerprint(base_metric) == fq.fingerprint(other_metric)


def test_step2_stale_failed_marker_ignored_after_cache_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    metric_cfg = step2_metric_config(cfg)
    fp = fq.fingerprint(metric_cfg)
    results_dir = fq.fold_results_dir(metric_cfg)
    fq.write_failed(
        results_dir,
        "CV2",
        0,
        {"traceback": "missing Step 1 cache", "cache_fp": "absent-cache"},
        fingerprint=fp,
    )
    _write_step1_artifacts(cfg)
    current_cache_fp = step2_fold_fingerprint(cfg, "CV2", 0)
    assert read_step2_failed_cache_fp(results_dir, "CV2", 0) == "absent-cache"
    assert current_cache_fp != "absent-cache"

    import sl_dl_model.exp08b_step2_runner as runner

    calls: list[tuple[str, int]] = []

    def fake_run_fold(frame, split_type, fold_id, config, producer):
        calls.append((split_type, fold_id))
        return [
            {
                "split_type": split_type,
                "fold_id": fold_id,
                "model": "exp08b",
                "slice": "full_universe",
                "metric": "ndcg@10",
                "value": 1.0,
            }
        ]

    class DummyProducer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(runner, "run_fold_with_producer", fake_run_fold)
    monkeypatch.setattr(runner, "CachedEmbeddingPairHeadProducer", DummyProducer)

    runner.run_train_sl_head(cfg)

    assert calls == [("CV2", 0)]
    assert read_step2_result_cache_fp(results_dir, "CV2", 0) == current_cache_fp
