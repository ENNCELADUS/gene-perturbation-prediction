"""Tests for default log file and per-fold epoch-metrics CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sl_dl_model.scoring import write_epoch_metrics


def test_write_epoch_metrics_csv(tmp_path: Path):
    rows = [
        {
            "epoch": 0.0,
            "mean_train_loss": 0.7,
            "val_pair_auroc": 0.5,
            "peak_gpu_mem_mb": 0.0,
        },
        {
            "epoch": 1.0,
            "mean_train_loss": 0.6,
            "val_pair_auroc": 0.55,
            "peak_gpu_mem_mb": 0.0,
        },
    ]
    out = write_epoch_metrics(tmp_path, "CV2", 3, rows)
    assert out == tmp_path / "CV2" / "epoch_metrics_fold3.csv"
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "split_type",
        "fold_id",
        "epoch",
        "mean_train_loss",
        "val_pair_auroc",
        "peak_gpu_mem_mb",
    ]
    assert len(df) == 2
    assert (df["split_type"] == "CV2").all()
    assert (df["fold_id"] == 3).all()


def test_append_epoch_metric_row_writes_header_once(tmp_path: Path):
    from sl_dl_model.scoring import append_epoch_metric_row

    r0 = {
        "epoch": 0.0,
        "mean_train_loss": 0.7,
        "val_pair_auroc": 0.5,
        "peak_gpu_mem_mb": 0.0,
    }
    r1 = {
        "epoch": 1.0,
        "mean_train_loss": 0.6,
        "val_pair_auroc": 0.55,
        "peak_gpu_mem_mb": 0.0,
    }

    out = append_epoch_metric_row(tmp_path, "CV2", 3, r0)
    assert out == tmp_path / "CV2" / "epoch_metrics_fold3.csv"
    # After the first row the file is readable with exactly one data row.
    df0 = pd.read_csv(out)
    assert list(df0.columns) == [
        "split_type",
        "fold_id",
        "epoch",
        "mean_train_loss",
        "val_pair_auroc",
        "peak_gpu_mem_mb",
    ]
    assert len(df0) == 1

    append_epoch_metric_row(tmp_path, "CV2", 3, r1)
    df1 = pd.read_csv(out)
    # Second call appends (not overwrites) and does NOT add a second header.
    assert len(df1) == 2
    assert df1["epoch"].tolist() == [0.0, 1.0]
    assert (df1["split_type"] == "CV2").all()
    assert (df1["fold_id"] == 3).all()


def test_default_log_file_path(tmp_path: Path, monkeypatch):
    """main() with no --log-file targets output_dir/train.log on main process."""
    import logging
    import sl_dl_model.__main__ as cli

    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "run"
    cfg_path.write_text(f"output_dir: {out_dir}\nsplit_types: [CV2]\nfolds: [0]\n")

    captured = {}

    def fake_run_cv(config, producer):
        captured["log_files"] = [
            h.baseFilename
            for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ]
        return None

    monkeypatch.setattr(cli, "_resolve_run_cv", lambda: fake_run_cv, raising=False)
    # Patch the lazy import target used in main().
    import sl_dl_model.evaluate as ev

    monkeypatch.setattr(ev, "run_cv", fake_run_cv)

    # logging.basicConfig is a no-op when the root logger already has handlers
    # (pytest's logging plugin attaches some). Clear them so main()'s handler
    # setup takes effect in-process; a real CLI run starts with no handlers.
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        cli.main(["run-cv", "--config", str(cfg_path), "--producer", "zero"])
    finally:
        root.handlers.clear()
        root.handlers.extend(saved)
    assert any(str(out_dir / "train.log") == p for p in captured["log_files"])
