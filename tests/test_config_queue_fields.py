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
