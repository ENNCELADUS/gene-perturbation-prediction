from __future__ import annotations

import pytest

from src import main
from src.utils.config import validate_config


def base_config(model: str = "pca_knn", stages: list[str] | None = None) -> dict:
    return {
        "run_config": {"stages": stages or ["evaluate"], "seed": 47},
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {"dataloader": {}},
        "model_config": {"model": model},
    }


def test_parse_args_accepts_only_config() -> None:
    args = main.parse_args(["--config", "src/scgpt/configs/norman.yaml"])

    assert args.config == "src/scgpt/configs/norman.yaml"


def test_parse_args_rejects_stage_like_cli_args() -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--config", "config.yaml", "--mode", "train"])


def test_validate_config_requires_top_level_sections() -> None:
    config = base_config()
    del config["data_config"]

    with pytest.raises(ValueError, match="data_config"):
        validate_config(config)


def test_validate_config_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model_config.model"):
        validate_config(base_config(model="tga"))


def test_validate_config_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="run_config.stages"):
        validate_config(base_config(stages=["train", "score"]))


def test_run_from_config_dispatches_stages_in_config_order(monkeypatch) -> None:
    config = base_config(model="pca_knn", stages=["prepare", "train", "evaluate"])
    calls: list[tuple[str, str]] = []

    def fake_import_stage_runner(model_name: str, stage_name: str):
        def run_stage(stage_config: dict) -> dict:
            assert stage_config is config
            calls.append((model_name, stage_name))
            return {"stage": stage_name}

        return run_stage

    monkeypatch.setattr(main, "import_stage_runner", fake_import_stage_runner)

    results = main.run_from_config(config)

    assert calls == [
        ("pca_knn", "prepare"),
        ("pca_knn", "train"),
        ("pca_knn", "evaluate"),
    ]
    assert [item["stage"] for item in results["stages"]] == [
        "prepare",
        "train",
        "evaluate",
    ]
