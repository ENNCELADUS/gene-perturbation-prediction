from __future__ import annotations

from pathlib import Path

import torch

from src import main
from src.utils.runtime import AccelerateRuntime


def test_accelerate_runtime_saves_unwrapped_state_dict(tmp_path: Path) -> None:
    runtime = AccelerateRuntime(
        {
            "device_config": {
                "device": "cpu",
                "ddp_enabled": False,
                "use_mixed_precision": False,
            }
        }
    )
    model = torch.nn.Linear(2, 1)
    output_path = tmp_path / "model.pt"

    runtime.save_state_dict(model, output_path)

    state = torch.load(output_path, map_location="cpu")
    assert sorted(state) == ["bias", "weight"]


def test_scgpt_dispatch_invokes_stage_boundaries(monkeypatch) -> None:
    calls: list[str] = []

    def fake_import_stage_runner(model_name: str, stage_name: str):
        assert model_name == "scgpt"

        def run_stage(config: dict) -> dict:
            calls.append(stage_name)
            return {"ok": True}

        return run_stage

    monkeypatch.setattr(main, "import_stage_runner", fake_import_stage_runner)
    config = {
        "run_config": {"stages": ["train", "evaluate"], "seed": 7},
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {"dataloader": {}},
        "model_config": {"model": "scgpt"},
    }

    results = main.run_from_config(config)

    assert calls == ["train", "evaluate"]
    assert [stage["stage"] for stage in results["stages"]] == ["train", "evaluate"]
