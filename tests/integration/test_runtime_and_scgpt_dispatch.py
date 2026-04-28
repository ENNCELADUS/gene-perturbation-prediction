from __future__ import annotations

from pathlib import Path

import torch

from src import main
from src.scgpt import evaluate as scgpt_evaluate
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


def test_scgpt_evaluate_uses_accelerate_runtime(monkeypatch, tmp_path: Path) -> None:
    class FakeRuntime:
        instances: list["FakeRuntime"] = []

        def __init__(self, config: dict) -> None:
            self.config = config
            self.device = torch.device("cpu")
            self.is_main_process = True
            self.prepare_calls = 0
            self.gather_calls = 0
            self.wait_calls = 0
            FakeRuntime.instances.append(self)

        def prepare(self, *objects):
            self.prepare_calls += 1
            return objects

        def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
            self.gather_calls += 1
            return tensor

        def wait_for_everyone(self) -> None:
            self.wait_calls += 1

    class FakeDataset(torch.utils.data.Dataset):
        gene_ids = torch.tensor([0, 1])
        gene_name_to_idx = {"A": 0, "B": 1}

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            return {"condition": "A+ctrl"}

    class FakeModel(torch.nn.Module):
        def forward(
            self,
            gene_ids: torch.Tensor,
            values: torch.Tensor,
            padding_mask: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            return torch.tensor([[2.0, -1.0]])

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    (pretrained_dir / "vocab.json").write_text('{"<pad>": 0, "A": 1, "B": 2}')

    monkeypatch.setattr(scgpt_evaluate, "AccelerateRuntime", FakeRuntime, raising=False)
    monkeypatch.setattr(
        scgpt_evaluate,
        "load_adata",
        lambda path: type("Adata", (), {"n_vars": 2})(),
    )
    monkeypatch.setattr(
        scgpt_evaluate,
        "get_condition_splits",
        lambda config: {"train": [], "validation": [], "test": ["A"]},
    )
    monkeypatch.setattr(
        scgpt_evaluate,
        "GeneScoreDataset",
        lambda **kwargs: FakeDataset(),
    )
    monkeypatch.setattr(
        scgpt_evaluate,
        "_build_model",
        lambda *args, **kwargs: FakeModel(),
    )

    def fake_collate(batch: list[dict], vocab: dict, n_genes: int) -> dict:
        return {
            "genes": torch.tensor([[1, 2]]),
            "values": torch.tensor([[1, 1]]),
            "padding_mask": torch.tensor([[False, False]]),
            "control_genes": torch.tensor([[1, 2]]),
            "control_values": torch.tensor([[1, 1]]),
            "control_padding_mask": torch.tensor([[False, False]]),
            "control_counts": 1,
            "targets": torch.tensor([[1.0, 0.0]]),
            "conditions": ["A+ctrl"],
        }

    monkeypatch.setattr(scgpt_evaluate, "collate_gene_score_batch", fake_collate)
    config = {
        "run_config": {"stages": ["evaluate"], "seed": 7},
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {"h5ad_path": "unused.h5ad", "condition_split": {"test": ["A"]}},
        "model_config": {"model": "scgpt", "pretrained_dir": str(pretrained_dir)},
        "evaluation_config": {"top_k_values": [1]},
    }

    result = scgpt_evaluate.run(config)

    runtime = FakeRuntime.instances[0]
    assert runtime.prepare_calls == 1
    assert runtime.gather_calls == 2
    assert runtime.wait_calls == 1
    assert result["metrics"]["exact_hit@1"] == 1.0
