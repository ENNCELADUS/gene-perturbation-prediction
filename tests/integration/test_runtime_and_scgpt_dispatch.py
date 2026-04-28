from __future__ import annotations

import logging
from pathlib import Path

import torch

from src import main
from src.scgpt import evaluate as scgpt_evaluate
from src.scgpt import train as scgpt_train
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
        "run_config": {
            "stages": ["train", "evaluate"],
            "seed": 7,
            "study_name": "test",
        },
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


def test_scgpt_dispatch_logs_stage_boundaries_only_on_primary_rank(
    monkeypatch,
    caplog,
) -> None:
    calls: list[str] = []

    def fake_import_stage_runner(model_name: str, stage_name: str):
        def run_stage(config: dict) -> dict:
            calls.append(f"{model_name}.{stage_name}")
            return {"ok": True}

        return run_stage

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(main, "import_stage_runner", fake_import_stage_runner)
    config = {
        "run_config": {
            "stages": ["prepare", "train"],
            "seed": 7,
            "study_name": "test",
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {"dataloader": {}},
        "model_config": {"model": "scgpt"},
    }

    with caplog.at_level(logging.INFO, logger=main.LOGGER.name):
        results = main.run_from_config(config)

    assert calls == ["scgpt.prepare", "scgpt.train"]
    assert [stage["stage"] for stage in results["stages"]] == ["prepare", "train"]
    assert "Running scgpt.prepare" not in caplog.text
    assert "Running scgpt.train" not in caplog.text


def test_scgpt_dispatch_logs_stage_boundaries_on_primary_rank(
    monkeypatch,
    caplog,
) -> None:
    def fake_import_stage_runner(model_name: str, stage_name: str):
        def run_stage(config: dict) -> dict:
            return {"ok": f"{model_name}.{stage_name}"}

        return run_stage

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(main, "import_stage_runner", fake_import_stage_runner)
    config = {
        "run_config": {
            "stages": ["prepare"],
            "seed": 7,
            "study_name": "test",
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {"dataloader": {}},
        "model_config": {"model": "scgpt"},
    }

    with caplog.at_level(logging.INFO, logger=main.LOGGER.name):
        main.run_from_config(config)

    assert "Running scgpt.prepare" in caplog.text


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
        "run_config": {"stages": ["evaluate"], "seed": 7, "study_name": "test"},
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


def test_scgpt_train_reports_epoch_progress_and_memory(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    class FakeRuntime:
        instances: list["FakeRuntime"] = []

        def __init__(self, config: dict) -> None:
            self.device = torch.device("cpu")
            self.backward_calls = 0
            self.clip_calls = 0
            self.saved_paths: list[Path] = []
            FakeRuntime.instances.append(self)

        def prepare(self, *objects):
            return objects

        def backward(self, loss: torch.Tensor) -> None:
            self.backward_calls += 1
            loss.backward()

        def clip_grad_norm_(self, parameters, max_norm: float) -> None:
            self.clip_calls += 1

        def save_state_dict(self, model: torch.nn.Module, path: str | Path) -> None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("saved")
            self.saved_paths.append(output_path)

    class FakeDataset(torch.utils.data.Dataset):
        gene_ids = torch.tensor([0, 1])

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict:
            return {"index": index}

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            gene_ids: torch.Tensor,
            values: torch.Tensor,
            padding_mask: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            return self.bias.expand(values.shape[0], 2)

    def fake_collate(batch: list[dict], vocab: dict, n_genes: int) -> dict:
        return {
            "genes": torch.tensor([[1, 2]]),
            "values": torch.tensor([[1.0, 1.0]]),
            "padding_mask": torch.tensor([[False, False]]),
            "control_genes": torch.tensor([[1, 2]]),
            "control_values": torch.tensor([[1.0, 1.0]]),
            "control_padding_mask": torch.tensor([[False, False]]),
            "control_counts": 1,
            "targets": torch.tensor([[1.0, 0.0]]),
        }

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    (pretrained_dir / "vocab.json").write_text('{"<pad>": 0, "A": 1, "B": 2}')

    monkeypatch.setattr(scgpt_train, "AccelerateRuntime", FakeRuntime, raising=False)
    monkeypatch.setattr(
        scgpt_train,
        "load_adata",
        lambda path: type("Adata", (), {"n_vars": 2})(),
    )
    monkeypatch.setattr(
        scgpt_train,
        "get_condition_splits",
        lambda config: {"train": ["A"], "validation": [], "test": []},
    )
    monkeypatch.setattr(scgpt_train, "GeneScoreDataset", lambda **kwargs: FakeDataset())
    monkeypatch.setattr(
        scgpt_train,
        "_build_model",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(scgpt_train, "collate_gene_score_batch", fake_collate)
    config = {
        "run_config": {
            "stages": ["train"],
            "seed": 7,
            "study_name": "test",
            "save_checkpoint_path": str(tmp_path / "model.pt"),
            "disable_tqdm": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "h5ad_path": "unused.h5ad",
            "condition_split": {"train": ["A"]},
        },
        "model_config": {"model": "scgpt", "pretrained_dir": str(pretrained_dir)},
        "training_config": {"epochs": 2, "batch_size": 1},
    }

    with caplog.at_level(logging.INFO, logger=scgpt_train.LOGGER.name):
        result = scgpt_train.run(config)

    runtime = FakeRuntime.instances[0]
    assert runtime.backward_calls == 4
    assert runtime.clip_calls == 4
    assert result["n_train"] == 2
    assert "scGPT train epoch 1/2 started" in caplog.text
    assert "scGPT train epoch 2/2 started" in caplog.text
    assert "scGPT train epoch 1/2 complete: batches=2" in caplog.text
    assert "mean_loss=" in caplog.text
    assert "gpu_memory=not_available" in caplog.text
