from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

from src.scgpt.model import ScGPTBackbone
from src.scgpt import evaluate as scgpt_evaluate
from src.scgpt import factory as scgpt_factory
from src.scgpt import train as scgpt_train


class FakeTransformerModel(torch.nn.Module):
    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        FakeTransformerModel.last_kwargs = kwargs
        self.transformer_encoder = types.SimpleNamespace(layers=[])


def test_scgpt_backbone_does_not_enable_flash_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_scgpt_module(monkeypatch)
    paths = _write_minimal_scgpt_files(tmp_path)
    monkeypatch.setattr(ScGPTBackbone, "_load_checkpoint", lambda self: None)

    ScGPTBackbone(
        checkpoint_path=paths["checkpoint"],
        vocab_path=paths["vocab"],
        args_path=paths["args"],
        freeze_encoder=False,
    )

    assert FakeTransformerModel.last_kwargs is not None
    assert FakeTransformerModel.last_kwargs["use_fast_transformer"] is False
    assert FakeTransformerModel.last_kwargs["fast_transformer_backend"] == "flash"


def test_scgpt_train_and_evaluate_pass_configured_flash_backend(
    monkeypatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict] = []

    class FakeGeneScoreModel:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.append(kwargs)

    monkeypatch.setattr(scgpt_factory, "GeneScoreModel", FakeGeneScoreModel)
    config = {
        "model_config": {
            "model": "scgpt",
            "pretrained_dir": str(tmp_path),
            "use_fast_transformer": True,
            "fast_transformer_backend": "flash",
        }
    }

    scgpt_train._build_model(
        config,
        n_genes=2,
        gene_ids=torch.tensor([1, 2]),
        device="cpu",
    )
    scgpt_evaluate._build_model(
        config, n_genes=2, gene_ids=torch.tensor([1, 2]), device=torch.device("cpu")
    )

    assert [kwargs["use_fast_transformer"] for kwargs in captured_kwargs] == [
        True,
        True,
    ]
    assert [kwargs["fast_transformer_backend"] for kwargs in captured_kwargs] == [
        "flash",
        "flash",
    ]


def test_scgpt_train_build_model_passes_causal_set_graph_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: list[dict] = []

    class FakeGeneScoreModel:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.append(kwargs)

    edge_path = tmp_path / "edges.csv"
    edge_path.write_text("source,target,weight\nA,B,0.5\n")
    monkeypatch.setattr(scgpt_factory, "GeneScoreModel", FakeGeneScoreModel)
    config = {
        "run_config": {"study_name": "causal_set_test"},
        "model_config": {
            "model": "scgpt",
            "pretrained_dir": str(tmp_path),
            "score_mode": "causal_set",
            "graph": {
                "enabled": True,
                "source": "edge_list",
                "path": str(edge_path),
                "directed": False,
            },
            "architecture": {
                "use_graph_encoder": True,
                "use_contrast_encoder": True,
                "use_slots": True,
                "use_cardinality_head": True,
                "use_cycle_loss": True,
            },
        },
    }

    scgpt_train._build_model(
        config,
        n_genes=2,
        gene_ids=torch.tensor([1, 2]),
        device=torch.device("cpu"),
        gene_names=["A", "B"],
    )

    kwargs = captured_kwargs[0]
    assert kwargs["score_mode"] == "causal_set"
    assert kwargs["use_graph_encoder"] is True
    assert kwargs["use_slots"] is True
    assert kwargs["use_cardinality_head"] is True
    assert kwargs["use_cycle_loss"] is True
    assert kwargs["gene_graph_edge_index"].tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(
        kwargs["gene_graph_edge_weight"],
        torch.tensor([0.5, 0.5]),
    )


def _install_fake_scgpt_module(monkeypatch) -> None:
    scgpt_module = types.ModuleType("scgpt")
    model_package = types.ModuleType("scgpt.model")
    model_module = types.ModuleType("scgpt.model.model")
    model_module.TransformerModel = FakeTransformerModel
    monkeypatch.setitem(sys.modules, "scgpt", scgpt_module)
    monkeypatch.setitem(sys.modules, "scgpt.model", model_package)
    monkeypatch.setitem(sys.modules, "scgpt.model.model", model_module)


def _write_minimal_scgpt_files(tmp_path: Path) -> dict[str, Path]:
    vocab_path = tmp_path / "vocab.json"
    args_path = tmp_path / "args.json"
    checkpoint_path = tmp_path / "best_model.pt"
    vocab_path.write_text(json.dumps({"<pad>": 0, "A": 1}))
    args_path.write_text(
        json.dumps(
            {
                "embsize": 8,
                "nheads": 2,
                "d_hid": 16,
                "nlayers": 1,
                "dropout": 0.1,
                "pad_token": "<pad>",
                "pad_value": -2,
                "input_emb_style": "category",
                "n_bins": 51,
            }
        )
    )
    checkpoint_path.write_bytes(b"")
    return {"vocab": vocab_path, "args": args_path, "checkpoint": checkpoint_path}
