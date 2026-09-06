"""CLI validation and config boundaries require no model/GPU imports."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.experiments.config import load_config, validate_config
from src.train import parse_args


def test_approved_defaults_and_conflicts():
    config = load_config(Path("configs/geneeffect_joint.yaml"))
    assert config["seeds"] == {"train": 0, "collator": 0, "projection": 0}
    assert config["train"]["max_epochs"] == 50
    assert config["train"]["dependency_batch_size"] == 1024
    assert config["train"]["response_batch_size"] == 64
    for group, key, value in [
        ("selection", "metric", "val_total_loss"),
        ("seeds", "train", 1),
        ("train", "response_batch_size", 5),
        ("train", "head_learning_rate", float("nan")),
        ("train", "max_epochs", True),
        ("features", "variable_gene_percentile", 101),
        ("preparation", "response_holdout_seed", 0),
        ("train", "response_interval", 3),
    ]:
        changed = copy.deepcopy(config)
        changed[group][key] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    for mutation in (
        lambda c: c["train"].pop("patience"),
        lambda c: c.update(stage1="retired"),
    ):
        changed = copy.deepcopy(config)
        mutation(changed)
        with pytest.raises(ValueError, match="missing|unknown"):
            validate_config(changed)


def test_run_mode_parser():
    for args in (["config.yaml"], ["config.yaml", "--run-id", "a", "--resume", "b"]):
        with pytest.raises(SystemExit):
            parse_args(args)
    args = parse_args(["config with spaces.yaml", "--resume", "last with spaces.pt"])
    assert args.config == Path("config with spaces.yaml")
    assert args.resume == Path("last with spaces.pt")


@pytest.mark.parametrize(
    "module",
    [
        "src.train",
        "src.evaluate",
        "src.experiments.prepare",
        "src.experiments.baselines",
    ],
)
def test_help_does_not_import_torch(module):
    code = (
        f'import runpy,sys; sys.argv=["{module}","--help"];\n'
        f'try: runpy.run_module("{module}",run_name="__main__")\n'
        "except SystemExit as e: assert e.code==0\n"
        'assert "torch" not in sys.modules'
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_dispatch_keeps_arguments(monkeypatch):
    from src import train, evaluate
    from src.experiments import geneeffect

    calls = []
    monkeypatch.setattr(
        geneeffect, "run_training", lambda *a, **kw: calls.append((a, kw))
    )
    monkeypatch.setattr(
        geneeffect, "evaluate_checkpoint", lambda *a, **kw: calls.append((a, kw))
    )
    train.main(["a b.yaml", "--run-id", "run 1"])
    evaluate.main(["--checkpoint", "a b.pt", "--split", "val"])
    assert calls == [
        ((Path("a b.yaml"),), {"run_id": "run 1", "resume": None}),
        ((Path("a b.pt"),), {"split": "val"}),
    ]


def test_missing_cache_training_fails_without_raw_rebuild(tmp_path, monkeypatch):
    # Keep production Accelerator construction on the synthetic suite's CPU,
    # including unsandboxed macOS runs where MPS is visible.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    from src.experiments import geneeffect, prepare
    from src.data import response

    config = load_config(Path("configs/geneeffect_joint.yaml"))
    config["prepared_root"] = str(tmp_path / "missing")
    config["output_root"] = str(tmp_path / "runs")
    config["precision"] = "no"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    def forbidden(*a, **kw):
        pytest.fail("training invoked raw assembly")

    monkeypatch.setattr(prepare, "prepare_inputs", forbidden)
    monkeypatch.setattr(response, "assemble_train_response_gene_bags", forbidden)
    with pytest.raises(FileNotFoundError, match="hpc/run.sh prepare"):
        geneeffect.run_training(path, run_id="missing")
    record = json.loads((tmp_path / "runs/missing/run.json").read_text())
    assert record["environment"]["device"] == "cpu"
    assert record["training"]["status"] == "failed"
    assert record["evaluation"]["status"] == "not_started"
    assert not (tmp_path / "missing").exists()


def test_frozen_tx1_collator_seed_does_not_depend_on_previous_lines(monkeypatch):
    import random
    import types
    import numpy as np
    import pandas as pd
    import torch
    from src.model import tx1

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def predict(self, loader, *, return_outputs):
            values = [random.random(), float(np.random.rand()), float(torch.rand(()))]
            return [{"cell_emb": torch.tensor([values])}]

    monkeypatch.setitem(sys.modules, "composer", types.SimpleNamespace(Trainer=Trainer))
    monkeypatch.setitem(
        sys.modules,
        "tahoe_x1.utils.util",
        types.SimpleNamespace(loader_from_adata=lambda **kwargs: None),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tx1, "install_padding_metadata_fallback", lambda: None)
    monkeypatch.setattr(
        tx1, "load_local_safetensors", lambda _: (object(), {"E": 0}, {}, {})
    )
    encoder, _ = tx1._build_tx1_encoder(Path("unused"), 2, 4)
    data = types.SimpleNamespace(var=pd.DataFrame({"ensembl_id": ["E"]}))
    first = encoder(data)
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    np.testing.assert_array_equal(first, encoder(data))
