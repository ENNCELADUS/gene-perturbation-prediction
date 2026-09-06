"""Actual tiny STATE training, common evaluation, logging and loss selection."""

import copy
import json
import random

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from src.data.prepared import load_inputs
from src.model.initialization import build_joint_model, _suppress_checkpoint_output
from src.training.checkpoint import load_checkpoint
from src.training import trainer
from test_joint_data import make_prepared_fixture


def tiny_training_config(root, *, transformer_dropout=0.1):
    """Reusable prepared fixture with the installed STATE class, no saved classes."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    config = make_prepared_fixture(root, hvg_width=2000)
    hparams = dict(
        input_dim=2560,
        hidden_dim=8,
        output_dim=2000,
        pert_dim=2,
        cell_set_len=4,
        predict_residual=False,
        embed_key="X_hvg",
        transformer_backbone_kwargs={
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "resid_pdrop": transformer_dropout,
            "embd_pdrop": transformer_dropout,
            "attn_pdrop": transformer_dropout,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(0)
    with _suppress_checkpoint_output():
        state = StateTransitionPerturbationModel(**copy.deepcopy(hparams))
    checkpoint = root / "released.ckpt"
    torch.save(
        {"hyper_parameters": hparams, "state_dict": state.state_dict()}, checkpoint
    )
    config["paths"]["state_checkpoint"] = str(checkpoint)
    config["model"] = {
        "cell_sentence_len": 4,
        "esm2_adapter_hidden": 4,
        "head_hidden": 8,
        "head_layers": 1,
    }
    config["train"].update(
        max_epochs=2,
        patience=5,
        response_interval=4,
        response_weight=1.0,
        state_learning_rate=1e-6,
        adapter_learning_rate=1e-5,
        head_learning_rate=1e-4,
        weight_decay=0.01,
        response_batch_size=4,
    )
    config["precision"] = "no"  # CPU numerical regression; production uses bf16.
    return config


def fresh_model(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    inputs = load_inputs(config)
    return build_joint_model(config, inputs), inputs


def restore_model(saved):
    inputs = load_inputs(saved["config"], preprocessing=saved["preprocessing"])
    model = build_joint_model(
        saved["config"],
        inputs,
        **{
            name: saved[name]
            for name in (
                "architecture",
                "model_state",
                "projection_state",
                "normalization_state",
            )
        },
    )
    return model, inputs


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_two_epoch_real_state_common_evaluator_and_geneeffect_selection(
    tmp_path, monkeypatch
):
    config = tiny_training_config(tmp_path / "data")
    config["train"].update(max_epochs=4, patience=1)
    model, inputs = fresh_model(config)
    accelerator = Accelerator(cpu=True)
    actual_evaluate = trainer.evaluate_model
    calls = []

    def evaluate(*args, **kwargs):
        result = actual_evaluate(*args, **kwargs)
        calls.append(dict(result.metrics))
        result.metrics.update(
            val_geneeffect_loss=[1.0, 2.0][len(calls) - 1],
            val_total_loss=[100.0, 1.0][len(calls) - 1],
            val_geneeffect_pearson_macro_per_line=[-1.0, 1.0][len(calls) - 1],
        )
        return result

    monkeypatch.setattr(trainer, "evaluate_model", evaluate)
    state = trainer.fit(model, inputs, config, tmp_path / "run", accelerator)
    assert len(calls) == 2
    assert (
        state.next_epoch,
        state.global_step,
        state.best_epoch,
        state.bad_epochs,
    ) == (2, 14, 0, 1)
    best = load_checkpoint(tmp_path / "run" / "best.pt")
    last = load_checkpoint(tmp_path / "run" / "last.pt")
    assert best["train_state"]["next_epoch"] == 1
    assert last["train_state"]["next_epoch"] == 2
    assert [g["name"] for g in last["optimizer"]["param_groups"]] == [
        "state",
        "adapter",
        "head",
    ]
    assert [g["lr"] for g in last["optimizer"]["param_groups"]] == [1e-6, 1e-5, 1e-4]
    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    validation = [r for r in records if "val_geneeffect_loss" in r]
    assert len(validation) == 2
    assert all(set(calls[i]) <= validation[i].keys() for i in range(2))
    training = [r for r in records if "train_geneeffect_loss" in r]
    assert [
        r["global_step"] - 1 for r in training if r["train_response_loss"] is not None
    ] == [0, 4, 8, 12]
    assert sum(r["train_dependency_rows"] for r in training) == 28
    assert all(
        r["train_response_mean_delta_mse"] is None
        for r in training
        if r["train_response_loss"] is None
    )
    assert accelerator.ddp_handler.find_unused_parameters
    assert not accelerator.ddp_handler.static_graph
    assert model.training


def test_finite_gradient_gate_accepts_zero_and_rejects_nan(tmp_path):
    config = tiny_training_config(tmp_path / "data")
    model, inputs = fresh_model(config)
    from src.model.normalization import fit_startup_standardizer
    from src.training.sampling import make_training_loaders

    accelerator = Accelerator(cpu=True)
    fit_startup_standardizer(model, inputs, batch_size=2)
    optimizer = trainer.make_optimizer(model, config)
    model, optimizer = accelerator.prepare(model, optimizer)
    batch = next(iter(make_training_loaders(inputs, config, 0, accelerator)[0]))
    handles = [
        p.register_hook(lambda g: g * 0) for p in model.parameters() if p.requires_grad
    ]
    trainer.train_update(model, optimizer, batch, None, config, accelerator)
    for handle in handles:
        handle.remove()
    parameter = next(model.head.parameters())
    handle = parameter.register_hook(lambda g: g * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        trainer.train_update(model, optimizer, batch, None, config, accelerator)
    handle.remove()
