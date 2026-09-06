"""Joint objectives, connected gradients and self-contained model initialization."""

from __future__ import annotations

import copy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from src.data.batches import ResponseForwardBatch
from src.data.datasets import DependencyDataset, ResponseDataset
from src.data.prepared import load_inputs
from src.model.initialization import build_joint_model
from src.model.losses import geneeffect_loss
from src.model.normalization import BlockStandardizer, fit_startup_standardizer
from src.model.response import energy_distance, mean_delta_mse, response_terms
from src.model.state import LinearMockStateModel
from test_joint_data import make_prepared_fixture


class TinyState(LinearMockStateModel):
    def __init__(self, input_dim, output_dim, pert_dim, cell_set_len=4, **kwargs):
        super().__init__(input_dim, output_dim, pert_dim)
        self.cell_sentence_len = cell_set_len
        # Mirror STATE's mutation of nested constructor arguments.
        kwargs.get("transformer_backbone_kwargs", {})["n_positions"] = cell_set_len


@pytest.fixture
def joint_setup(tmp_path):
    torch.manual_seed(0)
    config = make_prepared_fixture(tmp_path, hvg_width=2000)
    inputs = load_inputs(config)
    hparams = {
        "input_dim": 2560,
        "output_dim": 2000,
        "pert_dim": 2,
        "cell_set_len": 4,
        "transformer_backbone_kwargs": {"n_positions": 99},
    }
    reference = TinyState(**copy.deepcopy(hparams))
    checkpoint = tmp_path / "released.ckpt"
    torch.save(
        {"hyper_parameters": hparams, "state_dict": reference.state_dict()}, checkpoint
    )
    config["paths"]["state_checkpoint"] = str(checkpoint)
    config["model"] = {
        "cell_sentence_len": 4,
        "esm2_adapter_hidden": 4,
        "head_hidden": 8,
        "head_layers": 1,
    }
    model = build_joint_model(config, inputs, model_cls=TinyState)
    # Identity scaling keeps gradient checks independent of startup numerics.
    model.standardizer = BlockStandardizer.from_state(
        {
            "version": 1,
            "blocks": {
                name: {"mean": [0.0] * width, "scale": [1.0] * width}
                for name, width in vars(model.head.dims).items()
            },
        }
    )
    dependency = DependencyDataset(inputs, "train")
    batch = dependency.collate([0, 1])
    replay = ResponseDataset(inputs, holdout=False)
    response = replay.collate(replay.indices[:2])
    return model, inputs, config, batch, response


def test_huber_matches_absolute_geneeffect():
    pred = torch.tensor([0.2, -0.4], requires_grad=True)
    target = torch.tensor([0.0, -0.1])
    mean = torch.tensor([-0.7, 0.3])
    valid = torch.tensor([True, True])
    torch.testing.assert_close(
        geneeffect_loss(pred, target, valid),
        geneeffect_loss(pred + mean, target + mean, valid),
    )


def test_huber_masks_missing_values_and_reduces_in_float32():
    pred = torch.tensor([2.0, float("nan")], dtype=torch.bfloat16, requires_grad=True)
    loss = geneeffect_loss(
        pred, torch.tensor([0.0, float("nan")]), torch.tensor([True, False])
    )
    assert loss.dtype == torch.float32
    assert loss.item() == 1.5
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([1.0, 0.0], dtype=pred.dtype))


@pytest.mark.parametrize(
    "prediction,target,valid,match",
    [
        ([0.0], [0.0], [False], "no labeled"),
        ([float("nan")], [0.0], [True], "predictions"),
        ([0.0], [float("inf")], [True], "targets"),
    ],
)
def test_huber_rejects_invalid_labeled_batches(prediction, target, valid, match):
    with pytest.raises(ValueError, match=match):
        geneeffect_loss(
            torch.tensor(prediction), torch.tensor(target), torch.tensor(valid)
        )


def _grad_vector(module):
    return torch.cat(
        [
            parameter.grad.flatten()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
    )


def test_regression_updates_state_adapter_and_head_through_live_features(joint_setup):
    model, _, _, batch, _ = joint_setup
    output = model(batch.conditions)
    assert output.raw_features.delta_proj.requires_grad
    assert output.raw_features.s.requires_grad
    geneeffect_loss(output.delta_hat, batch.residual, batch.valid).backward()
    for module in (
        model.backbone.state_adapter,
        model.backbone.perturbations,
        model.head,
    ):
        gradient = _grad_vector(module)
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_replay_adds_reconstruction_gradients_in_same_model_call(joint_setup):
    model, _, _, batch, response = joint_setup
    alone = model(batch.conditions)
    geneeffect_loss(alone.delta_hat, batch.residual, batch.valid).backward()
    regression_gradient = _grad_vector(model.backbone).clone()
    head_gradient = _grad_vector(model.head).clone()
    model.zero_grad(set_to_none=True)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(True))
    output = model(
        batch.conditions, ResponseForwardBatch(response.controls_tx1, response.genes)
    )
    terms = response_terms(output.response_predicted, response)
    (
        geneeffect_loss(output.delta_hat, batch.residual, batch.valid)
        + sum(term.mean() for term in terms.values())
    ).backward()
    handle.remove()
    assert calls == [True]
    assert not torch.equal(_grad_vector(model.backbone), regression_gradient)
    torch.testing.assert_close(_grad_vector(model.head), head_gradient)
    for index, predicted in enumerate(output.response_predicted):
        torch.testing.assert_close(
            terms["mean_delta_mse"][index],
            mean_delta_mse(
                predicted,
                response.observed_hvg[index],
                response.control_hvg[index].mean(0),
            ),
        )
        torch.testing.assert_close(
            terms["energy_distance"][index],
            energy_distance(predicted, response.observed_hvg[index]),
        )


def test_strict_restore_uses_saved_metadata_and_actual_esm_without_upstream(
    joint_setup, monkeypatch
):
    model, inputs, config, batch, _ = joint_setup
    model.eval()
    expected = model(batch.conditions).delta_hat.detach()
    projection = model.projection.to_state()
    projection["components"][0][0] = 123.0

    # A saved matrix is restored as-is; no regeneration/hash identity gate.
    def no_reads(*args, **kwargs):
        raise AssertionError("resume reread an upstream checkpoint")

    monkeypatch.setattr(torch, "load", no_reads)
    restored = build_joint_model(
        config,
        inputs,
        architecture=model.architecture,
        model_state=model.state_dict(),
        projection_state=model.projection.to_state(),
        normalization_state=model.standardizer.to_state(),
        model_cls=TinyState,
    )
    restored.eval()
    torch.testing.assert_close(
        restored(batch.conditions).delta_hat, expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.backbone.perturbations.esm_matrix,
        torch.from_numpy(inputs.esm2_vectors),
        rtol=0,
        atol=0,
    )
    assert (
        model.architecture["state_hparams"]["transformer_backbone_kwargs"][
            "n_positions"
        ]
        == 99
    )
    broken = dict(model.state_dict())
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError, match="Missing key"):
        build_joint_model(
            config,
            inputs,
            architecture=model.architecture,
            model_state=broken,
            projection_state=projection,
            normalization_state=model.standardizer.to_state(),
            model_cls=TinyState,
        )
    with pytest.raises(ValueError, match="ESM2 order"):
        build_joint_model(
            config,
            replace(inputs, esm2_symbols=tuple(reversed(inputs.esm2_symbols))),
            architecture=model.architecture,
            model_state=model.state_dict(),
            projection_state=projection,
            normalization_state=model.standardizer.to_state(),
            model_cls=TinyState,
        )


def test_startup_fit_is_train_only_deterministic_and_preserves_rng(joint_setup):
    model, inputs, _, _, _ = joint_setup
    model.standardizer = BlockStandardizer()
    model.train()
    torch.manual_seed(121)
    random.seed(122)
    np.random.seed(123)
    before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    seen = []
    original = model.condition_features

    def record(batch):
        assert not torch.is_grad_enabled()
        assert not model.training
        seen.extend(batch.model_ids)
        return original(batch)

    model.condition_features = record
    fit_startup_standardizer(model, inputs, batch_size=4)
    assert set(seen) == (set(inputs.split.train) - set(inputs.split.unlabeled_train))
    assert "ACH-VAL" not in seen
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), before[0], rtol=0, atol=0)
    assert random.getstate() == before[1]
    np.testing.assert_array_equal(np.random.get_state()[1], before[2][1])
    first = model.standardizer.to_state()
    model.standardizer = BlockStandardizer()
    fit_startup_standardizer(model, inputs, batch_size=3)
    for name in first["blocks"]:
        np.testing.assert_allclose(
            model.standardizer.to_state()["blocks"][name]["mean"],
            first["blocks"][name]["mean"],
            atol=1e-8,
        )


def test_startup_failure_restores_modes_rng_and_reports_error(joint_setup):
    model, inputs, _, _, _ = joint_setup
    model.standardizer = BlockStandardizer()
    before = torch.get_rng_state().clone()

    def fail(batch):
        torch.rand(20)
        raise ValueError("bad fit inputs")

    model.condition_features = fail
    with pytest.raises(RuntimeError, match="bad fit inputs"):
        fit_startup_standardizer(model, inputs)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_real_installed_state_constructs_and_restores_without_released_weights(
    joint_setup, tmp_path
):
    from src.model.initialization import _suppress_checkpoint_output
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    _, inputs, config, batch, _ = joint_setup
    hparams = dict(
        input_dim=2560,
        hidden_dim=8,
        output_dim=2000,
        pert_dim=2,
        cell_set_len=4,
        predict_residual=False,
        embed_key="X_hvg",
        transformer_backbone_kwargs={"n_embd": 8, "n_layer": 1, "n_head": 2},
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    with _suppress_checkpoint_output():
        state = StateTransitionPerturbationModel(**copy.deepcopy(hparams))
    checkpoint = tmp_path / "tiny-real-state.ckpt"
    torch.save(
        {"hyper_parameters": hparams, "state_dict": state.state_dict()}, checkpoint
    )
    config["paths"]["state_checkpoint"] = str(checkpoint)
    model = build_joint_model(config, inputs)
    fit_startup_standardizer(model, inputs, batch_size=3)
    model.eval()
    prediction = model(batch.conditions).delta_hat
    checkpoint.unlink()
    restored = build_joint_model(
        config,
        inputs,
        architecture=model.architecture,
        model_state=model.state_dict(),
        projection_state=model.projection.to_state(),
        normalization_state=model.standardizer.to_state(),
    )
    restored.eval()
    torch.testing.assert_close(
        restored(batch.conditions).delta_hat, prediction, rtol=0, atol=0
    )


def test_startup_selects_at_most_32_unique_conditions_per_line(
    joint_setup, monkeypatch
):
    import pandas as pd
    import src.data.datasets as datasets

    model, inputs, _, _, _ = joint_setup
    real = DependencyDataset(inputs, "train")
    selected = []

    class ManyConditions:
        rows = pd.DataFrame({"model_id": ["ACH-A0"] * 40 + ["ACH-A1"] * 10})

        def __init__(self, supplied, split):
            assert supplied is inputs
            assert split == "train"

        def collate(self, indices):
            selected.extend(indices)
            return real.collate([index % len(real) for index in indices])

    monkeypatch.setattr(datasets, "DependencyDataset", ManyConditions)
    model.standardizer = BlockStandardizer()
    fit_startup_standardizer(model, inputs, batch_size=7)
    assert len(set(selected)) == len(selected) == 42
    assert sum(index < 40 for index in selected) == 32
    first = selected.copy()
    selected.clear()
    torch.manual_seed(912)
    np.random.seed(913)
    model.standardizer = BlockStandardizer()
    fit_startup_standardizer(model, inputs, batch_size=9)
    assert selected == first
