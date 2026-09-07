from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn

from src.model.geneeffect import GeneEffectE2EModel
from src.data.batches import OnlineConditionBatch, FeatureBatch, ResponseForwardBatch
from src.model.normalization import BlockStandardizer
from src.model.features import FixedSparseProjection
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectResidualHead,
)


class _MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(0.1))

    def forward(self, controls, genes, batch_indices):
        del genes, batch_indices
        return tuple(control[:, :2000] + self.shift for control in controls)


def _standardizer(dims: GeneEffectFeatureDims) -> BlockStandardizer:
    rng = np.random.default_rng(4)
    return BlockStandardizer().fit(
        {
            "delta_proj": rng.normal(size=(4, dims.delta_proj)),
            "s": rng.normal(size=(4, dims.s)),
            "q_sc": rng.normal(size=(4, dims.q_sc)),
            "e_g": rng.normal(size=(4, dims.e_g)),
            "z_c": rng.normal(size=(4, dims.z_c)),
        }
    )


def _model(blocks=GeneEffectBlockConfig()) -> GeneEffectE2EModel:
    dims = GeneEffectFeatureDims(delta_proj=256, s=6, q_sc=3, e_g=2, z_c=3)
    return GeneEffectE2EModel(
        _MockBackbone(),
        GeneEffectResidualHead(dims=dims, blocks=blocks, hidden=8, n_hidden_layers=1),
        FixedSparseProjection(),
        _standardizer(dims),
        collator_seed=7,
    )


def _precomputed(batch: int = 3) -> FeatureBatch:
    return FeatureBatch(
        delta_proj=torch.randn(batch, 256),
        s=torch.randn(batch, 6),
        q_sc=torch.randn(batch, 3),
        e_g=torch.randn(batch, 2),
        z_c=torch.randn(batch, 3),
        q_sc_mask=torch.ones(batch, dtype=torch.bool),
        hvg_panel_mask=torch.tensor([True, False, True][:batch]),
        own_gene_shift_mask=torch.tensor([True, False, False][:batch]),
        gene_symbols=tuple(f"G{i}" for i in range(batch)),
        model_ids=tuple(f"ACH-{i:06d}" for i in range(batch)),
    )


@pytest.mark.parametrize("use_response", [True, False])
def test_online_gradient_reaches_trainable_backbone(use_response) -> None:
    model = _model(
        GeneEffectBlockConfig(use_delta_proj=use_response, use_s=use_response)
    )
    model.train()
    controls = tuple(torch.randn(4, 2000) for _ in range(2))
    batch = OnlineConditionBatch(
        controls_tx1=controls,
        basal_hvg=tuple(control.detach().clone() for control in controls),
        genes=("A", "B"),
        model_ids=("ACH-1", "ACH-2"),
        q_sc=torch.randn(2, 3),
        e_g=torch.randn(2, 2),
        z_c=torch.randn(2, 3),
        q_sc_mask=torch.ones(2, dtype=torch.bool),
        gene_in_hvg_panel=torch.tensor([True, False]),
        own_gene_hvg_indices=(10, None),
        own_gene_shift_available=torch.tensor([True, False]),
    )
    output = model(batch)
    assert output.delta_hat.shape == (2,)
    output.delta_hat.sum().backward()
    if use_response:
        assert model.backbone.shift.grad is not None
        assert torch.isfinite(model.backbone.shift.grad)
    else:
        assert model.backbone.shift.grad is None
        assert not output.raw_features.delta_proj.requires_grad
        assert not output.raw_features.s.requires_grad
        # Removing response from the readout does not disable response supervision.
        output = model(batch, response=ResponseForwardBatch(controls, batch.genes))
        sum(value.sum() for value in output.response_predicted).backward()
        assert model.backbone.shift.grad.abs() > 0


@pytest.mark.parametrize(
    "disabled",
    [
        ("delta_proj",),
        ("s",),
        ("q_sc",),
        ("e_g",),
        ("z_c",),
        ("delta_proj", "s"),
    ],
)
def test_disabled_features_and_masks_do_not_enter_standardization_or_readout(disabled):
    model = _model(GeneEffectBlockConfig(**{f"use_{name}": False for name in disabled}))
    state = model.standardizer.to_state()
    for name in disabled:
        del state["blocks"][name]
    model.standardizer = BlockStandardizer.from_state(state)
    features = _precomputed()
    expected = model.forward_features(features)
    changes = {
        name: torch.full_like(getattr(features, name), float("nan"), requires_grad=True)
        for name in disabled
    }
    for name, mask_names in (
        ("q_sc", ("q_sc_mask",)),
        ("s", ("hvg_panel_mask", "own_gene_shift_mask")),
    ):
        if name in disabled:
            changes.update({mask: ~getattr(features, mask) for mask in mask_names})
    actual = model.forward_features(replace(features, **changes))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert all(changes[name].grad is None for name in disabled)


def test_online_batch_rejects_non_panel_gene_with_index() -> None:
    controls = (torch.randn(2, 2000),)
    batch = OnlineConditionBatch(
        controls_tx1=controls,
        basal_hvg=controls,
        genes=("A",),
        model_ids=("ACH-1",),
        q_sc=torch.randn(1, 3),
        e_g=torch.randn(1, 2),
        z_c=torch.randn(1, 3),
        q_sc_mask=torch.ones(1, dtype=torch.bool),
        gene_in_hvg_panel=torch.tensor([False]),
        own_gene_hvg_indices=(1,),
        own_gene_shift_available=torch.tensor([False]),
    )
    with pytest.raises(ValueError, match="non-HVG"):
        batch.validate()


def test_add_train_gene_mean_is_aligned_and_fail_closed() -> None:
    delta = torch.tensor([0.5, -0.5])
    absolute = GeneEffectE2EModel.add_train_gene_mean(
        ["A", "B"], delta, {"A": 1.0, "B": 2.0}
    )
    assert torch.equal(absolute, torch.tensor([1.5, 1.5]))
    with pytest.raises(KeyError, match="absent"):
        GeneEffectE2EModel.add_train_gene_mean(["A", "C"], delta, {"A": 1.0})


def test_precomputed_requires_boolean_masks() -> None:
    features = _precomputed()
    bad = FeatureBatch(
        **{
            **features.__dict__,
            "q_sc_mask": torch.ones(features.batch_size),
        }
    )
    with pytest.raises(ValueError, match="q_sc_mask"):
        bad.validate()
