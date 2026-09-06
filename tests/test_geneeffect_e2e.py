from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.model.geneeffect import GeneEffectE2EModel
from src.data.batches import OnlineConditionBatch, FeatureBatch
from src.model.normalization import BlockStandardizer
from src.model.features import FixedSparseProjection
from src.model.head import GeneEffectFeatureDims, GeneEffectResidualHead


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


def _model() -> GeneEffectE2EModel:
    dims = GeneEffectFeatureDims(delta_proj=256, s=6, q_sc=3, e_g=2, z_c=3)
    return GeneEffectE2EModel(
        _MockBackbone(),
        GeneEffectResidualHead(dims=dims, hidden=8, n_hidden_layers=1),
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


def test_frozen_warmup_updates_head_only_and_keeps_backbone_eval() -> None:
    model = _model()
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training
    loss = model.forward_precomputed(_precomputed()).square().mean()
    loss.backward()
    model.assert_frozen_backbone_clean()
    assert any(parameter.grad is not None for parameter in model.head.parameters())


def test_unfreeze_enables_online_gradient_to_backbone() -> None:
    model = _model()
    model.freeze_backbone()
    model.unfreeze_backbone()
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
    assert model.backbone.shift.grad is not None
    assert torch.isfinite(model.backbone.shift.grad)


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
