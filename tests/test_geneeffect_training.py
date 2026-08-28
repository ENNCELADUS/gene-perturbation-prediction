from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from aivc_model.geneeffect_e2e import PrecomputedFeatureBatch
from aivc_model.geneeffect_training import (
    PrecomputedSupervisedBatch,
    SupervisedMatrix,
    calibrate_lambda_dep,
    warmup_step,
)


class _TinyE2E(nn.Module):
    """Warmup protocol double exposing the real wrapper's freeze API."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self._backbone_frozen = False

    @property
    def backbone_frozen(self):
        return self._backbone_frozen

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self._backbone_frozen = True

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def assert_frozen_backbone_clean(self):
        assert all(
            not p.requires_grad and p.grad is None for p in self.backbone.parameters()
        )

    def forward_precomputed(self, features):
        return self.head(features.e_g).squeeze(-1)


def _features() -> PrecomputedFeatureBatch:
    pairs = 6
    return PrecomputedFeatureBatch(
        delta_proj=torch.randn(pairs, 256),
        s=torch.randn(pairs, 6),
        q_sc=torch.randn(pairs, 3),
        e_g=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]] * 2),
        z_c=torch.randn(pairs, 3),
        q_sc_mask=torch.ones(pairs, dtype=torch.bool),
        hvg_panel_mask=torch.ones(pairs, dtype=torch.bool),
        own_gene_shift_mask=torch.ones(pairs, dtype=torch.bool),
        gene_symbols=("G1", "G1", "G1", "G2", "G2", "G2"),
        model_ids=("C1", "C2", "C3", "C1", "C2", "C3"),
    )


def _supervision() -> SupervisedMatrix:
    return SupervisedMatrix(
        target=torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
        label_mask=torch.ones(2, 3, dtype=torch.bool),
        g_var_mask=torch.ones(2, dtype=torch.bool),
        gene_symbols=("G1", "G2"),
        context_model_ids_by_gene=(("C1", "C2", "C3"), ("C1", "C2", "C3")),
        target_kind="train_mean_residual",
        residual_target_sha256="0" * 64,
        centering_fit_model_ids_sha256="1" * 64,
    )


def test_warmup_step_updates_only_head() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    before_backbone = [
        parameter.detach().clone() for parameter in model.backbone.parameters()
    ]
    before_head = [parameter.detach().clone() for parameter in model.head.parameters()]
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
    metrics = warmup_step(
        model,  # type: ignore[arg-type]
        PrecomputedSupervisedBatch(_features(), _supervision()),
        optimizer,
    )
    assert metrics.n_valid_pairs == 6
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            before_backbone, model.backbone.parameters(), strict=True
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_head, model.head.parameters(), strict=True)
    )


def test_flat_feature_alignment_is_enforced() -> None:
    bad = replace(_features(), e_g=torch.randn(5, 2))
    with pytest.raises(ValueError, match="batch"):
        PrecomputedSupervisedBatch(bad, _supervision()).validate()


def test_same_cardinality_identity_permutation_is_rejected() -> None:
    features = replace(
        _features(),
        gene_symbols=("G1", "G1", "G2", "G1", "G2", "G2"),
        model_ids=("C1", "C2", "C1", "C3", "C2", "C3"),
    )
    with pytest.raises(ValueError, match="gene-major"):
        PrecomputedSupervisedBatch(features, _supervision()).validate()


def test_row_specific_context_identities_are_supported() -> None:
    supervision = replace(
        _supervision(),
        context_model_ids_by_gene=(("C1", "C2", "C3"), ("C3", "C1", "C2")),
    )
    features = replace(
        _features(),
        model_ids=("C1", "C2", "C3", "C3", "C1", "C2"),
    )
    PrecomputedSupervisedBatch(features, supervision).validate()


def test_repeated_gene_rows_allow_only_masked_padding_duplicates() -> None:
    supervision = replace(
        _supervision(),
        gene_symbols=("G1", "G1"),
        g_var_mask=torch.tensor([True, True]),
        label_mask=torch.tensor([[True, True, True], [False, False, False]]),
    )
    features = replace(
        _features(),
        gene_symbols=("G1",) * 6,
    )
    PrecomputedSupervisedBatch(features, supervision).validate()
    duplicate = replace(
        supervision,
        label_mask=torch.tensor([[True, True, True], [True, False, False]]),
    )
    with pytest.raises(ValueError, match="duplicate labeled"):
        PrecomputedSupervisedBatch(features, duplicate).validate()


def test_lambda_calibration_uses_median_and_clips() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))

    def pair(response_scale, dependency_scale):
        return (
            lambda: parameter.square() * response_scale,
            lambda: parameter.square() * dependency_scale,
        )

    report = calibrate_lambda_dep(
        [pair(1.0, 2.0), pair(2.0, 2.0), pair(6.0, 2.0)],
        [parameter],
        clip_min=0.1,
        clip_max=2.0,
    )
    assert report.raw_ratios == pytest.approx((0.5, 1.0, 3.0))
    assert report.lambda_dep == pytest.approx(1.0)


def test_lambda_calibration_rejects_zero_gradient() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    with pytest.raises(ValueError, match="positive"):
        calibrate_lambda_dep(
            [(lambda: parameter * 0.0, lambda: parameter.square())], [parameter]
        )


class _MultiRankAccelerator:
    num_processes = 2


def test_warmup_rejects_multi_rank_accelerator() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="single-process"):
        warmup_step(
            model,  # type: ignore[arg-type]
            PrecomputedSupervisedBatch(_features(), _supervision()),
            optimizer,
            accelerator=_MultiRankAccelerator(),  # type: ignore[arg-type]
        )
