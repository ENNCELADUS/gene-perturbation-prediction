from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import aivc_model.geneeffect_stage2_runner as runner
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

    def _standardize(self, features):
        return {"e_g": features.e_g}


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
    assert model.backbone.training is False


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


def test_duplicate_context_ids_are_allowed_only_when_masked() -> None:
    masked = replace(
        _supervision(),
        context_model_ids_by_gene=(("C1", "C2", "C2"), ("C1", "C2", "C3")),
        label_mask=torch.tensor([[True, True, False], [True, True, True]]),
    )
    features = replace(
        _features(),
        model_ids=("C1", "C2", "C2", "C1", "C2", "C3"),
    )
    PrecomputedSupervisedBatch(features, masked).validate()

    labeled = replace(
        masked,
        label_mask=torch.tensor([[True, True, True], [True, True, True]]),
    )
    with pytest.raises(ValueError, match="duplicate labeled"):
        PrecomputedSupervisedBatch(features, labeled).validate()


def test_only_labeled_targets_must_be_finite() -> None:
    masked_nonfinite = replace(
        _supervision(),
        target=torch.tensor([[0.0, 1.0, float("nan")], [2.0, 1.0, 0.0]]),
        label_mask=torch.tensor([[True, True, False], [True, True, True]]),
    )
    masked_nonfinite.validate()

    labeled_nonfinite = replace(
        masked_nonfinite,
        label_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="targets must be finite"):
        labeled_nonfinite.validate()


def test_supervision_allows_batch_without_g_var_gene() -> None:
    replace(_supervision(), g_var_mask=torch.zeros(2, dtype=torch.bool)).validate()


def test_sampler_tail_duplicate_feeds_precomputed_supervision() -> None:
    context_count = 170
    data = SimpleNamespace(
        genes=("G0", "G1"),
        model_ids=tuple(f"C{index}" for index in range(context_count)),
        targets=np.arange(2 * context_count, dtype=np.float32).reshape(
            2, context_count
        ),
        label_mask=np.ones((2, context_count), dtype=bool),
        g_var_mask=np.ones(2, dtype=bool),
        residual_target_sha256="0" * 64,
        centering_fit_model_ids_sha256="1" * 64,
    )
    config = SimpleNamespace(
        joint=SimpleNamespace(genes_per_batch=2, contexts_per_gene=32),
        seeds=SimpleNamespace(train=7),
    )
    indices = runner._epoch_batch_indices(data, data.model_ids, config, epoch=0)
    tail_index = next(
        batch for batch in indices if any(not all(row.label_mask) for row in batch.rows)
    )
    row = next(row for row in tail_index.rows if not all(row.label_mask))
    assert len(set(row.context_indices)) < len(row.context_indices)

    labeled_pairs: list[tuple[str, str]] = []
    for index in indices:
        supervision = runner._supervision_from_index(data, index)
        gene_symbols = tuple(
            gene
            for gene in supervision.gene_symbols
            for _ in range(supervision.shape[1])
        )
        model_ids = tuple(
            model_id
            for context_row in supervision.context_model_ids_by_gene
            for model_id in context_row
        )
        pair_count = supervision.pair_count
        features = PrecomputedFeatureBatch(
            delta_proj=torch.zeros(pair_count, 256),
            s=torch.zeros(pair_count, 6),
            q_sc=torch.zeros(pair_count, 3),
            e_g=torch.zeros(pair_count, 2),
            z_c=torch.zeros(pair_count, 3),
            q_sc_mask=torch.ones(pair_count, dtype=torch.bool),
            hvg_panel_mask=torch.ones(pair_count, dtype=torch.bool),
            own_gene_shift_mask=torch.ones(pair_count, dtype=torch.bool),
            gene_symbols=gene_symbols,
            model_ids=model_ids,
        )
        PrecomputedSupervisedBatch(features, supervision).validate()
        labeled_pairs.extend(
            (gene, model_id)
            for row_index, gene in enumerate(supervision.gene_symbols)
            for column, model_id in enumerate(
                supervision.context_model_ids_by_gene[row_index]
            )
            if bool(supervision.label_mask[row_index, column])
        )
    assert len(labeled_pairs) == len(set(labeled_pairs)) == 2 * context_count


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
    device = torch.device("cpu")

    def unwrap_model(self, value):
        return value.module

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value * self.num_processes

    def backward(self, value):
        value.backward()

    def clip_grad_norm_(self, parameters, limit):
        return torch.nn.utils.clip_grad_norm_(parameters, limit)


class _PreparedHead(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *, e_g, **_masks):
        return self.module(e_g).squeeze(-1)


class _PaddingAccelerator(_MultiRankAccelerator):
    def __init__(self) -> None:
        self.reductions = iter((torch.tensor([6.0, 2.0]), torch.tensor([3.0, 1.0])))

    def reduce(self, _value, reduction):
        assert reduction == "sum"
        return next(self.reductions)


def test_warmup_requires_prepared_head_for_multi_rank() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="prepared forward_head"):
        warmup_step(
            model,  # type: ignore[arg-type]
            PrecomputedSupervisedBatch(_features(), _supervision()),
            optimizer,
            accelerator=_MultiRankAccelerator(),  # type: ignore[arg-type]
        )


def test_warmup_uses_prepared_head_without_reassigning_model_head() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    raw_head = model.head
    forward_head = _PreparedHead(raw_head)
    optimizer = torch.optim.AdamW(raw_head.parameters(), lr=1e-2)
    metrics = warmup_step(
        model,  # type: ignore[arg-type]
        PrecomputedSupervisedBatch(_features(), _supervision()),
        optimizer,
        accelerator=_MultiRankAccelerator(),  # type: ignore[arg-type]
        forward_head=forward_head,
    )
    assert model.head is raw_head
    assert metrics.n_valid_pairs == 12
    assert metrics.n_genes_scored == 4


def test_warmup_rejects_forward_head_without_multi_rank_accelerator() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    with pytest.raises(ValueError, match="only valid for multi-rank"):
        warmup_step(
            model,  # type: ignore[arg-type]
            PrecomputedSupervisedBatch(_features(), _supervision()),
            torch.optim.AdamW(model.head.parameters(), lr=1e-2),
            forward_head=_PreparedHead(nn.Linear(2, 1)),
        )


def test_warmup_padding_reports_only_remote_real_pairs() -> None:
    model = _TinyE2E()
    model.freeze_backbone()
    forward_head = _PreparedHead(model.head)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-2)
    metrics = warmup_step(
        model,  # type: ignore[arg-type]
        PrecomputedSupervisedBatch(_features(), _supervision(), objective_weight=0.0),
        optimizer,
        accelerator=_PaddingAccelerator(),  # type: ignore[arg-type]
        forward_head=forward_head,
    )
    assert metrics.n_valid_pairs == 6
    assert metrics.n_genes_scored == 2
