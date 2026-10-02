"""Response features of the joint head and train-only block standardization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.features import (
    DELTA_WIDTH,
    HVG_WIDTH,
    PROJECTION_WIDTH,
    FixedSparseProjection,
    compute_condition_feature_batch,
)
from src.model.normalization import BlockStandardizer
from src.model.response import energy_distance


def _features(predicted, basal, *, panel, index, available):
    return compute_condition_feature_batch(
        (predicted,),
        (basal,),
        projection=FixedSparseProjection(),
        gene_in_hvg_panel=torch.tensor([panel]),
        own_gene_hvg_indices=(index,),
        own_gene_available=torch.tensor([available]),
    )


def test_projection_is_seeded_sparse_round_trippable_and_differentiable():
    first, second = FixedSparseProjection(), FixedSparseProjection(seed=0)
    assert np.array_equal(first.components, second.components)
    assert not np.array_equal(first.components, FixedSparseProjection(7).components)
    assert first.components.shape == (PROJECTION_WIDTH, DELTA_WIDTH)
    assert np.count_nonzero(first.components) < first.components.size // 10
    restored = FixedSparseProjection.from_state(first.to_state())
    assert np.array_equal(restored.components, first.components)
    assert restored.metadata == first.metadata
    delta = torch.randn(DELTA_WIDTH, requires_grad=True)
    restored.transform(delta).square().sum().backward()
    assert torch.count_nonzero(delta.grad) > 0


def test_condition_features_match_hand_computable_shift_and_summaries():
    basal = torch.zeros(2, HVG_WIDTH)
    basal[:, 0] = torch.tensor([0.0, 2.0])
    predicted = torch.zeros(2, HVG_WIDTH)
    predicted[:, 0] = torch.tensor([2.0, 4.0])
    predicted.requires_grad_()
    result = _features(predicted, basal, panel=True, index=0, available=True)
    expected_delta = torch.zeros(DELTA_WIDTH)
    expected_delta[0] = 2.0  # mean shift 2, variance unchanged
    torch.testing.assert_close(
        result.delta_proj[0], FixedSparseProjection().transform(expected_delta)
    )
    expected_s = torch.tensor(
        [
            energy_distance(predicted.detach(), basal),
            1.0 / HVG_WIDTH,
            0.5,
            2.0,
            1.0,
            2.0,
        ]
    )
    torch.testing.assert_close(result.s[0].detach(), expected_s, atol=1e-6, rtol=0)
    (result.delta_proj.sum() + result.s[0, [0, 1, 3, 4, 5]].sum()).backward()
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize(("panel", "index"), [(False, None), (True, 0)])
def test_unavailable_own_gene_shift_is_zero_with_false_mask(panel, index):
    bag = torch.ones(2, HVG_WIDTH)
    shifted = bag + 1.0
    result = _features(shifted, bag, panel=panel, index=index, available=False)
    assert result.s[0, -1].item() == 0.0
    assert not result.own_gene_shift_mask.item()


def test_standardizer_fits_training_statistics_keeps_constants_and_round_trips():
    train = {
        "s": np.array([[1.0, 5.0], [3.0, 5.0]]),
        "q_sc": torch.tensor([[2.0], [4.0]]),
    }
    standardizer = BlockStandardizer().fit(train)
    assert standardizer.constant_columns == {"s": (1,), "q_sc": ()}
    probe = torch.tensor([[5.0, 8.0]])
    torch.testing.assert_close(
        standardizer.transform("s", probe), torch.tensor([[3.0, 3.0]])
    )
    restored = BlockStandardizer.from_state(standardizer.to_state())
    assert restored.to_state() == standardizer.to_state()
    with pytest.raises(RuntimeError, match="cannot be refit"):
        standardizer.fit(train)


def test_streamed_standardizer_matches_materialized_fit():
    first = {"s": np.array([[1.0, 5.0], [3.0, 5.0]])}
    second = {"s": np.array([[7.0, 5.0]])}
    streamed = BlockStandardizer().fit_batches([first, second])
    materialized = BlockStandardizer().fit(
        {"s": np.concatenate([first["s"], second["s"]])}
    )
    probe = torch.tensor([[5.0, 9.0]])
    torch.testing.assert_close(
        streamed.transform("s", probe), materialized.transform("s", probe)
    )
