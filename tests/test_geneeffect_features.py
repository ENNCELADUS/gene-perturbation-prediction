from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.model.features import (
    DELTA_WIDTH,
    HVG_WIDTH,
    PROJECTION_WIDTH,
    FixedSparseProjection,
    compute_condition_features,
)
from src.model.normalization import BlockStandardizer
from src.model.response import energy_distance


def test_projection_is_deterministic_seeded_round_trippable_and_differentiable() -> (
    None
):
    first = FixedSparseProjection()
    second = FixedSparseProjection()
    different = FixedSparseProjection(seed=7)
    assert np.array_equal(first.components, second.components)
    assert not np.array_equal(first.components, different.components)
    assert first.components.shape == (PROJECTION_WIDTH, DELTA_WIDTH)
    assert np.count_nonzero(first.components) < first.components.size // 10
    restored = FixedSparseProjection.from_state(first.to_state())
    assert np.array_equal(restored.components, first.components)
    assert restored.metadata == first.metadata

    delta = torch.randn(DELTA_WIDTH, requires_grad=True)
    restored.transform(delta).square().sum().backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert torch.count_nonzero(delta.grad) > 0


def test_projection_restores_actual_values_and_rejects_nonfinite_state() -> None:
    state = FixedSparseProjection().to_state()
    state["components"][0][0] = 123.0
    assert FixedSparseProjection.from_state(state).components[0, 0] == 123.0
    state["components"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FixedSparseProjection.from_state(state)


def test_condition_features_match_hand_computable_delta_and_summaries() -> None:
    basal = torch.zeros(2, HVG_WIDTH)
    basal[:, 0] = torch.tensor([0.0, 2.0])
    predicted = torch.zeros(2, HVG_WIDTH, requires_grad=True)
    with torch.no_grad():
        predicted[:, 0] = torch.tensor([2.0, 4.0])
    projection = FixedSparseProjection()
    result = compute_condition_features(
        predicted,
        basal,
        projection=projection,
        gene_in_hvg_panel=True,
        own_gene_hvg_index=0,
        own_gene_available=True,
    )

    expected_delta = torch.zeros(DELTA_WIDTH)
    expected_delta[0] = 2.0
    assert torch.allclose(result.delta_proj, projection.transform(expected_delta))
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
    assert torch.allclose(result.s.detach(), expected_s, atol=1e-6)
    assert result.metadata["shift_threshold_basal_l2_p95"] == pytest.approx(1.0)
    assert result.own_gene_shift_mask.item() is True
    (result.delta_proj.sum() + result.s[[0, 1, 3, 4, 5]].sum()).backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_unavailable_own_gene_is_zero_with_false_mask() -> None:
    bag = torch.ones(2, HVG_WIDTH)
    for gene_in_hvg_panel, index in ((False, None), (True, 0)):
        result = compute_condition_features(
            bag,
            bag,
            projection=FixedSparseProjection(),
            gene_in_hvg_panel=gene_in_hvg_panel,
            own_gene_hvg_index=index,
            own_gene_available=False,
        )
        assert result.s[-1].item() == 0.0
        assert result.own_gene_shift_mask.item() is False


def test_standardizer_uses_training_statistics_keeps_constants_and_round_trips() -> (
    None
):
    train = {
        "s": np.array([[1.0, 5.0], [3.0, 5.0]]),
        "q_sc": torch.tensor([[2.0], [4.0]]),
    }
    standardizer = BlockStandardizer().fit(train)
    assert standardizer.constant_columns == {"s": (1,), "q_sc": ()}
    test = torch.tensor([[5.0, 8.0]], requires_grad=True)
    transformed = standardizer.transform("s", test)
    assert torch.allclose(transformed, torch.tensor([[3.0, 3.0]]))
    transformed.sum().backward()
    assert torch.equal(test.grad, torch.ones_like(test))
    with pytest.raises(RuntimeError, match="cannot be refit"):
        standardizer.fit(train)

    state = standardizer.to_state()
    restored = BlockStandardizer.from_state(state)
    assert restored.to_state() == standardizer.to_state()
    assert torch.equal(restored.transform("s", test.detach()), transformed.detach())


def test_standardizer_rejects_masks_invalid_and_tampered_inputs() -> None:
    with pytest.raises(ValueError, match="mask blocks"):
        BlockStandardizer().fit({"own_gene_shift_mask": np.ones((2, 1), dtype=bool)})
    with pytest.raises(ValueError, match="finite"):
        BlockStandardizer().fit({"s": np.array([[np.nan]])})
    fitted = BlockStandardizer().fit({"s": np.ones((2, 1))})
    with pytest.raises(ValueError, match="2-D"):
        fitted.transform("s", torch.ones(1))
    state = copy.deepcopy(fitted.to_state())
    state["blocks"]["s"]["scale"][0] = 0.0
    with pytest.raises(ValueError, match="statistics"):
        BlockStandardizer.from_state(state)


def test_streamed_standardizer_matches_materialized_fit() -> None:
    first = {
        "s": np.array([[1.0, 5.0], [3.0, 5.0]]),
        "q_sc": torch.tensor([[2.0], [4.0]]),
    }
    second = {
        "s": np.array([[7.0, 5.0]]),
        "q_sc": torch.tensor([[8.0]]),
    }
    streamed = BlockStandardizer().fit_batches([first, second])
    materialized = BlockStandardizer().fit(
        {
            "s": np.concatenate([first["s"], second["s"]]),
            "q_sc": torch.cat([first["q_sc"], second["q_sc"]]),
        }
    )
    probe = torch.tensor([[5.0, 9.0]])
    assert torch.allclose(
        streamed.transform("s", probe), materialized.transform("s", probe)
    )
    assert streamed.constant_columns == materialized.constant_columns


def test_streamed_standardizer_rejects_schema_drift_and_empty_stream() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BlockStandardizer().fit_batches([])
    with pytest.raises(ValueError, match="same blocks"):
        BlockStandardizer().fit_batches(
            [{"s": np.ones((1, 2))}, {"q_sc": np.ones((1, 1))}]
        )


@pytest.mark.parametrize(
    ("predicted", "basal", "match"),
    [
        (torch.zeros(0, HVG_WIDTH), torch.zeros(1, HVG_WIDTH), "at least one row"),
        (torch.zeros(1, 3), torch.zeros(1, HVG_WIDTH), "2000 HVGs"),
        (
            torch.full((1, HVG_WIDTH), float("nan")),
            torch.zeros(1, HVG_WIDTH),
            "non-finite",
        ),
    ],
)
def test_condition_features_reject_invalid_bags(
    predicted: torch.Tensor, basal: torch.Tensor, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_condition_features(
            predicted,
            basal,
            projection=FixedSparseProjection(),
            gene_in_hvg_panel=False,
            own_gene_hvg_index=None,
            own_gene_available=False,
        )


def test_condition_features_require_explicit_consistent_own_gene_availability() -> None:
    bag = torch.zeros(1, HVG_WIDTH)
    with pytest.raises(ValueError, match="explicit bool"):
        compute_condition_features(
            bag,
            bag,
            projection=FixedSparseProjection(),
            gene_in_hvg_panel=np.bool_(False),
            own_gene_hvg_index=None,
            own_gene_available=False,
        )
    with pytest.raises(ValueError, match="valid HVG index"):
        compute_condition_features(
            bag,
            bag,
            projection=FixedSparseProjection(),
            gene_in_hvg_panel=True,
            own_gene_hvg_index=None,
            own_gene_available=False,
        )


def test_default_projection_uses_zero_seed():
    np.testing.assert_array_equal(
        FixedSparseProjection().components, FixedSparseProjection(seed=0).components
    )
