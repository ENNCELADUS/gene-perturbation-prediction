"""Tests for the Stage 1 response-model training objective.

The loss is the part worth testing hard: it is distributional, so the usual
"does it go down" check cannot distinguish a model that matches the mean
while collapsing spread -- exactly the failure the retired backbone showed.
These cover each term's discriminating behavior, the per-line held-out gene
split, and that training actually moves the module it is handed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from aivc_model.response_training import (
    ResponseLoss,
    ResponseLossWeights,
    TrainingConfig,
    energy_distance,
    evaluate_response_model,
    mean_delta_mse,
    split_heldout_genes,
    train_response_model,
)


class _ShiftModel(nn.Module):
    """Adds a learnable per-gene shift to control cells; no head, like ST."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim))

    def predict_response_chunks(self, chunks, gene, batch_index_chunks):
        return tuple(chunk + self.shift for chunk in chunks)


def _bag(rows: int, dim: int, offset: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(rows * 100 + dim)
    return torch.randn(rows, dim, generator=generator) * scale + offset


# --- the two loss terms -------------------------------------------------


def test_mean_delta_mse_is_zero_when_the_mean_shift_matches() -> None:
    control_mean = torch.zeros(4)
    observed = torch.full((6, 4), 2.0)
    predicted = torch.full((3, 4), 2.0)
    assert float(mean_delta_mse(predicted, observed, control_mean)) == pytest.approx(
        0.0
    )


def test_mean_delta_mse_ignores_spread() -> None:
    """The mean term alone cannot see a collapsed prediction -- hence energy."""
    control_mean = torch.zeros(4)
    observed = _bag(64, 4, offset=1.0, scale=1.0)
    collapsed = observed.mean(dim=0, keepdim=True).repeat(64, 1)
    assert float(mean_delta_mse(collapsed, observed, control_mean)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_energy_distance_detects_the_collapse_the_mean_term_misses() -> None:
    observed = _bag(64, 4, offset=1.0, scale=1.0)
    collapsed = observed.mean(dim=0, keepdim=True).repeat(64, 1)
    assert float(energy_distance(collapsed, observed)) > 0.1


def test_energy_distance_is_near_zero_for_samples_of_one_distribution() -> None:
    left = _bag(128, 3, offset=0.0, scale=1.0)
    right = torch.randn(128, 3, generator=torch.Generator().manual_seed(7))
    assert float(energy_distance(left, right)) < 0.15


def test_energy_distance_rejects_an_empty_bag() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        energy_distance(torch.zeros(0, 3), _bag(4, 3))


def test_response_loss_reports_both_terms() -> None:
    loss_fn = ResponseLoss()
    total, parts = loss_fn(_bag(8, 3), _bag(9, 3, offset=1.0), torch.zeros(3))
    assert {"mean_delta_mse", "energy_distance", "loss"} <= set(parts)
    assert float(total) == pytest.approx(parts["loss"])


def test_response_loss_weights_reject_an_all_zero_objective() -> None:
    with pytest.raises(ValueError, match="at least one loss term"):
        ResponseLossWeights(mean_delta=0.0, energy=0.0)


# --- held-out gene split ------------------------------------------------


def test_heldout_genes_are_split_per_line_and_are_deterministic() -> None:
    genes_by_line = {
        "ACH-000551": [f"G{i}" for i in range(20)],
        "ACH-000995": [f"G{i}" for i in range(20)],
    }
    first = split_heldout_genes(genes_by_line, fraction=0.2, seed=1)
    again = split_heldout_genes(genes_by_line, fraction=0.2, seed=1)
    assert first == again
    assert len(first["ACH-000551"]) == 4
    # Same gene names, different lines: the split is per line, so the two
    # held-out sets are chosen independently rather than mirrored.
    assert first["ACH-000551"] != first["ACH-000995"]


def test_heldout_split_is_stable_when_a_gene_is_added() -> None:
    """Adding a gene must not reshuffle the existing assignment."""
    base = {"L": [f"G{i}" for i in range(50)]}
    grown = {"L": [f"G{i}" for i in range(51)]}
    kept = split_heldout_genes(base, fraction=0.2, seed=3)["L"]
    after = split_heldout_genes(grown, fraction=0.2, seed=3)["L"]
    assert len(kept & after) >= len(kept) - 1


def test_heldout_split_rejects_a_line_too_small_to_hold_out() -> None:
    with pytest.raises(ValueError, match="cannot yield a held-out set"):
        split_heldout_genes({"L": ["A", "B"]}, fraction=0.2, seed=1)


# --- training loop ------------------------------------------------------


def _batches(dim: int = 3, n: int = 4) -> list[dict]:
    out = []
    for index in range(n):
        control = _bag(16, dim)
        out.append(
            {
                "gene": f"G{index}",
                "model_id": "ACH-000551",
                "control": control,
                "control_target": control,
                "observed": control + 1.5,
            }
        )
    return out


def test_training_reduces_the_loss_and_updates_the_module() -> None:
    model = _ShiftModel(3)
    before = model.shift.detach().clone()
    report = train_response_model(
        model,
        _batches(),
        config=TrainingConfig(epochs=6, learning_rate=0.2, max_bag=16, log_every=0),
    )
    losses = [epoch["loss"] for epoch in report["epochs"]]
    assert losses[-1] < losses[0]
    assert not torch.allclose(before, model.shift.detach())


def test_training_rejects_an_empty_batch_list() -> None:
    with pytest.raises(ValueError, match="no response batches"):
        train_response_model(_ShiftModel(3), [])


def test_evaluate_reports_the_basal_copy_floor() -> None:
    """A trained shift must beat 'nothing happens', and the report must say so."""
    model = _ShiftModel(3)
    batches = _batches()
    train_response_model(
        model,
        batches,
        config=TrainingConfig(epochs=8, learning_rate=0.2, max_bag=16, log_every=0),
    )
    report = evaluate_response_model(model, batches)
    assert report["basal_copy_loss"] > report["model_loss"]
    assert report["improvement_over_basal_copy"] > 0
    assert "ACH-000551" in report["per_line_model_loss"]
    assert np.isfinite(report["model_loss"])
