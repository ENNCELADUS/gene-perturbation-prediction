"""Strict GeneEffect-loss-only checkpoint selection."""

from dataclasses import asdict
import math

import pytest

from src.training.checkpoint import TrainState, record_validation


def test_only_geneeffect_loss_controls_selection():
    state = TrainState()
    assert record_validation(state, {"val_geneeffect_loss": 0.4}, 0)
    assert not record_validation(
        state,
        {
            "val_geneeffect_loss": 0.5,
            "val_total_loss": 0.1,
            "val_residual_spearman_macro_per_gene": 0.99,
        },
        1,
    )
    assert state.best_epoch == 0 and state.bad_epochs == 1
    assert record_validation(
        state,
        {
            "val_geneeffect_loss": 0.3,
            "val_total_loss": 9.0,
        },
        2,
    )
    assert state.best_epoch == 2 and state.bad_epochs == 0
    assert not record_validation(state, {"val_geneeffect_loss": 0.3}, 3)
    assert state.bad_epochs == 1 and state.next_epoch == 4


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf])
def test_nonfinite_selector_does_not_mutate_state(loss):
    state = TrainState(global_step=40, best_loss=0.4, best_epoch=2)
    before = asdict(state)
    with pytest.raises(ValueError, match="must be finite"):
        record_validation(state, {"val_geneeffect_loss": loss}, 3)
    assert asdict(state) == before


def test_missing_selection_loss_cannot_use_other_metric():
    with pytest.raises(KeyError, match="val_geneeffect_loss"):
        record_validation(TrainState(), {"val_total_loss": 0.1}, 0)
