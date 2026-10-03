"""Strict selective-gene-Spearman-only checkpoint selection."""

from dataclasses import asdict
import math

import pytest

from src.training.checkpoint import TrainState, record_validation


def test_only_selective_spearman_controls_selection():
    state = TrainState()
    assert state.best_score == -math.inf
    assert record_validation(state, {"val_selective_spearman": -0.2}, 0)
    assert not record_validation(
        state,
        {
            "val_selective_spearman": -0.3,
            "val_geneeffect_loss": 0.0,
            "val_residual_spearman_macro_per_gene": 0.99,
        },
        1,
    )
    assert state.best_epoch == 0 and state.bad_epochs == 1
    assert record_validation(
        state, {"val_selective_spearman": 0.1, "val_geneeffect_loss": 9.0}, 2
    )
    assert state.best_epoch == 2 and state.bad_epochs == 0
    assert state.best_score == 0.1
    assert not record_validation(state, {"val_selective_spearman": 0.1}, 3)
    assert state.bad_epochs == 1 and state.next_epoch == 4


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf])
def test_undefined_selector_does_not_mutate_state(score):
    state = TrainState(global_step=40, best_score=0.4, best_epoch=2)
    before = asdict(state)
    with pytest.raises(ValueError, match="must be finite"):
        record_validation(state, {"val_selective_spearman": score}, 3)
    assert asdict(state) == before


def test_missing_selector_cannot_use_other_metric():
    with pytest.raises(KeyError, match="val_selective_spearman"):
        record_validation(TrainState(), {"val_geneeffect_loss": 0.1}, 0)
