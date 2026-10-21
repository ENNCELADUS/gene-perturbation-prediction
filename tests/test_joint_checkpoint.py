"""Strict single-metric checkpoint selection on the named selector."""

from dataclasses import asdict
import math

import pytest

from src.training.checkpoint import TrainState, record_validation

KEY = "val_selective_spearman"


def test_only_selective_spearman_controls_selection():
    state = TrainState()
    assert state.best_score == -math.inf
    assert record_validation(state, {"val_selective_spearman": -0.2}, 0, key=KEY)
    assert not record_validation(
        state,
        {
            "val_selective_spearman": -0.3,
            "val_geneeffect_loss": 0.0,
            "val_residual_spearman_macro_per_gene": 0.99,
        },
        1,
        key=KEY,
    )
    assert state.best_epoch == 0 and state.bad_epochs == 1
    assert record_validation(
        state, {"val_selective_spearman": 0.1, "val_geneeffect_loss": 9.0}, 2, key=KEY
    )
    assert state.best_epoch == 2 and state.bad_epochs == 0
    assert state.best_score == 0.1
    assert not record_validation(state, {"val_selective_spearman": 0.1}, 3, key=KEY)
    assert state.bad_epochs == 1 and state.next_epoch == 4


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf])
def test_undefined_selector_does_not_mutate_state(score):
    state = TrainState(global_step=40, best_score=0.4, best_epoch=2)
    before = asdict(state)
    with pytest.raises(ValueError, match="must be finite"):
        record_validation(state, {"val_selective_spearman": score}, 3, key=KEY)
    assert asdict(state) == before


def test_missing_selector_cannot_use_other_metric():
    with pytest.raises(KeyError, match="val_selective_spearman"):
        record_validation(TrainState(), {"val_geneeffect_loss": 0.1}, 0, key=KEY)


def test_the_selector_is_the_key_named():
    """Stage one selects on the per-line residual Spearman; the selective
    Spearman beside it decides nothing."""
    state = TrainState()
    assert record_validation(
        state,
        {"val_line_spearman": 0.1, "val_selective_spearman": 0.9},
        0,
        key="val_line_spearman",
    )
    assert not record_validation(
        state,
        {"val_line_spearman": 0.05, "val_selective_spearman": 0.99},
        1,
        key="val_line_spearman",
    )
    assert state.best_epoch == 0 and state.best_score == 0.1
