"""Epoch-boundary training state and GeneEffect-loss checkpoint selection."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


@dataclass
class TrainState:
    next_epoch: int = 0
    global_step: int = 0
    best_loss: float = math.inf
    best_epoch: int = -1
    bad_epochs: int = 0


def record_validation(
    state: TrainState, metrics: Mapping[str, Any], epoch: int
) -> bool:
    """Record a completed epoch; only a strict GeneEffect-loss decrease wins."""
    loss = float(metrics["val_geneeffect_loss"])
    if not math.isfinite(loss):
        raise ValueError("val_geneeffect_loss must be finite")
    improved = loss < state.best_loss
    if improved:
        state.best_loss, state.best_epoch, state.bad_epochs = loss, epoch, 0
    else:
        state.bad_epochs += 1
    state.next_epoch = epoch + 1
    return improved
