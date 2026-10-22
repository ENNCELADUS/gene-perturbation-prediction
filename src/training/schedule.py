"""The learning-rate schedule shared by the joint and stage-one trainers, apart from
``src.training.trainer`` (which imports accelerate) so ``.venv-pie`` can import it."""

from collections.abc import Mapping
import math
from typing import Any

import torch


def make_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any], updates_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_epochs`` epochs of updates, then cosine to zero.

    Stepped once per update: the last warmup update runs at the full rate and the
    rate reaches zero at the end of ``max_epochs``.
    """
    train = config["train"]
    warmup = train["warmup_epochs"] * updates_per_epoch
    decay = max(1, train["max_epochs"] * updates_per_epoch - warmup)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / decay)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
