"""GeneEffect regression loss."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def geneeffect_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean FP32 Huber loss (delta 1) over labelled GeneEffect residuals."""
    return F.huber_loss(prediction.float(), target.float(), delta=1.0)
