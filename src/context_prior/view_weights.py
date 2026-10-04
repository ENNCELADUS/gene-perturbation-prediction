"""Gene-conditioned view weights: the prior's only gradient-trained candidate.

The context stages' predictions are mixed per gene with weights
``V * softmax(M e_g + b)`` over the V views, so uniform weights reproduce the
plain stagewise sum. Genes without an embedding keep uniform weights. Trained on
out-of-fold stage predictions; other stages pass through unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ViewWeights:
    blocks: tuple[str, ...]
    weights: pd.DataFrame  # genes x blocks

    def combine(self, stage_predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        out = None
        for block, frame in stage_predictions.items():
            scaled = frame
            if block in self.blocks:
                scaled = (
                    frame * self.weights.loc[frame.columns, block].to_numpy()[None, :]
                )
            out = scaled if out is None else out + scaled
        return out


def fit_view_weights(
    stage_predictions: Mapping[str, pd.DataFrame],
    residual: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    blocks: Sequence[str],
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> ViewWeights:
    torch.manual_seed(seed)
    genes = list(residual.columns)
    lines = list(residual.index)
    views = torch.tensor(
        np.stack([stage_predictions[b].loc[lines, genes].to_numpy() for b in blocks]),
        dtype=torch.float32,
    )  # views x lines x genes
    target = residual.to_numpy(dtype=np.float64)
    observed = torch.tensor(np.isfinite(target))
    target = torch.tensor(
        np.where(np.isfinite(target), target, 0.0), dtype=torch.float32
    )
    width = len(next(iter(embeddings.values())))
    has = torch.tensor([g in embeddings for g in genes])
    embedding = torch.tensor(
        np.stack([embeddings.get(g, np.zeros(width)) for g in genes]),
        dtype=torch.float32,
    )
    mixer = torch.nn.Linear(width, len(blocks))
    torch.nn.init.zeros_(mixer.weight)
    torch.nn.init.zeros_(mixer.bias)
    optimizer = torch.optim.Adam(mixer.parameters(), lr=learning_rate)

    def weights() -> torch.Tensor:
        learned = len(blocks) * torch.softmax(mixer(embedding), dim=1)
        return torch.where(has[:, None], learned, torch.ones_like(learned))

    for _ in range(steps):
        optimizer.zero_grad()
        prediction = torch.einsum("vng,gv->ng", views, weights())
        loss = ((prediction - target) ** 2)[observed].mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        values = weights().numpy()
    return ViewWeights(
        tuple(blocks), pd.DataFrame(values, index=genes, columns=list(blocks))
    )
