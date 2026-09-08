"""Head-only readouts for the fixed-backbone P1-A diagnostic."""

from copy import deepcopy

import torch
from torch import nn

from src.data.batches import FeatureBatch
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectResidualHead,
)

ARMS = ("A0", "A1", "A2", "A3")


class FixedReadout(nn.Module):
    """Consume standardized features and optionally add per-gene PCA slopes."""

    def __init__(self, mlp: GeneEffectResidualHead, n_genes: int, explicit: bool):
        super().__init__()
        self.mlp = mlp
        self.slopes = nn.Parameter(torch.zeros(n_genes, 8)) if explicit else None

    def forward(self, features: FeatureBatch, gene_indices, contexts):
        response = self.mlp.blocks.use_s
        prediction = self.mlp(
            delta_proj=features.delta_proj if response else None,
            s=features.s if response else None,
            hvg_panel_mask=features.hvg_panel_mask if response else None,
            own_gene_shift_mask=features.own_gene_shift_mask if response else None,
            q_sc=features.q_sc,
            q_sc_mask=features.q_sc_mask,
            e_g=features.e_g,
            z_c=features.z_c,
        )
        if self.slopes is not None:
            prediction = prediction + (self.slopes[gene_indices] * contexts).sum(-1)
        return prediction

    def regularization(self):
        """Sum over eight slopes, mean over all covered genes, lambda 0.01."""
        if self.slopes is None:
            return next(self.parameters()).new_zeros(())
        return 0.01 * self.slopes.square().sum(dim=1).mean()


def make_readout(
    arm: str, dims: GeneEffectFeatureDims, n_genes: int, *, seed: int = 0
) -> FixedReadout:
    """Seeded canonical MLP; copy common columns when removing the response block."""
    if arm not in ARMS or n_genes < 1:
        raise ValueError("require A0-A3 and at least one training-covered gene")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = GeneEffectResidualHead(dims)
        if arm in {"A1", "A3"}:
            mlp = canonical
        else:
            mlp = GeneEffectResidualHead(
                dims, GeneEffectBlockConfig(use_delta_proj=False, use_s=False)
            )
            state = deepcopy(canonical.state_dict())
            offset = dims.delta_proj + dims.s + 2
            state["net.0.weight"] = state["net.0.weight"][:, offset:].clone()
            mlp.load_state_dict(state)
    return FixedReadout(mlp, n_genes, arm in {"A2", "A3"})
