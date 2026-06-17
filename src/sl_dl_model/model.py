"""End-to-end SL-pair model: frozen-STATE encoder + pooling + pair head.

The model composes three components:
- :class:`~sl_dl_model.encoder.StateEncoder`: ESM2->adapter->frozen STATE,
  maps (esm_vec, control_cells) to a predicted response bag.
- A pooling head from :func:`~sl_dl_model.pooling.build_pool`: bag -> e_g.
- :class:`~sl_dl_model.pair_head.SymmetricPairHead`: (e_a, e_b, ge) -> logit.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sl_dl_model.encoder import StateEncoder
from sl_dl_model.pair_head import SymmetricPairHead
from sl_dl_model.pooling import build_pool
from sl_dl_model.pooling import output_dim as _output_dim


def output_dim_fn(pooling: str, dim: int) -> int:
    """Return the pooled embedding dimension for a given pooling strategy.

    Wraps :func:`~sl_dl_model.pooling.output_dim` so callers can import a
    single helper without pulling the full pooling module.

    Args:
        pooling: Pooling strategy name (e.g. ``"mean_std"``).
        dim: STATE encoder output dimension.

    Returns:
        Pooled embedding dimensionality.
    """
    return _output_dim(pooling, dim)


class SlDlModel(nn.Module):
    """Frozen-STATE DL model for SL-pair ranking.

    Encodes each gene as e_g = pool(STATE(adapter(esm_vec), control_cells)),
    then scores gene pairs with a symmetric pair head.

    Args:
        backend: ``"linear_mock"`` for tests; ``"state_checkpoint"`` otherwise.
        checkpoint: Path to the STATE checkpoint (required unless mock backend).
        esm_dim: Dimensionality of the input ESM2 embedding.
        adapter_hidden: Hidden width of the PertAdapter MLP.
        pert_dim: Pert token dimensionality (must match STATE's expectation).
        input_dim: Control cell feature dimension fed to STATE.
        output_dim: STATE encoder output dimension (= input_dim for the HVG
            checkpoint).
        pooling: Pooling strategy name passed to :func:`build_pool`.
        pair_hidden: Hidden layer widths for the :class:`SymmetricPairHead`.
        include_coverage_flag: Whether to pass coverage flags to the pair head.
    """

    def __init__(
        self,
        *,
        backend: str,
        checkpoint: Path | None,
        esm_dim: int,
        adapter_hidden: int,
        pert_dim: int,
        input_dim: int,
        output_dim: int,
        pooling: str = "mean_std",
        pair_hidden: tuple[int, ...] = (256, 64),
        include_coverage_flag: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            backend=backend,
            checkpoint=checkpoint,
            esm_dim=esm_dim,
            adapter_hidden=adapter_hidden,
            pert_dim=pert_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.pool = build_pool(pooling, output_dim)
        self.emb_dim: int = output_dim_fn(pooling, output_dim)
        self.pair_head = SymmetricPairHead(
            emb_dim=self.emb_dim,
            hidden=pair_hidden,
            include_coverage_flag=include_coverage_flag,
        )

    def embed_gene(
        self,
        esm_vec: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """Produce a per-gene embedding e_g from ESM2 + control cells.

        Args:
            esm_vec: ESM2 embedding for the perturbation gene, shape
                ``(esm_dim,)``.
            control: Control cell embeddings, shape ``(T, input_dim)``.

        Returns:
            Gene embedding of shape ``(emb_dim,)``.
        """
        bag = self.encoder(esm_vec, control)
        return self.pool(bag)

    def score_pairs(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
        ge_features: torch.Tensor,
        cov_a: torch.Tensor | None = None,
        cov_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of gene pairs using the symmetric pair head.

        Args:
            e_a: Per-gene embeddings for gene A, shape ``(B, emb_dim)``.
            e_b: Per-gene embeddings for gene B, shape ``(B, emb_dim)``.
            ge_features: GeneEffect feature block, shape ``(B, 5)``.
            cov_a: Coverage flag for gene A, shape ``(B,)``.  Required when
                ``include_coverage_flag=True``.
            cov_b: Coverage flag for gene B, shape ``(B,)``.  Required when
                ``include_coverage_flag=True``.

        Returns:
            Logits of shape ``(B,)``.
        """
        return self.pair_head(e_a, e_b, ge_features, cov_a, cov_b)
