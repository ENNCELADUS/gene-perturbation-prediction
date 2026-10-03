"""Joint GeneEffect model: STATE response features feeding the residual head."""

from __future__ import annotations

import torch
from torch import nn

from src.data.batches import (
    E2EForwardOutput,
    FeatureBatch,
    OnlineConditionBatch,
    ResponseForwardBatch,
)
from src.model.features import FixedSparseProjection, compute_condition_feature_batch
from src.model.head import GeneEffectNestedHead
from src.model.normalization import STANDARDIZED_BLOCKS, BlockStandardizer
from src.model.response import predict_bags


class GeneEffectE2EModel(nn.Module):
    """A STATE response model and the nested low-rank residual head.

    STATE sees only each line's log-space basal HVG cells; the Tx1 context
    ``z_c`` (eigen-scaled context PCA scores) reaches the head directly and is
    not standardized. With both response blocks (``delta_proj``
    and ``s``) disabled, STATE is never called. The head predicts in units of
    the per-gene training residual SD ``residual_scale``; :meth:`forward`
    multiplies it back, so ``delta_hat`` is in residual units.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: GeneEffectNestedHead,
        projection: FixedSparseProjection,
        standardizer: BlockStandardizer,
        *,
        collator_seed: int,
        residual_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if tuple(residual_scale.shape) != (head.n_genes,):
            raise ValueError(
                f"residual_scale must be shaped ({head.n_genes},), got "
                f"{tuple(residual_scale.shape)}"
            )
        if not bool(
            torch.isfinite(residual_scale).all() and (residual_scale > 0).all()
        ):
            raise ValueError("residual_scale must be finite and positive")
        self.backbone = backbone
        self.head = head
        self.projection = projection
        self.standardizer = standardizer
        self.collator_seed = int(collator_seed)
        self.register_buffer(
            "residual_scale", residual_scale.detach().to(torch.float32).clone()
        )

    @property
    def uses_state(self) -> bool:
        """Whether any head block is computed from STATE's predicted response."""
        return self.head.blocks.use_delta_proj or self.head.blocks.use_s

    def forward_features(
        self, features: FeatureBatch, gene_index: torch.Tensor
    ) -> torch.Tensor:
        """Standardize the enabled blocks except ``z_c``; the head's output in
        residual-SD units."""
        blocks = self.head.blocks
        return self.head(
            gene_index=gene_index,
            **{
                name: self.standardizer.transform(name, getattr(features, name))
                for name in STANDARDIZED_BLOCKS
                if getattr(blocks, f"use_{name}")
            },
            z_c=features.z_c,
            q_sc_mask=features.q_sc_mask if blocks.use_q_sc else None,
            hvg_panel_mask=features.hvg_panel_mask if blocks.use_s else None,
            own_gene_shift_mask=features.own_gene_shift_mask if blocks.use_s else None,
        )

    def condition_features(self, batch: OnlineConditionBatch) -> FeatureBatch:
        """The raw blocks, keeping the STATE graph for backpropagation.

        Without STATE the response blocks and their masks are ``None``.
        """
        state_blocks: dict[str, torch.Tensor | None] = dict.fromkeys(
            ("delta_proj", "s", "hvg_panel_mask", "own_gene_shift_mask")
        )
        if self.uses_state:
            basal = tuple(value.float() for value in batch.basal_hvg)
            predicted = predict_bags(
                self.backbone, basal, batch.genes, seed=self.collator_seed
            )
            built = compute_condition_feature_batch(
                predicted,
                basal,
                projection=self.projection,
                gene_in_hvg_panel=batch.gene_in_hvg_panel,
                own_gene_hvg_indices=batch.own_gene_hvg_indices,
                own_gene_available=batch.own_gene_shift_available,
            )
            state_blocks = {name: getattr(built, name) for name in state_blocks}
        return FeatureBatch(
            **state_blocks,
            q_sc=batch.q_sc,
            e_g=batch.e_g,
            z_c=batch.z_c,
            q_sc_mask=batch.q_sc_mask,
            gene_symbols=batch.genes,
            model_ids=batch.model_ids,
        )

    def forward(
        self,
        batch: OnlineConditionBatch,
        response: ResponseForwardBatch | None = None,
    ) -> E2EForwardOutput:
        """GeneEffect residuals and, on replay updates, response predictions.

        Both go through one module call so DDP synchronises every gradient.
        """
        if response is not None and not self.uses_state:
            raise ValueError("response replay needs a model that uses STATE")
        standardized = self.forward_features(
            self.condition_features(batch), batch.gene_index
        )
        delta_hat = self.residual_scale[batch.gene_index] * standardized
        response_predicted = None
        if response is not None:
            response_predicted = predict_bags(
                self.backbone,
                tuple(value.float() for value in response.basal_hvg),
                response.genes,
                seed=self.collator_seed,
            )
        return E2EForwardOutput(delta_hat, response_predicted)
