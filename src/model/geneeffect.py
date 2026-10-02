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
from src.model.head import GeneEffectResidualHead
from src.model.normalization import BlockStandardizer
from src.model.response import predict_bags

BLOCKS = ("delta_proj", "s", "q_sc", "e_g", "z_c")


class GeneEffectE2EModel(nn.Module):
    """A trainable STATE response model and the five-block residual head.

    STATE sees only each line's log-space basal HVG cells; the Tx1 context
    ``z_c`` reaches the head directly.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: GeneEffectResidualHead,
        projection: FixedSparseProjection,
        standardizer: BlockStandardizer,
        *,
        collator_seed: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.projection = projection
        self.standardizer = standardizer
        self.collator_seed = int(collator_seed)

    def forward_features(self, features: FeatureBatch) -> torch.Tensor:
        """Standardize the enabled blocks and predict the GeneEffect residual."""
        blocks = self.head.blocks
        return self.head(
            **{
                name: self.standardizer.transform(name, getattr(features, name))
                for name in BLOCKS
                if getattr(blocks, f"use_{name}")
            },
            q_sc_mask=features.q_sc_mask if blocks.use_q_sc else None,
            hvg_panel_mask=features.hvg_panel_mask if blocks.use_s else None,
            own_gene_shift_mask=features.own_gene_shift_mask if blocks.use_s else None,
        )

    def condition_features(self, batch: OnlineConditionBatch) -> FeatureBatch:
        """All five raw blocks, keeping the STATE graph for backpropagation."""
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
        return FeatureBatch(
            delta_proj=built.delta_proj,
            s=built.s,
            q_sc=batch.q_sc,
            e_g=batch.e_g,
            z_c=batch.z_c,
            q_sc_mask=batch.q_sc_mask,
            hvg_panel_mask=built.hvg_panel_mask,
            own_gene_shift_mask=built.own_gene_shift_mask,
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
        delta_hat = self.forward_features(self.condition_features(batch))
        response_predicted = None
        if response is not None:
            response_predicted = predict_bags(
                self.backbone,
                tuple(value.float() for value in response.basal_hvg),
                response.genes,
                seed=self.collator_seed,
            )
        return E2EForwardOutput(delta_hat, response_predicted)
