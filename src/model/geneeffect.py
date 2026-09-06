"""model / geneeffect."""

from __future__ import annotations

from typing import Mapping, Sequence
import torch
from torch import nn
from src.model.normalization import BlockStandardizer
from src.model.features import (
    ConditionFeatures,
    FixedSparseProjection,
    compute_condition_features,
)
from src.model.head import GeneEffectResidualHead
from src.model.response import predict_bags
from src.data.batches import E2EForwardOutput
from src.data.batches import OnlineConditionBatch
from src.data.batches import PrecomputedFeatureBatch
from src.data.batches import ResponseForwardBatch


class GeneEffectE2EModel(nn.Module):
    """Compose a Stage-1 backbone and the five-block residual head."""

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
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def freeze_backbone(self) -> None:
        """Freeze Stage 1 and keep it in eval mode during head warmup."""
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Enable joint Stage-2 optimization of every learned backbone weight."""
        self.backbone.requires_grad_(True)
        self._backbone_frozen = False
        self.backbone.train(self.training)

    def train(self, mode: bool = True) -> GeneEffectE2EModel:
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def assert_frozen_backbone_clean(self) -> None:
        """Fail if a warmup step enabled or accumulated a backbone gradient."""
        offenders = [
            name
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad or parameter.grad is not None
        ]
        if offenders:
            raise RuntimeError(
                "frozen backbone contains trainable parameters or gradients: "
                f"{offenders[:10]}"
            )

    def _standardize(
        self, features: PrecomputedFeatureBatch
    ) -> dict[str, torch.Tensor]:
        features.validate()
        return {
            name: self.standardizer.transform(name, value)
            for name, value in (
                ("delta_proj", features.delta_proj),
                ("s", features.s),
                ("q_sc", features.q_sc),
                ("e_g", features.e_g),
                ("z_c", features.z_c),
            )
        }

    def forward_precomputed(self, features: PrecomputedFeatureBatch) -> torch.Tensor:
        """Run the head on raw cached features during frozen warmup."""
        blocks = self._standardize(features)
        return self.head(
            **blocks,
            q_sc_mask=features.q_sc_mask,
            hvg_panel_mask=features.hvg_panel_mask,
            own_gene_shift_mask=features.own_gene_shift_mask,
        )

    def forward(
        self,
        batch: OnlineConditionBatch,
        response: ResponseForwardBatch | None = None,
    ) -> E2EForwardOutput:
        """Generate dependency and optional response-anchor outputs in one call."""
        batch.validate()
        predicted = predict_bags(
            self.backbone,
            batch.controls_tx1,
            batch.genes,
            seed=self.collator_seed,
        )
        built: list[ConditionFeatures] = []
        for position, (predicted_bag, basal_bag) in enumerate(
            zip(predicted, batch.basal_hvg, strict=True)
        ):
            built.append(
                compute_condition_features(
                    predicted_bag.float(),
                    basal_bag.float(),
                    projection=self.projection,
                    gene_in_hvg_panel=bool(batch.gene_in_hvg_panel[position]),
                    own_gene_hvg_index=batch.own_gene_hvg_indices[position],
                    own_gene_available=bool(batch.own_gene_shift_available[position]),
                )
            )
        features = PrecomputedFeatureBatch(
            delta_proj=torch.stack([item.delta_proj for item in built]),
            s=torch.stack([item.s for item in built]),
            q_sc=batch.q_sc,
            e_g=batch.e_g,
            z_c=batch.z_c,
            q_sc_mask=batch.q_sc_mask,
            hvg_panel_mask=torch.stack([item.hvg_panel_mask for item in built]),
            own_gene_shift_mask=torch.stack(
                [item.own_gene_shift_mask for item in built]
            ),
            gene_symbols=batch.genes,
            model_ids=batch.model_ids,
        )
        response_predicted = None
        if response is not None:
            response.validate()
            response_predicted = predict_bags(
                self.backbone,
                response.controls_tx1,
                response.genes,
                seed=self.collator_seed,
            )
        return E2EForwardOutput(
            delta_hat=self.forward_precomputed(features),
            raw_features=features,
            feature_metadata=tuple(item.metadata for item in built),
            response_predicted=response_predicted,
        )

    @staticmethod
    def add_train_gene_mean(
        genes: Sequence[str],
        delta_hat: torch.Tensor,
        mu_train: Mapping[str, float],
    ) -> torch.Tensor:
        """Return absolute GeneEffect while failing on an unregistered gene."""
        if delta_hat.shape != (len(genes),):
            raise ValueError("delta_hat must align one-to-one with genes")
        missing = [str(gene) for gene in genes if str(gene) not in mu_train]
        if missing:
            raise KeyError(f"genes absent from train-only mu_g: {missing[:10]}")
        mean = torch.as_tensor(
            [mu_train[str(gene)] for gene in genes],
            dtype=delta_hat.dtype,
            device=delta_hat.device,
        )
        return delta_hat + mean
