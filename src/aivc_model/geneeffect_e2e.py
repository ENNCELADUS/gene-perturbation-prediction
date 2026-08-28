"""End-to-end Stage-1 response backbone plus GeneEffect residual head.

The model boundary starts from cached per-cell Tx1 embeddings and the matched
2000-HVG basal view.  Raw-count to Tx1 encoding remains the input-preparation
stage.  No cell-line identity enters this module: a context is represented only
by its basal bags, ``q_sc`` values, and moment-pooled Tx1 context vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from aivc_model.geneeffect_features import (
    BlockStandardizer,
    ConditionFeatures,
    FixedSparseProjection,
    compute_condition_features,
)
from aivc_model.geneeffect_head import GeneEffectResidualHead
from aivc_model.response_training import predict_bags


@dataclass(frozen=True)
class PrecomputedFeatureBatch:
    """Raw, unstandardized head features for frozen-backbone warmup."""

    delta_proj: torch.Tensor
    s: torch.Tensor
    q_sc: torch.Tensor
    e_g: torch.Tensor
    z_c: torch.Tensor
    q_sc_mask: torch.Tensor
    hvg_panel_mask: torch.Tensor
    own_gene_shift_mask: torch.Tensor
    gene_symbols: tuple[str, ...]
    model_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.delta_proj.shape[0])

    def validate(self) -> None:
        if (
            len(self.gene_symbols) != self.batch_size
            or len(self.model_ids) != self.batch_size
        ):
            raise ValueError("feature identities must align with batch_size")
        if any(not value for value in (*self.gene_symbols, *self.model_ids)):
            raise ValueError("feature identities cannot contain empty strings")
        blocks = {
            "delta_proj": self.delta_proj,
            "s": self.s,
            "q_sc": self.q_sc,
            "e_g": self.e_g,
            "z_c": self.z_c,
        }
        for name, value in blocks.items():
            if value.dim() != 2 or value.shape[0] != self.batch_size:
                raise ValueError(
                    f"{name} must be 2-D with batch={self.batch_size}, got "
                    f"{tuple(value.shape)}"
                )
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point")
        for name, value in (
            ("q_sc_mask", self.q_sc_mask),
            ("hvg_panel_mask", self.hvg_panel_mask),
            ("own_gene_shift_mask", self.own_gene_shift_mask),
        ):
            if value.shape != (self.batch_size,) or value.dtype != torch.bool:
                raise ValueError(
                    f"{name} must be boolean [{self.batch_size}], got "
                    f"shape={tuple(value.shape)} dtype={value.dtype}"
                )


@dataclass(frozen=True)
class OnlineConditionBatch:
    """Inputs for differentiable Stage-2 response-feature generation."""

    controls_tx1: tuple[torch.Tensor, ...]
    basal_hvg: tuple[torch.Tensor, ...]
    genes: tuple[str, ...]
    model_ids: tuple[str, ...]
    q_sc: torch.Tensor
    e_g: torch.Tensor
    z_c: torch.Tensor
    q_sc_mask: torch.Tensor
    gene_in_hvg_panel: torch.Tensor
    own_gene_hvg_indices: tuple[int | None, ...]
    own_gene_shift_available: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.genes)

    def validate(self) -> None:
        size = self.batch_size
        if size == 0:
            raise ValueError("online condition batch cannot be empty")
        if not (
            len(self.controls_tx1)
            == len(self.basal_hvg)
            == len(self.own_gene_hvg_indices)
            == len(self.model_ids)
            == size
        ):
            raise ValueError("online condition sequence fields must have equal length")
        if any(not value for value in (*self.genes, *self.model_ids)):
            raise ValueError("online condition identities cannot be empty")
        for name, values in (
            ("controls_tx1", self.controls_tx1),
            ("basal_hvg", self.basal_hvg),
        ):
            for value in values:
                if value.dim() != 2 or value.shape[0] == 0:
                    raise ValueError(f"every {name} bag must be non-empty and 2-D")
                if not value.is_floating_point() or not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError(f"every {name} bag must be finite floating point")
        for name, value in (
            ("q_sc", self.q_sc),
            ("e_g", self.e_g),
            ("z_c", self.z_c),
        ):
            if value.dim() != 2 or value.shape[0] != size:
                raise ValueError(f"{name} must be 2-D with batch={size}")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point")
        for name, value in (
            ("q_sc_mask", self.q_sc_mask),
            ("gene_in_hvg_panel", self.gene_in_hvg_panel),
            ("own_gene_shift_available", self.own_gene_shift_available),
        ):
            if value.shape != (size,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean [{size}]")
        for position, (in_panel, index, available) in enumerate(
            zip(
                self.gene_in_hvg_panel.tolist(),
                self.own_gene_hvg_indices,
                self.own_gene_shift_available.tolist(),
                strict=True,
            )
        ):
            if not in_panel and (index is not None or available):
                raise ValueError(
                    f"condition {position}: a non-HVG gene cannot have an own-gene "
                    "index or shift"
                )
            if in_panel and index is None:
                raise ValueError(
                    f"condition {position}: an HVG-panel gene requires its index"
                )
            if available and index is None:
                raise ValueError(
                    f"condition {position}: available own-gene shift requires an index"
                )


@dataclass(frozen=True)
class ResponseForwardBatch:
    """Response-anchor inputs carried through the same DDP forward call."""

    controls_tx1: tuple[torch.Tensor, ...]
    genes: tuple[str, ...]

    def validate(self) -> None:
        if not self.genes or len(self.controls_tx1) != len(self.genes):
            raise ValueError("response forward controls and genes must align")
        for control in self.controls_tx1:
            if control.dim() != 2 or control.shape[0] == 0:
                raise ValueError("response controls must be non-empty 2-D bags")
            if not control.is_floating_point() or not bool(
                torch.isfinite(control).all()
            ):
                raise ValueError("response controls must be finite floating point")


@dataclass(frozen=True)
class E2EForwardOutput:
    delta_hat: torch.Tensor
    raw_features: PrecomputedFeatureBatch
    feature_metadata: tuple[Mapping[str, object], ...]
    response_predicted: tuple[torch.Tensor, ...] | None = None


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
