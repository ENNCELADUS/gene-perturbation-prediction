"""data / batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class FeatureBatch:
    """Raw, unstandardized live head features."""

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

    metadata: tuple[Mapping[str, object], ...] = ()

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

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> FeatureBatch:
        """Move tensor fields while preserving row identifiers."""
        return FeatureBatch(
            delta_proj=self.delta_proj.to(device, non_blocking=non_blocking),
            s=self.s.to(device, non_blocking=non_blocking),
            q_sc=self.q_sc.to(device, non_blocking=non_blocking),
            e_g=self.e_g.to(device, non_blocking=non_blocking),
            z_c=self.z_c.to(device, non_blocking=non_blocking),
            q_sc_mask=self.q_sc_mask.to(device, non_blocking=non_blocking),
            hvg_panel_mask=self.hvg_panel_mask.to(device, non_blocking=non_blocking),
            own_gene_shift_mask=self.own_gene_shift_mask.to(
                device, non_blocking=non_blocking
            ),
            gene_symbols=self.gene_symbols,
            model_ids=self.model_ids,
            metadata=self.metadata,
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

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> OnlineConditionBatch:
        """Move every tensor field while preserving condition identifiers."""
        return OnlineConditionBatch(
            controls_tx1=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.controls_tx1
            ),
            basal_hvg=tuple(
                value.to(device, non_blocking=non_blocking) for value in self.basal_hvg
            ),
            genes=self.genes,
            model_ids=self.model_ids,
            q_sc=self.q_sc.to(device, non_blocking=non_blocking),
            e_g=self.e_g.to(device, non_blocking=non_blocking),
            z_c=self.z_c.to(device, non_blocking=non_blocking),
            q_sc_mask=self.q_sc_mask.to(device, non_blocking=non_blocking),
            gene_in_hvg_panel=self.gene_in_hvg_panel.to(
                device, non_blocking=non_blocking
            ),
            own_gene_hvg_indices=self.own_gene_hvg_indices,
            own_gene_shift_available=self.own_gene_shift_available.to(
                device, non_blocking=non_blocking
            ),
        )


@dataclass(frozen=True)
class DependencyBatch:
    """One finite GeneEffect minibatch and its live model inputs."""

    conditions: OnlineConditionBatch
    residual: torch.Tensor
    gene_mean: torch.Tensor
    valid: torch.Tensor

    def validate(self) -> None:
        self.conditions.validate()
        size = self.conditions.batch_size
        for name, value in (
            ("residual", self.residual),
            ("gene_mean", self.gene_mean),
        ):
            if value.shape != (size,) or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point [{size}]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if self.valid.shape != (size,) or self.valid.dtype != torch.bool:
            raise ValueError(f"valid must be boolean [{size}]")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> DependencyBatch:
        """Move nested tensors while preserving genes and ModelIDs."""
        return DependencyBatch(
            conditions=self.conditions.to(device, non_blocking=non_blocking),
            residual=self.residual.to(device, non_blocking=non_blocking),
            gene_mean=self.gene_mean.to(device, non_blocking=non_blocking),
            valid=self.valid.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ResponseBatch:
    """Observed response conditions with anchor-matched basal controls."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    controls_tx1: tuple[torch.Tensor, ...]
    observed_hvg: tuple[torch.Tensor, ...]
    control_hvg: tuple[torch.Tensor, ...]

    def validate(self) -> None:
        size = len(self.genes)
        if size == 0 or not (
            len(self.model_ids)
            == len(self.controls_tx1)
            == len(self.observed_hvg)
            == len(self.control_hvg)
            == size
        ):
            raise ValueError("response batch fields must be non-empty and aligned")
        if any(not value for value in (*self.model_ids, *self.genes)):
            raise ValueError("response batch identifiers cannot be empty")
        for name, bags in (
            ("controls_tx1", self.controls_tx1),
            ("observed_hvg", self.observed_hvg),
            ("control_hvg", self.control_hvg),
        ):
            for bag in bags:
                if bag.dim() != 2 or bag.shape[0] == 0:
                    raise ValueError(f"every {name} bag must be non-empty and 2-D")
                if not bag.is_floating_point() or not bool(torch.isfinite(bag).all()):
                    raise ValueError(f"every {name} bag must be finite floating point")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> ResponseBatch:
        """Move response tensors while preserving ModelID/gene keys."""
        return ResponseBatch(
            model_ids=self.model_ids,
            genes=self.genes,
            controls_tx1=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.controls_tx1
            ),
            observed_hvg=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.observed_hvg
            ),
            control_hvg=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.control_hvg
            ),
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

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> ResponseForwardBatch:
        """Move control tensors while preserving bare gene identifiers."""
        return ResponseForwardBatch(
            controls_tx1=tuple(
                value.to(device, non_blocking=non_blocking)
                for value in self.controls_tx1
            ),
            genes=self.genes,
        )


@dataclass(frozen=True)
class E2EForwardOutput:
    delta_hat: torch.Tensor
    raw_features: FeatureBatch
    feature_metadata: tuple[Mapping[str, object], ...]
    response_predicted: tuple[torch.Tensor, ...] | None = None
