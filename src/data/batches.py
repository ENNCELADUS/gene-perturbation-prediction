"""Batch records shared by the joint GeneEffect data, model and training code."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeatureBatch:
    """Raw, unstandardized live head features.

    The STATE-derived blocks ``delta_proj`` and ``s`` and their masks
    ``hvg_panel_mask`` and ``own_gene_shift_mask`` are all ``None`` for a model
    that does not use STATE, and all present otherwise.
    """

    delta_proj: torch.Tensor | None
    s: torch.Tensor | None
    q_sc: torch.Tensor
    e_g: torch.Tensor
    z_c: torch.Tensor
    q_sc_mask: torch.Tensor
    hvg_panel_mask: torch.Tensor | None
    own_gene_shift_mask: torch.Tensor | None
    gene_symbols: tuple[str, ...]
    model_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.q_sc.shape[0])

    def validate(self) -> None:
        if (
            len(self.gene_symbols) != self.batch_size
            or len(self.model_ids) != self.batch_size
        ):
            raise ValueError("feature identities must align with batch_size")
        if any(not value for value in (*self.gene_symbols, *self.model_ids)):
            raise ValueError("feature identities cannot contain empty strings")
        state_fields = (
            self.delta_proj,
            self.s,
            self.hvg_panel_mask,
            self.own_gene_shift_mask,
        )
        if len({value is None for value in state_fields}) != 1:
            raise ValueError(
                "delta_proj, s, hvg_panel_mask and own_gene_shift_mask must be "
                "all present or all None"
            )
        blocks = {
            "delta_proj": self.delta_proj,
            "s": self.s,
            "q_sc": self.q_sc,
            "e_g": self.e_g,
            "z_c": self.z_c,
        }
        for name, value in blocks.items():
            if value is None:
                continue
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
            if value is None:
                continue
            if value.shape != (self.batch_size,) or value.dtype != torch.bool:
                raise ValueError(
                    f"{name} must be boolean [{self.batch_size}], got "
                    f"shape={tuple(value.shape)} dtype={value.dtype}"
                )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> FeatureBatch:
        """Move tensor fields while preserving row identifiers."""

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.to(device, non_blocking=non_blocking)

        return FeatureBatch(
            delta_proj=move(self.delta_proj),
            s=move(self.s),
            q_sc=move(self.q_sc),
            e_g=move(self.e_g),
            z_c=move(self.z_c),
            q_sc_mask=move(self.q_sc_mask),
            hvg_panel_mask=move(self.hvg_panel_mask),
            own_gene_shift_mask=move(self.own_gene_shift_mask),
            gene_symbols=self.gene_symbols,
            model_ids=self.model_ids,
        )


def _move(values: tuple[torch.Tensor, ...], device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in values)


@dataclass(frozen=True)
class OnlineConditionBatch:
    """Inputs for differentiable response-feature generation, one row per condition.

    ``basal_hvg`` holds each condition's line's log-space basal HVG cells, STATE's
    only cell input; ``z_c`` is the line's eigen-scaled Tx1 context PCA scores
    for the head.
    ``gene_index`` is each row's gene position in ``inputs.genes``.
    """

    basal_hvg: tuple[torch.Tensor, ...]
    genes: tuple[str, ...]
    gene_index: torch.Tensor
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

    def to(self, device: torch.device | str) -> OnlineConditionBatch:
        return OnlineConditionBatch(
            basal_hvg=_move(self.basal_hvg, device),
            genes=self.genes,
            gene_index=self.gene_index.to(device),
            model_ids=self.model_ids,
            q_sc=self.q_sc.to(device),
            e_g=self.e_g.to(device),
            z_c=self.z_c.to(device),
            q_sc_mask=self.q_sc_mask.to(device),
            gene_in_hvg_panel=self.gene_in_hvg_panel.to(device),
            own_gene_hvg_indices=self.own_gene_hvg_indices,
            own_gene_shift_available=self.own_gene_shift_available.to(device),
        )


@dataclass(frozen=True)
class DependencyBatch:
    """GeneEffect rows: targets centred on the fold-fit mean, plus the fixed mean.

    ``residual`` is the training target; ``gene_mean`` is the fixed train mean added
    back to a predicted residual; ``gene_effect`` is the measured absolute value.
    ``residual_scale`` is each row's gene's training residual SD and ``selective``
    whether that gene is selective.
    """

    conditions: OnlineConditionBatch
    residual: torch.Tensor
    gene_effect: torch.Tensor
    gene_mean: torch.Tensor
    residual_scale: torch.Tensor
    selective: torch.Tensor

    def to(self, device: torch.device | str) -> DependencyBatch:
        return DependencyBatch(
            conditions=self.conditions.to(device),
            residual=self.residual.to(device),
            gene_effect=self.gene_effect.to(device),
            gene_mean=self.gene_mean.to(device),
            residual_scale=self.residual_scale.to(device),
            selective=self.selective.to(device),
        )


@dataclass(frozen=True)
class ResponseBatch:
    """Observed anchor perturbation bags with the anchor's basal control cells."""

    model_ids: tuple[str, ...]
    genes: tuple[str, ...]
    control_hvg: tuple[torch.Tensor, ...]
    observed_hvg: tuple[torch.Tensor, ...]

    def to(self, device: torch.device | str) -> ResponseBatch:
        return ResponseBatch(
            model_ids=self.model_ids,
            genes=self.genes,
            control_hvg=_move(self.control_hvg, device),
            observed_hvg=_move(self.observed_hvg, device),
        )


@dataclass(frozen=True)
class ResponseForwardBatch:
    """Response-anchor basal cells and genes carried through the joint forward call."""

    basal_hvg: tuple[torch.Tensor, ...]
    genes: tuple[str, ...]


@dataclass(frozen=True)
class E2EForwardOutput:
    delta_hat: torch.Tensor
    response_predicted: tuple[torch.Tensor, ...] | None = None
