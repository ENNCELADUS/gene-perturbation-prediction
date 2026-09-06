"""model / features."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from src.model.losses import moment_pool
from src.model.response import energy_distance

HVG_WIDTH = 2_000


DELTA_WIDTH = 2 * HVG_WIDTH


PROJECTION_WIDTH = 256


PROJECTION_SEED = 0


SUMMARY_WIDTH = 6


def _require_finite_2d(name: str, value: torch.Tensor) -> None:
    if value.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if value.shape[1] != HVG_WIDTH:
        raise ValueError(
            f"{name} must have {HVG_WIDTH} HVGs, got width {value.shape[1]}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class FeatureSchema:
    """Fixed ordering and widths of the joint response features."""

    hvg_width: int = HVG_WIDTH
    delta_fields: tuple[str, str] = ("mean_shift", "population_variance_shift")
    summary_fields: tuple[str, ...] = (
        "energy_distance",
        "mean_predicted_population_variance",
        "fraction_cells_beyond_basal_p95",
        "mean_shift_l2",
        "mean_cosine",
        "own_gene_mean_shift",
    )

    @property
    def delta_width(self) -> int:
        return self.hvg_width * len(self.delta_fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "hvg_width": self.hvg_width,
            "delta_fields": list(self.delta_fields),
            "summary_fields": list(self.summary_fields),
        }


FEATURE_SCHEMA = FeatureSchema()


class FixedSparseProjection:
    """A deterministic, data-independent sparse JL projection.

    Components use the Achlioptas distribution at density ``1/sqrt(4000)``.
    They are generated once from the stated seed; :meth:`transform` remains a
    plain torch matrix multiply so gradients flow to its input. Treat components
    as fixed after construction; use ``from_state`` to restore a different matrix.
    Device tensors are retained so per-condition calls do not recopy the matrix.
    """

    def __init__(self, seed: int = PROJECTION_SEED) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        density = 1.0 / np.sqrt(DELTA_WIDTH)
        nonzero = rng.random((PROJECTION_WIDTH, DELTA_WIDTH)) < density
        signs = rng.integers(0, 2, size=nonzero.shape, dtype=np.int8) * 2 - 1
        scale = 1.0 / np.sqrt(density * PROJECTION_WIDTH)
        self.components = np.where(nonzero, signs * scale, 0.0).astype(np.float32)
        self._tensors: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "algorithm": "achlioptas_sparse_jl_v1",
            "input_width": DELTA_WIDTH,
            "output_width": PROJECTION_WIDTH,
            "seed": self.seed,
        }

    def transform(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-1:] != (DELTA_WIDTH,):
            raise ValueError(
                f"delta must end in width {DELTA_WIDTH}, got {tuple(delta.shape)}"
            )
        if not delta.is_floating_point():
            raise ValueError("delta must be floating point")
        key = (delta.device, delta.dtype)
        components = self._tensors.get(key)
        if components is None:
            components = torch.as_tensor(
                self.components, device=delta.device, dtype=delta.dtype
            )
            self._tensors[key] = components
        projected = delta @ components.transpose(0, 1)
        return projected

    def to_state(self) -> dict[str, object]:
        return {"metadata": self.metadata, "components": self.components.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> FixedSparseProjection:
        metadata = state.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("projection state metadata must be a mapping")
        seed = metadata.get("seed")
        if not isinstance(seed, int):
            raise ValueError("projection seed must be an integer")
        restored = cls.__new__(cls)
        restored.seed = seed
        components = np.asarray(state.get("components"), dtype=np.float32)
        if components.shape != (PROJECTION_WIDTH, DELTA_WIDTH):
            raise ValueError(f"invalid projection component shape {components.shape}")
        if not np.isfinite(components).all():
            raise ValueError("projection components must be finite")
        restored.components = components.copy()
        restored._tensors = {}
        return restored


@dataclass(frozen=True)
class ConditionFeatures:
    delta_proj: torch.Tensor
    s: torch.Tensor
    hvg_panel_mask: torch.Tensor
    own_gene_shift_mask: torch.Tensor


def compute_condition_features(
    predicted: torch.Tensor,
    basal: torch.Tensor,
    *,
    projection: FixedSparseProjection,
    gene_in_hvg_panel: bool,
    own_gene_hvg_index: int | None,
    own_gene_available: bool,
) -> ConditionFeatures:
    """Build ``Delta_proj`` and the six scalar summaries for one condition."""
    _require_finite_2d("predicted", predicted)
    if basal.ndim != 2 or basal.shape[0] == 0 or basal.shape[1] != HVG_WIDTH:
        raise ValueError(f"basal must have at least one row and {HVG_WIDTH} HVGs")
    if predicted.device != basal.device:
        raise ValueError("predicted and basal must be on the same device")
    if predicted.dtype != basal.dtype or not predicted.is_floating_point():
        raise ValueError("predicted and basal must share a floating-point dtype")

    if not isinstance(gene_in_hvg_panel, bool):
        raise ValueError("gene_in_hvg_panel must be an explicit bool")
    if not isinstance(own_gene_available, bool):
        raise ValueError("own_gene_available must be an explicit bool")
    if gene_in_hvg_panel:
        if own_gene_hvg_index is None or not 0 <= own_gene_hvg_index < HVG_WIDTH:
            raise ValueError("gene in the HVG panel requires a valid HVG index")
    elif own_gene_hvg_index is not None:
        raise ValueError("gene outside the HVG panel must use own_gene_hvg_index=None")
    if own_gene_available and not gene_in_hvg_panel:
        raise ValueError("own-gene shift cannot be available outside the HVG panel")

    pred_moments = moment_pool(predicted, moments=2)
    basal_moments = moment_pool(basal, moments=2)
    delta = pred_moments - basal_moments
    delta_mean = delta[:HVG_WIDTH]
    pred_variance = pred_moments[HVG_WIDTH:]
    delta_proj = projection.transform(delta)

    basal_mean = basal_moments[:HVG_WIDTH]
    predicted_mean = pred_moments[:HVG_WIDTH]
    basal_distances = torch.linalg.vector_norm(basal - basal_mean, dim=1)
    shift_threshold = torch.quantile(basal_distances, 0.95)
    predicted_distances = torch.linalg.vector_norm(predicted - basal_mean, dim=1)
    shifted_fraction = (
        (predicted_distances > shift_threshold).to(predicted.dtype).mean()
    )
    own_shift = (
        delta_mean[own_gene_hvg_index]
        if own_gene_available and own_gene_hvg_index is not None
        else delta_mean.new_zeros(())
    )
    s = torch.stack(
        (
            energy_distance(predicted, basal),
            pred_variance.mean(),
            shifted_fraction,
            torch.linalg.vector_norm(delta_mean),
            F.cosine_similarity(predicted_mean, basal_mean, dim=0),
            own_shift,
        )
    )

    return ConditionFeatures(
        delta_proj=delta_proj,
        s=s,
        hvg_panel_mask=torch.tensor(
            gene_in_hvg_panel, dtype=torch.bool, device=predicted.device
        ),
        own_gene_shift_mask=torch.tensor(
            own_gene_available, dtype=torch.bool, device=predicted.device
        ),
    )
