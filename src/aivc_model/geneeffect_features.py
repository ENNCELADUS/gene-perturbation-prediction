"""Pure Stage-2 feature construction for the Exp13 GeneEffect head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from aivc_model.geneeffect_head import moment_pool
from aivc_model.response_training import energy_distance

HVG_WIDTH = 2_000
DELTA_WIDTH = 2 * HVG_WIDTH
PROJECTION_WIDTH = 256
PROJECTION_SEED = 20_260_828
SUMMARY_WIDTH = 6

_CONTINUOUS_BLOCKS = frozenset({"delta_proj", "s", "q_sc", "e_g", "z_c"})


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    """Fixed ordering and widths of the Stage-2 response features."""

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

    @property
    def schema_hash(self) -> str:
        return _json_hash(self.to_dict())


FEATURE_SCHEMA = FeatureSchema()


class FixedSparseProjection:
    """A deterministic, data-independent sparse JL projection.

    Components use the Achlioptas distribution at density ``1/sqrt(4000)``.
    They are generated once from the stated seed; :meth:`transform` remains a
    plain torch matrix multiply so gradients flow to its input.
    """

    def __init__(self, seed: int = PROJECTION_SEED) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        density = 1.0 / np.sqrt(DELTA_WIDTH)
        nonzero = rng.random((PROJECTION_WIDTH, DELTA_WIDTH)) < density
        signs = rng.integers(0, 2, size=nonzero.shape, dtype=np.int8) * 2 - 1
        scale = 1.0 / np.sqrt(density * PROJECTION_WIDTH)
        self.components = np.where(nonzero, signs * scale, 0.0).astype(np.float32)

    @property
    def components_hash(self) -> str:
        return hashlib.sha256(self.components.tobytes(order="C")).hexdigest()

    @property
    def schema_seed_hash(self) -> str:
        return _json_hash(
            {"schema_hash": FEATURE_SCHEMA.schema_hash, "seed": self.seed}
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "algorithm": "achlioptas_sparse_jl_v1",
            "input_width": DELTA_WIDTH,
            "output_width": PROJECTION_WIDTH,
            "seed": self.seed,
            "schema_hash": FEATURE_SCHEMA.schema_hash,
            "schema_seed_hash": self.schema_seed_hash,
            "components_hash": self.components_hash,
        }

    def transform(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[-1:] != (DELTA_WIDTH,):
            raise ValueError(
                f"delta must end in width {DELTA_WIDTH}, got {tuple(delta.shape)}"
            )
        if not delta.is_floating_point():
            raise ValueError("delta must be floating point")
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("delta contains non-finite values")
        components = torch.as_tensor(
            self.components, device=delta.device, dtype=delta.dtype
        )
        projected = delta @ components.transpose(0, 1)
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("projected delta contains non-finite values")
        return projected

    def to_state(self) -> dict[str, object]:
        return {"metadata": self.metadata, "components": self.components.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> FixedSparseProjection:
        metadata = state.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("projection state metadata must be a mapping")
        expected_schema = metadata.get("schema_hash")
        if expected_schema != FEATURE_SCHEMA.schema_hash:
            raise ValueError("projection schema hash does not match the current schema")
        seed = metadata.get("seed")
        if not isinstance(seed, int):
            raise ValueError("projection seed must be an integer")
        restored = cls(seed=seed)
        components = np.asarray(state.get("components"), dtype=np.float32)
        if components.shape != (PROJECTION_WIDTH, DELTA_WIDTH):
            raise ValueError(f"invalid projection component shape {components.shape}")
        restored.components = components
        if metadata.get("components_hash") != restored.components_hash:
            raise ValueError("projection components hash mismatch")
        expected_metadata = restored.metadata
        if dict(metadata) != expected_metadata:
            raise ValueError("projection metadata is inconsistent with its components")
        return restored


@dataclass(frozen=True)
class ConditionFeatures:
    delta_proj: torch.Tensor
    s: torch.Tensor
    hvg_panel_mask: torch.Tensor
    own_gene_shift_mask: torch.Tensor
    metadata: dict[str, object]


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
    _require_finite_2d("basal", basal)
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
    if s.shape != (SUMMARY_WIDTH,) or not bool(torch.isfinite(s).all()):
        raise ValueError("constructed summary features are invalid or non-finite")

    return ConditionFeatures(
        delta_proj=delta_proj,
        s=s,
        hvg_panel_mask=torch.tensor(
            gene_in_hvg_panel, dtype=torch.bool, device=predicted.device
        ),
        own_gene_shift_mask=torch.tensor(
            own_gene_available, dtype=torch.bool, device=predicted.device
        ),
        metadata={
            "feature_schema_hash": FEATURE_SCHEMA.schema_hash,
            "projection_components_hash": projection.components_hash,
            "projection_seed": projection.seed,
            "shift_threshold_basal_l2_p95": float(shift_threshold.detach().cpu()),
        },
    )


@dataclass(frozen=True)
class _BlockStats:
    mean: np.ndarray
    scale: np.ndarray
    constant_columns: tuple[int, ...]


class BlockStandardizer:
    """Fit once on training rows and standardize each continuous block."""

    def __init__(self) -> None:
        self._stats: dict[str, _BlockStats] = {}
        self._fitted = False

    @property
    def constant_columns(self) -> dict[str, tuple[int, ...]]:
        self._require_fitted()
        return {name: stats.constant_columns for name, stats in self._stats.items()}

    def fit(
        self, train_blocks: Mapping[str, np.ndarray | torch.Tensor]
    ) -> BlockStandardizer:
        if self._fitted:
            raise RuntimeError("standardizer is train-only and cannot be refit")
        if not train_blocks:
            raise ValueError("at least one continuous block is required")
        unknown = set(train_blocks) - _CONTINUOUS_BLOCKS
        if unknown:
            raise ValueError(
                f"unknown or mask blocks cannot be standardized: {sorted(unknown)}"
            )
        stats: dict[str, _BlockStats] = {}
        for name, value in train_blocks.items():
            array = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(f"{name} fit data must be a non-empty 2-D array")
            if (
                not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} fit data must be finite numeric values")
            array = array.astype(np.float64, copy=False)
            mean = array.mean(axis=0)
            raw_scale = array.std(axis=0, ddof=0)
            constant = tuple(np.flatnonzero(raw_scale == 0.0).tolist())
            scale = raw_scale.copy()
            scale[raw_scale == 0.0] = 1.0
            stats[name] = _BlockStats(mean, scale, constant)
        self._stats = stats
        self._fitted = True
        return self

    def fit_batches(
        self,
        train_batches: Iterable[Mapping[str, np.ndarray | torch.Tensor]],
    ) -> BlockStandardizer:
        """Fit exact population statistics without materializing all train rows."""
        if self._fitted:
            raise RuntimeError("standardizer is train-only and cannot be refit")
        counts: dict[str, int] = {}
        means: dict[str, np.ndarray] = {}
        m2: dict[str, np.ndarray] = {}
        expected_blocks: set[str] | None = None
        for batch in train_batches:
            blocks = set(batch)
            if not blocks or blocks - _CONTINUOUS_BLOCKS:
                raise ValueError(
                    "streamed standardizer batches must contain only continuous blocks"
                )
            if expected_blocks is None:
                expected_blocks = blocks
            elif blocks != expected_blocks:
                raise ValueError(
                    "every standardizer batch must contain the same blocks"
                )
            for name, value in batch.items():
                array = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                if (
                    array.ndim != 2
                    or array.shape[0] == 0
                    or array.shape[1] == 0
                    or not np.issubdtype(array.dtype, np.number)
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(
                        f"{name} streamed fit data must be a non-empty finite 2-D array"
                    )
                array = array.astype(np.float64, copy=False)
                batch_count = int(array.shape[0])
                batch_mean = array.mean(axis=0)
                batch_m2 = ((array - batch_mean) ** 2).sum(axis=0)
                if name not in counts:
                    counts[name] = batch_count
                    means[name] = batch_mean
                    m2[name] = batch_m2
                    continue
                if means[name].shape != batch_mean.shape:
                    raise ValueError(f"{name} width changed across streamed batches")
                prior_count = counts[name]
                total = prior_count + batch_count
                delta = batch_mean - means[name]
                m2[name] += batch_m2 + delta**2 * prior_count * batch_count / total
                means[name] += delta * batch_count / total
                counts[name] = total
        if expected_blocks is None:
            raise ValueError("at least one streamed training batch is required")
        stats: dict[str, _BlockStats] = {}
        for name in sorted(expected_blocks):
            raw_scale = np.sqrt(m2[name] / counts[name])
            constant = tuple(np.flatnonzero(raw_scale == 0.0).tolist())
            scale = raw_scale.copy()
            scale[raw_scale == 0.0] = 1.0
            stats[name] = _BlockStats(means[name], scale, constant)
        self._stats = stats
        self._fitted = True
        return self

    def transform(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        if name not in self._stats:
            raise ValueError(f"block {name!r} was not fitted")
        if value.dim() != 2:
            raise ValueError(f"{name} transform data must be 2-D")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} transform data must be finite floating point")
        stats = self._stats[name]
        if value.shape[1] != stats.mean.size:
            raise ValueError(
                f"{name} width mismatch: fitted {stats.mean.size}, got {value.shape[1]}"
            )
        mean = torch.as_tensor(stats.mean, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(stats.scale, device=value.device, dtype=value.dtype)
        return (value - mean) / scale

    @property
    def state_hash(self) -> str:
        state = self._state_without_hash()
        return _json_hash(state)

    def to_state(self) -> dict[str, object]:
        state = self._state_without_hash()
        state["state_hash"] = _json_hash(state)
        return state

    def _state_without_hash(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "version": 1,
            "blocks": {
                name: {
                    "mean": stats.mean.tolist(),
                    "scale": stats.scale.tolist(),
                    "constant_columns": list(stats.constant_columns),
                }
                for name, stats in sorted(self._stats.items())
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> BlockStandardizer:
        payload = {key: value for key, value in state.items() if key != "state_hash"}
        if state.get("state_hash") != _json_hash(payload):
            raise ValueError("standardizer state hash mismatch")
        if payload.get("version") != 1 or not isinstance(
            payload.get("blocks"), Mapping
        ):
            raise ValueError("invalid standardizer state")
        restored = cls()
        stats: dict[str, _BlockStats] = {}
        for name, raw in payload["blocks"].items():
            if name not in _CONTINUOUS_BLOCKS or not isinstance(raw, Mapping):
                raise ValueError("invalid standardized block state")
            mean = np.asarray(raw.get("mean"), dtype=np.float64)
            scale = np.asarray(raw.get("scale"), dtype=np.float64)
            constant = tuple(int(i) for i in raw.get("constant_columns", []))
            if (
                mean.ndim != 1
                or mean.size == 0
                or scale.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
                or any(i < 0 or i >= mean.size for i in constant)
            ):
                raise ValueError("invalid standardizer statistics")
            stats[name] = _BlockStats(mean, scale, constant)
        if not stats:
            raise ValueError("standardizer state contains no blocks")
        restored._stats = stats
        restored._fitted = True
        return restored

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("standardizer has not been fitted")


__all__ = [
    "HVG_WIDTH",
    "DELTA_WIDTH",
    "PROJECTION_WIDTH",
    "PROJECTION_SEED",
    "SUMMARY_WIDTH",
    "FeatureSchema",
    "FEATURE_SCHEMA",
    "FixedSparseProjection",
    "ConditionFeatures",
    "compute_condition_features",
    "BlockStandardizer",
]
