"""Train-only standardization of the head's feature blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs
    from src.model.geneeffect import GeneEffectE2EModel
from collections.abc import Iterable, Mapping
import numpy as np
import torch

_CONTINUOUS_BLOCKS = frozenset({"delta_proj", "s", "q_sc", "e_g", "z_c"})


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
        self._tensors: dict[
            tuple[str, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

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
        stats = self._stats[name]
        if value.shape[1] != stats.mean.size:
            raise ValueError(
                f"{name} width mismatch: fitted {stats.mean.size}, got {value.shape[1]}"
            )
        key = (name, value.device, value.dtype)
        tensors = self._tensors.get(key)
        if tensors is None:
            tensors = tuple(
                torch.as_tensor(array, device=value.device, dtype=value.dtype)
                for array in (stats.mean, stats.scale)
            )
            self._tensors[key] = tensors
        mean, scale = tensors
        return (value - mean) / scale

    def to_state(self) -> dict[str, object]:
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
        payload = state
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


def fit_startup_standardizer(
    model: GeneEffectE2EModel,
    inputs: PreparedInputs,
    *,
    accelerator: Any = None,
    batch_size: int = 32,
) -> BlockStandardizer:
    """Fit on up to 32 rows per supervised training line, once, before training.

    Rank zero fits from the fresh model's live features; other ranks receive its
    statistics so every rank standardizes identically. Resume restores saved
    statistics instead of calling this.
    """
    from src.data.datasets import DependencyDataset

    payload: list[Any] = [None]
    if accelerator is None or accelerator.is_main_process:
        device = next(model.parameters()).device
        dataset = DependencyDataset(inputs, "train", device=device)
        rng = np.random.default_rng(0)
        selected = [
            index
            for indices in dataset.rows.groupby("model_id", sort=True).indices.values()
            for index in rng.choice(indices, size=min(32, len(indices)), replace=False)
        ]
        enabled = [
            name
            for name in sorted(_CONTINUOUS_BLOCKS)
            if getattr(model.head.blocks, f"use_{name}")
        ]

        def blocks():
            for start in range(0, len(selected), batch_size):
                batch = dataset.collate(selected[start : start + batch_size])
                features = model.condition_features(batch.to(device).conditions)
                yield {name: getattr(features, name) for name in enabled}

        training = model.training
        model.eval()
        with torch.no_grad():
            payload[0] = BlockStandardizer().fit_batches(blocks()).to_state()
        model.train(training)
    if accelerator is not None and accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(payload, src=0)
    model.standardizer = BlockStandardizer.from_state(payload[0])
    return model.standardizer
