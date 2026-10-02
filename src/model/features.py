"""Response features of one condition: projected expression shift and summaries.

``Delta`` = [mean shift, population-variance shift] of STATE's predicted cells
against the line's basal cells, both in log-space HVG expression.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

HVG_WIDTH = 2_000
DELTA_WIDTH = 2 * HVG_WIDTH
PROJECTION_WIDTH = 256
PROJECTION_SEED = 0
SUMMARY_FIELDS = (
    "energy_distance",
    "mean_predicted_population_variance",
    "fraction_cells_beyond_basal_p95",
    "mean_shift_l2",
    "mean_cosine",
    "own_gene_mean_shift",
)


class FixedSparseProjection:
    """A deterministic, data-independent sparse JL projection 4000 -> 256.

    Components follow the Achlioptas distribution at density ``1/sqrt(4000)``,
    generated once from the seed; :meth:`transform` is a plain matrix multiply so
    gradients reach its input. Device copies are cached across calls.
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
        key = (delta.device, delta.dtype)
        if key not in self._tensors:
            self._tensors[key] = torch.as_tensor(
                self.components, device=delta.device, dtype=delta.dtype
            )
        return delta @ self._tensors[key].transpose(0, 1)

    def to_state(self) -> dict[str, object]:
        return {"metadata": self.metadata, "components": self.components.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> FixedSparseProjection:
        restored = cls.__new__(cls)
        restored.seed = int(state["metadata"]["seed"])
        restored.components = np.asarray(state["components"], dtype=np.float32)
        restored._tensors = {}
        return restored


@dataclass(frozen=True)
class ConditionFeatureBatch:
    """Response features for a batch of equal-sized condition bags."""

    delta_proj: torch.Tensor
    s: torch.Tensor
    hvg_panel_mask: torch.Tensor
    own_gene_shift_mask: torch.Tensor


def compute_condition_feature_batch(
    predicted: Sequence[torch.Tensor],
    basal: Sequence[torch.Tensor],
    *,
    projection: FixedSparseProjection,
    gene_in_hvg_panel: torch.Tensor,
    own_gene_hvg_indices: Sequence[int | None],
    own_gene_available: torch.Tensor,
) -> ConditionFeatureBatch:
    """``Delta_proj`` and the six scalar summaries for every condition at once.

    Rows of one line share its basal tensor object; the basal-only statistics are
    computed once per distinct tensor and indexed back to the rows.
    """
    predicted_batch = torch.stack(tuple(predicted))
    dtype = predicted_batch.dtype
    distinct: dict[int, int] = {}
    bags: list[torch.Tensor] = []
    for bag in basal:
        if id(bag) not in distinct:
            distinct[id(bag)] = len(bags)
            bags.append(bag)
    distinct_batch = torch.stack(bags)
    row_bag = torch.tensor(
        [distinct[id(bag)] for bag in basal],
        dtype=torch.long,
        device=distinct_batch.device,
    )
    basal_batch = distinct_batch[row_bag]

    predicted_mean = predicted_batch.mean(dim=1)
    distinct_mean = distinct_batch.mean(dim=1)
    basal_mean = distinct_mean[row_bag]
    predicted_variance = (
        (predicted_batch - predicted_mean[:, None]).square().mean(dim=1)
    )
    basal_variance = (
        (distinct_batch - distinct_mean[:, None]).square().mean(dim=1)[row_bag]
    )
    delta_mean = predicted_mean - basal_mean
    delta = torch.cat((delta_mean, predicted_variance - basal_variance), dim=1)

    cross = torch.cdist(predicted_batch, basal_batch).mean(dim=(1, 2))
    within_predicted = torch.cdist(predicted_batch, predicted_batch).mean(dim=(1, 2))
    within_basal = torch.cdist(distinct_batch, distinct_batch).mean(dim=(1, 2))
    energy = 2.0 * cross - within_predicted - within_basal[row_bag]
    basal_distances = torch.linalg.vector_norm(
        distinct_batch - distinct_mean[:, None], dim=2
    )
    shift_threshold = torch.quantile(basal_distances, 0.95, dim=1)[row_bag]
    predicted_distances = torch.linalg.vector_norm(
        predicted_batch - basal_mean[:, None], dim=2
    )
    shifted_fraction = (
        (predicted_distances > shift_threshold[:, None]).to(dtype).mean(1)
    )
    safe_indices = torch.tensor(
        [0 if index is None else index for index in own_gene_hvg_indices],
        dtype=torch.long,
        device=predicted_batch.device,
    )
    own_shift = delta_mean.gather(1, safe_indices[:, None]).squeeze(1)
    own_shift = torch.where(own_gene_available, own_shift, torch.zeros_like(own_shift))
    summaries = torch.stack(
        (
            energy,
            predicted_variance.mean(dim=1),
            shifted_fraction,
            torch.linalg.vector_norm(delta_mean, dim=1),
            F.cosine_similarity(predicted_mean, basal_mean, dim=1),
            own_shift,
        ),
        dim=1,
    )
    return ConditionFeatureBatch(
        delta_proj=projection.transform(delta),
        s=summaries,
        hvg_panel_mask=gene_in_hvg_panel,
        own_gene_shift_mask=own_gene_available,
    )
