"""model / response."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import nn


def _chunk_control_cell_indices(
    n_cells: int, cell_set_len: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Split ``n_cells`` basal-cell row indices into equal ``cell_set_len`` windows.

    ``StateForwardAdapter.forward_chunks`` requires every chunk to be the
    identical size, so a cell count not evenly divisible by ``cell_set_len``
    needs its final window padded. Padding resamples (with replacement,
    deterministically from ``seed``) from the same line's own basal cells,
    mirroring the retired ``make_cell_set_chunks``'s ``pad_short`` behavior. The
    padded rows only satisfy ST's fixed-window-size contract --
    :func:`generate_predicted_response` trims the output back to ``n_cells``.
    """
    if n_cells < 1:
        raise ValueError("at least one basal cell is required")
    if cell_set_len < 1:
        raise ValueError("cell_set_len must be at least 1")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_cells)
    chunks: list[np.ndarray] = []
    for start in range(0, n_cells, cell_set_len):
        window = all_indices[start : start + cell_set_len]
        if len(window) < cell_set_len:
            padding = rng.choice(
                all_indices, size=cell_set_len - len(window), replace=True
            )
            window = np.concatenate([window, padding])
        chunks.append(window)
    return tuple(chunks)


@dataclass(frozen=True)
class ResponseLossWeights:
    """Relative weights of the two ``L_resp`` terms (``01`` §4).

    Attributes:
        mean_delta: Weight on the mean-delta MSE term.
        energy: Weight on the energy-distance term. Set to ``0.0`` to train
            on the mean alone, which is strictly weaker -- it cannot see a
            change in spread -- and is only useful as an ablation.
    """

    mean_delta: float = 1.0
    energy: float = 1.0

    def __post_init__(self) -> None:
        if self.mean_delta < 0 or self.energy < 0:
            raise ValueError("loss weights must be non-negative")
        if self.mean_delta == 0 and self.energy == 0:
            raise ValueError("at least one loss term must carry weight")


class ResponseLoss(nn.Module):
    """``L_resp`` = ``w_mean * mean-delta MSE + w_energy * energy distance``."""

    def __init__(self, weights: ResponseLossWeights = ResponseLossWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        predicted: torch.Tensor,
        observed: torch.Tensor,
        control_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one (gene, line) bag pair.

        Args:
            predicted: ``[n_pred, genes]`` predicted post-perturbation cells.
            observed: ``[n_obs, genes]`` observed post-perturbation cells.
            control_mean: ``[genes]`` mean of this line's control cells.

        Returns:
            ``(loss, parts)`` where ``parts`` carries the detached terms for
            logging.
        """
        total, tensor_parts = self.tensor_parts(predicted, observed, control_mean)
        return total, {
            name: float(value.detach()) for name, value in tensor_parts.items()
        }

    def tensor_parts(
        self,
        predicted: torch.Tensor,
        observed: torch.Tensor,
        control_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return tensor-valued terms without synchronizing CUDA to Python."""
        mean_term = mean_delta_mse(predicted, observed, control_mean)
        parts = {"mean_delta_mse": mean_term}
        total = self.weights.mean_delta * mean_term
        energy_term = energy_distance(predicted, observed)
        parts["energy_distance"] = energy_term
        total = total + self.weights.energy * energy_term
        parts["loss"] = total
        return total, parts


def mean_delta_mse(
    predicted: torch.Tensor, observed: torch.Tensor, control_mean: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted and observed mean shift from the control mean.

    The control mean cancels algebraically, but it is kept explicit because
    the quantity being matched is a perturbation *effect*: dropping it would
    silently turn this into a test of absolute expression, which a model can
    win by reproducing the cell line and ignoring the perturbation.
    """
    delta_pred = predicted.mean(dim=0) - control_mean
    delta_obs = observed.mean(dim=0) - control_mean
    return torch.mean((delta_pred - delta_obs) ** 2)


def energy_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Energy distance between two cell bags, ``2 E|x-y| - E|x-x'| - E|y-y'|``.

    Zero iff the two samples come from the same distribution, and unlike the
    mean term it responds to a change in spread -- the case where a model
    predicts the right average shift while collapsing cell-to-cell variation.
    That collapse is a documented failure of the retired backbone
    (``docs/results/exp05-hct116-frozen-backbone-transport.md``), so the term
    is not optional in practice.

    Cost is ``O(n*m*d)``; keep bags small (see ``TrainingConfig.max_bag``).
    """
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError("energy distance needs at least one cell per bag")
    cross = torch.cdist(left, right).mean()
    within_left = torch.cdist(left, left).mean()
    within_right = torch.cdist(right, right).mean()
    return 2.0 * cross - within_left - within_right


def _state_window(model: nn.Module) -> int | None:
    """Resolve STATE's fixed sentence length through an optional DDP wrapper."""
    inner = getattr(model, "module", model)
    adapter = getattr(inner, "state_adapter", None)
    state_model = getattr(adapter, "state_model", None)
    window = (
        getattr(state_model, "cell_sentence_len", None)
        or getattr(adapter, "cell_sentence_len", None)
        or getattr(inner, "cell_sentence_len", None)
    )
    if adapter is not None and not window:
        raise ValueError(
            "the ST adapter declares no cell_sentence_len; refusing to fall "
            "back to a single chunk. forward_chunks rejects that, and the "
            "silent fallback is what disguised a missing attribute as a "
            "chunk-size bug."
        )
    return int(window) if window else None


def _cuda_rng_devices(
    model: nn.Module, controls: Sequence[torch.Tensor]
) -> tuple[int, ...]:
    """Return CUDA devices whose RNG a STATE forward may consume."""
    devices: set[int] = set()
    tensors = [*controls, *model.parameters(), *model.buffers()]
    for tensor in tensors:
        if tensor.device.type == "cuda":
            devices.add(
                torch.cuda.current_device()
                if tensor.device.index is None
                else tensor.device.index
            )
    return tuple(sorted(devices))


def _seeded_model_forward(
    model: nn.Module,
    chunks: Sequence[torch.Tensor],
    genes: Sequence[str],
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Run STATE with a private, deterministic CPU/CUDA torch RNG stream."""
    cuda_devices = _cuda_rng_devices(model, chunks)
    with torch.random.fork_rng(devices=list(cuda_devices)):
        torch.random.default_generator.manual_seed(seed)
        for device_index in cuda_devices:
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)
        return tuple(model(tuple(chunks), tuple(genes), tuple(None for _ in chunks)))


def predict_bags(
    model: nn.Module,
    controls: Sequence[torch.Tensor],
    genes: Sequence[str],
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Forward multiple gene conditions through one padded STATE call.

    ``StateForwardAdapter`` requires every chunk to be exactly
    ``cell_sentence_len``; a bag that is not a multiple of it must be padded
    by resampling and the output trimmed back. The caller's ``seed`` governs
    two independent deterministic streams: NumPy chooses padding indices and
    a forked torch RNG governs stochastic STATE collator/model operations.
    The fork restores the caller's CPU and participating CUDA RNG states after
    the forward. Training happened to satisfy the window contract only because
    ``max_bag`` equalled the window size, which is accidental -- routing both
    training and evaluation through this helper makes the contract explicit.
    """

    if len(controls) != len(genes):
        raise ValueError("controls and genes must have equal length")
    if not controls:
        return ()
    window = _state_window(model)
    if not window:
        # Only a genuine test double -- no state adapter at all -- may take
        # the single-chunk path.
        return _seeded_model_forward(
            model,
            controls,
            genes,
            seed=seed,
        )

    chunks: list[torch.Tensor] = []
    chunk_genes: list[str] = []
    chunks_per_control: list[int] = []
    cell_counts: list[int] = []
    for control, gene in zip(controls, genes, strict=True):
        n_cells = int(control.shape[0])
        index_chunks = _chunk_control_cell_indices(n_cells, window, seed)
        condition_chunks = [
            control[torch.as_tensor(idx, dtype=torch.long, device=control.device)]
            for idx in index_chunks
        ]
        chunks.extend(condition_chunks)
        chunk_genes.extend(str(gene) for _ in condition_chunks)
        chunks_per_control.append(len(condition_chunks))
        cell_counts.append(n_cells)
    # The padding-index and model-forward streams intentionally share the
    # caller's pinned seed, but live in NumPy and a private torch RNG fork,
    # respectively. Call the module, never the bound method: a DDP-wrapped
    # model proxies only ``forward``, and unwrapping would skip all-reduce.
    predicted_chunks = _seeded_model_forward(
        model,
        chunks,
        chunk_genes,
        seed=seed,
    )
    if len(predicted_chunks) != len(chunks):
        raise ValueError("STATE returned a different number of chunks than supplied")
    predicted: list[torch.Tensor] = []
    offset = 0
    for n_chunks, n_cells in zip(chunks_per_control, cell_counts, strict=True):
        predicted.append(
            torch.cat(predicted_chunks[offset : offset + n_chunks], dim=0)[:n_cells]
        )
        offset += n_chunks
    return tuple(predicted)


def predict_bag(
    model: nn.Module, control: torch.Tensor, gene: str, *, seed: int
) -> torch.Tensor:
    """Forward one bag through the same vectorized path used by training."""
    return predict_bags(model, [control], [gene], seed=seed)[0]
