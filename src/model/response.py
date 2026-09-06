"""model / response."""

from __future__ import annotations

from src.data.batches import ResponseBatch
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
    :func:`predict_bags` trims the output back to ``n_cells``.
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

    Cost is ``O(n*m*d)``; preparation fixes the bag sizes.
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
    """Use ordinary rank RNG in training and isolated seeded RNG in evaluation."""
    if model.training:
        return tuple(model(tuple(chunks), tuple(genes), tuple(None for _ in chunks)))
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

    Padding indices always use the explicit collator seed. Training stochastic
    layers consume the normal per-rank torch RNG; evaluation uses an isolated
    seeded RNG and restores the caller's state.
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
        if n_cells < 1:
            raise ValueError("at least one basal cell is required")
        # Complete windows are already consecutive: indexing them creates a
        # host-to-device index transfer and a GPU copy for every condition.
        condition_chunks = list(control.split(window, dim=0))
        if n_cells % window:
            index = _chunk_control_cell_indices(n_cells, window, seed)[-1]
            condition_chunks[-1] = control[
                torch.as_tensor(index, dtype=torch.long, device=control.device)
            ]
        chunks.extend(condition_chunks)
        chunk_genes.extend(str(gene) for _ in condition_chunks)
        chunks_per_control.append(len(condition_chunks))
        cell_counts.append(n_cells)
    # Call the module so DDP observes the forward and synchronizes gradients.
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


def response_terms(
    predicted: Sequence[torch.Tensor], batch: ResponseBatch
) -> dict[str, torch.Tensor]:
    """Return FP32 losses per condition; callers perform anchor reduction."""
    if not predicted or len(predicted) != len(batch.genes):
        raise ValueError("response predictions must match a non-empty condition batch")
    means, energies = [], []
    for pred, observed, control in zip(
        predicted, batch.observed_hvg, batch.control_hvg, strict=True
    ):
        pred, observed, control = pred.float(), observed.float(), control.float()
        if pred.ndim != 2 or observed.ndim != 2 or control.ndim != 2:
            raise ValueError("response bags must be two-dimensional")
        if not (pred.shape[1] == observed.shape[1] == control.shape[1]):
            raise ValueError("response bag gene widths must match")
        if pred.shape[0] == 0 or observed.shape[0] == 0 or control.shape[0] == 0:
            raise ValueError("response bags must be non-empty")
        means.append(mean_delta_mse(pred, observed, control.mean(dim=0)))
        energies.append(energy_distance(pred, observed))
    terms = {
        "mean_delta_mse": torch.stack(means),
        "energy_distance": torch.stack(energies),
    }
    if any(not bool(torch.isfinite(value).all()) for value in terms.values()):
        raise ValueError("non-finite response loss")
    return terms
