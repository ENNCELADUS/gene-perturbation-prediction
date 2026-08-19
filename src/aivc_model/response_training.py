"""Train the response model: Tx1 basal input -> ST -> HVG expression, no head.

This fits the Stage 1 objective ``L_resp`` from ``01-blueprint.md`` §4 on the
four Perturb-seq anchor lines. The module under training is
:class:`~aivc_model.tx1_predicted_response.ForwardOnlyStateModel` -- the ST
state adapter plus the ESM2 perturbation adapter, and deliberately nothing
else. No GeneEffect head is constructed here and no dependency label is read,
so a run of this module cannot leak a dependency signal into the backbone.

``L_resp`` is **distributional**. A predicted bag of cells has no
correspondence to the observed bag -- different cells, often different counts
-- so a per-cell reconstruction loss is not defined. §4 specifies mean-delta
MSE plus energy distance, and both are implemented here against the *delta*
from the line's own control mean rather than against absolute expression:

    delta_pred = mean(F(control, g)) - mean(control)
    delta_obs  = mean(observed_g)    - mean(control)

Centering on the control mean is what makes the target a perturbation effect
instead of a cell-line identity, which the model would otherwise be graded on
recovering -- the same failure mode the dependency side documents for
``mu_g`` (``residual_metrics.py``).

Held-out perturbation genes are split **per anchor line**, not globally: a
gene screened in three lines would otherwise appear in train for one line and
in the held-out set for another, and the generalization check would be
reading a gene the model has already fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aivc_model.distributed import assert_all_ranks_stepped

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "ResponseLossWeights",
    "ResponseLoss",
    "energy_distance",
    "mean_delta_mse",
    "evaluate_response_model",
    "split_heldout_genes",
    "basal_copy_baseline_loss",
    "TrainingConfig",
    "train_response_model",
    "predict_bag",
    "SELECTION_METRIC_NAME",
]

#: The per-epoch key -- and checkpoint-metadata field -- that best-epoch
#: selection and early stopping read. Named once so a checkpoint's
#: ``metadata.json`` can record the metric it was actually selected by
#: instead of a hand-typed string that could drift from the real
#: computation.
SELECTION_METRIC_NAME: Final[str] = "heldout_anchor_weighted_L_resp"


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
        mean_term = mean_delta_mse(predicted, observed, control_mean)
        parts = {"mean_delta_mse": float(mean_term.detach())}
        total = self.weights.mean_delta * mean_term
        if self.weights.energy:
            energy_term = energy_distance(predicted, observed)
            parts["energy_distance"] = float(energy_term.detach())
            total = total + self.weights.energy * energy_term
        parts["loss"] = float(total.detach())
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


def predict_bag(
    model: nn.Module, control: torch.Tensor, gene: str, *, seed: int
) -> torch.Tensor:
    """Forward one control bag through ST in fixed-size windows.

    ``StateForwardAdapter`` requires every chunk to be exactly
    ``cell_sentence_len``; a bag that is not a multiple of it must be padded
    by resampling and the output trimmed back. Training happened to satisfy
    this only because ``max_bag`` equalled the window size, which is
    accidental -- routing both training and evaluation through this helper
    makes the contract explicit rather than a coincidence of configuration.
    """
    from aivc_model.tx1_predicted_response import _chunk_control_cell_indices

    # Resolve the window through the DDP wrapper's ``.module``: reading an
    # attribute needs the inner module, while the FORWARD below must still go
    # through the wrapper so gradients all-reduce.
    inner = getattr(model, "module", model)
    adapter = getattr(inner, "state_adapter", None)
    # Read the window off ``state_model`` -- the object
    # ``StateForwardAdapter.forward_chunks`` reads it from. The adapter
    # carries no ``cell_sentence_len`` of its own, so looking for one there
    # resolves to None on every real model, and the fallback below then
    # silently sent a whole bag through as one chunk. Reading it from the
    # same object that checks it means the two cannot disagree.
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
    if not window:
        # Only a genuine test double -- no state adapter at all -- may take
        # the single-chunk path.
        return torch.cat(tuple(model((control,), gene, (None,))), dim=0)
    n_cells = int(control.shape[0])
    index_chunks = _chunk_control_cell_indices(n_cells, int(window), seed)
    chunks = tuple(
        control[torch.as_tensor(idx, dtype=torch.long, device=control.device)]
        for idx in index_chunks
    )
    # Call the module, never the bound method: a DDP-wrapped model proxies
    # only ``forward``, and unwrapping would skip the gradient all-reduce.
    predicted = model(chunks, gene, tuple(None for _ in chunks))
    return torch.cat(tuple(predicted), dim=0)[:n_cells]


def split_heldout_genes(
    genes_by_line: Mapping[str, Sequence[str]],
    *,
    fraction: float,
    seed: int,
) -> dict[str, frozenset[str]]:
    """Choose a held-out perturbation-gene set per line, deterministically.

    Selection is by ``sha256(seed|line|gene)`` rank rather than an RNG draw,
    so the same line and seed yield the same held-out genes regardless of
    how many genes the caller passes or in what order -- a re-run that adds
    one gene does not reshuffle the split.

    Args:
        genes_by_line: Perturbation genes available per ``model_id``.
        fraction: Fraction of each line's genes to hold out, in ``(0, 1)``.
        seed: Seed mixed into the hash.

    Returns:
        Held-out gene set per line.

    Raises:
        ValueError: ``fraction`` is outside ``(0, 1)``, or a line has too
            few genes to hold any out.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    heldout: dict[str, frozenset[str]] = {}
    for model_id, genes in genes_by_line.items():
        unique = sorted({str(gene) for gene in genes})
        n_hold = int(len(unique) * fraction)
        if not unique or n_hold < 1:
            raise ValueError(
                f"{model_id}: {len(unique)} genes cannot yield a held-out set "
                f"at fraction {fraction}"
            )
        ranked = sorted(
            unique,
            key=lambda gene: hashlib.sha256(
                f"{seed}|{model_id}|{gene}".encode("utf-8")
            ).hexdigest(),
        )
        heldout[model_id] = frozenset(ranked[:n_hold])
    return heldout


def basal_copy_baseline_loss(
    observed: torch.Tensor,
    control: torch.Tensor,
    control_mean: torch.Tensor,
    loss_fn: ResponseLoss,
) -> float:
    """``L_resp`` of predicting "nothing happens" -- the control cells verbatim.

    The Stage 1 freeze thresholds require beating this by a declared margin
    (Exp13 spec §7). It is the honest floor: a model that ignores the
    perturbation entirely still scores here, so any improvement over it is
    the only part attributable to perturbation modeling.
    """
    with torch.no_grad():
        loss, _ = loss_fn(control, observed, control_mean)
    return float(loss)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_response_model`.

    ``mixed_precision`` is deliberately absent: it comes from the
    ``accelerate launch --mixed_precision`` flag, not from code, exactly as
    the deleted trainer did -- a config field here would silently diverge
    from whatever the launcher actually applied.
    """

    max_epochs: int = 5
    patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_bag: int = 128
    grad_clip: float = 1.0
    seed: int = 20260818
    log_every: int = 50
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.max_bag < 2:
            raise ValueError("max_bag must be >= 2 for a distributional loss")


def _subsample(
    bag: torch.Tensor, max_bag: int, generator: torch.Generator
) -> torch.Tensor:
    """Cap a bag at ``max_bag`` cells, sampling without replacement."""
    if bag.shape[0] <= max_bag:
        return bag
    index = torch.randperm(bag.shape[0], generator=generator)[:max_bag]
    return bag[index]


def make_accelerator(config: TrainingConfig, *, cpu: bool = False) -> Accelerator:
    """Build the single :class:`Accelerator` every rank must share.

    ``even_batches=False`` turns off Accelerate's own batch padding so it
    cannot fight the explicit padding in :func:`_pad_gene_indices`.
    ``use_seedable_sampler`` plus ``data_seed`` makes the train shuffle
    order reproducible across ranks and restarts.

    ``cpu=True`` pins CPU even where CUDA/MPS exists; leaving it ``False``
    lets Accelerate auto-detect the device, which is what ``--device auto``
    means. Mixed precision is never set here -- it comes from the
    ``accelerate launch`` flag.
    """
    dataloader_config = DataLoaderConfiguration(
        even_batches=False,
        use_seedable_sampler=True,
        data_seed=config.seed,
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.ddp_find_unused_parameters,
        static_graph=config.ddp_static_graph,
    )
    return Accelerator(
        cpu=cpu,
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs],
    )


class _GeneIndexDataset(Dataset):
    """Index-only dataset: the heavy per-batch dicts stay in the parent list.

    ``accelerator.prepare`` shards and (for DDP) collectively touches
    whatever a ``DataLoader`` yields; a numpy array of cells inside that
    path would be copied/communicated pointlessly. Each item is looked up
    by index against the materialized batch list instead.
    """

    def __init__(self, indices: np.ndarray, is_padding: np.ndarray) -> None:
        if len(indices) != len(is_padding):
            raise ValueError("indices and is_padding must have the same length")
        self._indices = [int(index) for index in indices]
        self._is_padding = [bool(value) for value in is_padding]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, int | bool]:
        return {"index": self._indices[index], "is_padding": self._is_padding[index]}


def _pad_gene_indices(
    indices: np.ndarray, *, world_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ``indices`` to a multiple of ``world_size`` by repeating entries.

    Every DDP rank must take the same number of optimizer steps per epoch
    (``static_graph=True`` requires every rank to run the identical sequence
    of backward passes, and an uneven step count desyncs the collective
    all-reduce). Padding repeats real indices rather than inventing empty
    ones, so a padded step is a genuine forward/backward on real data whose
    loss is multiplied by zero before backward -- never a skipped step.

    Returns:
        ``(padded_indices, is_padding)``, same length, ``is_padding`` marking
        the appended entries.
    """
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    normalized = np.asarray(indices, dtype=np.int64)
    if len(normalized) == 0:
        raise ValueError("no batches to shard")
    is_padding = np.zeros(len(normalized), dtype=bool)
    remainder = len(normalized) % world_size
    if remainder == 0:
        return normalized, is_padding
    pad_count = world_size - remainder
    repeats = int(math.ceil(pad_count / len(normalized)))
    padding = np.tile(normalized, repeats)[:pad_count]
    return (
        np.concatenate([normalized, padding]),
        np.concatenate([is_padding, np.ones(pad_count, dtype=bool)]),
    )


def _shard_loader(
    n_records: int, *, accelerator: Accelerator, seed: int, shuffle: bool
) -> DataLoader:
    """Build the even-step-sharded, ``accelerator``-prepared index loader."""
    padded_indices, is_padding = _pad_gene_indices(
        np.arange(n_records), world_size=accelerator.num_processes
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        _GeneIndexDataset(padded_indices, is_padding),
        batch_size=1,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )
    return accelerator.prepare(loader)


def _validate_anchor_weights(
    anchor_weights: Mapping[str, float],
    heldout_records: Sequence[Mapping[str, object]],
) -> None:
    """Require ``anchor_weights`` and the held-out set to name the same lines.

    A held-out batch naming an anchor absent from ``anchor_weights``, or a
    weighted anchor with no held-out batch, would each silently change the
    selection objective without saying so -- both raise instead.
    """
    total = sum(anchor_weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"anchor_weights must sum to 1, got {total}")
    heldout_models = {str(record["model_id"]) for record in heldout_records}
    for record in heldout_records:
        model_id = str(record["model_id"])
        if model_id not in anchor_weights:
            raise ValueError(
                f"held-out batch names model_id {model_id!r}, "
                "absent from anchor_weights"
            )
    for model_id, weight in anchor_weights.items():
        if weight > 0 and model_id not in heldout_models:
            raise ValueError(
                f"anchor {model_id!r} has weight {weight} but no held-out batch"
            )


def _score_heldout_batch(
    model: nn.Module,
    record: Mapping[str, object],
    loss_fn: ResponseLoss,
    device: torch.device,
) -> tuple[str, float, float]:
    """Score one held-out (gene, line) batch: model loss and basal-copy floor.

    Shared by the per-epoch in-loop validation and the post-hoc
    :func:`evaluate_response_model` report so the two scoring paths cannot
    drift apart. Must be called under ``torch.no_grad()``.
    """
    control = torch.as_tensor(record["control"], dtype=torch.float32).to(device)
    observed = torch.as_tensor(record["observed"], dtype=torch.float32).to(device)
    control_target = torch.as_tensor(record["control_target"], dtype=torch.float32).to(
        device
    )
    predicted = predict_bag(model, control, str(record["gene"]), seed=0)
    # The launcher passes `--mixed_precision bf16`, so `predicted` may be
    # bf16; energy_distance's three torch.cdist calls are numerically
    # fragile in reduced precision, so force fp32 immediately before scoring.
    predicted_f = predicted.float()
    observed_f = observed.float()
    control_mean = control_target.mean(dim=0).float()
    loss, _ = loss_fn(predicted_f, observed_f, control_mean)
    # The floor is the control cells in the OUTPUT space (`control_target`),
    # not `control`: the latter is the 2560-d Tx1 basal embedding, while
    # `observed` and `control_mean` live in the ~2000-d HVG space, so feeding
    # it to the loss is a dimension mismatch, not a conservative baseline.
    floor = basal_copy_baseline_loss(
        observed_f, control_target.float(), control_mean, loss_fn
    )
    return str(record["model_id"]), float(loss), floor


def _per_anchor_mean_from_gathered(
    sum_matrix: torch.Tensor, count_matrix: torch.Tensor
) -> torch.Tensor:
    """Per-anchor mean loss from gathered per-rank sums and counts.

    Sums across the gathered rank dimension (dim 0) before dividing --
    averaging each rank's own per-anchor mean first would silently weight
    ranks unequally whenever they hold different held-out counts for an
    anchor, which even-step padding does nothing to prevent (it equalizes
    total step count, not per-anchor counts per rank).
    """
    total_sum = sum_matrix.sum(dim=0)
    total_count = count_matrix.sum(dim=0)
    return total_sum / total_count


def _is_better_metric(value: float, best_value: float, *, mode: str) -> bool:
    """Whether ``value`` improves on the incumbent ``best_value``.

    A non-finite candidate never wins; a non-finite incumbent always loses
    to any finite candidate. Without this, a metric that starts non-finite
    (e.g. a NaN first-epoch validation loss) would make ``value < best_value``
    False forever, silently freezing checkpoint selection on epoch 1.
    """
    if not math.isfinite(value):
        return False
    if not math.isfinite(best_value):
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"unknown selection mode: {mode}")


def _select_best_epoch(history: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Pick the epoch with the lowest :data:`SELECTION_METRIC_NAME`.

    Ties keep the earliest epoch (``min`` is stable and iterates in epoch
    order). A non-finite entry is never selected, matching
    :func:`_is_better_metric`'s NaN handling -- if every entry is non-finite
    this raises rather than returning an arbitrary epoch.
    """
    if not history:
        raise ValueError("history must be non-empty to select a best epoch")
    finite = [
        row for row in history if math.isfinite(float(row[SELECTION_METRIC_NAME]))
    ]
    if not finite:
        raise ValueError("no epoch produced a finite selection metric")
    return min(finite, key=lambda row: float(row[SELECTION_METRIC_NAME]))


def _save_model_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    path: Path,
    metadata: dict[str, object],
) -> None:
    """Save one checkpoint: unwrapped state dict plus ``metadata.json``.

    Rank-0 only; callers must follow this with
    ``accelerator.wait_for_everyone()`` so other ranks do not race ahead
    while rank 0 is still writing.

    ``Esm2PerturbationAdapter`` resolves every gene through one shared MLP
    and has no positional-binding hazard, unlike the deleted trainer's
    ``PerturbationVectorAdapter`` -- so unlike that trainer, this never
    writes a ``gene_vocabulary.json`` sidecar.
    """
    if not accelerator.is_main_process:
        return
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), path / "pytorch_model.bin")
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_train_log(
    history: Sequence[Mapping[str, object]], out_dir: Path, accelerator: Accelerator
) -> None:
    if not accelerator.is_main_process:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(history)).to_csv(out_dir / "train_log.csv", index=False)


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    records: Sequence[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    loss_fn: ResponseLoss,
    config: TrainingConfig,
    accelerator: Accelerator,
    generator: torch.Generator,
    *,
    epoch: int,
) -> tuple[dict[str, float], int]:
    """Run one training epoch; returns (mean loss terms, local step count).

    A padded step still runs the full forward and backward with its loss
    multiplied by ``0.0`` -- skipping it would desync the DDP collective
    under ``static_graph=True`` -- but is excluded from the reported mean,
    since it duplicates a real record and would otherwise double-count it.
    """
    model.train()
    totals: dict[str, list[float]] = {}
    local_steps = 0
    progress = tqdm(
        loader,
        desc=f"epoch {epoch}/{config.max_epochs}",
        disable=not accelerator.is_main_process,
    )
    for step, batch in enumerate(progress):
        index = int(batch["index"][0])
        is_padding = bool(batch["is_padding"][0])
        record = records[index]
        control = torch.as_tensor(
            record["control"], dtype=torch.float32, device=accelerator.device
        )
        observed = torch.as_tensor(
            record["observed"], dtype=torch.float32, device=accelerator.device
        )
        control_target = torch.as_tensor(
            record["control_target"], dtype=torch.float32, device=accelerator.device
        )
        observed = _subsample(observed, config.max_bag, generator)
        control = _subsample(control, config.max_bag, generator)

        predicted = predict_bag(model, control, str(record["gene"]), seed=config.seed)
        # The launcher passes `--mixed_precision bf16`, so `predicted` may be
        # bf16; energy_distance's three torch.cdist calls are numerically
        # fragile in reduced precision, so force fp32 immediately before
        # scoring.
        predicted_f = predicted.float()
        observed_f = observed.float()
        control_mean = control_target.mean(dim=0).float()
        loss, parts = loss_fn(predicted_f, observed_f, control_mean)
        backward_loss = loss * 0.0 if is_padding else loss

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(backward_loss)
        if config.grad_clip:
            accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        local_steps += 1

        if not is_padding:
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
            if config.log_every and step % config.log_every == 0:
                _LOGGER.info(
                    "epoch %d step %d gene %s loss %.6f",
                    epoch,
                    step,
                    record["gene"],
                    parts["loss"],
                )
    if not totals:
        raise ValueError("epoch produced no non-padding training steps")
    # Each rank's loader holds only its own shard, so a local mean would
    # describe rank zero's records rather than the epoch. Only rank zero
    # writes train_log.csv, so an ungathered value would silently report a
    # fraction of the training set as if it were all of it. Gather per-term
    # sums and counts and divide once, never averaging per-rank means, which
    # would misweight ranks holding different record counts.
    keys = sorted(totals)
    local_sums = torch.tensor(
        [float(np.sum(totals[key])) for key in keys],
        device=accelerator.device,
        dtype=torch.float32,
    )
    local_counts = torch.tensor(
        [float(len(totals[key])) for key in keys],
        device=accelerator.device,
        dtype=torch.float32,
    )
    total_sums = accelerator.gather(local_sums.unsqueeze(0)).sum(dim=0)
    total_counts = accelerator.gather(local_counts.unsqueeze(0)).sum(dim=0)
    return {
        key: float(total_sums[position] / total_counts[position])
        for position, key in enumerate(keys)
    }, local_steps


def _run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    records: Sequence[Mapping[str, object]],
    loss_fn: ResponseLoss,
    anchor_weights: Mapping[str, float],
    accelerator: Accelerator,
) -> tuple[dict[str, float], float]:
    """Score held-out ``records``; returns (per-anchor loss, selection value).

    Padded entries are skipped here (not zeroed-and-run): validation takes
    no backward pass, so there is no DDP collective for a skipped forward to
    desync, unlike training. Per-anchor sums and counts are gathered and
    combined via :func:`_per_anchor_mean_from_gathered` so every rank
    computes the identical selection value -- required so the early-stop
    decision cannot diverge across ranks.
    """
    model.eval()
    anchor_ids = sorted(anchor_weights)
    anchor_index = {anchor: position for position, anchor in enumerate(anchor_ids)}
    local_sum = torch.zeros(len(anchor_ids), device=accelerator.device)
    local_count = torch.zeros(len(anchor_ids), device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            if bool(batch["is_padding"][0]):
                continue
            record = records[int(batch["index"][0])]
            model_id, model_loss, _floor = _score_heldout_batch(
                model, record, loss_fn, accelerator.device
            )
            position = anchor_index[model_id]
            local_sum[position] += model_loss
            local_count[position] += 1.0
    gathered_sum = accelerator.gather(local_sum.unsqueeze(0))
    gathered_count = accelerator.gather(local_count.unsqueeze(0))
    per_anchor_mean = _per_anchor_mean_from_gathered(gathered_sum, gathered_count)
    per_anchor = {
        anchor: float(per_anchor_mean[position])
        for anchor, position in anchor_index.items()
    }
    selection_value = sum(
        anchor_weights[anchor] * per_anchor[anchor]
        for anchor in anchor_ids
        if anchor_weights[anchor] > 0
    )
    return per_anchor, float(selection_value)


def train_response_model(
    model: nn.Module,
    train_batches: Iterable[Mapping[str, object]],
    heldout_batches: Iterable[Mapping[str, object]],
    *,
    anchor_weights: Mapping[str, float],
    out_dir: Path,
    config: TrainingConfig = TrainingConfig(),
    loss_fn: ResponseLoss | None = None,
    accelerator: Accelerator | None = None,
) -> dict[str, object]:
    """Fit ``model`` on response batches with DDP, held-out validation, and
    early stopping; returns a JSON-ready training report.

    Each batch is a mapping with ``gene`` (str), ``model_id`` (str),
    ``control`` (``[n_ctrl, in_dim]`` Tx1 basal cells), ``observed``
    (``[n_obs, out_dim]`` observed perturbed cells) and ``control_target``
    (``[n_ctrl, out_dim]`` the same control cells in output space). The
    caller supplies these; this function does no I/O beyond ``out_dir`` so
    it stays testable without a checkpoint or a cache.

    Args:
        model: Module whose ``forward`` runs the ST + perturbation pair (
            perturbation pair; no head).
        train_batches: Per-(gene, line) training batches, re-iterated each
            epoch.
        heldout_batches: Per-(gene, line) held-out batches, scored every
            epoch for checkpoint selection and early stopping.
        anchor_weights: ``model_id`` -> weight (summing to 1) combining
            per-anchor held-out loss into the selection metric.
        out_dir: Directory for ``train_log.csv`` and the ``best``/``final``
            checkpoints.
        config: Hyperparameters.
        loss_fn: Loss module; defaults to an equally weighted ``L_resp``.
        accelerator: Pre-built :class:`Accelerator`; if ``None``, one is
            constructed from ``config`` so single-process callers and tests
            need not build one.

    Raises:
        ValueError: Either batch iterable is empty, ``anchor_weights`` does
            not sum to 1, a held-out batch names an anchor absent from
            ``anchor_weights``, or a weighted anchor has no held-out batch.
    """
    loss_fn = loss_fn or ResponseLoss()
    accelerator = accelerator or make_accelerator(config)
    train_records = list(train_batches)
    heldout_records = list(heldout_batches)
    if not train_records:
        raise ValueError("no response batches were supplied")
    if not heldout_records:
        raise ValueError("no held-out response batches were supplied")
    _validate_anchor_weights(anchor_weights, heldout_records)

    model = accelerator.prepare(model)
    optimizer = accelerator.prepare(
        torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    )
    train_loader = _shard_loader(
        len(train_records), accelerator=accelerator, seed=config.seed, shuffle=True
    )
    heldout_loader = _shard_loader(
        len(heldout_records), accelerator=accelerator, seed=config.seed, shuffle=False
    )
    generator = torch.Generator().manual_seed(config.seed)

    out_dir = Path(out_dir)
    history: list[dict[str, object]] = []
    best_value = math.nan
    best_epoch = -1
    epochs_since_improvement = 0
    stopped_early = False
    stopped_at_epoch = 0

    for epoch in range(config.max_epochs):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        train_metrics, local_steps = _run_train_epoch(
            model,
            train_loader,
            train_records,
            optimizer,
            loss_fn,
            config,
            accelerator,
            generator,
            epoch=epoch,
        )
        assert_all_ranks_stepped(accelerator, local_steps)
        per_anchor, selection_value = _run_validation_epoch(
            model, heldout_loader, heldout_records, loss_fn, anchor_weights, accelerator
        )
        row: dict[str, object] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"heldout_{anchor}_loss": value for anchor, value in per_anchor.items()},
            SELECTION_METRIC_NAME: selection_value,
        }
        history.append(row)
        _write_train_log(history, out_dir, accelerator)
        stopped_at_epoch = epoch

        if _is_better_metric(selection_value, best_value, mode="min"):
            best_value = selection_value
            best_epoch = epoch
            epochs_since_improvement = 0
            _save_model_checkpoint(
                accelerator,
                model,
                out_dir / "best",
                {
                    "checkpoint_kind": "best",
                    "epoch": epoch,
                    "selection_metric": SELECTION_METRIC_NAME,
                    "selection_mode": "min",
                    "metric_value": best_value,
                },
            )
            accelerator.wait_for_everyone()
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= config.patience:
            stopped_early = True
            break

    if best_epoch < 0:
        raise RuntimeError("no epoch produced a finite selection metric")
    selected = _select_best_epoch(history)
    if int(selected["epoch"]) != best_epoch:
        raise RuntimeError(
            "best-epoch tracking disagreed with _select_best_epoch: "
            f"{best_epoch} vs {selected['epoch']}"
        )

    _save_model_checkpoint(
        accelerator,
        model,
        out_dir / "final",
        {
            "checkpoint_kind": "final",
            "epoch": stopped_at_epoch,
            "selection_metric": SELECTION_METRIC_NAME,
            "selection_mode": "min",
            "metric_value": history[-1][SELECTION_METRIC_NAME],
            "best_metric_value": best_value,
            "stopped_early": stopped_early,
            "patience": config.patience,
        },
    )
    accelerator.wait_for_everyone()

    return {
        "epochs": history,
        "best_epoch": best_epoch,
        "best_metric_value": best_value,
        "selection_metric": SELECTION_METRIC_NAME,
        "stopped_early": stopped_early,
        "stopped_at_epoch": stopped_at_epoch,
        "n_train_batches": len(train_records),
        "n_heldout_batches": len(heldout_records),
        "world_size": accelerator.num_processes,
        "config": vars(config),
    }


def _null_shuffle_records(
    records: Sequence[Mapping[str, object]], seed: int
) -> list[Mapping[str, object]]:
    """Re-pair each bag with a DIFFERENT gene, keeping everything else fixed.

    Only the ``gene`` field moves: the same control cells, the same observed
    bag, the same line. So the difference against the unshuffled score is
    attributable to perturbation identity alone. With fewer than two records
    no derangement exists and the shuffle is a no-op, which is reported as
    such rather than silently scoring the identity permutation.
    """
    if len(records) < 2:
        return list(records)
    generator = np.random.default_rng(seed)
    order = np.arange(len(records))
    for _ in range(100):
        generator.shuffle(order)
        if all(order != np.arange(len(records))):
            break
    else:  # pragma: no cover - a derangement is found long before 100 draws
        order = np.roll(np.arange(len(records)), 1)
    return [
        {**record, "gene": str(records[int(position)]["gene"])}
        for record, position in zip(records, order, strict=True)
    ]


@torch.no_grad()
def evaluate_response_model(
    model: nn.Module,
    batches: Iterable[Mapping[str, object]],
    *,
    loss_fn: ResponseLoss | None = None,
    device: torch.device | str = "cpu",
    null_shuffle_seed: int = 0,
) -> dict[str, object]:
    """Score ``model`` against the basal-copy floor and a null shuffle.

    Reporting both floors is the point: Exp13 spec §7 gates Stage 1 on
    beating a basal-copy prediction AND a null shuffle by declared margins,
    and a loss number alone cannot show either. Uses
    :func:`_score_heldout_batch`, the same per-batch scoring the in-loop
    training validation uses, so the two paths cannot drift apart.

    The null shuffle re-scores every bag under a permuted gene assignment.
    It is the arm that catches a model scoring well because the anchors'
    bags merely resemble one another: such a model clears the basal-copy
    floor while its predictions barely move when the perturbation identity
    is scrambled. The permutation is a derangement where one exists, so no
    bag keeps its own gene and the null is not diluted by fixed points.
    """
    loss_fn = loss_fn or ResponseLoss()
    device = torch.device(device)
    model = model.to(device).eval()
    records = list(batches)
    if not records:
        raise ValueError("no evaluation batches were supplied")
    model_losses: list[float] = []
    floor_losses: list[float] = []
    per_line: dict[str, list[float]] = {}
    for batch in records:
        model_id, model_loss, floor_loss = _score_heldout_batch(
            model, batch, loss_fn, device
        )
        model_losses.append(model_loss)
        floor_losses.append(floor_loss)
        per_line.setdefault(model_id, []).append(model_loss)

    shuffled_losses = [
        _score_heldout_batch(model, record, loss_fn, device)[1]
        for record in _null_shuffle_records(records, null_shuffle_seed)
    ]
    return {
        "n_batches": len(model_losses),
        "model_loss": float(np.mean(model_losses)),
        "basal_copy_loss": float(np.mean(floor_losses)),
        "null_shuffle_loss": float(np.mean(shuffled_losses)),
        "null_shuffle_seed": int(null_shuffle_seed),
        "improvement_over_basal_copy": float(
            np.mean(floor_losses) - np.mean(model_losses)
        ),
        "improvement_over_null_shuffle": float(
            np.mean(shuffled_losses) - np.mean(model_losses)
        ),
        "per_line_model_loss": {k: float(np.mean(v)) for k, v in per_line.items()},
    }
