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
import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

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
]


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

    adapter = getattr(model, "state_adapter", None)
    window = getattr(adapter, "cell_sentence_len", None) or getattr(
        model, "cell_sentence_len", None
    )
    if not window:
        # A model with no fixed-window contract (test doubles) takes one chunk.
        return torch.cat(
            tuple(model.predict_response_chunks((control,), gene, (None,))), dim=0
        )
    n_cells = int(control.shape[0])
    index_chunks = _chunk_control_cell_indices(n_cells, int(window), seed)
    chunks = tuple(
        control[torch.as_tensor(idx, dtype=torch.long, device=control.device)]
        for idx in index_chunks
    )
    predicted = model.predict_response_chunks(chunks, gene, tuple(None for _ in chunks))
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
    """Hyperparameters for :func:`train_response_model`."""

    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_bag: int = 128
    grad_clip: float = 1.0
    seed: int = 20260818
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
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


def train_response_model(
    model: nn.Module,
    batches: Iterable[Mapping[str, object]],
    *,
    config: TrainingConfig = TrainingConfig(),
    loss_fn: ResponseLoss | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Fit ``model`` on response batches and return a training report.

    Each batch is a mapping with ``gene`` (str), ``model_id`` (str),
    ``control`` (``[n_ctrl, in_dim]`` Tx1 basal cells), ``observed``
    (``[n_obs, out_dim]`` observed perturbed cells) and ``control_target``
    (``[n_ctrl, out_dim]`` the same control cells in output space). The
    caller supplies these; this function does no I/O so it stays testable
    without a checkpoint or a cache.

    Args:
        model: Module exposing ``predict_response_chunks`` (the ST +
            perturbation pair; no head).
        batches: Iterable of per-(gene, line) batches, re-iterated each epoch.
        config: Hyperparameters.
        loss_fn: Loss module; defaults to an equally weighted ``L_resp``.
        device: Device to train on.

    Returns:
        A JSON-ready report: per-epoch mean loss and term breakdown.
    """
    loss_fn = loss_fn or ResponseLoss()
    device = torch.device(device)
    model = model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    materialized = list(batches)
    if not materialized:
        raise ValueError("no response batches were supplied")

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        totals: dict[str, list[float]] = {}
        for step, batch in enumerate(materialized):
            control = torch.as_tensor(batch["control"], dtype=torch.float32).to(device)
            observed = torch.as_tensor(batch["observed"], dtype=torch.float32).to(
                device
            )
            control_target = torch.as_tensor(
                batch["control_target"], dtype=torch.float32
            ).to(device)
            observed = _subsample(observed, config.max_bag, generator)
            control = _subsample(control, config.max_bag, generator)

            predicted = predict_bag(
                model, control, str(batch["gene"]), seed=config.seed
            )
            loss, parts = loss_fn(predicted, observed, control_target.mean(dim=0))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
            if config.log_every and step % config.log_every == 0:
                _LOGGER.info(
                    "epoch %d step %d gene %s loss %.6f",
                    epoch,
                    step,
                    batch["gene"],
                    parts["loss"],
                )
        history.append(
            {"epoch": epoch, **{k: float(np.mean(v)) for k, v in totals.items()}}
        )
        _LOGGER.info("epoch %d mean loss %.6f", epoch, history[-1]["loss"])
    return {"epochs": history, "batches": len(materialized), "config": vars(config)}


@torch.no_grad()
def evaluate_response_model(
    model: nn.Module,
    batches: Iterable[Mapping[str, object]],
    *,
    loss_fn: ResponseLoss | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Score ``model`` and the basal-copy floor on held-out batches.

    Reporting the floor next to the model is the point: Exp13 spec §7 gates
    Stage 1 on beating a basal-copy prediction by a declared margin, and a
    loss number alone cannot show that.
    """
    loss_fn = loss_fn or ResponseLoss()
    device = torch.device(device)
    model = model.to(device).eval()
    model_losses: list[float] = []
    floor_losses: list[float] = []
    per_line: dict[str, list[float]] = {}
    for batch in batches:
        control = torch.as_tensor(batch["control"], dtype=torch.float32).to(device)
        observed = torch.as_tensor(batch["observed"], dtype=torch.float32).to(device)
        control_target = torch.as_tensor(
            batch["control_target"], dtype=torch.float32
        ).to(device)
        control_mean = control_target.mean(dim=0)
        predicted = predict_bag(model, control, str(batch["gene"]), seed=0)
        loss, _ = loss_fn(predicted, observed, control_mean)
        model_losses.append(float(loss))
        floor_losses.append(
            basal_copy_baseline_loss(observed, control, control_mean, loss_fn)
        )
        per_line.setdefault(str(batch["model_id"]), []).append(float(loss))
    if not model_losses:
        raise ValueError("no evaluation batches were supplied")
    return {
        "n_batches": len(model_losses),
        "model_loss": float(np.mean(model_losses)),
        "basal_copy_loss": float(np.mean(floor_losses)),
        "improvement_over_basal_copy": float(
            np.mean(floor_losses) - np.mean(model_losses)
        ),
        "per_line_model_loss": {k: float(np.mean(v)) for k, v in per_line.items()},
    }
