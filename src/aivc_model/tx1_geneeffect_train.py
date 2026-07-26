"""Phase D Task 3: train ``Tx1GeneEffectHead`` over the 28/5/9 split (core).

Trains ``tx1_geneeffect_head.Tx1GeneEffectHead`` on ``(predicted response
bag, basal bag) -> GeneEffect scalar`` over a set of training lines,
selecting the returned checkpoint on a disjoint set of validation lines
(D12). This module is the pure-tensor half: it consumes already-
materialized per-line, per-gene bags (:class:`LineExamples`) and has no
disk/cache dependency. The sibling module
:mod:`aivc_model.tx1_geneeffect_train_io` is the I/O half -- assembling
:class:`LineExamples` from Phase B/Task 2 caches via Task 1's line-role
selector, and writing the trained checkpoint plus its provenance record.
Split the same way Phase B split ``tx1_basal.py``/``tx1_embed_cache.py`` and
Phase D Task 2 split ``tx1_predicted_response.py``/
``tx1_predicted_response_cache.py``: one file for this diff came to well
over CLAUDE.md's 600-line file rule.

Two design points carried over verbatim from the wave's constraints:

* **D3 -- the objective is fixed.** Training minimizes ``rank_variance_loss``
  (imported, not reimplemented) and nothing else. No MSE term is added "to
  stabilize training" -- that is exactly the shortcut the frozen head design
  was built to prevent (spec Sec 2.2).
* **D12/D6 -- validation and test lines must never enter training.**
  :func:`train_tx1_geneeffect_head` re-derives the train/validation
  disjointness check itself (:func:`_assert_examples_admissible`), on top of
  whatever guarantee the caller's line-selection code already gives, and
  every admitted line's manifest ``role`` is re-checked via
  ``tx1_geneeffect_data.assert_training_role``. A ``test``-role line, or a
  line that appears in both partitions, raises before any forward pass runs.

D2 (Phase E needs the pooled feature vector, not just the scalar) is
satisfied by :func:`compute_line_features_and_predictions`, which returns
both the pre-MLP-trunk pooled vector (width ``(response_dim + basal_dim) *
moments``) and the scalar prediction, via the same code path used during
validation -- so Phase E's calibrator consumes exactly what the trained head
actually saw, not a re-derived stand-in.

No GPU training is performed by this module in this task; it is exercised
only against synthetic fixtures.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import torch

from aivc_model.tx1_geneeffect_data import assert_training_role
from aivc_model.tx1_geneeffect_head import (
    Tx1GeneEffectHead,
    moment_pool,
    rank_variance_loss,
)

_LOGGER = logging.getLogger(__name__)

#: The selection metric name recorded in provenance. Named for exactly what
#: it computes -- ``rank_variance_loss`` evaluated on the validation
#: lines -- so the string and the value it is paired with can never drift
#: apart the way Wave 2's "val_c_loss"/``generation_loss`` mismatch did.
SELECTION_METRIC_NAME: Final[str] = "val_rank_variance_loss"


@dataclass(frozen=True)
class LineExamples:
    """All per-gene training/validation examples for one cell line.

    Attributes:
        model_id: The line's ACH model id.
        role: The line's manifest ``role`` (re-checked by
            ``assert_training_role`` at every admission point).
        arm: ``"tx1_arm"`` or ``"hvg_arm"`` -- which basal view
            ``basal_bag`` holds (D7: both arms train the identical head and
            data, just a different basal/response feature space).
        basal_bag: This line's basal (control) per-cell bag, shape ``[M,
            basal_dim]``. Not gene-specific -- reused unmodified across
            every gene in ``genes``.
        genes: The genes this line has both a cached predicted response and
            a finite GeneEffect target for, sorted.
        response_bags: One ragged ``[N_i, response_dim]`` predicted-response
            bag per gene in ``genes``, same order.
        targets: GeneEffect targets, shape ``[len(genes)]``, same order as
            ``genes``.
    """

    model_id: str
    role: str
    arm: str
    basal_bag: torch.Tensor
    genes: tuple[str, ...]
    response_bags: tuple[torch.Tensor, ...]
    targets: torch.Tensor


def _line_pooled_features(head: Tx1GeneEffectHead, line: LineExamples) -> torch.Tensor:
    """The pooled response+basal feature matrix for every gene in ``line``.

    Shape ``[G, (response_dim + basal_dim) * moments]``: response-moments
    columns first, then basal-moments columns, matching the concatenation
    order ``Tx1GeneEffectHead.forward`` uses internally, so this is
    bit-for-bit the vector the head's MLP trunk actually consumes. The
    basal bag is pooled once and broadcast across every gene (it is not
    gene-specific), rather than recomputed per gene.
    """
    pooled_basal = moment_pool(line.basal_bag, moments=head.moments)
    pooled_responses = torch.stack(
        [moment_pool(bag, moments=head.moments) for bag in line.response_bags],
        dim=0,
    )
    basal_expanded = pooled_basal.unsqueeze(0).expand(pooled_responses.shape[0], -1)
    return torch.cat([pooled_responses, basal_expanded], dim=1)


def compute_line_features_and_predictions(
    head: Tx1GeneEffectHead, line: LineExamples
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled features (D2) and scalar predictions for every gene in ``line``.

    This is the export Phase E's calibrator needs: a per-gene ``features``
    matrix in the same space the head used internally, alongside the
    zero-shot scalar prediction. Runs in eval mode under ``no_grad`` --
    intended for post-training feature/inference export, not for gradient
    steps (see the training loop in :func:`train_tx1_geneeffect_head` for
    that).

    Args:
        head: A (typically trained) :class:`Tx1GeneEffectHead`.
        line: One line's :class:`LineExamples`.

    Returns:
        ``(features, predictions)``: ``features`` has shape ``[G,
        (response_dim + basal_dim) * moments]``; ``predictions`` has shape
        ``[G]``, identical to calling ``head.forward`` once per gene.
    """
    head.eval()
    with torch.no_grad():
        features = _line_pooled_features(head, line)
        predictions = head.net(features).squeeze(-1)
    return features, predictions


def _predict_line(head: Tx1GeneEffectHead, line: LineExamples) -> torch.Tensor:
    """Gradient-bearing scalar predictions for every gene in ``line``.

    Same computation as :func:`compute_line_features_and_predictions`,
    without forcing ``eval``/``no_grad`` -- used inside the training loop's
    forward/backward step, where the caller's ``head.train()``/grad context
    must stay in effect.
    """
    features = _line_pooled_features(head, line)
    return head.net(features).squeeze(-1)


def _assert_examples_admissible(
    train_lines: Sequence[LineExamples], validation_lines: Sequence[LineExamples]
) -> None:
    """D6/D12 defense-in-depth guard, re-checked independently of how
    ``train_lines``/``validation_lines`` were assembled.

    Every line in both partitions must carry an admissible training role
    (``assert_training_role`` rejects ``test``); and no ``model_id`` may
    appear in both partitions -- a validation line reaching training would
    not crash and would not look wrong, it would just silently destroy the
    only checkpoint-selection signal this design has (D12).

    Raises:
        ValueError: A line's role is not admissible, or a ``model_id``
            appears in both ``train_lines`` and ``validation_lines``.
    """
    for line in train_lines:
        assert_training_role(line.role, line.model_id)
    for line in validation_lines:
        assert_training_role(line.role, line.model_id)

    train_ids = {line.model_id for line in train_lines}
    validation_ids = {line.model_id for line in validation_lines}
    leaked = sorted(train_ids & validation_ids)
    if leaked:
        raise ValueError(
            f"validation line(s) {leaked} also appear in the training set; a "
            "validation line reaching head training silently destroys the "
            "only checkpoint-selection signal this design has (D12) -- "
            "refusing to train"
        )


def _infer_and_validate_dims(
    train_lines: Sequence[LineExamples], validation_lines: Sequence[LineExamples]
) -> tuple[int, int]:
    """Infer ``(response_dim, basal_dim)``, raising if any line disagrees."""
    all_lines = list(train_lines) + list(validation_lines)
    basal_dims = {int(line.basal_bag.shape[-1]) for line in all_lines}
    if len(basal_dims) != 1:
        raise ValueError(
            f"basal_bag feature width disagrees across lines: {sorted(basal_dims)}"
        )
    response_dims = {
        int(bag.shape[-1]) for line in all_lines for bag in line.response_bags
    }
    if len(response_dims) != 1:
        raise ValueError(
            "response_bag feature width disagrees across examples: "
            f"{sorted(response_dims)}"
        )
    return next(iter(response_dims)), next(iter(basal_dims))


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for :func:`train_tx1_geneeffect_head`.

    Attributes:
        hidden: ``Tx1GeneEffectHead`` MLP trunk width.
        moments: Moment count forwarded to ``moment_pool`` (see
            ``tx1_geneeffect_head.moment_pool``).
        lam: Weight on ``variance_matching_loss`` inside
            ``rank_variance_loss`` (D3's only knob; default 1.0 per the
            registered objective).
        learning_rate: Adam learning rate.
        epochs: Number of full passes over ``train_lines``.
        seed: Seed for ``torch.manual_seed`` (head weight initialization).
    """

    hidden: int = 256
    moments: int = 2
    lam: float = 1.0
    learning_rate: float = 1e-3
    epochs: int = 200
    seed: int = 0


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch's train/validation loss."""

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """The outcome of one :func:`train_tx1_geneeffect_head` run.

    Attributes:
        best_epoch: The epoch whose state was restored into the returned
            head (lowest ``validation_loss`` in ``history``).
        best_validation_metric: That epoch's ``validation_loss``.
        selection_metric: Always :data:`SELECTION_METRIC_NAME` -- named
            explicitly (rather than left implicit) so provenance can record
            the metric actually used, not a hand-typed string that could
            drift from the real computation.
        history: Every epoch's metrics, in epoch order.
        train_model_ids: The training lines used, sorted.
        validation_model_ids: The validation lines used, sorted.
    """

    best_epoch: int
    best_validation_metric: float
    selection_metric: str
    history: tuple[EpochMetrics, ...]
    train_model_ids: tuple[str, ...]
    validation_model_ids: tuple[str, ...]


def _select_best_epoch(history: Sequence[EpochMetrics]) -> EpochMetrics:
    """Pick the epoch with the lowest ``validation_loss`` (D12's rule).

    Reads only ``validation_loss`` -- never ``train_loss`` -- since
    checkpoint selection must be driven exclusively by the validation
    lines. Split into its own function so that property is directly
    unit-testable against an adversarial ``history`` where the
    train-loss-best and validation-loss-best epochs disagree. Ties keep the
    earliest epoch (``min`` is stable).

    Raises:
        ValueError: ``history`` is empty.
    """
    if not history:
        raise ValueError("history must be non-empty to select a best epoch")
    return min(history, key=lambda entry: entry.validation_loss)


def train_tx1_geneeffect_head(
    train_lines: Sequence[LineExamples],
    validation_lines: Sequence[LineExamples],
    config: TrainingConfig = TrainingConfig(),
) -> tuple[Tx1GeneEffectHead, TrainingResult]:
    """Train ``Tx1GeneEffectHead`` over ``train_lines``, selecting on
    ``validation_lines``.

    D3: the loss is ``rank_variance_loss`` and nothing else -- no MSE term
    is added. D12: ``validation_lines`` is used only to pick which epoch's
    weights are returned (via :func:`_select_best_epoch`); it never
    contributes a gradient. One optimizer step is taken per training line
    per epoch (line-level batches, each spanning that line's full gene
    panel), since ``rank_variance_loss`` needs a same-line/same-batch
    correlation signal across genes to be meaningful.

    Args:
        train_lines: The 28 training lines' examples (order-independent;
            sorted internally for determinism).
        validation_lines: The 5 validation lines' examples.
        config: Hyperparameters.

    Returns:
        ``(head, result)``: ``head`` has the epoch-``result.best_epoch``
        weights loaded; ``result`` records the full training/selection
        history.

    Raises:
        ValueError: Either line list is empty, a line's role is not
            admissible, a validation-line ``model_id`` also appears in
            ``train_lines``, or response/basal feature widths disagree
            across lines.
    """
    if not train_lines:
        raise ValueError("train_lines must be non-empty")
    if not validation_lines:
        raise ValueError(
            "validation_lines must be non-empty -- D12 selection has no "
            "signal without at least one validation line"
        )
    _assert_examples_admissible(train_lines, validation_lines)
    response_dim, basal_dim = _infer_and_validate_dims(train_lines, validation_lines)

    torch.manual_seed(int(config.seed))
    head = Tx1GeneEffectHead(
        response_dim=response_dim,
        basal_dim=basal_dim,
        hidden=config.hidden,
        moments=config.moments,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)

    ordered_train = sorted(train_lines, key=lambda line: line.model_id)
    ordered_validation = sorted(validation_lines, key=lambda line: line.model_id)

    history: list[EpochMetrics] = []
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    for epoch in range(config.epochs):
        head.train()
        train_losses: list[float] = []
        for line in ordered_train:
            optimizer.zero_grad()
            predictions = _predict_line(head, line)
            loss = rank_variance_loss(predictions, line.targets, lam=config.lam)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        validation_losses: list[float] = []
        for line in ordered_validation:
            _, predictions = compute_line_features_and_predictions(head, line)
            loss = rank_variance_loss(predictions, line.targets, lam=config.lam)
            validation_losses.append(float(loss))

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(np.mean(train_losses)),
                validation_loss=float(np.mean(validation_losses)),
            )
        )
        epoch_states[epoch] = copy.deepcopy(head.state_dict())

    best_entry = _select_best_epoch(history)
    head.load_state_dict(epoch_states[best_entry.epoch])
    head.eval()

    _LOGGER.info(
        "tx1_geneeffect_head training complete: selected epoch %d/%d "
        "(%s=%.6f) over %d train / %d validation lines",
        best_entry.epoch,
        config.epochs - 1,
        SELECTION_METRIC_NAME,
        best_entry.validation_loss,
        len(ordered_train),
        len(ordered_validation),
    )
    result = TrainingResult(
        best_epoch=best_entry.epoch,
        best_validation_metric=best_entry.validation_loss,
        selection_metric=SELECTION_METRIC_NAME,
        history=tuple(history),
        train_model_ids=tuple(line.model_id for line in ordered_train),
        validation_model_ids=tuple(line.model_id for line in ordered_validation),
    )
    return head, result


__all__ = [
    "SELECTION_METRIC_NAME",
    "EpochMetrics",
    "LineExamples",
    "TrainingConfig",
    "TrainingResult",
    "compute_line_features_and_predictions",
    "train_tx1_geneeffect_head",
]
