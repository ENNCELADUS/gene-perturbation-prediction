"""model / initialization."""

from __future__ import annotations
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any
import torch
from torch import nn
from typing import Sequence
from src.data.embeddings import load_esm2_embeddings
from src.model.perturbation import Esm2PerturbationAdapter
from src.model.state import StateForwardAdapter
from src.model.state import ForwardOnlyStateModel

logger = logging.getLogger(__name__)


LEGAL_OUTPUT_SPACES = frozenset({"embedding", "gene", "all"})


def validate_output_space(output_space: str | None) -> None:
    """Raise if ``output_space`` is set but not one of STATE's legal values.

    Args:
        output_space: A candidate override, or ``None`` (always legal --
            ``None`` means "leave the checkpoint's own value alone").

    Raises:
        ValueError: If ``output_space`` is not ``None`` and not one of
            ``LEGAL_OUTPUT_SPACES``.
    """
    if output_space is not None and output_space not in LEGAL_OUTPUT_SPACES:
        msg = (
            f"Unsupported output_space {output_space!r}. Expected one of "
            f"{sorted(LEGAL_OUTPUT_SPACES)} "
            "(state.tx.models.base.PerturbationModel)."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class WarmStartReport:
    """Key-by-key outcome of a shape-filtered checkpoint warm start.

    Mirrors ``src.model.tx1.validate_load_result``'s
    honest-load-reporting shape, adapted for a shape filter rather than a
    strict key-name-only load. Every destination-model parameter/buffer name
    is accounted for in exactly one of ``loaded_keys``, ``shape_skipped_keys``,
    or ``missing_keys``; ``unexpected_keys`` are checkpoint keys the
    destination model does not have at all.
    """

    loaded_keys: tuple[str, ...]
    shape_skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def warm_start_state_dict(model: nn.Module, checkpoint_path: Path) -> WarmStartReport:
    """Warm-start ``model`` from a checkpoint, skipping shape-mismatched keys.

    Loads ``checkpoint_path`` with a raw ``torch.load`` (bypassing Lightning's
    ``load_from_checkpoint``, which loads the full state dict unfiltered and
    raises ``RuntimeError`` on any shape mismatch), filters the checkpoint's
    ``state_dict`` down to keys whose shape matches ``model``'s own, and
    loads only those with ``strict=False``. Keys left unmatched (by name or
    shape) keep ``model``'s freshly initialized values untouched.

    Args:
        model: Destination module, already constructed with its final
            shapes (e.g. a fresh 2560-input
            ``StateTransitionPerturbationModel``).
        checkpoint_path: Path to a Lightning ``.ckpt`` file with a top-level
            ``state_dict`` key.

    Returns:
        A :class:`WarmStartReport` naming every loaded, shape-skipped,
        missing, and unexpected key.

    Raises:
        ValueError: If zero keys are loadable. A warm start that silently
            loads nothing is worse than a crash: it would produce a
            randomly-initialized model that trains, converges to garbage,
            and reports no error.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["state_dict"]
    model_state = model.state_dict()

    loaded: dict[str, torch.Tensor] = {}
    shape_skipped: list[str] = []
    unexpected: list[str] = []
    for name, tensor in checkpoint_state.items():
        destination = model_state.get(name)
        if destination is None:
            unexpected.append(name)
        elif tuple(tensor.shape) != tuple(destination.shape):
            shape_skipped.append(name)
        else:
            loaded[name] = tensor
    missing = sorted(set(model_state) - set(loaded) - set(shape_skipped))

    if not loaded:
        msg = (
            f"Warm start from {checkpoint_path} loaded zero keys into "
            f"{type(model).__name__} -- refusing a silent no-op warm start "
            "that would train a randomly-initialized model with no error."
        )
        raise ValueError(msg)

    model.load_state_dict(loaded, strict=False)

    report = WarmStartReport(
        loaded_keys=tuple(sorted(loaded)),
        shape_skipped_keys=tuple(sorted(shape_skipped)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(unexpected)),
    )
    logger.info(
        "STATE warm start from %s: %d loaded, %d shape-skipped, %d missing, "
        "%d unexpected. shape_skipped=%s missing=%s unexpected=%s",
        checkpoint_path,
        len(report.loaded_keys),
        len(report.shape_skipped_keys),
        len(report.missing_keys),
        len(report.unexpected_keys),
        report.shape_skipped_keys,
        report.missing_keys,
        report.unexpected_keys,
    )
    return report


def build_warm_started_state_model(
    *,
    model_cls: type[nn.Module],
    hparams_checkpoint_path: Path,
    warm_start_from: Path,
    input_dim: int,
    output_dim: int,
    pert_dim: int,
    output_space: str | None,
    emit_checkpoint_output: bool,
) -> tuple[nn.Module, WarmStartReport]:
    """Construct a fresh STATE model from a checkpoint's hparams, then warm start.

    Reads ``hparams_checkpoint_path``'s saved ``hyper_parameters`` (hidden
    size, transformer backbone, batch count, ...) and overrides only
    ``input_dim``/``output_dim``/``pert_dim``/``output_space`` with the
    caller's values, so the transformer body's other dimensions come from
    the reference checkpoint rather than being hardcoded here. The model is
    built directly (not through ``load_from_checkpoint``), because
    Lightning's loader would then try to load the reference checkpoint's
    state dict unfiltered and raise ``RuntimeError`` on the (deliberately
    mismatched) input projection.

    Args:
        model_cls: ``StateTransitionPerturbationModel`` (or a compatible
            class taking these hparams as constructor kwargs).
        hparams_checkpoint_path: Checkpoint whose non-input hparams (hidden
            size, transformer backbone, batch count, ...) seed construction.
        warm_start_from: Checkpoint whose weights are warm-started via
            :func:`warm_start_state_dict` (usually the same file as
            ``hparams_checkpoint_path``, but not required to be).
        input_dim: Fresh input (ST basal-encoder) dimension, e.g. 2560 for
            the Tx1 arm.
        output_dim: Fresh output dimension.
        pert_dim: Fresh perturbation-token dimension.
        output_space: STATE ``output_space`` override, or ``None`` to keep
            the reference checkpoint's own saved value.
        emit_checkpoint_output: Whether to let the model constructor print
            its architecture summary to stdout/stderr.

    Returns:
        A ``(model, report)`` pair: the constructed-and-warm-started module,
        and the :class:`WarmStartReport` from the warm-start step.
    """
    validate_output_space(output_space)
    reference = torch.load(
        hparams_checkpoint_path, map_location="cpu", weights_only=False
    )
    hparams = dict(reference["hyper_parameters"])
    hparams["input_dim"] = int(input_dim)
    hparams["output_dim"] = int(output_dim)
    hparams["pert_dim"] = int(pert_dim)
    if output_space is not None:
        hparams["output_space"] = output_space
    output_context = (
        nullcontext() if emit_checkpoint_output else _suppress_checkpoint_output()
    )
    with output_context:
        model = model_cls(**hparams)
    report = warm_start_state_dict(model, warm_start_from)
    return model, report


@contextmanager
def _suppress_checkpoint_output() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def construct_forward_only_model(
    *,
    model_cls: type[nn.Module],
    hparams_checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    pert_dim: int,
    genes: Sequence[str],
    esm2_embeddings_path: Path,
    esm2_adapter_hidden: int = 512,
    output_space: str | None = None,
    emit_checkpoint_output: bool = False,
) -> ForwardOnlyStateModel:
    """Build a fresh ST + ``Esm2PerturbationAdapter`` pair, ready to warm-load.

    Reuses ``build_warm_started_state_model`` to construct ST from
    ``hparams_checkpoint_path`` (the RELEASED ST checkpoint's architecture
    hparams, not a Phase C arm's own trained ``pytorch_model.bin``), self-
    warm-started from that same file (harmless: every ``state_adapter.*``
    weight is overwritten by :func:`load_forward_only_checkpoint` next).
    ``p_g = A_phi(E_ESM2(protein(g)))`` (``01-blueprint.md`` §3): the K562
    one-hot ``state_onehot`` tokenizer is retired, so this always builds an
    :class:`~src.model.perturbation.Esm2PerturbationAdapter`.

    ``Esm2PerturbationAdapter`` looks its per-gene vector up by symbol, not
    construction-time list position, so (unlike the retired one-hot adapter)
    a different ``genes`` order cannot silently bind the WRONG gene's vector
    into a slot. It still fails loudly and immediately, at construction, if
    ``esm2_embeddings_path`` is missing a vector for any requested gene --
    never a zero vector, never a silently dropped gene.

    Args:
        esm2_embeddings_path: Precomputed ESM2 ``.npz`` (``keys`` ``symbols``/
            ``vectors``/``resolved``, see :func:`~src.data.embeddings.
            load_esm2_embeddings`); must resolve every gene in ``genes``.
        esm2_adapter_hidden: Hidden width of the ESM2 -> STATE pert-space
            adapter MLP (``PertAdapter``, 1280 -> ``esm2_adapter_hidden`` ->
            ``pert_dim``).

    Returns:
        A :class:`ForwardOnlyStateModel` with freshly constructed (not yet
        trained) weights.

    Raises:
        ValueError: ``esm2_embeddings_path`` has no vector for one or more
            genes in ``genes``.
    """
    model, _self_warm_start_report = build_warm_started_state_model(
        model_cls=model_cls,
        hparams_checkpoint_path=hparams_checkpoint_path,
        warm_start_from=hparams_checkpoint_path,
        input_dim=input_dim,
        output_dim=output_dim,
        pert_dim=pert_dim,
        output_space=output_space,
        emit_checkpoint_output=emit_checkpoint_output,
    )
    esm2_table = load_esm2_embeddings(esm2_embeddings_path)
    perturbations = Esm2PerturbationAdapter(
        list(genes),
        esm2_table,
        adapter_hidden=esm2_adapter_hidden,
        pert_dim=int(pert_dim),
    )
    return ForwardOnlyStateModel(StateForwardAdapter(model), perturbations)
