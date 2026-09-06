"""experiments / exp13 legacy / tx1 predicted response."""

from __future__ import annotations

import logging
from types import MappingProxyType
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence
import numpy as np
import torch
from torch import nn
from src.data.splits import assert_fit_eligible
from src.data.splits import FixedSplit
from src.data.gene_order import encode_batch_labels
from src.experiments.exp13_legacy.stage1_artifact import (
    Stage1ArtifactLoadReport,
    load_stage1_artifact,
)
from src.data.tx1_cache import load_line_cache
from src.model.state import ForwardOnlyStateModel
from src.model.response import _chunk_control_cell_indices
from src.model.initialization import construct_forward_only_model

_LOGGER = logging.getLogger(__name__)


_DROPPED_PREFIXES: Final[tuple[str, ...]] = (
    "response_encoder.",
    "response_pooler.",
    "c_head.",
)


_DROPPED_EXACT_KEYS: Final[frozenset[str]] = frozenset({"control_expression_mean"})


ARM_TX1: Final[str] = "tx1_arm"


ARM_HVG: Final[str] = "hvg_arm"


_VALID_ARMS: Final[frozenset[str]] = frozenset({ARM_TX1, ARM_HVG})


SLICE_SYMBOL_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {"CRIPTO": "TDGF1", "HEMK2": "N6AMT1"}
)


class UnknownPerturbationGeneError(ValueError):
    """A requested gene is outside the ST perturbation vocabulary.

    Raised instead of letting ``Esm2PerturbationAdapter.forward``'s bare
    ``KeyError`` (``state_core.py``) surface from deep inside a forward-pass
    loop.
    """


@dataclass(frozen=True)
class ForwardOnlyLoadReport:
    """Key-by-key outcome of a forward-only load.

    Mirrors ``validate_load_result``'s allowed-vs-disallowed-unexpected
    split and ``WarmStartReport``'s honest-reporting style.
    """

    loaded_keys: tuple[str, ...]
    dropped_keys: tuple[str, ...]


def construct_stage2_model_from_stage1_artifact(
    *,
    model_cls: type[nn.Module],
    checkpoint_path: Path,
    hparams_checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    pert_dim: int,
    target_genes: Sequence[str],
    target_esm_embeddings_path: Path,
    trainable: bool,
    esm2_adapter_hidden: int = 512,
    output_space: str | None = None,
    emit_checkpoint_output: bool = False,
) -> tuple[ForwardOnlyStateModel, Stage1ArtifactLoadReport]:
    """Build a target-universe model and strictly restore Stage 1 weights."""
    model = construct_forward_only_model(
        model_cls=model_cls,
        hparams_checkpoint_path=hparams_checkpoint_path,
        input_dim=input_dim,
        output_dim=output_dim,
        pert_dim=pert_dim,
        genes=target_genes,
        esm2_embeddings_path=target_esm_embeddings_path,
        esm2_adapter_hidden=esm2_adapter_hidden,
        output_space=output_space,
        emit_checkpoint_output=emit_checkpoint_output,
    )
    report = load_stage1_artifact(
        model,
        checkpoint_path=checkpoint_path,
        target_esm_embeddings_path=target_esm_embeddings_path,
        trainable=trainable,
    )
    return model, report


def load_forward_only_checkpoint(
    model: ForwardOnlyStateModel, checkpoint_path: Path
) -> ForwardOnlyLoadReport:
    """Load Phase C's trained ``state_adapter``/``perturbations`` weights only.

    ``checkpoint_path`` is a flat, buffer-free ``AivcModel.state_dict()``
    (a raw ``torch.save``, not a nested Lightning checkpoint). The legacy
    Phase C checkpoint persists ``perturbations.esm_matrix`` and must instead
    go through the sealed-artifact loader, which authenticates it before
    dropping it. This helper loads with ``strict=False`` and validates like
    ``validate_load_result``: a missing destination weight or a disallowed
    unexpected key (outside the known GMM/MLP-head drop list) both raise --
    a partial load must never be silent. Sets ``eval()`` and freezes every
    parameter.

    Raises:
        ValueError: A destination key failed to load, an unexpected
            checkpoint key falls outside the known drop list, or zero keys
            loaded in total.
    """
    checkpoint_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if {
        "perturbations.esm_matrix",
        "perturbations.gene_vocabulary_sha256",
    } & set(checkpoint_state):
        raise ValueError(
            "Authenticated perturbation vocabulary requires a sealed Stage-1 "
            "artifact; "
            "use construct_stage2_model_from_stage1_artifact"
        )
    result = model.load_state_dict(checkpoint_state, strict=False)
    missing = sorted(
        key
        for key in result.missing_keys
        if key != "perturbations.gene_vocabulary_sha256"
    )
    unexpected = sorted(result.unexpected_keys)
    dropped = sorted(key for key in unexpected if _is_expected_drop(key))
    disallowed = sorted(set(unexpected) - set(dropped))
    loaded = sorted(
        key
        for key in model.state_dict()
        if key != "perturbations.gene_vocabulary_sha256" and key not in missing
    )
    if missing or disallowed or not loaded:
        raise ValueError(
            f"Incomplete forward-only ST checkpoint load from {checkpoint_path}: "
            f"missing={missing}, disallowed_unexpected={disallowed}, "
            f"loaded_count={len(loaded)}"
        )
    model.eval()
    model.requires_grad_(False)
    _LOGGER.info(
        "forward-only ST load from %s: %d loaded, %d dropped (expected head keys)",
        checkpoint_path,
        len(loaded),
        len(dropped),
    )
    return ForwardOnlyLoadReport(loaded_keys=tuple(loaded), dropped_keys=tuple(dropped))


def _is_expected_drop(key: str) -> bool:
    return key in _DROPPED_EXACT_KEYS or any(
        key.startswith(prefix) for prefix in _DROPPED_PREFIXES
    )


def vocabulary_genes(perturbations: nn.Module) -> frozenset[str]:
    """Return the closed gene set a perturbation adapter can forward.

    Works for any adapter exposing a ``genes`` attribute (currently only
    ``Esm2PerturbationAdapter``).
    """
    genes = getattr(perturbations, "genes", None)
    if genes is None:
        raise TypeError(
            f"{type(perturbations).__name__} has no gene vocabulary "
            "(`genes` attribute); cannot pre-filter requested genes"
        )
    return frozenset(str(gene) for gene in genes)


def resolve_genes_against_vocabulary(
    genes: Sequence[str],
    perturbations: nn.Module,
    *,
    alias_map: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve requested genes to ST-vocabulary keys, or raise a named error.

    ``Esm2PerturbationAdapter.forward`` is a closed, exact-string dict lookup
    with no fallback: a gene outside its vocabulary raises a bare
    ``KeyError`` with no context. This pre-filters the whole requested list
    and raises :class:`UnknownPerturbationGeneError` naming every missing
    gene, instead of failing on whichever comes up first deep inside a loop.

    ``alias_map`` resolves symbol-history renames (e.g. the frozen slice's
    ``CRIPTO``/``HEMK2`` are perturbed under current HGNC symbols
    ``TDGF1``/``N6AMT1`` in every anchor -- see ``task-0-coverage.md`` §2).
    No case-folding is performed -- membership is an exact string match,
    matching ``Esm2PerturbationAdapter``'s own lookup (case-normalized to
    upper-case internally, see ``state_core.py``).

    Raises:
        UnknownPerturbationGeneError: One or more requested genes (after
            alias resolution) are outside the vocabulary.
    """
    vocabulary = vocabulary_genes(perturbations)
    aliases = dict(SLICE_SYMBOL_ALIASES if alias_map is None else alias_map)
    resolved: list[str] = []
    unresolved: list[str] = []
    for gene in genes:
        key = aliases.get(str(gene), str(gene))
        if key in vocabulary:
            resolved.append(key)
        else:
            unresolved.append(str(gene))
    if unresolved:
        raise UnknownPerturbationGeneError(
            "requested gene(s) are outside the ST perturbation vocabulary "
            f"(after alias resolution, if any): {sorted(unresolved)}"
        )
    return tuple(resolved)


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a configured device string the same way exp05's own
    ``train.TrainConfig.device`` convention does (Wave 3 Codex gate P1-5).

    ``"auto"`` (the default) picks CUDA when available, CPU otherwise --
    Phase D's forward-only ST inference generates roughly ``33 + 9`` lines
    times the full ~587-gene slice worth of forward passes
    (:func:`generate_predicted_response_for_line` calls), which stayed
    CPU-only before this fix because nothing ever resolved or threaded a
    device through it. This is a resolution, not a hard requirement: CPU
    keeps working unchanged for tests and for machines with no GPU.

    Args:
        requested: ``"auto"``, an explicit device string (e.g. ``"cpu"``,
            ``"cuda"``, ``"cuda:0"``), or an already-constructed
            :class:`torch.device`.

    Returns:
        The resolved :class:`torch.device`.
    """
    if isinstance(requested, torch.device):
        return requested
    if str(requested) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(str(requested))


def generate_predicted_response(
    model: ForwardOnlyStateModel,
    control_input: np.ndarray,
    gene: str,
    *,
    cell_set_len: int,
    window_macro_batch_size: int = 1,
    seed: int,
    batch_labels: np.ndarray | None = None,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Forward ST on one line's basal cells for one gene's perturbation.

    ``control_input`` is this line's basal Tx1 embeddings (or, for the HVG
    arm, its basal HVG matrix), shape ``(n_basal_cells, input_dim)``. ``gene``
    is re-checked against the vocabulary here too (raises
    :class:`UnknownPerturbationGeneError`, never a bare ``KeyError``), so
    this is safe to call without pre-resolving. ``seed`` is the deterministic
    padding-resample seed (see :func:`_chunk_control_cell_indices`) and is
    recorded in provenance, since it changes which cells get duplicated into
    the final window and therefore the output for those rows. ``batch_labels``
    are optional raw per-cell batch labels (this
    codebase keys basal control cells by ``model_id`` --
    ``tx1_response_data.py:603-625``); ``None`` omits ``batch`` entirely.

    ``device`` (Wave 3 Codex gate P1-5) is where the forward pass itself
    runs -- ``model`` must already have been moved there by the caller
    (``model.to(device)``; not repeated per call here, since this function
    is invoked once per gene). The returned tensor is always moved back to
    CPU regardless of ``device`` (below), so every downstream consumer
    (including legacy diagnostic callers) stays device-oblivious.

    Returns:
        The predicted response, shape ``(n_basal_cells, response_dim)`` --
        exactly ``control_input``'s row count, padding trimmed off, always
        on CPU. No gradient is tracked (runs under ``torch.no_grad()``).

    Raises:
        UnknownPerturbationGeneError: ``gene`` is outside the ST vocabulary.
    """
    if gene not in vocabulary_genes(model.perturbations):
        raise UnknownPerturbationGeneError(
            f"gene {gene!r} is outside the ST perturbation vocabulary"
        )
    control_input = np.asarray(control_input, dtype=np.float32)
    n_cells = int(control_input.shape[0])
    chunk_indices = _chunk_control_cell_indices(n_cells, cell_set_len, seed)
    if window_macro_batch_size < 1:
        raise ValueError("window_macro_batch_size must be at least 1")
    batch_chunks = _batch_index_chunks(
        batch_labels, batch_lookup, chunk_indices, device
    )
    response: torch.Tensor | None = None
    write_offset = 0
    with torch.no_grad():
        for macro_start in range(0, len(chunk_indices), window_macro_batch_size):
            macro_indices = chunk_indices[
                macro_start : macro_start + window_macro_batch_size
            ]
            macro_batch_chunks = batch_chunks[
                macro_start : macro_start + window_macro_batch_size
            ]
            control_chunks = tuple(
                torch.as_tensor(
                    control_input[indices], dtype=torch.float32, device=device
                )
                for indices in macro_indices
            )
            outputs = model.predict_response_chunks(
                control_chunks, gene, macro_batch_chunks
            )
            if len(outputs) != len(macro_indices):
                raise RuntimeError(
                    f"model returned {len(outputs)} chunks for "
                    f"{len(macro_indices)} inputs"
                )
            for output_chunk_device in outputs:
                output_chunk = output_chunk_device.detach().cpu()
                if response is None:
                    response = torch.empty(
                        (n_cells, int(output_chunk.shape[-1])),
                        dtype=output_chunk.dtype,
                    )
                valid_rows = min(int(output_chunk.shape[0]), n_cells - write_offset)
                response[write_offset : write_offset + valid_rows].copy_(
                    output_chunk[:valid_rows]
                )
                write_offset += valid_rows
            del control_chunks, outputs
    if response is None or write_offset != n_cells:
        raise RuntimeError(
            f"incomplete response generation for {gene}: "
            f"wrote {write_offset}/{n_cells} rows"
        )
    # Always return CPU: `device` is only where the (expensive) forward pass
    # itself runs. Legacy bag-level callers expect a CPU
    # tensor; without this, a non-CPU `device` would silently leave GPU
    # tensors flowing into `LineExamples`/`moment_pool`, which either raises
    # a device-mismatch error against the CPU-resident head or basal bag, or
    # (worse) silently succeeds only when everything happens to already be
    # on the same device.
    return response


def generate_pooled_predicted_response(
    model: ForwardOnlyStateModel,
    control_input: np.ndarray | torch.Tensor,
    gene: str,
    *,
    cell_set_len: int,
    window_macro_batch_size: int,
    seed: int,
    batch_labels: np.ndarray | None = None,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Macro-batch STATE windows and return mean plus population variance.

    The reduction is performed online on the forward device and the compact
    pooled feature remains on that device for the GPU-resident GeneEffect head.
    No full predicted-response bag is retained or written to disk. Padding rows
    from the final cell window are excluded.
    """
    if gene not in vocabulary_genes(model.perturbations):
        raise UnknownPerturbationGeneError(
            f"gene {gene!r} is outside the ST perturbation vocabulary"
        )
    if window_macro_batch_size < 1:
        raise ValueError("window_macro_batch_size must be at least 1")
    resolved_device = torch.device(device)
    if isinstance(control_input, torch.Tensor):
        resident_control = control_input.to(device=resolved_device, dtype=torch.float32)
        n_cells = int(resident_control.shape[0])
        cpu_control: np.ndarray | None = None
    else:
        resident_control = None
        cpu_control = np.asarray(control_input, dtype=np.float32)
        n_cells = int(cpu_control.shape[0])
    chunk_indices = _chunk_control_cell_indices(n_cells, cell_set_len, seed)
    batch_chunks = _batch_index_chunks(
        batch_labels, batch_lookup, chunk_indices, device
    )
    running_mean: torch.Tensor | None = None
    running_m2: torch.Tensor | None = None
    seen = 0
    response_dim: int | None = None
    with torch.no_grad():
        for macro_start in range(0, len(chunk_indices), window_macro_batch_size):
            macro_indices = chunk_indices[
                macro_start : macro_start + window_macro_batch_size
            ]
            macro_batch_chunks = batch_chunks[
                macro_start : macro_start + window_macro_batch_size
            ]
            if resident_control is None:
                assert cpu_control is not None
                control_chunks = tuple(
                    torch.as_tensor(
                        cpu_control[indices],
                        dtype=torch.float32,
                        device=resolved_device,
                    )
                    for indices in macro_indices
                )
            else:
                control_chunks_list: list[torch.Tensor] = []
                for indices in macro_indices:
                    start = int(indices[0])
                    valid_rows = min(cell_set_len, n_cells - start)
                    chunk = resident_control[start : start + valid_rows]
                    if valid_rows < cell_set_len:
                        padding_indices = torch.as_tensor(
                            indices[valid_rows:],
                            dtype=torch.long,
                            device=resolved_device,
                        )
                        chunk = torch.cat(
                            [chunk, resident_control.index_select(0, padding_indices)],
                            dim=0,
                        )
                    control_chunks_list.append(chunk)
                control_chunks = tuple(control_chunks_list)
            outputs = model.predict_response_chunks(
                control_chunks, gene, macro_batch_chunks
            )
            if len(outputs) != len(macro_indices):
                raise RuntimeError(
                    f"model returned {len(outputs)} chunks for "
                    f"{len(macro_indices)} inputs"
                )
            valid_parts: list[torch.Tensor] = []
            for output in outputs:
                valid_rows = min(int(output.shape[0]), n_cells - seen)
                if valid_rows > 0:
                    valid_parts.append(output[:valid_rows].detach().float())
                    seen += valid_rows
            macro_output = torch.cat(valid_parts, dim=0)
            macro_var, macro_mean = torch.var_mean(macro_output, dim=0, unbiased=False)
            macro_count = int(macro_output.shape[0])
            response_dim = int(macro_output.shape[-1])
            if running_mean is None:
                running_mean = macro_mean
                running_m2 = macro_var * macro_count
            else:
                assert running_m2 is not None
                total = seen
                previous = total - macro_count
                delta = macro_mean - running_mean
                running_mean = running_mean + delta * (macro_count / total)
                running_m2 = (
                    running_m2
                    + macro_var * macro_count
                    + delta.square() * (previous * macro_count / total)
                )
            del control_chunks, outputs, valid_parts, macro_output
    if running_mean is None or running_m2 is None or seen != n_cells:
        raise RuntimeError(
            f"incomplete pooled response generation for {gene}: "
            f"saw {seen}/{n_cells} rows"
        )
    variance = (running_m2 / seen).clamp_min(0.0)
    pooled = torch.cat([running_mean, variance], dim=0).detach()
    assert response_dim is not None
    return pooled, response_dim


def _batch_index_chunks(
    batch_labels: np.ndarray | None,
    batch_lookup: Mapping[str, int] | None,
    chunk_indices: tuple[np.ndarray, ...],
    device: torch.device | str,
) -> tuple[torch.Tensor | None, ...]:
    if batch_labels is None:
        return tuple(None for _ in chunk_indices)
    encoded = encode_batch_labels(np.asarray(batch_labels), dict(batch_lookup or {}))
    if encoded is None:
        return tuple(None for _ in chunk_indices)
    return tuple(
        torch.as_tensor(encoded[indices], dtype=torch.long, device=device)
        for indices in chunk_indices
    )


def generate_predicted_response_for_line(
    tx1_cache_dir: Path,
    model_id: str,
    model: ForwardOnlyStateModel,
    gene: str,
    *,
    arm: str,
    cell_set_len: int,
    seed: int,
    split: FixedSplit | None = None,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
    require_fit_eligible: bool = True,
) -> torch.Tensor:
    """Load one line's cached basal view and forward ST for one gene.

    ``arm`` selects which of ``load_line_cache``'s two arrays feeds ST's
    input (``tx1_arm`` -> Tx1 embeddings, ``hvg_arm`` -> HVG matrix).
    ``require_fit_eligible`` (default ``True``) calls
    ``benchmark_split.assert_fit_eligible`` against ``split`` (the
    ``cell_line_geneeffect_226_split`` membership authority) first, refusing
    a ``val``/``test``/``unlabeled_train`` line -- reaching model *training*
    with one is a Critical defect. A held-out line's *inference* is the one
    legitimate caller that must pass ``require_fit_eligible=False`` -- an
    explicit, auditable opt-out, since ``verify_cache``/``load_line_cache``
    are membership-agnostic and serve every line, including val/test.

    Args:
        split: The loaded ``cell_line_geneeffect_226_split`` membership
            authority (:func:`~src.data.splits.
            load_geneeffect_226_split`). Required whenever
            ``require_fit_eligible`` is ``True``.

    Raises:
        ValueError: ``require_fit_eligible`` is ``True`` and ``split`` is
            ``None`` or ``model_id`` is not a labeled train member, or
            ``arm`` is not recognized.
        UnknownPerturbationGeneError: ``gene`` is outside the ST vocabulary.
    """
    if require_fit_eligible:
        if split is None:
            raise ValueError(
                "require_fit_eligible=True requires `split` (the "
                "cell_line_geneeffect_226_split membership authority); pass "
                "require_fit_eligible=False for inference-only val/test calls"
            )
        assert_fit_eligible(model_id, split)
    embeddings, hvg_matrix, _obs = load_line_cache(tx1_cache_dir, model_id)
    if arm == ARM_TX1:
        basal_view = np.asarray(embeddings, dtype=np.float32)
    elif arm == ARM_HVG:
        basal_view = np.asarray(hvg_matrix, dtype=np.float32)
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_VALID_ARMS)}")
    batch_labels = np.full(basal_view.shape[0], model_id, dtype=object)
    return generate_predicted_response(
        model,
        basal_view,
        gene,
        cell_set_len=cell_set_len,
        seed=seed,
        batch_labels=batch_labels,
        batch_lookup=batch_lookup,
        device=device,
    )
