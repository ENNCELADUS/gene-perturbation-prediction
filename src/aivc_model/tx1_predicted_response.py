"""Phase D Task 2: forward-only ST loading and predicted-response generation.

Phase C's checkpoint (``pytorch_model.bin``) is the **entire** ``AivcModel``
state dict: ``state_adapter``/``perturbations`` (real trained signal -- the
29 basal-only Tahoe lines had every response-supervision loss zeroed, so
they never touch these) plus ``response_encoder``/``response_pooler``/
``c_head``/``control_expression_mean`` (the GMM/MLP head Phase D replaces --
see ``.superpowers/sdd/phase-d/discovery-b-data-labels.md`` §2).

No existing loader restores just the ST+perturbations half:
``train.py:_load_authorized_model_checkpoint`` is fold-authority-gated and
raises outside its own run; ``bridge_a.py:load_bridge_a_context`` loads the
*whole* model, head included. This module is that missing loader, following
``bridge_a.py``'s plain-``torch.load`` pattern and
``scripts/verify_tx1_obsm_width.py::validate_load_result``'s honest-load-
reporting convention (this repo's dominant failure mode is a silent
``strict=False`` partial load that leaves weights randomly initialized).

It also generates the predicted-response bag for one (line, gene) pair. The
fingerprinted predicted-response cache (Global Constraint D11) is the sibling
module :mod:`aivc_model.tx1_predicted_response_cache`, split out the same way
Phase B split ``tx1_basal.py`` (build) from ``tx1_embed_cache.py`` (cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from aivc_model.model import PerturbationVectorAdapter, StateForwardAdapter
from aivc_model.prepare import encode_batch_labels, load_perturbation_vectors
from aivc_model.state_warm_start import build_warm_started_state_model
from aivc_model.tx1_embed_cache import load_line_cache
from aivc_model.tx1_geneeffect_data import assert_training_role

_LOGGER = logging.getLogger(__name__)

#: Phase C's checkpoint's response_encoder/response_pooler/c_head submodules
#: and its control_expression_mean buffer are the GMM/MLP head this wave
#: replaces -- expected, intentional drops, not a partial load.
_DROPPED_PREFIXES: Final[tuple[str, ...]] = (
    "response_encoder.",
    "response_pooler.",
    "c_head.",
)
_DROPPED_EXACT_KEYS: Final[frozenset[str]] = frozenset({"control_expression_mean"})

#: The two encoder arms Phase D runs identically (D7): which cached basal
#: view (Tx1 embedding vs. HVG matrix) feeds ST's input for each.
ARM_TX1: Final[str] = "tx1_arm"
ARM_HVG: Final[str] = "hvg_arm"
_VALID_ARMS: Final[frozenset[str]] = frozenset({ARM_TX1, ARM_HVG})


class UnknownPerturbationGeneError(ValueError):
    """A requested gene is outside the ST perturbation vocabulary.

    Raised instead of letting ``PerturbationVectorAdapter.forward``'s bare
    ``KeyError`` (``model.py:291-294``) surface from deep inside a
    forward-pass loop.
    """


@dataclass(frozen=True)
class ForwardOnlyLoadReport:
    """Key-by-key outcome of a forward-only load.

    Mirrors ``validate_load_result``'s allowed-vs-disallowed-unexpected
    split and ``WarmStartReport``'s honest-reporting style.
    """

    loaded_keys: tuple[str, ...]
    dropped_keys: tuple[str, ...]


class ForwardOnlyStateModel(nn.Module):
    """ST + perturbation adapter only -- no response encoder/pooler/head.

    A deliberately small ``nn.Module`` (not ``AivcModel``), holding the two
    submodules Phase D needs under the same attribute names ``AivcModel``
    uses, so its ``state_dict()`` keys are a strict subset of Phase C's
    saved checkpoint and load straight out of it.
    """

    def __init__(
        self, state_adapter: StateForwardAdapter, perturbations: nn.Module
    ) -> None:
        super().__init__()
        self.state_adapter = state_adapter
        self.perturbations = perturbations

    def predict_response_chunks(
        self,
        control_chunks: tuple[torch.Tensor, ...],
        gene: str,
        batch_index_chunks: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Forward ST for one gene (``AivcModel.predict_response_chunks`` minus
        the response-encoder step; Phase D's own head takes over from the raw
        expression-space output).
        """
        perturbation = self.perturbations(gene)
        return self.state_adapter.forward_chunks(
            control_chunks, perturbation, gene, batch_index_chunks
        )


def construct_forward_only_model(
    *,
    model_cls: type[nn.Module],
    hparams_checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    pert_dim: int,
    genes: Sequence[str],
    known_perturbation_vectors: Path | None,
    output_space: str | None = None,
    emit_checkpoint_output: bool = False,
) -> ForwardOnlyStateModel:
    """Build a fresh ST + ``PerturbationVectorAdapter`` pair, ready to warm-load.

    Reuses ``build_warm_started_state_model`` to construct ST from
    ``hparams_checkpoint_path`` (the RELEASED ST checkpoint's architecture
    hparams, not Phase C's own trained ``pytorch_model.bin``), self-warm-
    started from that same file (harmless: every ``state_adapter.*`` weight
    is overwritten by :func:`load_forward_only_checkpoint` next). Only
    ``gene_tokenizer: state_onehot`` is supported (the only tokenizer either
    Phase C arm config uses).

    ``genes`` must be the exact vocabulary, in the exact order, Phase C's
    checkpoint was trained with: :class:`PerturbationVectorAdapter` keys its
    trainable "missing" parameters by construction-time list position, not
    gene identity, so a different order/vocabulary silently loads the WRONG
    gene's vector into each slot once the checkpoint loads.
    ``known_perturbation_vectors`` only sizes/construction-initializes the
    adapter -- every value is overwritten next.

    Returns:
        A :class:`ForwardOnlyStateModel` with freshly constructed (not yet
        Phase-C-trained) weights.
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
    known_vectors = load_perturbation_vectors(known_perturbation_vectors)
    perturbations = PerturbationVectorAdapter(list(genes), known_vectors, int(pert_dim))
    return ForwardOnlyStateModel(StateForwardAdapter(model), perturbations)


def load_forward_only_checkpoint(
    model: ForwardOnlyStateModel, checkpoint_path: Path
) -> ForwardOnlyLoadReport:
    """Load Phase C's trained ``state_adapter``/``perturbations`` weights only.

    ``checkpoint_path`` is the flat ``AivcModel.state_dict()`` Phase C's
    ``_save_model_checkpoint`` wrote (a raw ``torch.save``, not a nested
    Lightning checkpoint). Loads with ``strict=False`` and validates like
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
    result = model.load_state_dict(checkpoint_state, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    dropped = sorted(key for key in unexpected if _is_expected_drop(key))
    disallowed = sorted(set(unexpected) - set(dropped))
    loaded = sorted(set(model.state_dict()) - set(missing))
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

    Works for both ``PerturbationVectorAdapter`` and
    ``Esm2PerturbationAdapter`` -- both expose a ``genes`` attribute.
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

    ``PerturbationVectorAdapter.forward`` is a closed, exact-string dict
    lookup with no fallback: a gene outside its vocabulary raises a bare
    ``KeyError`` with no context. This pre-filters the whole requested list
    and raises :class:`UnknownPerturbationGeneError` naming every missing
    gene, instead of failing on whichever comes up first deep inside a loop.

    ``alias_map`` resolves symbol-history renames (e.g. the frozen slice's
    ``CRIPTO``/``HEMK2`` are perturbed under current HGNC symbols
    ``TDGF1``/``N6AMT1`` in every anchor -- see ``task-0-coverage.md`` §2).
    No case-folding is performed -- membership is an exact string match,
    matching ``PerturbationVectorAdapter``'s own lookup.

    Raises:
        UnknownPerturbationGeneError: One or more requested genes (after
            alias resolution) are outside the vocabulary.
    """
    vocabulary = vocabulary_genes(perturbations)
    aliases = dict(alias_map) if alias_map is not None else {}
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


def _chunk_control_cell_indices(
    n_cells: int, cell_set_len: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Split ``n_cells`` basal-cell row indices into equal ``cell_set_len`` windows.

    ``StateForwardAdapter.forward_chunks`` requires every chunk to be the
    identical size, so a cell count not evenly divisible by ``cell_set_len``
    needs its final window padded. Padding resamples (with replacement,
    deterministically from ``seed``) from the same line's own basal cells,
    mirroring ``prepare.make_cell_set_chunks``'s ``pad_short`` behavior. The
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


def generate_predicted_response(
    model: ForwardOnlyStateModel,
    control_input: np.ndarray,
    gene: str,
    *,
    cell_set_len: int,
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
    part of the D11 cache fingerprint, since it changes which cells get
    duplicated into the final window and therefore the output for those
    rows. ``batch_labels`` are optional raw per-cell batch labels (this
    codebase keys basal control cells by ``model_id`` --
    ``tx1_response_data.py:603-625``); ``None`` omits ``batch`` entirely.

    Returns:
        The predicted response, shape ``(n_basal_cells, response_dim)`` --
        exactly ``control_input``'s row count, padding trimmed off. No
        gradient is tracked (runs under ``torch.no_grad()``).

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
    control_chunks = tuple(
        torch.as_tensor(control_input[indices], dtype=torch.float32, device=device)
        for indices in chunk_indices
    )
    batch_chunks = _batch_index_chunks(
        batch_labels, batch_lookup, chunk_indices, device
    )
    with torch.no_grad():
        outputs = model.predict_response_chunks(control_chunks, gene, batch_chunks)
        response = torch.cat(outputs, dim=0)[:n_cells]
    return response


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
    role: str,
    model: ForwardOnlyStateModel,
    gene: str,
    *,
    arm: str,
    cell_set_len: int,
    seed: int,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
    require_training_role: bool = True,
) -> torch.Tensor:
    """Load one line's cached basal view and forward ST for one gene.

    ``arm`` selects which of ``load_line_cache``'s two arrays feeds ST's
    input (``tx1_arm`` -> Tx1 embeddings, ``hvg_arm`` -> HVG matrix).
    ``require_training_role`` (default ``True``) calls ``assert_training_role``
    first, refusing a ``test``-role line (D6 -- reaching model *training*
    with one is a Critical defect). Phase F/Task 4's held-out-line
    *inference* is the one legitimate caller that must pass ``False`` --
    an explicit, auditable opt-out, since ``verify_cache``/``load_line_cache``
    are role-agnostic and serve every line including the 9 held-out ones.

    Raises:
        ValueError: ``require_training_role`` is ``True`` and ``role`` is not
            an admissible training role, or ``arm`` is not recognized.
        UnknownPerturbationGeneError: ``gene`` is outside the ST vocabulary.
    """
    if require_training_role:
        assert_training_role(role, model_id)
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


__all__ = [
    "ARM_HVG",
    "ARM_TX1",
    "ForwardOnlyLoadReport",
    "ForwardOnlyStateModel",
    "UnknownPerturbationGeneError",
    "construct_forward_only_model",
    "generate_predicted_response",
    "generate_predicted_response_for_line",
    "load_forward_only_checkpoint",
    "resolve_genes_against_vocabulary",
    "vocabulary_genes",
]
