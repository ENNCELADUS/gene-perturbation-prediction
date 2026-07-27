"""Phase D Task 3: assembly, checkpoint, and provenance I/O (sibling of
:mod:`aivc_model.tx1_geneeffect_train`).

The I/O half of Task 3's split streams Task 2's forward-only predicted
responses directly into moment pooling, retaining only compact
:class:`~aivc_model.tx1_geneeffect_train.PooledLineExamples`. It never writes
or reads a Phase D predicted-response cache. It also saves the trained head's
checkpoint and provenance record. Split out of
``tx1_geneeffect_train.py`` the same way Phase B split ``tx1_basal.py``
(build) from ``tx1_embed_cache.py`` (cache) and Phase D Task 2 split
``tx1_predicted_response.py`` from ``tx1_predicted_response_cache.py`` --
one file for this diff came to well over CLAUDE.md's 600-line rule.

Provenance records every input hash the wave's brief requires: the ST
checkpoint, the Phase B cache manifest hashes for every line actually used,
the DepMap GeneEffect source hash, the seed, the validation-line set, and
the metric truly used for selection -- read directly off the training
result, never restated by hand (see :func:`build_provenance`), so this can
never reproduce Wave 2's bug of a recorded ``selection_metric`` string that
did not match what selection actually used.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_embed_cache import load_line_cache
from aivc_model.tx1_geneeffect_data import assert_training_role, build_line_role_split
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead, moment_pool
from aivc_model.tx1_geneeffect_train import (
    PooledLineExamples,
    TrainingConfig,
    TrainingResult,
)
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    ForwardOnlyStateModel,
    generate_predicted_response,
)

_VALID_ARMS = frozenset({ARM_TX1, ARM_HVG})
_LOGGER = logging.getLogger(__name__)


def assemble_line_features(
    model_id: str,
    role: str,
    arm: str,
    tx1_cache_dir: Path,
    gene_effect: pd.Series,
    model: ForwardOnlyStateModel,
    *,
    moments: int,
    cell_set_len: int,
    seed: int,
    batch_lookup: Mapping[str, int] | None = None,
    device: torch.device | str = "cpu",
) -> PooledLineExamples:
    """Generate and pool one line gene-by-gene without a response cache.

    Only the Phase B basal cache is read. Each raw predicted-response tensor
    is reduced to moments immediately, then released before the next gene.
    No Phase D tensor is read from or written to disk.
    """
    assert_training_role(role, model_id)
    if arm not in _VALID_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(_VALID_ARMS)}")

    embeddings, hvg_matrix, _obs = load_line_cache(tx1_cache_dir, model_id)
    raw_basal_view = embeddings if arm == ARM_TX1 else hvg_matrix
    basal_view = np.asarray(raw_basal_view, dtype=np.float32)
    finite_targets = gene_effect[gene_effect.notna()]
    genes = sorted(str(gene) for gene in finite_targets.index)
    if not genes:
        raise ValueError(f"line {model_id}: no finite GeneEffect target; cannot train")

    # Phase B arrays are read-only mmaps. Zero-copy is essential here (a Tx1
    # line can be tens of GB); moment_pool only reads the tensor.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        basal_tensor = torch.from_numpy(basal_view)
    pooled_basal = moment_pool(basal_tensor, moments=moments)
    batch_labels = np.full(basal_view.shape[0], model_id, dtype=object)
    pooled_responses: list[torch.Tensor] = []
    response_dim: int | None = None
    _LOGGER.info(
        "streaming Phase D features for %s/%s: %d genes", model_id, arm, len(genes)
    )
    for gene_index, gene in enumerate(genes, start=1):
        response = generate_predicted_response(
            model,
            basal_view,
            gene,
            cell_set_len=cell_set_len,
            seed=seed,
            batch_labels=batch_labels,
            batch_lookup=batch_lookup,
            device=device,
        )
        response_dim = int(response.shape[-1])
        pooled_responses.append(moment_pool(response, moments=moments))
        del response
        if gene_index % 50 == 0 or gene_index == len(genes):
            _LOGGER.info(
                "streamed %s/%s: %d/%d genes",
                model_id,
                arm,
                gene_index,
                len(genes),
            )
    pooled_response_matrix = torch.stack(pooled_responses, dim=0)
    features = torch.cat(
        [
            pooled_response_matrix,
            pooled_basal.unsqueeze(0).expand(len(genes), -1),
        ],
        dim=1,
    )
    targets = torch.as_tensor(
        [float(finite_targets[gene]) for gene in genes], dtype=torch.float32
    )
    return PooledLineExamples(
        model_id=str(model_id),
        role=str(role),
        arm=str(arm),
        genes=tuple(genes),
        features=features,
        targets=targets,
        response_dim=int(response_dim),
        basal_dim=int(basal_view.shape[-1]),
        moments=int(moments),
    )


def assemble_split_features(
    manifest: pd.DataFrame,
    validation_registration: Mapping[str, object],
    *,
    arm: str,
    tx1_cache_dir: Path,
    gene_effect_by_line: Mapping[str, pd.Series],
    model: ForwardOnlyStateModel,
    moments: int,
    cell_set_len: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[list[PooledLineExamples], list[PooledLineExamples]]:
    """Stream and pool train/validation features via Task 1's own split.

    Calls ``tx1_geneeffect_data.build_line_role_split`` -- never re-derives
    the 28/5/9 split -- then calls :func:`assemble_line_features` once per
    admitted line: the 28 training lines (24 ``train_head`` + 4 anchors) and
    the 5 validation lines. The 9 frozen ``test`` lines never appear in
    either returned list, by construction of ``build_line_role_split``.

    Args:
        manifest: The frozen manifest, as returned by
            ``tx1_basal.load_line_manifest``.
        validation_registration: The derived validation-line registration,
            as returned by
            ``tx1_geneeffect_data.load_validation_line_registration``.
        arm: ``"tx1_arm"`` or ``"hvg_arm"``.
        tx1_cache_dir: Root of the Phase B basal-embedding cache.
        gene_effect_by_line: ``model_id -> gene_effect`` Series (see
            :func:`assemble_line_features`), covering every training and
            validation line.

    Returns:
        ``(train_lines, validation_lines)``.
    """
    split = build_line_role_split(manifest, validation_registration)
    role_by_id = {
        str(row.model_id): str(row.role) for row in manifest.itertuples(index=False)
    }

    def _assemble(model_id: str) -> PooledLineExamples:
        return assemble_line_features(
            model_id,
            role_by_id[model_id],
            arm,
            tx1_cache_dir,
            gene_effect_by_line[model_id],
            model,
            moments=moments,
            cell_set_len=cell_set_len,
            seed=seed,
            device=device,
        )

    train_lines = [_assemble(model_id) for model_id in split.train_model_ids]
    validation_lines = [_assemble(model_id) for model_id in split.validation_model_ids]
    return train_lines, validation_lines


def _hidden_width(head: Tx1GeneEffectHead) -> int:
    """Recover the MLP trunk's hidden width from the head itself.

    ``Tx1GeneEffectHead`` does not store ``hidden`` as an attribute (only
    ``response_dim``/``basal_dim``/``moments``); it is only reflected in
    ``net[0]``'s output width. Reading it back from the object itself
    (rather than requiring the caller to pass ``config.hidden`` again)
    keeps the checkpoint payload self-describing from a single source of
    truth.
    """
    first_linear = head.net[0]
    return int(first_linear.out_features)


def save_checkpoint(head: Tx1GeneEffectHead, path: Path) -> Path:
    """Save a self-describing checkpoint: state dict plus constructor args.

    Args:
        head: The (typically best-epoch) trained head.
        path: Destination file path.

    Returns:
        ``path``.
    """
    payload = {
        "state_dict": head.state_dict(),
        "response_dim": int(head.response_dim),
        "basal_dim": int(head.basal_dim),
        "moments": int(head.moments),
        "hidden": _hidden_width(head),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path) -> Tx1GeneEffectHead:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is deliberate: this checkpoint is this module's
    own trusted output (a dict of tensors plus four plain ints), not an
    externally sourced/adversarial file.

    Returns:
        A reconstructed :class:`Tx1GeneEffectHead` in eval mode with the
        saved weights loaded.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    head = Tx1GeneEffectHead(
        response_dim=int(payload["response_dim"]),
        basal_dim=int(payload["basal_dim"]),
        hidden=int(payload["hidden"]),
        moments=int(payload["moments"]),
    )
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head


@dataclass(frozen=True)
class Tx1GeneEffectProvenance:
    """The provenance record accompanying one trained-head checkpoint.

    Names every input hash the wave's brief requires (D3/D12): the ST
    checkpoint, the Phase B cache manifest hashes for every line actually
    used, the DepMap GeneEffect source hash, the seed, the validation-line
    set, and the metric truly used for selection.
    """

    st_checkpoint_sha256: str
    phase_b_cache_manifest_hashes: Mapping[str, Mapping[str, str]]
    depmap_gene_effect_sha256: str
    seed: int
    validation_model_ids: tuple[str, ...]
    train_model_ids: tuple[str, ...]
    selection_metric: str
    selection_metric_value: float
    best_epoch: int
    arm: str
    config: Mapping[str, object]
    history: tuple[Mapping[str, float], ...]
    response_generation: Mapping[str, object] = field(default_factory=dict)


def _phase_b_line_hashes(phase_b_manifest_path: Path, model_id: str) -> dict[str, str]:
    """Read one line's recorded embeddings/HVG hashes from Phase B's
    cache-root ``manifest.json``.

    Mirrors, independently (not by import, since
    ``tx1_predicted_response_cache._phase_b_line_hashes`` is module-private),
    the exact same read Task 2's cache fingerprint performs -- so this
    provenance record's hashes come from the identical source of truth
    Task 2's own staleness check already relies on.

    Raises:
        ValueError: ``model_id`` is not recorded, or its entry is missing
            either array's hash.
    """
    manifest = json.loads(Path(phase_b_manifest_path).read_text())
    lines = manifest.get("lines") if isinstance(manifest, dict) else None
    entry = lines.get(str(model_id)) if isinstance(lines, dict) else None
    if entry is None:
        raise ValueError(
            f"line {model_id}: not recorded in Phase B cache manifest "
            f"{phase_b_manifest_path}"
        )
    arrays = entry.get("arrays") if isinstance(entry, dict) else None
    if not isinstance(arrays, dict):
        raise ValueError(f"line {model_id}: Phase B manifest entry has no arrays")
    try:
        return {
            "embeddings_sha256": str(arrays["embeddings.npy"]["sha256"]),
            "hvg_sha256": str(arrays["hvg.npy"]["sha256"]),
        }
    except KeyError as exc:
        raise ValueError(
            f"line {model_id}: Phase B manifest entry is missing {exc} hash"
        ) from exc


def build_provenance(
    *,
    st_checkpoint_path: Path,
    phase_b_manifest_path: Path,
    depmap_gene_effect_path: Path,
    arm: str,
    config: TrainingConfig,
    result: TrainingResult,
    response_generation: Mapping[str, object] | None = None,
) -> Tx1GeneEffectProvenance:
    """Assemble the provenance record for one
    ``tx1_geneeffect_train.train_tx1_geneeffect_head`` run.

    ``selection_metric``/``selection_metric_value`` are read directly off
    ``result`` (itself produced by ``tx1_geneeffect_train._select_best_epoch``)
    rather than passed in separately, so this can never reproduce Wave 2's
    bug of a recorded ``selection_metric`` string that did not match the
    metric selection actually used.

    Args:
        st_checkpoint_path: Path to Phase C's trained ``pytorch_model.bin``.
        phase_b_manifest_path: Path to Phase B's cache-root ``manifest.json``.
        depmap_gene_effect_path: Path to the local ``CRISPRGeneEffect.csv``.
        arm: ``"tx1_arm"`` or ``"hvg_arm"``.
        config: The hyperparameters this run used.
        result: This run's ``TrainingResult``.

    Returns:
        The assembled :class:`Tx1GeneEffectProvenance`.
    """
    all_model_ids = sorted(
        set(result.train_model_ids) | set(result.validation_model_ids)
    )
    phase_b_hashes = {
        model_id: _phase_b_line_hashes(Path(phase_b_manifest_path), model_id)
        for model_id in all_model_ids
    }
    return Tx1GeneEffectProvenance(
        st_checkpoint_sha256=sha256_file(Path(st_checkpoint_path)),
        phase_b_cache_manifest_hashes=phase_b_hashes,
        depmap_gene_effect_sha256=sha256_file(Path(depmap_gene_effect_path)),
        seed=int(config.seed),
        validation_model_ids=result.validation_model_ids,
        train_model_ids=result.train_model_ids,
        selection_metric=result.selection_metric,
        selection_metric_value=result.best_validation_metric,
        best_epoch=result.best_epoch,
        arm=str(arm),
        config=asdict(config),
        history=tuple(
            {
                "epoch": entry.epoch,
                "train_loss": entry.train_loss,
                "validation_loss": entry.validation_loss,
            }
            for entry in result.history
        ),
        response_generation=dict(response_generation or {}),
    )


def write_provenance(provenance: Tx1GeneEffectProvenance, path: Path) -> Path:
    """Write ``provenance`` as indented, sorted-key JSON.

    Returns:
        ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "Tx1GeneEffectProvenance",
    "assemble_line_features",
    "assemble_split_features",
    "build_provenance",
    "load_checkpoint",
    "save_checkpoint",
    "write_provenance",
]
