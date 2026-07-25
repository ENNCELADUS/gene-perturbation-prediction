"""Multi-line observed-response ``GeneBags`` assembly (Wave 2 Phase C, Task 5).

ST's supervision target is observed post-perturbation **gene expression**
(HVG width); ST's input is basal **Tx1 embeddings** (2560-d,
``tx1_embed_cache.EMBEDDING_WIDTH``). This assembles a two-view
``GeneBags`` (``prepare.GeneBags``, Task 1's ``target_*`` fields) spanning
exactly the frozen manifest's four ``train_response_and_head`` lines (C5):

- **input**: the Phase B Tx1 basal embedding cache's per-line control-cell
  embeddings (``tx1_embed_cache.load_line_cache``) -- already built, never
  recomputed;
- **target**: CPU-only raw-count extraction of *perturbed* cells' HVG
  expression from the raw sources in ``perturbseq_sources.json``, via the
  new ``build_*_response_adata`` siblings in ``tx1_basal.py``.

No Tx1 inference runs here. The target side is aligned to the checkpoint's
HVG gene order via ``prepare.resolve_state_gene_order``, the same
mechanism ``tx1_embed_cache._resolve_hvg_matrix`` uses on the basal side
(mirrored locally, not imported, to avoid a private cross-module dependency).

Three design decisions (expanded in the task report): (1) per-line bags,
not a cross-line gene intersection -- a gene tested in more than one line
gets one bag *per line*, disambiguated by ``metadata["model_id"]``, since
the four sources' gene panels differ wildly in size; (2) ``l2_normalize``
(C7) defaults to ``False`` (Phase B's measured un-normalized Tx1 output),
recorded in every ``metadata["l2_normalize"]`` row; (3)
``batch_bags``/``control_batch`` carry each bag's ``model_id`` so
``prepare._sample_control_indices``'s existing batch-matched control
sampling (already used by ``merge_gene_bag_pool``) draws a gene's control
cells from its *own* line once wired into training, not a pool blended
across lines. ``input_bags`` (unused once ``target_bags`` is set) are NaN
placeholders so an accidental read fails loudly rather than training on a
fabricated number. ``latent_bags``/``control_latent`` fall back to the
target (expression) view -- the legacy "expression-as-latent" convention,
pointed at the space actually meaningful in the two-view world.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from aivc_model.prepare import GeneBags, resolve_state_gene_order
from aivc_model.tx1_basal import (
    build_perturbseq_response_adata,
    build_xatlas_orion_response_adata,
    load_line_manifest,
)
from aivc_model.tx1_embed_cache import (
    EMBEDDING_WIDTH,
    PerturbseqSource,
    PerturbseqSourceConfig,
    XatlasOrionSource,
    load_hvg_gene_order,
    load_line_cache,
)

_LOGGER = logging.getLogger(__name__)

#: C5/C4: the only role this module ever reads from the frozen manifest.
_TRAIN_RESPONSE_ROLE = "train_response_and_head"

#: C1's Perturb-seq-derived ``basal_source`` value (Tahoe DMSO has no labels).
_PERTURBSEQ_BASAL_SOURCE = "Perturb-seq non-targeting control"

#: ``--perturbseq-source-config`` ``"source_type"`` values, mirroring
#: ``scripts/build_tx1_basal_embeddings.py`` (not imported: layering runs
#: scripts -> aivc_model, not the reverse).
_SOURCE_TYPE_H5AD = "h5ad"
_SOURCE_TYPE_XATLAS_PARQUET = "xatlas_orion_parquet"
_VALID_SOURCE_TYPES = frozenset({_SOURCE_TYPE_H5AD, _SOURCE_TYPE_XATLAS_PARQUET})
_H5AD_SOURCE_CONFIG_KEYS = (
    "h5ad_path",
    "perturbation_col",
    "control_label",
    "var_ensembl_col",
)
_XATLAS_SOURCE_CONFIG_REQUIRED_KEYS = (
    "shard_dir",
    "gene_metadata_path",
    "control_label",
)

#: Default ``var`` column carrying gene *symbols* (vs. ``var_ensembl_col``,
#: Ensembl ids) for target-side HVG alignment. Verified against the real
#: Replogle-lineage ``K562_essential_normalized_singlecell_01.h5ad`` (same
#: family as the 3 h5ad anchors): ``var["gene_name"]`` holds symbols (e.g.
#: ``"HES4"``) matching the released checkpoint's HVG order exactly.
#: Overridable per line via ``"target_gene_symbol_col"`` in the source JSON.
_DEFAULT_TARGET_GENE_SYMBOL_COL = "gene_name"


class _LinePerturbationBags(NamedTuple):
    """Per-gene response bags assembled for one line, before cross-line concat."""

    genes: list[str]
    input_bags: list[np.ndarray]
    target_bags: list[np.ndarray]
    batch_bags: list[np.ndarray]
    cell_type_bags: list[np.ndarray]
    metadata_rows: list[dict[str, object]]


class _ControlView(NamedTuple):
    """One line's control-cell arrays, before cross-line concatenation."""

    control_input: np.ndarray
    control_target: np.ndarray
    control_batch: np.ndarray
    control_cell_type: np.ndarray


def assemble_train_response_gene_bags(
    *,
    cell_line_manifest_path: Path,
    tx1_cache_dir: Path,
    hvg_state_model_dir: Path,
    perturbseq_sources_path: Path,
    l2_normalize: bool = False,
    genes: Sequence[str] | None = None,
) -> GeneBags:
    """Assemble a two-view ``GeneBags`` spanning the 4 ST response-training lines.

    Args:
        cell_line_manifest_path: Frozen ``cell_line_manifest.csv`` (C4).
        tx1_cache_dir: Root of the Phase B Tx1 basal embedding cache.
        hvg_state_model_dir: Released ST checkpoint dir (e.g.
            ``ST-HVG-Replogle/fewshot/k562``); its ``var_dims.pkl`` gives
            the target HVG gene order.
        perturbseq_sources_path: Path to ``perturbseq_sources.json``.
        l2_normalize: Whether to L2-normalize the Tx1 input embeddings (C7);
            recorded in ``GeneBags.metadata["l2_normalize"]``.
        genes: Optional restriction to these perturbation-gene labels.
            ``None`` processes every perturbed cell per line -- bound
            memory with an explicit subset for a genome-scale line.

    Returns:
        A two-view ``GeneBags``: ``input_bags``/``control_input`` are Tx1
        space, ``target_bags``/``control_target`` are HVG gene space, with
        per-line provenance in ``metadata``/``batch_bags``/``control_batch``.

    Raises:
        ValueError: No ``train_response_and_head`` row exists, a selected
            line lacks a cache entry/configured source, its
            ``basal_source`` is not Perturb-seq-derived, or the extracted
            target gene order does not match the cache's.
    """
    manifest = load_line_manifest(cell_line_manifest_path)
    selected = _select_train_response_lines(manifest)
    model_ids = selected["model_id"].astype(str).tolist()
    what = f"Tx1 embedding cache entry under {tx1_cache_dir}"
    _require_present(model_ids, _cache_line_ids(tx1_cache_dir), what)
    sources, symbol_cols = _load_perturbseq_sources(perturbseq_sources_path)
    _require_present(model_ids, set(sources), "perturbseq source config entry")
    checkpoint_gene_order = load_hvg_gene_order(hvg_state_model_dir)
    per_line: list[_LinePerturbationBags] = []
    control_views: list[_ControlView] = []
    for row in selected.itertuples(index=False):
        _assert_admissible_role(row)
        gene_bags, control_view = _assemble_line(
            row,
            sources[str(row.model_id)],
            symbol_cols.get(str(row.model_id), _DEFAULT_TARGET_GENE_SYMBOL_COL),
            checkpoint_gene_order,
            hvg_state_model_dir,
            tx1_cache_dir,
            genes,
        )
        per_line.append(gene_bags)
        control_views.append(control_view)
    return _combine(per_line, control_views, checkpoint_gene_order, l2_normalize)


def _assert_admissible_role(row: object) -> None:
    """C5 as a runtime check, not a comment: never admit a non-anchor line.

    Defense-in-depth beyond :func:`_select_train_response_lines`'s filter;
    raises explicitly rather than using a bare ``assert`` (stripped under
    ``python -O``, which would silently defeat a Critical data-leakage guard).
    """
    if str(row.role) != _TRAIN_RESPONSE_ROLE:
        raise ValueError(
            f"line {row.model_id}: refusing to admit role={row.role!r} into ST "
            f"response supervision; only {_TRAIN_RESPONSE_ROLE!r} is allowed (C5)"
        )
    if str(row.basal_source) != _PERTURBSEQ_BASAL_SOURCE:
        raise ValueError(
            f"line {row.model_id}: {_TRAIN_RESPONSE_ROLE!r} requires "
            f"basal_source={_PERTURBSEQ_BASAL_SOURCE!r}, got {row.basal_source!r}"
        )


def _select_train_response_lines(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the frozen manifest's ``train_response_and_head`` rows."""
    selected = manifest.loc[manifest["role"] == _TRAIN_RESPONSE_ROLE].reset_index(
        drop=True
    )
    if selected.empty:
        raise ValueError(
            f"cell line manifest has no role={_TRAIN_RESPONSE_ROLE!r} rows; ST "
            "response training requires at least one"
        )
    return selected


def _cache_line_ids(tx1_cache_dir: Path) -> set[str]:
    """Directory names directly under ``tx1_cache_dir`` (one per cached line)."""
    root = Path(tx1_cache_dir)
    if not root.is_dir():
        return set()
    return {entry.name for entry in root.iterdir() if entry.is_dir()}


def _require_present(model_ids: Sequence[str], present: set[str], what: str) -> None:
    """Raise, naming every affected line, if any selected line lacks ``what``."""
    missing = [model_id for model_id in model_ids if model_id not in present]
    if missing:
        raise ValueError(f"no {what} for line(s): {missing}")


def _load_perturbseq_sources(
    path: Path,
) -> tuple[dict[str, PerturbseqSourceConfig], dict[str, str]]:
    """Load the per-``model_id`` Perturb-seq source config JSON.

    Mirrors the CLI's own source-config parser (not imported: layering runs
    scripts -> aivc_model), plus reads each entry's optional
    ``"target_gene_symbol_col"``.

    Returns:
        ``(sources, target_gene_symbol_cols)``, both keyed by ``model_id``.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"perturbseq source config at {path} must be a JSON object")
    sources: dict[str, PerturbseqSourceConfig] = {}
    symbol_cols: dict[str, str] = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"perturbseq source config entry {model_id!r} must be a JSON object"
            )
        sources[str(model_id)] = _parse_source_entry(str(model_id), entry)
        symbol_cols[str(model_id)] = str(
            entry.get("target_gene_symbol_col", _DEFAULT_TARGET_GENE_SYMBOL_COL)
        )
    return sources, symbol_cols


def _parse_source_entry(
    model_id: str, entry: Mapping[str, object]
) -> PerturbseqSourceConfig:
    """Parse one source-config entry, dispatched on ``source_type``."""
    source_type = str(entry.get("source_type", _SOURCE_TYPE_H5AD))
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"perturbseq source config entry {model_id!r} has unknown "
            f"source_type {source_type!r}; expected one of "
            f"{sorted(_VALID_SOURCE_TYPES)}"
        )
    required_keys = (
        _H5AD_SOURCE_CONFIG_KEYS
        if source_type == _SOURCE_TYPE_H5AD
        else _XATLAS_SOURCE_CONFIG_REQUIRED_KEYS
    )
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise ValueError(
            f"perturbseq source config entry {model_id!r} is missing keys: {missing}"
        )
    if source_type == _SOURCE_TYPE_H5AD:
        return PerturbseqSource(
            h5ad_path=Path(entry["h5ad_path"]),
            control_label=str(entry["control_label"]),
            perturbation_col=str(entry["perturbation_col"]),
            var_ensembl_col=str(entry["var_ensembl_col"]),
        )
    return XatlasOrionSource(
        shard_dir=Path(entry["shard_dir"]),
        gene_metadata_path=Path(entry["gene_metadata_path"]),
        control_label=str(entry["control_label"]),
        shard_glob=str(entry.get("shard_glob", "*.parquet")),
        pass_guide_filter_value=int(entry.get("pass_guide_filter_value", 1)),
    )


def _build_response_adata(
    row: object,
    source: PerturbseqSourceConfig,
    genes: Sequence[str] | None,
) -> ad.AnnData:
    """Dispatch one manifest row to the matching response-cell builder."""
    if isinstance(source, PerturbseqSource):
        return build_perturbseq_response_adata(
            source.h5ad_path,
            control_label=source.control_label,
            perturbation_col=source.perturbation_col,
            cell_line_name=str(row.cell_line_name),
            model_id=str(row.model_id),
            cellosaurus_id=str(row.cellosaurus_id),
            var_ensembl_col=source.var_ensembl_col,
            genes=genes,
        )
    if isinstance(source, XatlasOrionSource):
        return build_xatlas_orion_response_adata(
            source.shard_dir,
            source.gene_metadata_path,
            cell_line_name=str(row.cell_line_name),
            model_id=str(row.model_id),
            cellosaurus_id=str(row.cellosaurus_id),
            shard_glob=source.shard_glob,
            control_label=source.control_label,
            pass_guide_filter_value=source.pass_guide_filter_value,
            genes=genes,
        )
    raise ValueError(  # pragma: no cover - exhaustive over PerturbseqSourceConfig
        f"line {row.model_id}: unsupported PerturbseqSourceConfig type {type(source)!r}"
    )


def _align_to_checkpoint_order(
    adata: ad.AnnData,
    hvg_state_model_dir: Path,
    checkpoint_gene_order: np.ndarray,
    gene_symbol_col: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve ``adata``'s expression to the checkpoint's HVG gene order.

    Mirrors ``tx1_embed_cache._resolve_hvg_matrix``'s zero-fill policy,
    applied here to *response* (perturbed) cells.

    Returns:
        ``(matrix, resolved_gene_order, fill_rate)``; the caller must
        compare ``resolved_gene_order`` against ``checkpoint_gene_order``
        (:func:`_assert_target_order_matches`).
    """
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(f"response AnnData var is missing column {gene_symbol_col!r}")
    source_symbols = set(adata.var[gene_symbol_col].astype(str))
    missing = [name for name in checkpoint_gene_order if name not in source_symbols]
    padded = (
        _pad_missing_target_genes(adata, missing, gene_symbol_col) if missing else adata
    )
    indices, resolved_names = resolve_state_gene_order(
        padded, hvg_state_model_dir, gene_symbol_col
    )
    matrix = padded.X[:, indices]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("extracted target expression contains non-finite values")
    n_checkpoint = len(checkpoint_gene_order)
    fill_rate = len(missing) / n_checkpoint if n_checkpoint else 0.0
    if fill_rate:
        _LOGGER.warning(
            "target extraction: %d/%d (%.1f%%) HVG checkpoint genes absent from "
            "response source, zero-filled",
            len(missing),
            n_checkpoint,
            100.0 * fill_rate,
        )
    return matrix, resolved_names, fill_rate


def _pad_missing_target_genes(
    adata: ad.AnnData, missing_genes: list[str], gene_symbol_col: str
) -> ad.AnnData:
    """Append zero-valued columns for checkpoint genes absent from ``adata``
    (mirrors ``tx1_embed_cache._pad_missing_genes``, duplicated locally)."""
    n_cells = adata.n_obs
    pad_matrix = csr_matrix((n_cells, len(missing_genes)), dtype=np.float32)
    source_matrix = adata.X if sparse.issparse(adata.X) else csr_matrix(adata.X)
    combined_matrix = sparse.hstack([source_matrix, pad_matrix], format="csr")
    pad_var = pd.DataFrame(
        {gene_symbol_col: missing_genes},
        index=[f"__target_fill__{index}" for index in range(len(missing_genes))],
    )
    combined_var = pd.concat([adata.var[[gene_symbol_col]], pad_var])
    return ad.AnnData(X=combined_matrix, var=combined_var)


def _assert_width(model_id: str, label: str, actual: int, expected: int) -> None:
    """Raise if a per-line array's feature width disagrees with its contract."""
    if actual != expected:
        raise ValueError(f"line {model_id}: {label} width {actual} != {expected}")


def _assert_target_order_matches(
    model_id: str,
    target_matrix: np.ndarray,
    resolved_names: np.ndarray,
    checkpoint_gene_order: np.ndarray,
) -> None:
    """Guard against a silent target-side gene reordering (requirement #3):
    checks width, then full order equality (a same-width, different-order
    array would pass a width check while still mispairing every gene)."""
    _assert_width(
        model_id, "target", target_matrix.shape[1], len(checkpoint_gene_order)
    )
    if not np.array_equal(resolved_names, checkpoint_gene_order):
        raise ValueError(
            f"line {model_id}: resolved target gene order does not match "
            "tx1_embed_cache.load_hvg_gene_order's cache order -- refusing a "
            "silent gene mispairing"
        )


def _group_indices_by_gene(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Group row indices by their observed perturbation-gene label."""
    frame = pd.DataFrame({"gene": np.asarray(labels).astype(str)})
    return {
        str(gene): group.index.to_numpy(dtype=np.int64)
        for gene, group in frame.groupby("gene", sort=True)
    }


def _build_gene_bags_for_line(
    groups: Mapping[str, np.ndarray],
    target_matrix: np.ndarray,
    *,
    model_id: str,
    cell_line_name: str,
    basal_source: str,
    fill_rate: float,
) -> _LinePerturbationBags:
    """Build one line's per-gene bags (target-space real, input-space placeholder)."""
    genes: list[str] = []
    input_bags: list[np.ndarray] = []
    target_bags: list[np.ndarray] = []
    batch_bags: list[np.ndarray] = []
    cell_type_bags: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    for gene, indices in groups.items():
        n_cells = int(len(indices))
        genes.append(gene)
        target_bags.append(target_matrix[indices])
        # NaN placeholder: see the module docstring's third design decision.
        input_bags.append(np.full((n_cells, EMBEDDING_WIDTH), np.nan, dtype=np.float32))
        batch_bags.append(np.full(n_cells, model_id, dtype=object))
        cell_type_bags.append(np.full(n_cells, cell_line_name, dtype=object))
        metadata_rows.append(
            {
                "perturbation_gene": gene,
                "model_id": model_id,
                "cell_line_name": cell_line_name,
                "basal_source": basal_source,
                "n_cells": n_cells,
                "target_hvg_fill_rate": float(fill_rate),
            }
        )
    return _LinePerturbationBags(
        genes=genes,
        input_bags=input_bags,
        target_bags=target_bags,
        batch_bags=batch_bags,
        cell_type_bags=cell_type_bags,
        metadata_rows=metadata_rows,
    )


def _line_control_view(
    model_id: str,
    cell_line_name: str,
    tx1_cache_dir: Path,
    checkpoint_gene_order: np.ndarray,
) -> _ControlView:
    """Read one cached control-cell Tx1/HVG view (already checkpoint-ordered)."""
    try:
        embeddings, hvg_matrix, _obs = load_line_cache(tx1_cache_dir, model_id)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(
            f"line {model_id}: no usable Tx1 embedding cache found under "
            f"{tx1_cache_dir} (required for role={_TRAIN_RESPONSE_ROLE!r})"
        ) from exc
    control_input = np.asarray(embeddings, dtype=np.float32)
    control_target = np.asarray(hvg_matrix, dtype=np.float32)
    width = len(checkpoint_gene_order)
    _assert_width(model_id, "cache embeddings", control_input.shape[1], EMBEDDING_WIDTH)
    _assert_width(model_id, "cache hvg", control_target.shape[1], width)
    n_control = control_input.shape[0]
    control_batch = np.full(n_control, model_id, dtype=object)
    control_cell_type = np.full(n_control, cell_line_name, dtype=object)
    return _ControlView(control_input, control_target, control_batch, control_cell_type)


def _assemble_line(
    row: object,
    source: PerturbseqSourceConfig,
    target_gene_symbol_col: str,
    checkpoint_gene_order: np.ndarray,
    hvg_state_model_dir: Path,
    tx1_cache_dir: Path,
    genes: Sequence[str] | None,
) -> tuple[_LinePerturbationBags, _ControlView]:
    """Assemble one line's per-gene response bags plus its control-cell view."""
    model_id = str(row.model_id)
    cell_line_name = str(row.cell_line_name)
    response_adata = _build_response_adata(row, source, genes)
    target_matrix, resolved_names, fill_rate = _align_to_checkpoint_order(
        response_adata,
        hvg_state_model_dir,
        checkpoint_gene_order,
        target_gene_symbol_col,
    )
    _assert_target_order_matches(
        model_id, target_matrix, resolved_names, checkpoint_gene_order
    )
    groups = _group_indices_by_gene(response_adata.obs["perturbation_gene"].to_numpy())
    gene_bags = _build_gene_bags_for_line(
        groups,
        target_matrix,
        model_id=model_id,
        cell_line_name=cell_line_name,
        basal_source=str(row.basal_source),
        fill_rate=fill_rate,
    )
    control_view = _line_control_view(
        model_id, cell_line_name, tx1_cache_dir, checkpoint_gene_order
    )
    return gene_bags, control_view


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize, leaving any exact-zero row untouched (no /0)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    return (matrix / safe_norms).astype(np.float32)


def _tx1_feature_names() -> np.ndarray:
    """Synthetic per-dimension names for the opaque Tx1 embedding space."""
    return np.asarray(
        [f"tx1_embedding_{i}" for i in range(EMBEDDING_WIDTH)], dtype=object
    )


def _combine(
    per_line: list[_LinePerturbationBags],
    control_views: list[_ControlView],
    checkpoint_gene_order: np.ndarray,
    l2_normalize: bool,
) -> GeneBags:
    """Concatenate every line's bags/control view into one ``GeneBags``."""
    genes = [gene for line in per_line for gene in line.genes]
    if not genes:
        raise ValueError("no perturbation-gene bags were assembled across any line")
    input_bags = tuple(bag for line in per_line for bag in line.input_bags)
    target_bags = tuple(bag for line in per_line for bag in line.target_bags)
    batch_bags = tuple(bag for line in per_line for bag in line.batch_bags)
    cell_type_bags = tuple(bag for line in per_line for bag in line.cell_type_bags)
    metadata_rows = [row for line in per_line for row in line.metadata_rows]
    control_input = np.concatenate(
        [view.control_input for view in control_views], axis=0
    ).astype(np.float32)
    control_target = np.concatenate(
        [view.control_target for view in control_views], axis=0
    ).astype(np.float32)
    control_batch = np.concatenate(
        [view.control_batch for view in control_views], axis=0
    )
    control_cell_type = np.concatenate(
        [view.control_cell_type for view in control_views], axis=0
    )
    if l2_normalize:
        control_input = _l2_normalize_rows(control_input)
    target_dim = int(control_target.shape[1])
    metadata = pd.DataFrame(metadata_rows)
    metadata["l2_normalize"] = bool(l2_normalize)
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.full(len(genes), np.nan, dtype=np.float32),
        input_bags=input_bags,
        latent_bags=target_bags,
        control_input=control_input,
        control_latent=control_target,
        cell_type_bags=cell_type_bags,
        control_cell_type=control_cell_type,
        batch_bags=batch_bags,
        control_batch=control_batch,
        feature_names=_tx1_feature_names(),
        metadata=metadata,
        input_dim=EMBEDDING_WIDTH,
        latent_dim=target_dim,
        target_bags=target_bags,
        control_target=control_target,
        target_dim=target_dim,
        target_feature_names=np.asarray(checkpoint_gene_order, dtype=object),
    )


__all__ = ["assemble_train_response_gene_bags"]
