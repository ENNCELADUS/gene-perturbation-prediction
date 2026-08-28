"""Encoder-agnostic Tx1-3B basal embedding cache: writer, reader, verifier.

Wave 1 Phase B, Task 2 (`.superpowers/sdd/briefs/task-2-brief.md`). Builds on
Task 1's CPU-only basal AnnData assembly (`tx1_basal.py`) by adding the cache
format, the ``verify_cache`` exit criterion, and the ``embed_lines``
orchestration loop. The Tx1-3B forward pass itself is injected as an
``EncoderFn`` so every function here is unit-testable without a GPU; the real
encoder is wired in ``scripts/build_tx1_basal_embeddings.py``.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from aivc_model.gene_splits import sha256_file
from aivc_model.state_core import resolve_state_gene_order, sha256_strings
from aivc_model.tx1_basal import (
    assert_tx1_input_contract,
    build_perturbseq_basal_adata,
    build_tahoe_basal_adata,
    build_xatlas_orion_basal_adata,
    load_line_manifest,
)

_LOGGER = logging.getLogger(__name__)

EncoderFn = Callable[[ad.AnnData], np.ndarray]

#: Verified Tx1-3B cell-embedding width (Global Constraint 2). Any code path
#: that could silently accept a different width is a defect.
EMBEDDING_WIDTH: Final[int] = 2560
MODEL_LABEL: Final[str] = "tahoe_x1_3b"
REJECTED_MODEL_LABEL: Final[str] = "tx-70m-merged"

_SCHEMA_VERSION: Final[int] = 1
_UNIT_NORM_ATOL: Final[float] = 1e-3
PINNED_TX1_REGISTRATION_SHA256: Final[str] = (
    "02afe8b49b4bef87aec3c86580f8373d78e24f8a35b41a794aef8ef50b64dc31"
)
_TX1_MODEL_FILES: Final[frozenset[str]] = frozenset(
    {"collator_config.yml", "model.safetensors", "model_config.yml", "vocab.json"}
)

#: Per-line sidecar recording the HVG gene-order hash a line's ``hvg.npy``
#: was written against (Critical 2). Not part of the ``arrays`` metadata
#: returned by :func:`write_line_cache` (its width is a gene count, not a
#: cell/row count, so mixing it into the generic per-array row-count-
#: agreement check in :func:`_verify_line` would be spurious); read
#: directly by :func:`_is_cached` instead.
_HVG_GENE_ORDER_FILENAME: Final[str] = "hvg_gene_order.json"

#: Per-line sidecar recording the sample-provenance signature (selected
#: cell identifiers plus seed and cap) a line was embedded against
#: (Codex P1-b). A **missing** sidecar is treated as a legacy cache written
#: before this check existed, not as a mismatch -- see
#: :func:`_cached_sample_signature_matches`.
_SAMPLE_PROVENANCE_FILENAME: Final[str] = "sample_provenance.json"
_SOURCE_PROVENANCE_FILENAME: Final[str] = "source_provenance.json"

#: Basal-source enum values, as documented in the frozen Phase-A manifest
#: contract (Global Constraint 1). Duplicated as literals rather than
#: importing ``tx1_basal``'s private equivalents, since they are part of the
#: data contract, not that module's implementation detail.
_TAHOE_BASAL_SOURCE: Final[str] = "Tahoe-100M DMSO"
_PERTURBSEQ_BASAL_SOURCE: Final[str] = "Perturb-seq non-targeting control"


@dataclass(frozen=True)
class PerturbseqSource:
    """Per-line configuration for a Perturb-seq non-targeting-control source.

    Attributes:
        h5ad_path: Path to the Perturb-seq h5ad file.
        control_label: Exact ``obs[perturbation_col]`` value identifying
            non-targeting controls.
        perturbation_col: ``obs`` column carrying the perturbation label.
        var_ensembl_col: ``var`` column carrying Ensembl gene ids.
    """

    h5ad_path: Path
    control_label: str
    perturbation_col: str
    var_ensembl_col: str


@dataclass(frozen=True)
class XatlasOrionSource:
    """Per-line configuration for an X-Atlas-Orion parquet Perturb-seq source.

    Counterpart to :class:`PerturbseqSource` for the one non-h5ad
    Perturb-seq-sourced line (HCT116), whose raw counts live in X-Atlas-Orion
    parquet shards rather than an h5ad file (see
    ``aivc_model.tx1_basal.build_xatlas_orion_basal_adata``).

    Attributes:
        shard_dir: Directory holding this line's X-Atlas-Orion parquet
            shards (may also hold other cell lines' shards).
        gene_metadata_path: Parquet file mapping ``gene_token_id`` to
            ``ensembl_id`` and ``gene_name``.
        control_label: Exact ``gene_target`` value identifying
            non-targeting-control cells.
        shard_glob: Glob pattern (relative to ``shard_dir``) selecting this
            line's shard files.
        pass_guide_filter_value: Required ``pass_guide_filter`` value.
    """

    shard_dir: Path
    gene_metadata_path: Path
    control_label: str
    shard_glob: str = "*.parquet"
    pass_guide_filter_value: int = 1


#: A configured Perturb-seq-derived line's source, either an h5ad file
#: (:class:`PerturbseqSource`) or X-Atlas-Orion parquet shards
#: (:class:`XatlasOrionSource`); dispatched on type in :func:`_build_basal_adata`.
PerturbseqSourceConfig = PerturbseqSource | XatlasOrionSource


def authenticate_tx1_registration(
    registration_path: Path,
) -> tuple[dict[str, object], str]:
    """Return the only authorized Tx1 source manifest for Stage 2."""
    registration_path = Path(registration_path)
    registration_sha256 = sha256_file(registration_path)
    if registration_sha256 != PINNED_TX1_REGISTRATION_SHA256:
        raise ValueError(
            "Tx1 Phase-A registration SHA-256 mismatch: "
            f"{registration_sha256} != {PINNED_TX1_REGISTRATION_SHA256}"
        )
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Tx1 Phase-A registration is unreadable: {exc}") from exc
    if not isinstance(registration, dict):
        raise ValueError("Tx1 Phase-A registration root must be an object")
    artifacts = registration.get("artifacts")
    evidence = registration.get("phase_a_gate_evidence")
    if not isinstance(artifacts, dict) or not isinstance(evidence, dict):
        raise ValueError("Tx1 Phase-A registration is missing authority records")
    source_manifest = evidence.get("tx1_source_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("Tx1 Phase-A registration has no embedded source manifest")
    canonical = json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    source_manifest_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if artifacts.get("tx1_source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("embedded Tx1 source manifest SHA-256 authority mismatch")
    if (
        source_manifest.get("status") != "verified"
        or source_manifest.get("model_label") != MODEL_LABEL
        or source_manifest.get("rejected_label") != REJECTED_MODEL_LABEL
        or source_manifest.get("expected_obsm_width") != EMBEDDING_WIDTH
        or source_manifest.get("model_repo") != "tahoebio/Tahoe-x1"
        or source_manifest.get("model_subdirectory") != "3b-model"
    ):
        raise ValueError("embedded Tx1 source manifest semantics are invalid")
    revision = source_manifest.get("model_revision")
    files = source_manifest.get("files")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(files, dict)
        or set(files) != _TX1_MODEL_FILES
    ):
        raise ValueError("embedded Tx1 checkpoint identity is invalid")
    for filename, record in files.items():
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise ValueError(f"embedded Tx1 file identity is invalid: {filename}")
    return dict(source_manifest), source_manifest_sha256


# --- cache writer -----------------------------------------------------------


def write_line_cache(
    cache_dir: Path,
    model_id: str,
    embeddings: np.ndarray,
    hvg_matrix: np.ndarray,
    obs: pd.DataFrame,
    *,
    hvg_gene_order: Sequence[str] | np.ndarray,
    sample_signature: Mapping[str, object] | None = None,
    source_signature: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate and atomically write one line's embedding + HVG cache.

    Writes ``embeddings.npy`` (float32), ``hvg.npy`` (float32), and
    ``obs.parquet`` into ``cache_dir/<model_id>/`` via a temp-dir-then-
    ``os.replace`` swap, so a crash mid-write never leaves a partial line
    directory at the final path (Global Constraint 5).

    Args:
        cache_dir: Root cache directory.
        model_id: DepMap model id identifying the line.
        embeddings: Tx1-3B cell embeddings, shape ``(n_cells, EMBEDDING_WIDTH)``.
        hvg_matrix: HVG-arm control matrix, shape ``(n_cells, len(hvg_gene_order))``.
        obs: Per-cell metadata, ``len(obs) == n_cells``.
        hvg_gene_order: The released gene order ``hvg_matrix`` columns align to.
        sample_signature: Optional sample-provenance signature (see
            :func:`_sample_provenance_signature`), written as a sidecar so a
            resumed run can detect a changed ``--max-cells-per-line``/
            ``--seed`` (Codex P1-b). ``None`` writes no sidecar, which a
            resumed run's :func:`_is_cached` treats as a legacy cache
            (accept, not mismatch) rather than skip it.

    Returns:
        Per-array metadata (``sha256``/``shape``/``dtype``) for the three files.

    Raises:
        ValueError: A width, row-count, or finiteness check fails (see
            :func:`_validate_line_cache_inputs`).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    hvg_matrix = np.asarray(hvg_matrix, dtype=np.float32)
    _validate_line_cache_inputs(model_id, embeddings, hvg_matrix, obs, hvg_gene_order)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir / f".tmp-{model_id}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    try:
        np.save(tmp_dir / "embeddings.npy", embeddings)
        np.save(tmp_dir / "hvg.npy", hvg_matrix)
        obs.to_parquet(tmp_dir / "obs.parquet")
        (tmp_dir / _HVG_GENE_ORDER_FILENAME).write_text(
            json.dumps(_hvg_gene_order_signature(hvg_gene_order), sort_keys=True) + "\n"
        )
        if sample_signature is not None:
            (tmp_dir / _SAMPLE_PROVENANCE_FILENAME).write_text(
                json.dumps(dict(sample_signature), sort_keys=True) + "\n"
            )
        if source_signature is not None:
            (tmp_dir / _SOURCE_PROVENANCE_FILENAME).write_text(
                json.dumps(dict(source_signature), sort_keys=True) + "\n"
            )
        metadata = _line_dir_array_metadata(tmp_dir)
        final_dir = cache_dir / model_id
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return metadata


def _validate_line_cache_inputs(
    model_id: str,
    embeddings: np.ndarray,
    hvg_matrix: np.ndarray,
    obs: pd.DataFrame,
    hvg_gene_order: Sequence[str] | np.ndarray,
) -> None:
    """Raise ``ValueError`` on any :func:`write_line_cache` contract violation."""
    actual_width = embeddings.shape[1] if embeddings.ndim == 2 else None
    if actual_width != EMBEDDING_WIDTH:
        raise ValueError(
            f"line {model_id}: embeddings width {actual_width} != required "
            f"width {EMBEDDING_WIDTH}"
        )
    hvg_width = hvg_matrix.shape[1] if hvg_matrix.ndim == 2 else None
    if hvg_width != len(hvg_gene_order):
        raise ValueError(
            f"line {model_id}: hvg_matrix width {hvg_width} != "
            f"hvg_gene_order length {len(hvg_gene_order)}"
        )
    n_cells = embeddings.shape[0]
    if hvg_matrix.shape[0] != n_cells or len(obs) != n_cells:
        raise ValueError(
            f"line {model_id}: row count mismatch: embeddings={n_cells}, "
            f"hvg_matrix={hvg_matrix.shape[0]}, obs={len(obs)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"line {model_id}: embeddings contain non-finite values")
    if not np.isfinite(hvg_matrix).all():
        raise ValueError(f"line {model_id}: hvg_matrix contains non-finite values")


def _hvg_gene_order_signature(
    hvg_gene_order: Sequence[str] | np.ndarray,
) -> dict[str, object]:
    """Hash + width signature identifying an HVG checkpoint gene order.

    Persisted per line (Critical 2) so a resumed :func:`embed_lines` pass
    can tell a genuinely still-valid cache from one written against a
    since-changed ``--hvg-state-model-dir``, and recorded per line in the
    run manifest so :func:`verify_cache` can flag the same staleness
    independently of :func:`_is_cached` having caught it mid-run.

    Args:
        hvg_gene_order: The resolved HVG checkpoint gene order.

    Returns:
        ``{"sha256": ..., "width": ...}``.
    """
    names = np.asarray(hvg_gene_order, dtype=object).astype(str)
    return {"sha256": sha256_strings(names), "width": int(len(names))}


def _sample_provenance_signature(
    obs: pd.DataFrame, *, seed: int, max_cells_per_line: int | None
) -> dict[str, object]:
    """Fingerprint the freshly resolved cell selection for one line (P1-b).

    Hashes the selected cells' identifiers (``obs.index`` -- real barcodes
    for Perturb-seq sources, deterministic reservoir-derived ids for Tahoe
    sources) together with the seed and cap that produced the selection.
    This lets a resumed run tell a genuinely unchanged selection from one
    whose ``--max-cells-per-line`` or ``--seed`` changed since the cache was
    written, even though the on-disk cache's own row/width counts stay
    internally self-consistent in both cases: resumability's existing
    structural checks (Global Constraint 5) cannot see that; only comparing
    against the freshly rebuilt AnnData can.

    Args:
        obs: The freshly built basal AnnData's ``obs``.
        seed: The seed used to build ``obs``.
        max_cells_per_line: The per-line cell cap used to build ``obs``.

    Returns:
        ``{"cell_ids_sha256", "n_cells", "seed", "max_cells_per_line"}``.
    """
    cell_ids = obs.index.astype(str).to_numpy()
    return {
        "cell_ids_sha256": sha256_strings(cell_ids),
        "n_cells": int(len(cell_ids)),
        "seed": int(seed),
        "max_cells_per_line": (
            None if max_cells_per_line is None else int(max_cells_per_line)
        ),
    }


def embedding_norm_stats(embeddings: np.ndarray) -> dict[str, float]:
    """Compute per-cell L2-norm statistics (Global Constraint 3).

    Records rather than assumes normalization: the cache stores raw
    embeddings, and Phase C decides whether to normalize using these stats.

    Args:
        embeddings: Cell embeddings, shape ``(n_cells, width)``.

    Returns:
        ``mean``, ``std``, ``min``, ``max`` of the per-cell L2 norm, and
        ``fraction_unit_norm`` (the fraction within ``1e-3`` of ``1.0``).

    Raises:
        ValueError: ``embeddings`` is not a non-empty 2-D array.
    """
    # Deliberately no `dtype=np.float64` upcast of the full array here: a
    # 140k-cell line's ~1.4 GB float32 (possibly mmap-backed) cache would
    # otherwise double to ~2.8 GB on every resumed pass, for every
    # already-cached line this is recomputed for. Only the tiny
    # (n_cells,)-length per-cell-norm reduction output is upcast, which is
    # numerically equivalent for these summary statistics.
    array = np.asarray(embeddings)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(
            f"embedding_norm_stats requires a non-empty 2-D array; "
            f"got shape {array.shape}"
        )
    norms = np.linalg.norm(array, axis=1).astype(np.float64)
    unit_norm = np.abs(norms - 1.0) <= _UNIT_NORM_ATOL
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "fraction_unit_norm": float(np.mean(unit_norm)),
    }


def write_run_manifest(
    cache_dir: Path,
    *,
    model_label: str,
    source_manifest: Mapping[str, object],
    line_entries: Mapping[str, Mapping[str, object]],
    config_snapshot: Mapping[str, object],
) -> Path:
    """Write the cache-root ``manifest.json`` (the run-level record).

    Merges ``line_entries`` into any existing ``manifest.json``'s recorded
    ``"lines"`` rather than overwriting them (Important 3), so the
    documented 4-GPU sharding workflow (``--only-line``, repeatable, each
    process writing a disjoint ``model_id`` subset) does not have the last
    process to finish erase the other three's records. An entry in
    ``line_entries`` overrides an existing entry recorded for the same
    ``model_id``.

    This merge is read-then-write and **not** protected by a file lock: two
    processes calling this function at truly the same moment can each read
    the manifest before the other's write lands, and the later ``os.replace``
    wins outright -- silently dropping whichever entries only the other
    process's read captured. The 4 shard processes in the intended workflow
    finish at different wall-clock times in practice (they process
    differently sized DMSO subsets), which makes an exact simultaneous
    write unlikely but does not eliminate it; a caller needing a hard
    guarantee should have one process merge every shard's ``line_entries``
    in a single final call instead of having each shard process call this
    function directly against the same ``cache_dir``.

    Args:
        cache_dir: Root cache directory.
        model_label: The Tx1 model label actually used (must not be
            :data:`REJECTED_MODEL_LABEL`).
        source_manifest: The Tx1 source identity to record verbatim (e.g.
            the parsed ``tx1_source_manifest.json`` contents), naming the
            weight/config/vocab file sha256s.
        line_entries: Per-``model_id`` entries, as returned by
            :func:`embed_lines`, to merge into the recorded lines.
        config_snapshot: Miscellaneous per-run scalars to record verbatim:
            at minimum the frozen line-manifest sha256, the subsample cap
            and seed actually used, and the HVG gene order sha256.

    Returns:
        The written ``manifest.json`` path.

    Raises:
        ValueError: ``model_label == REJECTED_MODEL_LABEL``.
    """
    if model_label == REJECTED_MODEL_LABEL:
        raise ValueError(
            f"model_label={model_label!r} is the rejected label "
            f"({REJECTED_MODEL_LABEL!r}); the registered label is {MODEL_LABEL!r}"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    merged_lines = _existing_manifest_lines(manifest_path)
    merged_lines.update(
        {model_id: dict(entry) for model_id, entry in line_entries.items()}
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_label": model_label,
        "tx1_source_manifest": dict(source_manifest),
        "config_snapshot": dict(config_snapshot),
        "lines": merged_lines,
    }
    tmp_path = cache_dir / f".tmp-manifest-{uuid.uuid4().hex}.json"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def _existing_manifest_lines(manifest_path: Path) -> dict[str, dict[str, object]]:
    """Read a prior run's recorded ``"lines"``, or ``{}`` if none exist yet."""
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        _LOGGER.warning(
            "existing manifest at %s is not valid JSON; not merging its lines",
            manifest_path,
        )
        return {}
    lines = manifest.get("lines") if isinstance(manifest, dict) else None
    return dict(lines) if isinstance(lines, dict) else {}


# --- cache reader -------------------------------------------------------


def load_line_cache(
    cache_dir: Path, model_id: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Memory-mapped read of one line's cached embeddings, HVG matrix, and obs.

    Args:
        cache_dir: Root cache directory.
        model_id: DepMap model id identifying the line.

    Returns:
        ``(embeddings, hvg_matrix, obs)``. The two arrays are memory-mapped
        (``mmap_mode="r"``); ``obs`` is fully materialized (parquet has no
        memory-mapped reader).

    Raises:
        FileNotFoundError: Any of the three per-line files is missing.
    """
    line_dir = Path(cache_dir) / model_id
    embeddings = np.load(line_dir / "embeddings.npy", mmap_mode="r")
    hvg_matrix = np.load(line_dir / "hvg.npy", mmap_mode="r")
    obs = pd.read_parquet(line_dir / "obs.parquet")
    return embeddings, hvg_matrix, obs


def load_hvg_gene_order(state_model_dir: Path) -> np.ndarray:
    """Load the released ST checkpoint's gene order from ``var_dims.pkl``.

    Args:
        state_model_dir: Directory holding the released checkpoint's
            ``var_dims.pkl`` (e.g. ``ST-HVG-Replogle``).

    Returns:
        The checkpoint's ``gene_names``, in order.
    """
    with (Path(state_model_dir) / "var_dims.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return np.asarray(payload["gene_names"], dtype=object).astype(str)


# --- verifier: the Phase B exit criterion --------------------------------


def verify_cache(
    cache_dir: Path,
    *,
    frozen_manifest_path: Path | None = None,
    only_lines: Sequence[str] | None = None,
    expected_model_ids: Sequence[str] | None = None,
    expected_source_sha256: Mapping[str, str] | None = None,
    expected_matrix_semantics: str | None = None,
    expected_tx1_source_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify every embedding cache under ``cache_dir`` against its manifest.

    This is the spec's "verified embedding caches" exit criterion. It
    recomputes every sha256 (not trusting any recorded value), re-asserts
    every embedding width against :data:`EMBEDDING_WIDTH`, re-asserts
    row-count agreement across ``embeddings``/``hvg``/``obs``, cross-checks
    each line's recorded ``n_cells`` against its arrays' actual row count
    (Codex P1-b -- catches e.g. a resumed run whose sampling inputs changed
    but whose cache was wrongly skipped), confirms the set of on-disk line
    directories matches exactly the set ``manifest.json`` declares, diffs
    the frozen Phase-A manifest's full ``model_id`` set against what this
    run actually recorded in both directions (Critical 1 -- a run's own
    ``manifest.json`` only records what it claims to have embedded, so
    without this a partial run, e.g. Tahoe-only, would report a perfect
    disk/manifest match and be mistaken for complete; Codex P2 -- an extra,
    out-of-contract ``model_id`` absent from the frozen manifest must also
    fail, not just a missing one), and confirms every line was recorded
    against the same HVG gene order (Critical 2 -- independent,
    run-manifest-level detection of the same staleness :func:`_is_cached`
    guards against mid-run). A caller must not be able to mistake a
    partial, stale, or corrupted cache for a good one: any discrepancy sets
    ``status="failed"``, never a partial "verified".

    Args:
        cache_dir: Root cache directory.
        frozen_manifest_path: Override for locating the frozen Phase-A line
            manifest (Global Constraint 1's
            ``results/phase_a_tx1_20260724/cell_line_manifest.csv``). If
            omitted, it is recovered from this run's own recorded
            ``manifest.json["config_snapshot"]["line_manifest_path"]``. An
            unlocatable, unloadable, or since-changed (sha256 mismatch)
            frozen manifest is itself a discrepancy -- never silently
            skipped.
        only_lines: Restrict verification to exactly these ``model_id``
            values (Codex P1-c -- the ``--only-line`` 4-GPU sharding
            workflow). When given, this is a **per-shard** check: it
            verifies only the requested lines' own on-disk correctness and
            that each is a legitimate frozen-manifest line, without
            requiring every other frozen-manifest line to already exist and
            without flagging other shards' directories under the same
            ``cache_dir`` as untracked. ``None`` (the default) performs the
            full, unrestricted check against the complete frozen manifest --
            the spec's actual "verified embedding caches" exit criterion,
            unweakened. A caller running the 4-GPU ``--only-line`` sharding
            workflow must still run one final unrestricted (or
            ``--verify-only``) pass to certify the whole cache.
        expected_model_ids: Explicit complete cache membership contract. This
            supports Exp13 caches whose line list does not use the legacy
            frozen Phase-A manifest schema. It must be a non-empty sequence of
            unique, non-empty strings and cannot be combined with
            ``frozen_manifest_path``. With ``only_lines``, each requested line
            must belong to this contract, but the check remains per-shard.

    Returns:
        A report with ``status`` (``"verified"`` or ``"failed"``),
        ``model_label``, ``lines_expected``, ``lines_present``, and
        ``discrepancies`` (empty iff ``status == "verified"``).
    """
    if expected_model_ids is not None and frozen_manifest_path is not None:
        raise ValueError(
            "expected_model_ids cannot be combined with frozen_manifest_path"
        )
    explicit_expected_ids = _validate_expected_model_ids(expected_model_ids)
    source_hashes = _validate_expected_source_sha256(expected_source_sha256)
    if source_hashes is not None:
        if explicit_expected_ids is None:
            raise ValueError("expected_source_sha256 requires expected_model_ids")
        if set(source_hashes) != explicit_expected_ids:
            raise ValueError(
                "expected_source_sha256 keys must exactly match expected_model_ids"
            )
        if not expected_matrix_semantics:
            raise ValueError(
                "expected_matrix_semantics is required with expected_source_sha256"
            )

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return _failed_report(cache_dir, [f"missing run manifest: {manifest_path}"])
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return _failed_report(cache_dir, [f"run manifest is not valid JSON: {exc}"])

    discrepancies = _manifest_contract_discrepancies(manifest)
    if expected_tx1_source_manifest is not None and manifest.get(
        "tx1_source_manifest"
    ) != dict(expected_tx1_source_manifest):
        discrepancies.append(
            "run manifest Tx1 source identity differs from pinned registration"
        )
    lines = manifest.get("lines") if isinstance(manifest, dict) else None
    if not isinstance(lines, dict) or not lines:
        discrepancies.append("run manifest has no recorded lines")
        return _failed_report(cache_dir, discrepancies)

    scope_ids = set(only_lines) if only_lines is not None else None
    if explicit_expected_ids is None:
        discrepancies.extend(
            _frozen_manifest_discrepancies(
                manifest, lines, frozen_manifest_path, scope_ids
            )
        )
    else:
        discrepancies.extend(
            _expected_model_id_discrepancies(lines, explicit_expected_ids, scope_ids)
        )
    if source_hashes is not None:
        discrepancies.extend(
            _source_binding_discrepancies(
                lines,
                source_hashes,
                str(expected_matrix_semantics),
                scope_ids,
            )
        )
    scoped_lines = (
        {model_id: entry for model_id, entry in lines.items() if model_id in scope_ids}
        if scope_ids is not None
        else lines
    )
    discrepancies.extend(_hvg_gene_order_consistency_discrepancies(scoped_lines))

    present_dirs = {
        entry.name
        for entry in cache_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    }
    expected_ids = (
        scope_ids
        if scope_ids is not None
        else explicit_expected_ids
        if explicit_expected_ids is not None
        else set(lines)
    )
    if scope_ids is not None:
        for missing_id in sorted(expected_ids - set(lines)):
            discrepancies.append(
                f"line {missing_id}: requested via --only-line but missing from "
                "run manifest.json lines"
            )
    check_ids = scope_ids & set(lines) if scope_ids is not None else set(lines)
    directory_required_ids = (
        check_ids if scope_ids is not None else expected_ids | set(lines)
    )
    for missing_id in sorted(directory_required_ids - present_dirs):
        discrepancies.append(f"line {missing_id}: directory missing from cache_dir")
    if scope_ids is None:
        for extra_id in sorted(present_dirs - expected_ids):
            discrepancies.append(
                f"line {extra_id}: directory not recorded in run manifest"
            )
    for model_id in sorted(check_ids & present_dirs):
        discrepancies.extend(_verify_line(cache_dir, model_id, lines[model_id]))

    status = "failed" if discrepancies else "verified"
    lines_present = (
        len(present_dirs) if scope_ids is None else len(check_ids & present_dirs)
    )
    tx1_source_manifest = manifest.get("tx1_source_manifest")
    tx1_source_manifest_sha256 = (
        hashlib.sha256(
            (json.dumps(tx1_source_manifest, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest()
        if isinstance(tx1_source_manifest, dict)
        else None
    )
    return {
        "status": status,
        "cache_dir": str(cache_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "tx1_source_manifest_sha256": tx1_source_manifest_sha256,
        "model_label": manifest.get("model_label"),
        "lines_expected": len(expected_ids),
        "lines_present": lines_present,
        "line_artifact_sha256": {
            str(model_id): {
                str(name): str(metadata["sha256"])
                for name, metadata in sorted(entry.get("arrays", {}).items())
                if isinstance(metadata, dict) and "sha256" in metadata
            }
            for model_id, entry in sorted(scoped_lines.items())
            if isinstance(entry, dict)
        },
        "discrepancies": discrepancies,
    }


def _validate_expected_model_ids(
    expected_model_ids: Sequence[str] | None,
) -> set[str] | None:
    if expected_model_ids is None:
        return None
    if isinstance(expected_model_ids, str) or not expected_model_ids:
        raise ValueError("expected_model_ids must be a non-empty sequence")
    if any(
        not isinstance(model_id, str) or not model_id for model_id in expected_model_ids
    ):
        raise ValueError("expected_model_ids must contain only non-empty strings")
    expected_ids = set(expected_model_ids)
    if len(expected_ids) != len(expected_model_ids):
        raise ValueError("expected_model_ids must contain unique values")
    return expected_ids


def _validate_expected_source_sha256(
    expected: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if expected is None:
        return None
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("expected_source_sha256 must be a non-empty mapping")
    result = {str(model_id): str(digest) for model_id, digest in expected.items()}
    invalid = [
        model_id
        for model_id, digest in result.items()
        if not model_id
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ]
    if invalid:
        raise ValueError(
            "expected_source_sha256 must map non-empty ModelIDs to lowercase "
            f"SHA-256 hex digests; invalid={sorted(invalid)}"
        )
    return result


def _source_binding_discrepancies(
    lines: Mapping[str, object],
    expected_hashes: Mapping[str, str],
    expected_semantics: str,
    scope_ids: set[str] | None,
) -> list[str]:
    model_ids = (
        set(expected_hashes) if scope_ids is None else scope_ids & set(expected_hashes)
    )
    problems: list[str] = []
    for model_id in sorted(model_ids & set(lines)):
        entry = lines[model_id]
        if not isinstance(entry, dict):
            problems.append(f"line {model_id}: manifest line entry is malformed")
            continue
        if entry.get("model_id") != model_id:
            problems.append(f"line {model_id}: manifest ModelID binding mismatch")
        if entry.get("source_sha256") != expected_hashes[model_id]:
            problems.append(f"line {model_id}: registry source SHA-256 mismatch")
        if entry.get("matrix_semantics") != expected_semantics:
            problems.append(f"line {model_id}: matrix semantics mismatch")
        if entry.get("source_provenance") != {
            "model_id": model_id,
            "source_sha256": expected_hashes[model_id],
            "matrix_semantics": expected_semantics,
        }:
            problems.append(f"line {model_id}: source provenance is missing or stale")
        sample = entry.get("sample_provenance")
        if not isinstance(sample, dict) or not sample.get("cell_ids_sha256"):
            problems.append(f"line {model_id}: sample provenance is missing")
    return problems


def _expected_model_id_discrepancies(
    lines: Mapping[str, object],
    expected_ids: set[str],
    scope_ids: set[str] | None,
) -> list[str]:
    if scope_ids is not None:
        return [
            f"line {model_id}: requested via --only-line but absent from "
            "expected_model_ids"
            for model_id in sorted(scope_ids - expected_ids)
        ]
    problems = [
        f"line {model_id}: expected by expected_model_ids but was never embedded "
        "(missing from run manifest.json lines)"
        for model_id in sorted(expected_ids - set(lines))
    ]
    problems.extend(
        f"line {model_id}: present in run manifest.json lines but absent from "
        "expected_model_ids (out-of-contract line)"
        for model_id in sorted(set(lines) - expected_ids)
    )
    return problems


def _manifest_contract_discrepancies(manifest: object) -> list[str]:
    problems: list[str] = []
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _SCHEMA_VERSION
    ):
        problems.append("run manifest schema_version mismatch or manifest malformed")
        return problems
    model_label = manifest.get("model_label")
    if model_label == REJECTED_MODEL_LABEL:
        problems.append(
            f"run manifest model_label is the rejected label {model_label!r}"
        )
    elif model_label != MODEL_LABEL:
        problems.append(
            f"run manifest model_label={model_label!r}; expected {MODEL_LABEL!r}"
        )
    return problems


def _frozen_manifest_discrepancies(
    manifest: Mapping[str, object],
    lines: Mapping[str, object],
    frozen_manifest_path: Path | None,
    scope_ids: set[str] | None = None,
) -> list[str]:
    """Diff the frozen Phase-A manifest's full ``model_id`` set against this run.

    Global Constraint 1: the frozen manifest is the single source of which
    lines exist and must be embedded. Resolves the frozen manifest from
    ``frozen_manifest_path`` if given, else from this run's own recorded
    ``config_snapshot["line_manifest_path"]`` (the CLI always points
    ``--line-manifest`` at the frozen file, so that recorded path is the
    frozen manifest regardless of any ``--only-line`` sharding used for a
    given run). An unlocatable, unloadable, or since-changed (sha256
    mismatch against ``config_snapshot["line_manifest_sha256"]``) frozen
    manifest is itself returned as a discrepancy -- the check is never
    silently skipped.

    Checks both set differences (Codex P2): a frozen line never embedded is
    a discrepancy, and so is an out-of-contract ``model_id`` recorded in
    ``lines`` that the frozen manifest does not name at all.

    Args:
        manifest: The parsed run ``manifest.json``.
        lines: ``manifest["lines"]``.
        frozen_manifest_path: See :func:`verify_cache`.
        scope_ids: If given (the ``--only-line`` restricted-verify case,
            Codex P1-c), skip the completeness check -- this run is not
            expected to have embedded every frozen line, only its own
            shard's scope -- and instead only check that every requested id
            is itself a legitimate frozen-manifest line. ``None`` performs
            the full completeness check plus the out-of-contract check
            across every line this run actually recorded.
    """
    config_snapshot = manifest.get("config_snapshot")
    recorded_sha256 = (
        config_snapshot.get("line_manifest_sha256")
        if isinstance(config_snapshot, dict)
        else None
    )
    path = frozen_manifest_path
    if path is None:
        recorded_path = (
            config_snapshot.get("line_manifest_path")
            if isinstance(config_snapshot, dict)
            else None
        )
        if not recorded_path:
            return [
                "cannot locate the frozen line manifest: no frozen_manifest_path "
                "override given and run manifest config_snapshot has no "
                "line_manifest_path"
            ]
        path = Path(recorded_path)
    if not path.is_file():
        return [f"frozen line manifest not found at {path}"]

    problems: list[str] = []
    actual_sha256 = sha256_file(path)
    if recorded_sha256 is not None and actual_sha256 != recorded_sha256:
        problems.append(
            f"frozen line manifest at {path} has sha256 {actual_sha256} but "
            f"the run manifest recorded {recorded_sha256}"
        )
    try:
        frozen = load_line_manifest(path)
    except ValueError as exc:
        problems.append(f"frozen line manifest at {path} failed to load: {exc}")
        return problems
    frozen_ids = set(frozen["model_id"].astype(str))
    if scope_ids is None:
        missing = sorted(frozen_ids - set(lines))
        problems.extend(
            f"line {model_id}: present in the frozen line manifest but was "
            "never embedded (missing from run manifest.json lines)"
            for model_id in missing
        )
        extra = sorted(set(lines) - frozen_ids)
    else:
        extra = sorted(scope_ids - frozen_ids)
    problems.extend(
        f"line {model_id}: present in run manifest.json lines but absent "
        "from the frozen line manifest (out-of-contract line)"
        for model_id in extra
    )
    return problems


def _hvg_gene_order_consistency_discrepancies(lines: Mapping[str, object]) -> list[str]:
    """Flag lines recorded with a disagreeing HVG gene-order hash (Critical 2).

    A single run resolves one HVG checkpoint gene order
    (``--hvg-state-model-dir``) and applies it to every line, so every
    entry's ``hvg_gene_order_sha256`` should agree. Disagreement means some
    lines were embedded (or last validly resumed) against a different HVG
    checkpoint than the others -- the same staleness :func:`_is_cached`
    guards against mid-run, detected here independently from that guard
    having worked correctly.
    """
    hashes = {
        model_id: entry.get("hvg_gene_order_sha256")
        for model_id, entry in lines.items()
        if isinstance(entry, dict)
    }
    if len({value for value in hashes.values()}) > 1:
        return [
            "lines were recorded with disagreeing hvg_gene_order_sha256 "
            f"values (stale HVG cache): {hashes}"
        ]
    return []


def _verify_line(cache_dir: Path, model_id: str, expected_entry: object) -> list[str]:
    """Recompute one line's on-disk state and diff it against the manifest."""
    if not isinstance(expected_entry, dict) or not isinstance(
        expected_entry.get("arrays"), dict
    ):
        return [f"line {model_id}: run manifest has no array metadata"]
    if set(expected_entry["arrays"]) != {
        "embeddings.npy",
        "hvg.npy",
        "obs.parquet",
    }:
        return [f"line {model_id}: manifest array membership is invalid"]
    line_dir = cache_dir / model_id
    problems: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    for filename, expected in expected_entry["arrays"].items():
        path = line_dir / filename
        if not path.is_file():
            problems.append(f"line {model_id}: {filename} is missing from disk")
            continue
        if sha256_file(path) != expected.get("sha256"):
            problems.append(f"line {model_id}: {filename} sha256 mismatch")
        shape = _file_shape(path)
        shapes[filename] = shape
        if list(shape) != expected.get("shape"):
            problems.append(f"line {model_id}: {filename} shape mismatch")
    if "embeddings.npy" in shapes:
        width = shapes["embeddings.npy"][-1] if shapes["embeddings.npy"] else None
        if width != EMBEDDING_WIDTH:
            problems.append(
                f"line {model_id}: embeddings width {width} != {EMBEDDING_WIDTH}"
            )
    row_counts = {name: shape[0] for name, shape in shapes.items() if shape}
    if len(set(row_counts.values())) > 1:
        problems.append(
            f"line {model_id}: row counts disagree across arrays: {row_counts}"
        )
    problems.extend(
        _n_cells_agreement_discrepancy(model_id, expected_entry, row_counts)
    )
    for filename, field in (
        (_SAMPLE_PROVENANCE_FILENAME, "sample_provenance"),
        (_SOURCE_PROVENANCE_FILENAME, "source_provenance"),
    ):
        expected_sidecar = expected_entry.get(field)
        if expected_sidecar is None:
            continue
        sidecar_path = line_dir / filename
        try:
            observed_sidecar = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"line {model_id}: {filename} is unreadable: {exc}")
            continue
        if observed_sidecar != expected_sidecar:
            problems.append(f"line {model_id}: {filename} disagrees with manifest")
    return problems


def _n_cells_agreement_discrepancy(
    model_id: str, expected_entry: Mapping[str, object], row_counts: Mapping[str, int]
) -> list[str]:
    """Cross-check the manifest's recorded ``n_cells`` against actual rows.

    Codex P1-b (part 2): a resumed run that wrongly skips a line whose
    sampling inputs changed (e.g. ``--max-cells-per-line``) can leave the
    on-disk arrays at the old, smaller row count while the manifest entry's
    ``n_cells`` reflects the newly resolved (larger) selection -- the two
    still self-agree across ``embeddings``/``hvg``/``obs`` (all still the
    old count), so the row-count-agreement check above cannot see it. Only
    an explicit comparison against ``n_cells`` catches this.
    """
    recorded_n_cells = expected_entry.get("n_cells")
    if recorded_n_cells is None or not row_counts:
        return []
    actual_counts = set(row_counts.values())
    if len(actual_counts) != 1:
        return []  # already reported as a row-count disagreement above
    actual_n_cells = next(iter(actual_counts))
    if int(recorded_n_cells) != actual_n_cells:
        return [
            f"line {model_id}: manifest n_cells={recorded_n_cells} but actual "
            f"array row count={actual_n_cells}"
        ]
    return []


def _file_shape(path: Path) -> tuple[int, ...]:
    if path.suffix == ".npy":
        return tuple(np.load(path, mmap_mode="r").shape)
    frame = pd.read_parquet(path)
    return (len(frame), len(frame.columns))


def _failed_report(cache_dir: Path, discrepancies: list[str]) -> dict[str, object]:
    return {
        "status": "failed",
        "cache_dir": str(cache_dir),
        "model_label": None,
        "lines_expected": 0,
        "lines_present": 0,
        "discrepancies": discrepancies,
    }


# --- orchestration --------------------------------------------------------


def embed_lines(
    manifest: pd.DataFrame,
    cache_dir: Path,
    *,
    encoder: EncoderFn,
    shard_dir: Path,
    gene_metadata_path: Path,
    hvg_state_model_dir: Path,
    hvg_gene_symbol_col: str = "gene_symbol",
    max_cells_per_line: int | None = None,
    seed: int = 0,
    perturbseq_sources: Mapping[str, PerturbseqSourceConfig] | None = None,
    only_lines: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Build (or resume) verified Tx1 basal embedding caches for every line.

    For each ``manifest`` row: build its basal AnnData via Task 1
    (``build_tahoe_basal_adata``, ``build_perturbseq_basal_adata``, or
    ``build_xatlas_orion_basal_adata``, dispatched on ``basal_source`` and,
    for Perturb-seq-sourced rows, on the configured source's type), resolve
    its HVG control matrix, and write the cache. A line whose cache already
    exists and is structurally valid skips the (expensive, GPU-bound)
    ``encoder`` call (Global Constraint 5: resumability) -- its basal AnnData
    is still rebuilt so the HVG fill rate is always freshly computed from the
    current source, since the encoder call is the one resource this guards.

    Args:
        manifest: The frozen line manifest (``load_line_manifest`` output).
        cache_dir: Root cache directory; one subdirectory per ``model_id``.
        encoder: ``AnnData -> (n_cells, EMBEDDING_WIDTH)`` callable, injected
            so tests substitute a stub.
        shard_dir: Tahoe-100M DMSO parquet shard directory.
        gene_metadata_path: Tahoe token-id -> Ensembl gene metadata parquet.
        hvg_state_model_dir: Released HVG checkpoint directory whose gene
            order the HVG cache is resolved against.
        hvg_gene_symbol_col: ``var`` column aligning basal AnnData columns to
            the checkpoint's gene order.
        max_cells_per_line: Optional per-line cell cap forwarded to Task 1.
        seed: Seed forwarded to Task 1's deterministic subsampling.
        perturbseq_sources: Per-``model_id`` Perturb-seq source config
            (``PerturbseqSource`` for an h5ad source, ``XatlasOrionSource``
            for an X-Atlas-Orion parquet source), for rows whose
            ``basal_source`` is Perturb-seq-derived.
        only_lines: If given, restrict processing to these ``model_id``
            values (sharding a run across multiple GPUs/processes).

    Returns:
        Per-``model_id`` entries (``arrays``, ``norm_stats``, ``n_cells``,
        ``hvg_fill_rate``, ``basal_source``, ``cellosaurus_id``,
        ``hvg_gene_order_sha256``), directly usable as
        :func:`write_run_manifest`'s ``line_entries``.
    """
    sources = perturbseq_sources or {}
    restrict = set(only_lines) if only_lines is not None else None
    line_entries: dict[str, dict[str, object]] = {}
    for row in manifest.itertuples(index=False):
        model_id = str(row.model_id)
        if restrict is not None and model_id not in restrict:
            continue
        line_entries[model_id] = _embed_one_line(
            row,
            cache_dir,
            encoder=encoder,
            shard_dir=shard_dir,
            gene_metadata_path=gene_metadata_path,
            hvg_state_model_dir=hvg_state_model_dir,
            hvg_gene_symbol_col=hvg_gene_symbol_col,
            max_cells_per_line=max_cells_per_line,
            seed=seed,
            perturbseq_sources=sources,
        )
    return line_entries


def embed_registry_lines(
    registry: pd.DataFrame,
    cache_dir: Path,
    *,
    encoder: EncoderFn,
    hvg_state_model_dir: Path,
    var_ensembl_col: str,
    hvg_gene_symbol_col: str = "gene_symbol",
    max_cells_per_line: int | None = None,
    seed: int = 0,
    only_lines: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Build Tx1 caches directly from the Exp13 raw-UMI source registry."""
    if registry.index.name != "model_id" or not registry.index.is_unique:
        raise ValueError("registry must be uniquely indexed by model_id")
    required = {"source_path", "source_kind", "matrix_semantics"}
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"registry is missing columns: {missing}")
    restrict = set(only_lines) if only_lines is not None else None
    if restrict is not None:
        unknown = sorted(restrict - set(registry.index.astype(str)))
        if unknown:
            raise ValueError(f"only_lines contains unknown ModelIDs: {unknown}")
    entries: dict[str, dict[str, object]] = {}
    for model_id, row in registry.iterrows():
        model_id = str(model_id)
        if restrict is not None and model_id not in restrict:
            continue
        if str(row["source_kind"]) != "h5ad":
            raise ValueError(f"line {model_id}: registry source_kind must be h5ad")
        semantics = str(row["matrix_semantics"])
        if semantics != "raw_umi_counts":
            raise ValueError(
                f"line {model_id}: registry matrix_semantics must be raw_umi_counts"
            )
        source_path = Path(str(row["source_path"]))
        source_sha256 = sha256_file(source_path)
        adata, sample_signature = _load_registry_basal_adata(
            source_path,
            model_id=model_id,
            var_ensembl_col=var_ensembl_col,
            max_cells=max_cells_per_line,
            seed=seed,
        )
        if sha256_file(source_path) != source_sha256:
            raise ValueError(f"line {model_id}: registry source changed while reading")
        hvg_matrix, hvg_gene_order, fill_rate = _resolve_hvg_matrix(
            adata, hvg_state_model_dir, hvg_gene_symbol_col
        )
        source_signature = {
            "model_id": model_id,
            "source_sha256": source_sha256,
            "matrix_semantics": semantics,
        }
        if _is_cached(
            cache_dir,
            model_id,
            hvg_gene_order=hvg_gene_order,
            sample_signature=sample_signature,
            source_signature=source_signature,
        ):
            _LOGGER.info("line %s: bound cache already exists", model_id)
            embeddings, _, _ = load_line_cache(cache_dir, model_id)
            arrays = _line_dir_array_metadata(Path(cache_dir) / model_id)
        else:
            embeddings = np.asarray(encoder(adata), dtype=np.float32)
            if sha256_file(source_path) != source_sha256:
                raise ValueError(
                    f"line {model_id}: registry source changed during encoding"
                )
            arrays = write_line_cache(
                cache_dir,
                model_id,
                embeddings,
                hvg_matrix,
                adata.obs,
                hvg_gene_order=hvg_gene_order,
                sample_signature=sample_signature,
                source_signature=source_signature,
            )
        entries[model_id] = {
            "model_id": model_id,
            "source_sha256": source_sha256,
            "matrix_semantics": semantics,
            "sample_provenance": sample_signature,
            "source_provenance": source_signature,
            "arrays": arrays,
            "norm_stats": embedding_norm_stats(np.asarray(embeddings)),
            "n_cells": int(adata.n_obs),
            "hvg_fill_rate": fill_rate,
            "hvg_gene_order_sha256": _hvg_gene_order_signature(hvg_gene_order)[
                "sha256"
            ],
        }
    return entries


def _load_registry_basal_adata(
    source_path: Path,
    *,
    model_id: str,
    var_ensembl_col: str,
    max_cells: int | None,
    seed: int,
) -> tuple[ad.AnnData, dict[str, object]]:
    backed = ad.read_h5ad(source_path, backed="r")
    try:
        if "model_id" not in backed.obs.columns:
            raise ValueError(f"line {model_id}: source obs is missing model_id")
        observed_ids = set(backed.obs["model_id"].astype(str))
        if observed_ids != {model_id}:
            raise ValueError(
                f"line {model_id}: source obs ModelIDs are {sorted(observed_ids)}"
            )
        if var_ensembl_col in backed.var.columns:
            ensembl_ids = backed.var[var_ensembl_col].astype(str).to_numpy()
        elif backed.var.index.name == var_ensembl_col:
            ensembl_ids = backed.var.index.astype(str).to_numpy()
        else:
            raise ValueError(
                f"line {model_id}: source var has no {var_ensembl_col!r} column/index"
            )
        cell_ids = backed.obs_names.astype(str).to_numpy()
        if not len(cell_ids):
            raise ValueError(f"line {model_id}: source contains no cells")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError(f"line {model_id}: source cell identities are not unique")
        if len(set(ensembl_ids)) != len(ensembl_ids):
            raise ValueError(f"line {model_id}: source Ensembl IDs are not unique")
        selected = _stable_registry_indices(cell_ids, model_id, max_cells, seed)
        materialized = backed[selected, :].to_memory()
    finally:
        backed.file.close()
    materialized.var.index = ensembl_ids
    materialized.var["ensembl_id"] = ensembl_ids
    materialized.obs["model_id"] = model_id
    if "cell_type" not in materialized.obs.columns:
        materialized.obs["cell_type"] = model_id
    data = (
        materialized.X.data
        if sparse.issparse(materialized.X)
        else np.asarray(materialized.X).ravel()
    )
    if data.size and not np.equal(data, np.floor(data)).all():
        raise ValueError(
            f"line {model_id}: source matrix is not raw integer UMI counts"
        )
    assert_tx1_input_contract(materialized)
    selected_ids = materialized.obs_names.astype(str).to_numpy()
    signature = _sample_provenance_signature(
        materialized.obs, seed=seed, max_cells_per_line=max_cells
    )
    signature["selection_algorithm"] = "sha256_model_id_cell_id_v1"
    signature["selected_cell_ids"] = selected_ids.tolist()
    return materialized, signature


def _stable_registry_indices(
    cell_ids: np.ndarray,
    model_id: str,
    max_cells: int | None,
    seed: int,
) -> np.ndarray:
    if max_cells is None or max_cells >= len(cell_ids):
        return np.arange(len(cell_ids), dtype=np.int64)
    if max_cells <= 0:
        raise ValueError("max_cells_per_line must be positive")
    ranked = sorted(
        range(len(cell_ids)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{model_id}\0{cell_ids[index]}".encode()
        ).digest(),
    )[:max_cells]
    return np.asarray(sorted(ranked), dtype=np.int64)


def _embed_one_line(
    row: object,
    cache_dir: Path,
    *,
    encoder: EncoderFn,
    shard_dir: Path,
    gene_metadata_path: Path,
    hvg_state_model_dir: Path,
    hvg_gene_symbol_col: str,
    max_cells_per_line: int | None,
    seed: int,
    perturbseq_sources: Mapping[str, PerturbseqSourceConfig],
) -> dict[str, object]:
    model_id = str(row.model_id)
    adata = _build_basal_adata(
        row,
        shard_dir=shard_dir,
        gene_metadata_path=gene_metadata_path,
        perturbseq_sources=perturbseq_sources,
        max_cells=max_cells_per_line,
        seed=seed,
    )
    hvg_matrix, hvg_gene_order, fill_rate = _resolve_hvg_matrix(
        adata, hvg_state_model_dir, hvg_gene_symbol_col
    )
    sample_signature = _sample_provenance_signature(
        adata.obs, seed=seed, max_cells_per_line=max_cells_per_line
    )
    if _is_cached(
        cache_dir,
        model_id,
        hvg_gene_order=hvg_gene_order,
        sample_signature=sample_signature,
    ):
        _LOGGER.info("line %s: valid cache already exists, skipping encoder", model_id)
        embeddings, _, _ = load_line_cache(cache_dir, model_id)
        arrays = _line_dir_array_metadata(Path(cache_dir) / model_id)
    else:
        _LOGGER.info("line %s: embedding %d basal cells", model_id, adata.n_obs)
        embeddings = np.asarray(encoder(adata), dtype=np.float32)
        arrays = write_line_cache(
            cache_dir,
            model_id,
            embeddings,
            hvg_matrix,
            adata.obs,
            hvg_gene_order=hvg_gene_order,
            sample_signature=sample_signature,
        )
    return {
        "arrays": arrays,
        "norm_stats": embedding_norm_stats(np.asarray(embeddings)),
        "n_cells": int(adata.n_obs),
        "hvg_fill_rate": fill_rate,
        "basal_source": str(row.basal_source),
        "cellosaurus_id": str(row.cellosaurus_id),
        "hvg_gene_order_sha256": _hvg_gene_order_signature(hvg_gene_order)["sha256"],
    }


def _build_basal_adata(
    row: object,
    *,
    shard_dir: Path,
    gene_metadata_path: Path,
    perturbseq_sources: Mapping[str, PerturbseqSourceConfig],
    max_cells: int | None,
    seed: int,
) -> ad.AnnData:
    """Dispatch one manifest row to the matching Task 1 basal AnnData builder."""
    basal_source = str(row.basal_source)
    if basal_source == _TAHOE_BASAL_SOURCE:
        return build_tahoe_basal_adata(
            shard_dir,
            gene_metadata_path,
            str(row.cellosaurus_id),
            cell_line_name=str(row.cell_line_name),
            model_id=str(row.model_id),
            max_cells=max_cells,
            seed=seed,
        )
    if basal_source == _PERTURBSEQ_BASAL_SOURCE:
        source = perturbseq_sources.get(str(row.model_id))
        if source is None:
            raise ValueError(
                f"line {row.model_id}: basal_source is "
                f"{_PERTURBSEQ_BASAL_SOURCE!r} but no PerturbseqSource or "
                "XatlasOrionSource was configured for it"
            )
        return _build_perturbseq_source_adata(
            row, source, max_cells=max_cells, seed=seed
        )
    raise ValueError(f"line {row.model_id}: unsupported basal_source={basal_source!r}")


def _build_perturbseq_source_adata(
    row: object,
    source: PerturbseqSourceConfig,
    *,
    max_cells: int | None,
    seed: int,
) -> ad.AnnData:
    """Dispatch a Perturb-seq-sourced row to its configured source's builder."""
    if isinstance(source, PerturbseqSource):
        return build_perturbseq_basal_adata(
            source.h5ad_path,
            control_label=source.control_label,
            perturbation_col=source.perturbation_col,
            cell_line_name=str(row.cell_line_name),
            model_id=str(row.model_id),
            cellosaurus_id=str(row.cellosaurus_id),
            var_ensembl_col=source.var_ensembl_col,
            max_cells=max_cells,
            seed=seed,
        )
    if isinstance(source, XatlasOrionSource):
        return build_xatlas_orion_basal_adata(
            source.shard_dir,
            source.gene_metadata_path,
            cell_line_name=str(row.cell_line_name),
            model_id=str(row.model_id),
            cellosaurus_id=str(row.cellosaurus_id),
            shard_glob=source.shard_glob,
            control_label=source.control_label,
            pass_guide_filter_value=source.pass_guide_filter_value,
            max_cells=max_cells,
            seed=seed,
        )
    raise ValueError(  # pragma: no cover - exhaustive over PerturbseqSourceConfig
        f"line {row.model_id}: unsupported PerturbseqSourceConfig type {type(source)!r}"
    )


def _is_cached(
    cache_dir: Path,
    model_id: str,
    *,
    hvg_gene_order: Sequence[str] | np.ndarray,
    sample_signature: Mapping[str, object] | None = None,
    source_signature: Mapping[str, object] | None = None,
) -> bool:
    """Check whether a line's on-disk cache is complete, valid, and current.

    Beyond the existing structural checks (files present, widths and row
    counts self-consistent), this requires the cached ``hvg.npy`` to have
    been written against ``hvg_gene_order`` -- the gene order this call just
    resolved from the *current* ``--hvg-state-model-dir`` (Critical 2).
    Without this, a line cached under one HVG checkpoint would be silently
    skipped forever on a resumed run after an operator corrects a
    misconfigured ``--hvg-state-model-dir``, even though its width and row
    counts still self-agree and its stale ``hvg.npy`` no longer means what
    the current run believes.

    It also requires the cached sample-provenance sidecar (if one is
    recorded) to match ``sample_signature`` -- the selection this call just
    resolved for the *current* ``--max-cells-per-line``/``--seed`` (Codex
    P1-b). A cache with **no** recorded sidecar is treated as a legacy
    cache written before this check existed and is accepted as-is (see
    :func:`_cached_sample_signature_matches`) -- this is deliberate so that
    switching to this fixed code does not force re-embedding of lines an
    in-flight run already wrote under the previous code.

    Args:
        cache_dir: Root cache directory.
        model_id: DepMap model id identifying the line.
        hvg_gene_order: The HVG gene order currently resolved for this line.
        sample_signature: The sample-provenance signature currently resolved
            for this line (see :func:`_sample_provenance_signature`).
            ``None`` skips this check entirely.

    Returns:
        Whether the on-disk cache may be trusted and the encoder skipped.
    """
    try:
        embeddings, hvg_matrix, obs = load_line_cache(cache_dir, model_id)
    except (FileNotFoundError, OSError, ValueError, EOFError):
        return False
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_WIDTH:
        return False
    if hvg_matrix.ndim != 2:
        return False
    n_cells = embeddings.shape[0]
    if hvg_matrix.shape[0] != n_cells or len(obs) != n_cells:
        return False
    if not _cached_hvg_gene_order_matches(cache_dir, model_id, hvg_gene_order):
        return False
    return _cached_sample_signature_matches(
        cache_dir, model_id, sample_signature
    ) and _cached_source_signature_matches(cache_dir, model_id, source_signature)


def _cached_hvg_gene_order_matches(
    cache_dir: Path, model_id: str, hvg_gene_order: Sequence[str] | np.ndarray
) -> bool:
    """Compare the on-disk HVG gene-order sidecar against the resolved order."""
    sidecar_path = Path(cache_dir) / model_id / _HVG_GENE_ORDER_FILENAME
    if not sidecar_path.is_file():
        return False
    try:
        recorded = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError:
        return False
    expected = _hvg_gene_order_signature(hvg_gene_order)
    return isinstance(recorded, dict) and recorded.get("sha256") == expected["sha256"]


def _cached_sample_signature_matches(
    cache_dir: Path, model_id: str, expected: Mapping[str, object] | None
) -> bool:
    """Compare the on-disk sample-provenance sidecar against ``expected``.

    A **missing** on-disk sidecar is treated as a legacy cache written
    before this check existed, not as a mismatch (Codex P1-b backward
    compatibility): forcing a re-embed here would silently discard
    already-completed GPU work for every line an in-flight run wrote before
    this fix shipped. Absence is trusted as-is; only a *recorded and
    disagreeing* signature invalidates the cache.

    Args:
        cache_dir: Root cache directory.
        model_id: DepMap model id identifying the line.
        expected: The sample-provenance signature currently resolved for
            this line. ``None`` disables the check (always matches).

    Returns:
        Whether the cached sample provenance may be trusted.
    """
    if expected is None:
        return True
    sidecar_path = Path(cache_dir) / model_id / _SAMPLE_PROVENANCE_FILENAME
    if not sidecar_path.is_file():
        return True  # legacy cache, written before this sidecar existed
    try:
        recorded = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError:
        return False
    return recorded == dict(expected)


def _cached_source_signature_matches(
    cache_dir: Path, model_id: str, expected: Mapping[str, object] | None
) -> bool:
    if expected is None:
        return True
    sidecar_path = Path(cache_dir) / model_id / _SOURCE_PROVENANCE_FILENAME
    if not sidecar_path.is_file():
        return False
    try:
        recorded = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError:
        return False
    return recorded == dict(expected)


# --- HVG gene-order resolution (zero-fill for absent genes) --------------


def _resolve_hvg_matrix(
    adata: ad.AnnData,
    hvg_state_model_dir: Path,
    gene_symbol_col: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve ``adata`` to the released checkpoint's HVG gene order.

    Genes the checkpoint expects but the basal source does not carry are
    zero-filled rather than raising, since ``resolve_state_gene_order`` is
    strict (all-or-nothing). This pads the missing genes as zero columns
    first, so the delegated call always succeeds and its duplicate/alignment
    checks still apply to every gene actually present in the source.

    Returns:
        ``(hvg_matrix, checkpoint_gene_order, fill_rate)`` where
        ``fill_rate`` is the fraction of checkpoint genes that were absent
        from ``adata`` and zero-filled.
    """
    checkpoint_names = load_hvg_gene_order(hvg_state_model_dir)
    if gene_symbol_col == "auto":
        candidates = [
            name
            for name in ("gene_symbol", "gene_symbols", "gene_name")
            if name in adata.var.columns
        ]
        if len(candidates) != 1:
            raise ValueError(
                "basal AnnData var must contain exactly one recognized HVG symbol "
                f"column in auto mode, found {candidates}"
            )
        gene_symbol_col = candidates[0]
    if gene_symbol_col not in adata.var.columns:
        raise ValueError(
            f"basal AnnData var is missing HVG symbol column {gene_symbol_col!r}"
        )
    symbols = adata.var[gene_symbol_col].astype(str).to_numpy()
    if len(set(symbols)) != len(symbols):
        ordered_symbols = tuple(dict.fromkeys(symbols))
        symbol_index = {symbol: index for index, symbol in enumerate(ordered_symbols)}
        if sparse.issparse(adata.X):
            columns = np.asarray([symbol_index[symbol] for symbol in symbols])
            mapper = sparse.csr_matrix(
                (
                    np.ones(len(symbols), dtype=np.float32),
                    (np.arange(len(symbols)), columns),
                ),
                shape=(len(symbols), len(ordered_symbols)),
            )
            collapsed_x = adata.X @ mapper
        else:
            source = np.asarray(adata.X)
            collapsed_x = np.zeros(
                (adata.n_obs, len(ordered_symbols)), dtype=source.dtype
            )
            for source_column, symbol in enumerate(symbols):
                collapsed_x[:, symbol_index[symbol]] += source[:, source_column]
        adata = ad.AnnData(
            X=collapsed_x,
            obs=adata.obs.copy(),
            var=pd.DataFrame({gene_symbol_col: ordered_symbols}, index=ordered_symbols),
        )
        _LOGGER.warning(
            "line: collapsed %d duplicate gene-symbol columns by summing raw counts",
            len(symbols) - len(ordered_symbols),
        )
    source_symbols = set(adata.var[gene_symbol_col].astype(str))
    missing = [name for name in checkpoint_names if name not in source_symbols]
    padded = _pad_missing_genes(adata, missing, gene_symbol_col) if missing else adata
    indices, resolved_names = resolve_state_gene_order(
        padded, hvg_state_model_dir, gene_symbol_col
    )
    matrix = padded.X[:, indices]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    fill_rate = len(missing) / len(checkpoint_names) if len(checkpoint_names) else 0.0
    if fill_rate:
        _LOGGER.warning(
            "line: %d/%d (%.1f%%) HVG checkpoint genes absent from basal "
            "source, zero-filled",
            len(missing),
            len(checkpoint_names),
            100.0 * fill_rate,
        )
    return matrix, resolved_names, fill_rate


def _pad_missing_genes(
    adata: ad.AnnData, missing_genes: list[str], gene_symbol_col: str
) -> ad.AnnData:
    """Append zero-valued columns for checkpoint genes absent from ``adata``."""
    n_cells = adata.n_obs
    pad_matrix = csr_matrix((n_cells, len(missing_genes)), dtype=np.float32)
    source_matrix = adata.X if sparse.issparse(adata.X) else csr_matrix(adata.X)
    combined_matrix = sparse.hstack([source_matrix, pad_matrix], format="csr")
    pad_var = pd.DataFrame(
        {gene_symbol_col: missing_genes},
        index=[f"__hvg_fill__{index}" for index in range(len(missing_genes))],
    )
    combined_var = pd.concat([adata.var[[gene_symbol_col]], pad_var])
    return ad.AnnData(X=combined_matrix, var=combined_var)


def _line_dir_array_metadata(line_dir: Path) -> dict[str, dict[str, object]]:
    """Recompute ``sha256``/``shape``/``dtype`` for a written line directory."""
    return {
        "embeddings.npy": _npy_metadata(line_dir / "embeddings.npy"),
        "hvg.npy": _npy_metadata(line_dir / "hvg.npy"),
        "obs.parquet": _parquet_metadata(line_dir / "obs.parquet"),
    }


def _npy_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _parquet_metadata(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    return {
        "sha256": sha256_file(path),
        "shape": [len(frame), len(frame.columns)],
        "dtype": "parquet",
    }


__all__ = [
    "EMBEDDING_WIDTH",
    "MODEL_LABEL",
    "PINNED_TX1_REGISTRATION_SHA256",
    "REJECTED_MODEL_LABEL",
    "EncoderFn",
    "PerturbseqSource",
    "PerturbseqSourceConfig",
    "XatlasOrionSource",
    "write_line_cache",
    "embedding_norm_stats",
    "write_run_manifest",
    "load_line_cache",
    "load_hvg_gene_order",
    "verify_cache",
    "embed_lines",
    "embed_registry_lines",
    "authenticate_tx1_registration",
]
