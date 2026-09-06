"""Fingerprinted cache for the EXPENSIVE, arm-independent half of Task 5's
assembled observed-response ``GeneBags`` (fix-round-3, Fix 2).

Observed post-perturbation responses are ARM-INDEPENDENT: only ST's INPUT
space differs between the Tx1 arm (2560-d Tx1 embeddings) and the HVG arm
(2000-d gene space); the target-space (HVG) response extracted from the raw
Perturb-seq/X-Atlas-Orion sources is IDENTICAL for both arms (spec C9, and
see ``tx1_response_data``'s own two-view design). Building it twice --
each arm's own ``train_tx1_st_response.py`` invocation independently calling
:func:`src.data.response.assemble_train_response_gene_bags` --
is pure waste; the 2026-07-26 incident additionally shows that running both
builds CONCURRENTLY doubles peak RAM for the SAME underlying computation
(see ``.superpowers/sdd/phase-c/progress.md``).

This caches exactly the per-gene TARGET (HVG) matrices and their per-bag
provenance metadata that :func:`src.data.response
._build_line_gene_bags` builds from raw sources across every
``train_response_and_head`` line, concatenated. Deliberately NOT cached,
because both are cheap to reconstruct and/or arm-dependent:

* ``input_bags``/``control_input`` -- the Tx1 arm's are Phase B's own
  already-cached, already-verified basal embeddings
  (``tx1_embed_cache.load_line_cache``), re-read fresh every time regardless
  (cheap: an mmap read, not a raw-source materialization); a bag's
  ``input_bags`` entry is always an all-NaN placeholder of shape ``(n_cells,
  EMBEDDING_WIDTH)`` (see ``tx1_response_data``'s own module docstring), a
  pure function of a cached bag's row count, not data that needs storing;
* ``batch_bags``/``cell_type_bags`` -- each bag's own ``model_id``/
  ``cell_line_name`` repeated ``n_cells`` times, both already present per
  bag in the cached ``metadata`` frame;
* ``l2_normalize`` -- only affects ``control_input`` (the ALWAYS-fresh half
  above), never this cache's own stored target arrays. Including it in the
  fingerprint would force a spurious rebuild between the two arms whenever
  they set it differently, which is the exact waste this cache exists to
  eliminate.

Layout mirrors ``gwps_cache.py``'s concatenated-rows-plus-offsets pattern:
``target_cells.npy`` (every gene bag's rows, concatenated, across every
line) + ``offsets.npy`` (per-bag row boundaries, length ``n_bags + 1``) +
``genes.npy`` (composite bag keys, ``tx1_response_data.composite_gene_key``)
+ ``metadata.parquet`` (the per-bag provenance frame Task 5 already builds)
+ ``manifest.json`` (``schema_version``, ``fingerprint``, ``target_dim``).

Fingerprint payload (:func:`response_targets_fingerprint`), following
``gwps_cache.py::source_fingerprint``'s and
``tx1_predicted_response_cache.py::predicted_response_fingerprint``'s
established discipline -- refuse-and-recompute on ANY mismatch, never
warn-and-continue, a ``schema_version`` bumped whenever the payload or file
layout changes:

* the frozen cell-line manifest file (content hash) -- changes the line set
  or roles;
* the Perturb-seq source-config JSON (content hash) -- changes which raw
  files/columns feed each line;
* every raw source file the config directly names
  (``tx1_response_data.referenced_source_paths``) -- a modified h5ad/parquet
  at the SAME configured path must still invalidate the cache, not just a
  changed path/glob string;
* the Tx1 embedding cache's own run manifest (content hash) -- Task 5
  cross-validates each line's cached HVG gene order against the checkpoint's
  (``_assert_cache_hvg_order_matches_checkpoint``); a re-embed changes this
  even when the checkpoint itself does not;
* the checkpoint's ``var_dims.pkl`` (content hash) -- the HVG gene order the
  target is aligned to;
* ``max_cells_per_gene``, ``total_cells_per_line`` (fix-round-3, Fix 1's
  caps) and ``seed`` -- the response builders' own sampling controls;
* ``genes`` (sorted, deduplicated) -- a different requested gene-panel
  restriction must never reuse another panel's cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from src.data.gene_splits import sha256_file

#: Bump whenever the fingerprint payload or cache file layout changes -- a
#: stale schema_version must refuse to load, same as a stale fingerprint.
_SCHEMA_VERSION: Final[int] = 1

_ARRAY_FILENAMES: Final[tuple[str, ...]] = (
    "target_cells.npy",
    "offsets.npy",
    "genes.npy",
)


def response_targets_fingerprint(
    *,
    cell_line_manifest_path: Path,
    perturbseq_sources_path: Path,
    referenced_source_paths: Sequence[Path],
    tx1_cache_manifest_path: Path,
    checkpoint_var_dims_path: Path,
    max_cells_per_gene: int | None,
    total_cells_per_line: int | None,
    seed: int,
    genes: Sequence[str] | None,
) -> str:
    """Hash every source that can change the assembled target data.

    See the module docstring for the field-by-field rationale.

    Returns:
        A 64-character hex sha256 digest.
    """
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "cell_line_manifest_sha256": sha256_file(Path(cell_line_manifest_path)),
        "perturbseq_sources_sha256": sha256_file(Path(perturbseq_sources_path)),
        "referenced_source_sha256": {
            str(path): sha256_file(Path(path))
            for path in sorted(referenced_source_paths, key=str)
        },
        "tx1_cache_manifest_sha256": (
            sha256_file(Path(tx1_cache_manifest_path))
            if Path(tx1_cache_manifest_path).is_file()
            else None
        ),
        "checkpoint_var_dims_sha256": sha256_file(Path(checkpoint_var_dims_path)),
        "max_cells_per_gene": max_cells_per_gene,
        "total_cells_per_line": total_cells_per_line,
        "seed": int(seed),
        "genes": sorted({str(gene) for gene in genes}) if genes is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_response_targets_cache(
    cache_dir: Path,
    fingerprint: str,
    *,
    genes: Sequence[str],
    target_bags: Sequence[np.ndarray],
    metadata: pd.DataFrame,
) -> Path:
    """Atomically write the concatenated per-bag target arrays + metadata.

    Mirrors ``tx1_predicted_response_cache.write_predicted_response_cache``'s
    temp-dir-then-``os.replace`` pattern (a crash mid-write must never leave
    a partial cache at the final path).

    Raises:
        ValueError: ``genes``/``target_bags``/``metadata`` disagree in
            length, or any two bags disagree in target width.
    """
    n_bags = len(genes)
    if len(target_bags) != n_bags or len(metadata) != n_bags:
        raise ValueError(
            f"genes ({n_bags}), target_bags ({len(target_bags)}), and metadata "
            f"({len(metadata)}) must all have the same length"
        )
    widths = {int(bag.shape[1]) for bag in target_bags} if target_bags else set()
    if len(widths) > 1:
        raise ValueError(f"every bag must share one target width; got {widths}")
    target_dim = widths.pop() if widths else 0
    concatenated = (
        np.concatenate(
            [np.asarray(bag, dtype=np.float32) for bag in target_bags], axis=0
        )
        if target_bags
        else np.empty((0, target_dim), dtype=np.float32)
    )
    lengths = [int(bag.shape[0]) for bag in target_bags]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir / f".tmp-response-targets-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    final_dir = cache_dir / "response_targets"
    try:
        np.save(tmp_dir / "target_cells.npy", concatenated)
        np.save(tmp_dir / "offsets.npy", offsets)
        np.save(
            tmp_dir / "genes.npy", np.asarray(genes, dtype=object), allow_pickle=True
        )
        metadata.to_parquet(tmp_dir / "metadata.parquet")
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "n_bags": n_bags,
            "target_dim": target_dim,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir / "manifest.json"


def load_response_targets_cache(
    cache_dir: Path, expected_fingerprint: str
) -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    """Load the cached target arrays + metadata, refusing any staleness.

    A ``schema_version`` or ``fingerprint`` mismatch always **refuses to
    load and raises** -- never a warning-and-continue, never a partial
    reuse (D11's discipline, applied here). The caller is expected to catch
    this, recompute, and re-write the cache.

    Returns:
        ``(genes, target_bags, metadata)`` -- ``genes`` is the array of
        composite bag keys in cached order; ``target_bags`` is split back
        into one array per bag via the stored ``offsets``; ``metadata`` is
        the per-bag provenance frame, same row order as ``genes``.

    Raises:
        FileNotFoundError: No cache entry exists at this path.
        ValueError: The cache's ``schema_version`` or ``fingerprint`` does
            not match, or its stored arrays disagree in shape.
    """
    final_dir = Path(cache_dir) / "response_targets"
    manifest_path = final_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no response-targets cache at {final_dir}")
    manifest = json.loads(manifest_path.read_text())
    recorded_schema = manifest.get("schema_version")
    if recorded_schema != _SCHEMA_VERSION:
        raise ValueError(
            f"response-targets cache at {final_dir} has schema_version "
            f"{recorded_schema!r}, expected {_SCHEMA_VERSION!r}; refusing to "
            "load -- recompute instead"
        )
    recorded_fingerprint = manifest.get("fingerprint")
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            f"response-targets cache at {final_dir} is stale: recorded "
            f"fingerprint {recorded_fingerprint!r} != expected "
            f"{expected_fingerprint!r}; refusing to load -- recompute instead "
            "(never warn-and-continue, never partially reuse)"
        )
    for filename in _ARRAY_FILENAMES:
        if not (final_dir / filename).is_file():
            raise ValueError(
                f"response-targets cache at {final_dir} is missing {filename}"
            )
    genes = np.load(final_dir / "genes.npy", allow_pickle=True)
    offsets = np.load(final_dir / "offsets.npy")
    target_cells = np.load(final_dir / "target_cells.npy", mmap_mode="r")
    metadata = pd.read_parquet(final_dir / "metadata.parquet")
    n_bags = int(manifest.get("n_bags", -1))
    if len(genes) != n_bags or len(metadata) != n_bags or len(offsets) != n_bags + 1:
        raise ValueError(
            f"response-targets cache at {final_dir}: recorded n_bags={n_bags} "
            f"disagrees with genes ({len(genes)}), metadata ({len(metadata)}), "
            f"or offsets ({len(offsets)})"
        )
    target_bags = [
        np.asarray(target_cells[offsets[i] : offsets[i + 1]]) for i in range(n_bags)
    ]
    return genes, target_bags, metadata


__all__ = [
    "load_response_targets_cache",
    "response_targets_fingerprint",
    "write_response_targets_cache",
]
