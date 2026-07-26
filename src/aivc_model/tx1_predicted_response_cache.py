"""Phase D Task 2 (D11): the fingerprinted predicted-response cache.

Split out from :mod:`aivc_model.tx1_predicted_response` the same way Phase B
split ``tx1_basal.py`` (build) from ``tx1_embed_cache.py`` (cache write/read/
verify) -- this module is the cache half, that module is the build half.

Cache predicted responses keyed on ``sha256(st_checkpoint_hash,
phase_b_cache_manifest_hashes, gene_list, seed, arm)`` (Global Constraint
D11, the human ruling). On a fingerprint mismatch, refuse to load and
recompute -- never warn-and-continue, never partially reuse -- following
``gwps_cache.py::source_fingerprint``'s pattern, including a schema version
bumped whenever the payload or file layout changes.

Fingerprint field-by-field rationale (:func:`predicted_response_fingerprint`):

* ``st_checkpoint_sha256`` -- Phase C's own trained ``state_adapter``/
  ``perturbations`` weights. Retraining or replacing the checkpoint must
  invalidate every cached response.
* ``phase_b_cache_manifest_hashes`` -- THIS line's recorded
  ``embeddings.npy``/``hvg.npy`` sha256 from Phase B's run manifest (not the
  whole 42-line manifest file, so an unrelated line's re-embed does not
  spuriously invalidate every other line's cache). Both hashes are always
  included regardless of ``arm``, since ``arm`` alone selects which one is
  the actual ST input for this entry, and hashing the unused one too costs
  nothing (D10: prefer over-inclusion).
* ``model_id`` -- NOT in the human ruling's literal field list; added as
  cheap belt-and-suspenders on top of the cache's own per-line directory
  scoping, catching a caller bug that pairs the wrong line's embeddings hash
  with a cache write for a different ``model_id``.
* ``gene_list`` -- sorted, deduplicated: a different requested gene panel
  must never reuse another panel's cache.
* ``vocabulary_gene_order`` -- Wave 3 Codex gate P1-4: the FULL, ORDERED
  ``PerturbationVectorAdapter``/forward-only model gene vocabulary the
  response was generated from (Phase D's ``state.gene_vocabulary_path``),
  preserved in construction order and NOT sorted or deduplicated here (unlike
  ``gene_list`` above). ``gene_list`` alone only captures the *requested*
  genes -- reordering the vocabulary a model was constructed with changes
  which positional ``missing_vectors`` slot binds to which gene (fix rounds
  5/6; ``tx1_geneeffect_pipeline_run``'s module docstring), which changes
  every generated response, WITHOUT changing ``gene_list`` at all. Hashing
  order-sensitively is deliberate and mirrors
  ``gwps_cache.py::source_fingerprint`` hashing ``state_input_view`` for the
  same reason: two constructions that could otherwise hash alike must not.
* ``seed`` -- the padding-resample seed
  (``tx1_predicted_response._chunk_control_cell_indices``); changing it
  changes which basal cells get duplicated into a line's final short window.
* ``arm`` -- ``"tx1_arm"``/``"hvg_arm"``. Even though the two arms use
  different checkpoint files (already covered by ``st_checkpoint_sha256``),
  this is included explicitly for the identical reason
  ``gwps_cache.py::source_fingerprint`` hashes ``state_input_view`` itself:
  two differently-intended runs must not hash coincidentally alike.
* ``cell_set_len`` -- also NOT in the literal list, added because ST is a
  cell-SET transformer: which OTHER cells share a window can change a
  cell's own predicted output, so a different window size changes results
  even at a fixed ``seed`` and fixed basal-cell content.

Deliberately excluded: the released ``pert_onehot_map.pt`` and the batch
one-hot map. Both are used only at :func:`~aivc_model.tx1_predicted_response
.construct_forward_only_model`'s fresh-construction time; every value they
seed is overwritten by ``st_checkpoint_sha256``'s own file once
``load_forward_only_checkpoint`` runs, and every basal line's
``model_id``-keyed batch label already falls back to a fixed index
regardless of the batch map's content (no ACH id is ever a batch-map key --
``tx1_response_data.py:603-625``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import torch

from aivc_model.gene_splits import sha256_file

#: Bump whenever the fingerprint payload or cache file layout changes -- a
#: stale schema_version must refuse to load, same as a stale fingerprint.
#: Bumped 1 -> 2 for Wave 3 Codex gate P1-4: schema 1 caches never hashed
#: ``vocabulary_gene_order``, so any pre-existing cache entry could be
#: silently stale against a reordered vocabulary; bumping forces every
#: existing entry to miss and regenerate rather than being trusted blind.
_SCHEMA_VERSION: Final[int] = 2


def predicted_response_fingerprint(
    *,
    st_checkpoint_path: Path,
    phase_b_manifest_path: Path,
    model_id: str,
    genes: Sequence[str],
    vocabulary_genes: Sequence[str],
    seed: int,
    arm: str,
    cell_set_len: int,
) -> str:
    """Fingerprint every source that can silently change a predicted response.

    See the module docstring for the field-by-field rationale.

    Args:
        vocabulary_genes: The FULL, ORDERED gene vocabulary the forward-only
            model's perturbation adapter was constructed with (e.g.
            ``PerturbationVectorAdapter.genes``) -- order preserved, not
            sorted or deduplicated (see module docstring).

    Returns:
        A 64-character hex sha256 digest.
    """
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "st_checkpoint_sha256": sha256_file(Path(st_checkpoint_path)),
        "phase_b_cache_manifest_hashes": _phase_b_line_hashes(
            Path(phase_b_manifest_path), model_id
        ),
        "model_id": str(model_id),
        "gene_list": sorted({str(gene) for gene in genes}),
        "vocabulary_gene_order": [str(gene) for gene in vocabulary_genes],
        "seed": int(seed),
        "arm": str(arm),
        "cell_set_len": int(cell_set_len),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _phase_b_line_hashes(phase_b_manifest_path: Path, model_id: str) -> dict[str, str]:
    manifest = json.loads(phase_b_manifest_path.read_text())
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


def write_predicted_response_cache(
    cache_dir: Path,
    model_id: str,
    arm: str,
    fingerprint: str,
    responses: Mapping[str, np.ndarray | torch.Tensor],
) -> Path:
    """Atomically write one line/arm's predicted-response cache.

    Layout mirrors ``tx1_embed_cache.write_line_cache``'s temp-dir-then-
    ``os.replace`` pattern (a crash mid-write must never leave a partial
    cache at the final path): ``cache_dir/<model_id>/<arm>/manifest.json``
    (``schema_version``, ``fingerprint``, stored gene order) plus
    ``responses.npy`` (stacked ``(n_genes, n_basal_cells, response_dim)``).
    Every gene's array must share one shape -- all genes address the same
    line's same basal cells.

    Raises:
        ValueError: ``responses`` is empty, or its arrays disagree in shape.
    """
    if not responses:
        raise ValueError("responses must contain at least one gene")
    ordered_genes = sorted(str(gene) for gene in responses)
    arrays = [_to_numpy(responses[gene]) for gene in ordered_genes]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"predicted-response arrays must share one shape: {shapes}")
    stacked = np.stack(arrays, axis=0).astype(np.float32)

    line_arm_dir = Path(cache_dir) / str(model_id) / arm
    line_arm_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = line_arm_dir.parent / f".tmp-{arm}-{uuid.uuid4().hex}"
    tmp_dir.mkdir()
    try:
        np.save(tmp_dir / "responses.npy", stacked)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "genes": ordered_genes,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if line_arm_dir.exists():
            shutil.rmtree(line_arm_dir)
        os.replace(tmp_dir, line_arm_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return line_arm_dir / "manifest.json"


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_predicted_response_cache(
    cache_dir: Path,
    model_id: str,
    arm: str,
    expected_fingerprint: str,
) -> dict[str, np.ndarray]:
    """Load a line/arm's predicted-response cache, refusing any staleness (D11).

    A ``schema_version`` or ``fingerprint`` mismatch always **refuses to
    load and raises** -- never a warning-and-continue, never a partial
    reuse. The caller is expected to recompute and re-write the cache on
    catching this.

    Returns:
        ``{gene: array}``, each array shaped ``(n_basal_cells, response_dim)``.

    Raises:
        FileNotFoundError: No cache entry exists at this path.
        ValueError: The cache's ``schema_version`` or ``fingerprint`` does
            not match, or its stored gene count disagrees with its array.
    """
    line_arm_dir = Path(cache_dir) / str(model_id) / arm
    manifest_path = line_arm_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no predicted-response cache at {line_arm_dir}")
    manifest = json.loads(manifest_path.read_text())
    recorded_schema = manifest.get("schema_version")
    if recorded_schema != _SCHEMA_VERSION:
        raise ValueError(
            f"predicted-response cache at {line_arm_dir} has schema_version "
            f"{recorded_schema!r}, expected {_SCHEMA_VERSION!r}; refusing to "
            "load -- recompute instead (D11)"
        )
    recorded_fingerprint = manifest.get("fingerprint")
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            f"predicted-response cache at {line_arm_dir} is stale: recorded "
            f"fingerprint {recorded_fingerprint!r} != expected "
            f"{expected_fingerprint!r}; refusing to load -- recompute instead "
            "(D11: never warn-and-continue, never partially reuse)"
        )
    genes = manifest.get("genes")
    if not isinstance(genes, list):
        raise ValueError(f"predicted-response cache at {line_arm_dir}: no gene list")
    stacked = np.load(line_arm_dir / "responses.npy", mmap_mode="r")
    if stacked.shape[0] != len(genes):
        raise ValueError(
            f"predicted-response cache at {line_arm_dir}: {len(genes)} genes "
            f"recorded but responses.npy has {stacked.shape[0]} rows"
        )
    return {gene: stacked[index] for index, gene in enumerate(genes)}


__all__ = [
    "load_predicted_response_cache",
    "predicted_response_fingerprint",
    "write_predicted_response_cache",
]
