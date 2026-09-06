"""Streamed, fail-closed storage for Exp13 precomputed condition features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid

import numpy as np
import torch

from src.data.batches import PrecomputedFeatureBatch


SCHEMA_VERSION = "exp13-geneeffect-feature-store-v1"
VALID_STAGES = frozenset({"stage1_frozen", "stage2_selected"})
GENE_WIDTH = 1_280
CONTEXT_WIDTH = 5_120
DELTA_PROJ_WIDTH = 256
SUMMARY_WIDTH = 6
Q_SC_WIDTH = 3

_GENE_FILE = "genes.npz"
_CONTEXT_FILE = "contexts.npz"
_MANIFEST_FILE = "manifest.json"
_SHARD_DIR = "shards"
_SHARD_ARRAYS = {
    "model_id",
    "gene_symbols",
    "delta_proj",
    "s",
    "q_sc",
    "q_sc_mask",
    "hvg_panel_mask",
    "own_gene_shift_mask",
    "source_sha256",
    "model_checkpoint_sha256",
    "feature_schema_sha256",
    "projection_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordered_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of strings")
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _validate_model_id(model_id: str) -> None:
    if (
        not model_id
        or Path(model_id).name != model_id
        or model_id.startswith(".")
        or model_id in {_GENE_FILE, _CONTEXT_FILE, _MANIFEST_FILE, _SHARD_DIR}
    ):
        raise ValueError(f"unsafe model_id for shard filename: {model_id!r}")


def _float32(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must have dtype float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _boolean(name: str, value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have dtype bool, got {array.dtype}")
    return array


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scalar_text(shard: Mapping[str, np.ndarray], name: str) -> str:
    value = shard[name]
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def _check_shard(
    path: Path,
    *,
    model_id: str,
    gene_symbols: tuple[str, ...],
    source_sha256: str,
    checkpoint_sha256: str,
    schema_sha256: str,
    projection_sha256: str,
) -> list[str]:
    problems: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != _SHARD_ARRAYS:
                return [f"{model_id}: shard keys mismatch"]
            shard = {name: loaded[name] for name in loaded.files}
        if _scalar_text(shard, "model_id") != model_id:
            problems.append(f"{model_id}: embedded model_id mismatch")
        symbols = shard["gene_symbols"]
        if symbols.ndim != 1 or symbols.dtype.kind not in {"U", "S"}:
            problems.append(
                f"{model_id}: gene_symbols must be a one-dimensional string array"
            )
        elif tuple(symbols.astype(str).tolist()) != gene_symbols:
            problems.append(f"{model_id}: gene order mismatch")
        size = len(gene_symbols)
        for name, width in (
            ("delta_proj", DELTA_PROJ_WIDTH),
            ("s", SUMMARY_WIDTH),
            ("q_sc", Q_SC_WIDTH),
        ):
            try:
                _float32(name, shard[name], (size, width))
            except ValueError as exc:
                problems.append(f"{model_id}: {exc}")
        for name in ("q_sc_mask", "hvg_panel_mask", "own_gene_shift_mask"):
            try:
                _boolean(name, shard[name], size)
            except ValueError as exc:
                problems.append(f"{model_id}: {exc}")
        hvg_mask = shard["hvg_panel_mask"]
        own_shift_mask = shard["own_gene_shift_mask"]
        if (
            hvg_mask.shape == (size,)
            and hvg_mask.dtype == np.dtype(bool)
            and own_shift_mask.shape == (size,)
            and own_shift_mask.dtype == np.dtype(bool)
            and np.any(own_shift_mask & ~hvg_mask)
        ):
            problems.append(f"{model_id}: own_gene_shift_mask requires hvg_panel_mask")
        expected_hashes = {
            "source_sha256": source_sha256,
            "model_checkpoint_sha256": checkpoint_sha256,
            "feature_schema_sha256": schema_sha256,
            "projection_sha256": projection_sha256,
        }
        for name, expected in expected_hashes.items():
            if _scalar_text(shard, name) != expected:
                problems.append(f"{model_id}: {name} mismatch")
    except Exception as exc:
        problems.append(f"{model_id}: unreadable shard: {exc}")
    return problems


class GeneEffectFeatureStoreWriter:
    """Write globals once and stream one condition shard at a time."""

    def __init__(
        self,
        root: Path,
        *,
        stage: str,
        model_ids: Sequence[str],
        gene_symbols: Sequence[str],
        e_g: np.ndarray,
        z_c: np.ndarray,
        gene_embedding_source_sha256: str,
        feature_schema_sha256: str,
        projection_sha256: str,
        resume: bool = False,
    ) -> None:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}")
        self.root = Path(root)
        self.stage = stage
        self.model_ids = _ordered_unique(model_ids, "model_ids")
        self.gene_symbols = _ordered_unique(gene_symbols, "gene_symbols")
        for model_id in self.model_ids:
            _validate_model_id(model_id)
        for name, value in (
            ("gene_embedding_source_sha256", gene_embedding_source_sha256),
            ("feature_schema_sha256", feature_schema_sha256),
            ("projection_sha256", projection_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        self.gene_embedding_source_sha256 = gene_embedding_source_sha256
        self.feature_schema_sha256 = feature_schema_sha256
        self.projection_sha256 = projection_sha256
        e_g = _float32("e_g", e_g, (len(self.gene_symbols), GENE_WIDTH))
        z_c = _float32("z_c", z_c, (len(self.model_ids), CONTEXT_WIDTH))

        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise FileExistsError(
                f"refusing to overwrite nonempty feature store {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / _SHARD_DIR).mkdir(exist_ok=True)
        self._prior_entries: Mapping[str, object] = {}
        prior_manifest_path = self.root / _MANIFEST_FILE
        if resume and prior_manifest_path.is_file():
            try:
                prior = json.loads(
                    (self.root / _MANIFEST_FILE).read_text(encoding="utf-8")
                )
                if not isinstance(prior, dict):
                    raise ValueError("manifest root is not an object")
                identity_matches = (
                    prior.get("schema_version") == SCHEMA_VERSION
                    and prior.get("stage") == stage
                    and prior.get("model_ids") == list(self.model_ids)
                    and prior.get("gene_symbols") == list(self.gene_symbols)
                    and prior.get("gene_embedding_source_sha256")
                    == gene_embedding_source_sha256
                    and prior.get("feature_schema_sha256") == feature_schema_sha256
                    and prior.get("projection_sha256") == projection_sha256
                )
                if not identity_matches or not isinstance(prior.get("shards"), dict):
                    raise ValueError(
                        "resume manifest identity does not match this build"
                    )
                with np.load(self.root / _GENE_FILE, allow_pickle=False) as prior_genes:
                    gene_globals_match = (
                        set(prior_genes.files)
                        == {"gene_symbols", "e_g", "source_sha256"}
                        and tuple(prior_genes["gene_symbols"].astype(str).tolist())
                        == self.gene_symbols
                        and np.array_equal(prior_genes["e_g"], e_g)
                        and _scalar_text(prior_genes, "source_sha256")
                        == gene_embedding_source_sha256
                    )
                with np.load(
                    self.root / _CONTEXT_FILE, allow_pickle=False
                ) as prior_contexts:
                    context_globals_match = (
                        set(prior_contexts.files) == {"model_ids", "z_c"}
                        and tuple(prior_contexts["model_ids"].astype(str).tolist())
                        == self.model_ids
                        and np.array_equal(prior_contexts["z_c"], z_c)
                    )
                prior_globals = prior.get("globals")
                digest_match = isinstance(prior_globals, dict) and all(
                    prior_globals.get(filename) == _sha256_file(self.root / filename)
                    for filename in (_GENE_FILE, _CONTEXT_FILE)
                )
                if gene_globals_match and context_globals_match and digest_match:
                    self._prior_entries = prior["shards"]
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"cannot authenticate resume manifest/globals: {exc}"
                ) from exc
            prior_manifest_path.unlink()
        _atomic_npz(
            self.root / _GENE_FILE,
            gene_symbols=np.asarray(self.gene_symbols),
            e_g=e_g,
            source_sha256=np.asarray(gene_embedding_source_sha256),
        )
        _atomic_npz(
            self.root / _CONTEXT_FILE,
            model_ids=np.asarray(self.model_ids),
            z_c=z_c,
        )
        self._global_digests = {
            _GENE_FILE: _sha256_file(self.root / _GENE_FILE),
            _CONTEXT_FILE: _sha256_file(self.root / _CONTEXT_FILE),
        }
        self._entries: dict[str, dict[str, str]] = {}
        self._checkpoint_sha256: str | None = None
        self._hvg_panel_mask: np.ndarray | None = None

    def write_shard(
        self,
        model_id: str,
        *,
        delta_proj: np.ndarray,
        s: np.ndarray,
        q_sc: np.ndarray,
        q_sc_mask: np.ndarray,
        hvg_panel_mask: np.ndarray,
        own_gene_shift_mask: np.ndarray,
        source_sha256: str,
        model_checkpoint_sha256: str,
    ) -> bool:
        """Write one shard, returning ``False`` only for a verified resume skip."""
        model_id = str(model_id)
        if model_id not in self.model_ids:
            raise ValueError(
                f"model_id is outside the frozen context order: {model_id}"
            )
        if model_id in self._entries:
            raise ValueError(f"model_id already written in this session: {model_id}")
        for name, value in (
            ("source_sha256", source_sha256),
            ("model_checkpoint_sha256", model_checkpoint_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            self._checkpoint_sha256 is not None
            and model_checkpoint_sha256 != self._checkpoint_sha256
        ):
            raise ValueError("all feature shards must use one model checkpoint hash")
        self._checkpoint_sha256 = model_checkpoint_sha256
        size = len(self.gene_symbols)
        arrays = {
            "model_id": np.asarray(model_id),
            "gene_symbols": np.asarray(self.gene_symbols),
            "delta_proj": _float32("delta_proj", delta_proj, (size, DELTA_PROJ_WIDTH)),
            "s": _float32("s", s, (size, SUMMARY_WIDTH)),
            "q_sc": _float32("q_sc", q_sc, (size, Q_SC_WIDTH)),
            "q_sc_mask": _boolean("q_sc_mask", q_sc_mask, size),
            "hvg_panel_mask": _boolean("hvg_panel_mask", hvg_panel_mask, size),
            "own_gene_shift_mask": _boolean(
                "own_gene_shift_mask", own_gene_shift_mask, size
            ),
            "source_sha256": np.asarray(source_sha256),
            "model_checkpoint_sha256": np.asarray(model_checkpoint_sha256),
            "feature_schema_sha256": np.asarray(self.feature_schema_sha256),
            "projection_sha256": np.asarray(self.projection_sha256),
        }
        if np.any(arrays["own_gene_shift_mask"] & ~arrays["hvg_panel_mask"]):
            raise ValueError("own_gene_shift_mask requires hvg_panel_mask")
        if self._hvg_panel_mask is None:
            self._hvg_panel_mask = arrays["hvg_panel_mask"].copy()
        elif not np.array_equal(self._hvg_panel_mask, arrays["hvg_panel_mask"]):
            raise ValueError("hvg_panel_mask must be identical across all contexts")
        path = self.root / _SHARD_DIR / f"{model_id}.npz"
        prior = self._prior_entries.get(model_id)
        can_skip = (
            isinstance(prior, dict)
            and prior.get("path") == f"{_SHARD_DIR}/{model_id}.npz"
            and path.is_file()
            and prior.get("sha256") == _sha256_file(path)
            and not _check_shard(
                path,
                model_id=model_id,
                gene_symbols=self.gene_symbols,
                source_sha256=source_sha256,
                checkpoint_sha256=model_checkpoint_sha256,
                schema_sha256=self.feature_schema_sha256,
                projection_sha256=self.projection_sha256,
            )
        )
        if not can_skip:
            _atomic_npz(path, **arrays)
        self._entries[model_id] = {
            "path": f"{_SHARD_DIR}/{model_id}.npz",
            "sha256": _sha256_file(path),
            "source_sha256": source_sha256,
            "model_checkpoint_sha256": model_checkpoint_sha256,
        }
        return not can_skip

    def finalize(self) -> Mapping[str, object]:
        """Write the manifest last, after every frozen context has a shard."""
        missing = [
            model_id for model_id in self.model_ids if model_id not in self._entries
        ]
        if missing:
            raise ValueError(f"cannot finalize with missing model shards: {missing}")
        expected_root = {_MANIFEST_FILE, _GENE_FILE, _CONTEXT_FILE, _SHARD_DIR}
        extras = {path.name for path in self.root.iterdir()} - expected_root
        if extras:
            raise ValueError(f"cannot finalize with extra root paths: {sorted(extras)}")
        expected_shards = {f"{model_id}.npz" for model_id in self.model_ids}
        observed_shards = {path.name for path in (self.root / _SHARD_DIR).iterdir()}
        if observed_shards != expected_shards:
            raise ValueError("cannot finalize with missing or extra shard paths")
        for filename, digest in self._global_digests.items():
            if _sha256_file(self.root / filename) != digest:
                raise ValueError(f"cannot finalize after {filename} changed")
        for model_id, entry in self._entries.items():
            path = self.root / str(entry["path"])
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"cannot finalize after {model_id} shard changed")
            problems = _check_shard(
                path,
                model_id=model_id,
                gene_symbols=self.gene_symbols,
                source_sha256=entry["source_sha256"],
                checkpoint_sha256=entry["model_checkpoint_sha256"],
                schema_sha256=self.feature_schema_sha256,
                projection_sha256=self.projection_sha256,
            )
            if problems:
                raise ValueError(
                    f"cannot finalize invalid {model_id} shard: {problems}"
                )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "model_ids": list(self.model_ids),
            "gene_symbols": list(self.gene_symbols),
            "gene_embedding_source_sha256": self.gene_embedding_source_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "projection_sha256": self.projection_sha256,
            "globals": self._global_digests,
            "shards": {
                model_id: self._entries[model_id] for model_id in self.model_ids
            },
        }
        _atomic_json(self.root / _MANIFEST_FILE, manifest)
        return manifest


def verify_geneeffect_feature_store(
    root: Path,
    *,
    expected_stage: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_feature_schema_sha256: str | None = None,
    expected_projection_sha256: str | None = None,
    expected_source_sha256: Mapping[str, str] | None = None,
    expected_gene_embedding_source_sha256: str | None = None,
    expected_model_ids: Sequence[str] | None = None,
    expected_gene_symbols: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """Verify the whole store, returning discrepancies instead of raising."""
    root = Path(root)
    problems: list[str] = []
    manifest: dict[str, object] = {}
    try:
        raw = json.loads((root / _MANIFEST_FILE).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest root is not an object")
        manifest = raw
    except Exception as exc:
        return {"status": "failed", "discrepancies": [f"manifest unreadable: {exc}"]}

    try:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            problems.append("manifest schema_version mismatch")
        stage = manifest.get("stage")
        if stage not in VALID_STAGES:
            problems.append("manifest stage is invalid")
        if expected_stage is not None and stage != expected_stage:
            problems.append("manifest stage mismatch")
        model_ids = _ordered_unique(manifest.get("model_ids", ()), "manifest model_ids")
        genes = _ordered_unique(
            manifest.get("gene_symbols", ()), "manifest gene_symbols"
        )
        if expected_model_ids is not None and tuple(expected_model_ids) != model_ids:
            problems.append("manifest model order mismatch against expected membership")
        if expected_gene_symbols is not None and tuple(expected_gene_symbols) != genes:
            problems.append("manifest gene order mismatch against expected universe")
        schema_hash = manifest.get("feature_schema_sha256")
        projection_hash = manifest.get("projection_sha256")
        gene_embedding_source_hash = manifest.get("gene_embedding_source_sha256")
        if not _is_sha256(gene_embedding_source_hash):
            problems.append("manifest gene_embedding_source_sha256 is invalid")
        if (
            expected_gene_embedding_source_sha256 is not None
            and gene_embedding_source_hash != expected_gene_embedding_source_sha256
        ):
            problems.append("stale gene embedding source hash")
        if not _is_sha256(schema_hash):
            problems.append("manifest feature_schema_sha256 is invalid")
        if not _is_sha256(projection_hash):
            problems.append("manifest projection_sha256 is invalid")
        if (
            expected_feature_schema_sha256 is not None
            and schema_hash != expected_feature_schema_sha256
        ):
            problems.append("stale feature schema hash")
        if (
            expected_projection_sha256 is not None
            and projection_hash != expected_projection_sha256
        ):
            problems.append("stale projection hash")
        globals_meta = manifest.get("globals")
        shard_meta = manifest.get("shards")
        if not isinstance(globals_meta, dict) or set(globals_meta) != {
            _GENE_FILE,
            _CONTEXT_FILE,
        }:
            problems.append("manifest globals are inconsistent")
            globals_meta = {}
        if not isinstance(shard_meta, dict) or set(shard_meta) != set(model_ids):
            problems.append("manifest shards are inconsistent with model order")
            shard_meta = {}

        expected_root = {_MANIFEST_FILE, _GENE_FILE, _CONTEXT_FILE, _SHARD_DIR}
        observed_root = {path.name for path in root.iterdir()}
        for name in sorted(expected_root - observed_root):
            problems.append(f"missing path: {name}")
        for name in sorted(observed_root - expected_root):
            problems.append(f"extra path: {name}")
        if any(
            not (root / name).is_file()
            for name in (_MANIFEST_FILE, _GENE_FILE, _CONTEXT_FILE)
        ):
            problems.append("manifest and global paths must be files")
        if not (root / _SHARD_DIR).is_dir():
            problems.append("shards path must be a directory")

        for filename in (_GENE_FILE, _CONTEXT_FILE):
            path = root / filename
            if path.is_file() and globals_meta.get(filename) != _sha256_file(path):
                problems.append(f"{filename}: SHA-256 mismatch")
        gene_path = root / _GENE_FILE
        if gene_path.is_file():
            with np.load(gene_path, allow_pickle=False) as loaded:
                if set(loaded.files) != {"gene_symbols", "e_g", "source_sha256"}:
                    problems.append("genes.npz keys mismatch")
                else:
                    symbols = loaded["gene_symbols"]
                    if symbols.ndim != 1 or symbols.dtype.kind not in {"U", "S"}:
                        problems.append("genes.npz gene_symbols dtype/shape mismatch")
                    elif tuple(symbols.astype(str).tolist()) != genes:
                        problems.append("genes.npz gene order mismatch")
                    try:
                        _float32("e_g", loaded["e_g"], (len(genes), GENE_WIDTH))
                    except ValueError as exc:
                        problems.append(f"genes.npz: {exc}")
                    if (
                        _scalar_text(loaded, "source_sha256")
                        != gene_embedding_source_hash
                    ):
                        problems.append("genes.npz source_sha256 mismatch")
        context_path = root / _CONTEXT_FILE
        if context_path.is_file():
            with np.load(context_path, allow_pickle=False) as loaded:
                if set(loaded.files) != {"model_ids", "z_c"}:
                    problems.append("contexts.npz keys mismatch")
                else:
                    contexts = loaded["model_ids"]
                    if contexts.ndim != 1 or contexts.dtype.kind not in {"U", "S"}:
                        problems.append("contexts.npz model_ids dtype/shape mismatch")
                    elif tuple(contexts.astype(str).tolist()) != model_ids:
                        problems.append("contexts.npz model order mismatch")
                    try:
                        _float32("z_c", loaded["z_c"], (len(model_ids), CONTEXT_WIDTH))
                    except ValueError as exc:
                        problems.append(f"contexts.npz: {exc}")

        shard_dir = root / _SHARD_DIR
        observed_checkpoint_hashes: set[object] = set()
        checkpoint_hvg_mask: np.ndarray | None = None
        checkpoint_hvg_model_id: str | None = None
        if shard_dir.is_dir():
            expected_files = {f"{model_id}.npz" for model_id in model_ids}
            observed_files = {path.name for path in shard_dir.iterdir()}
            for name in sorted(expected_files - observed_files):
                problems.append(f"missing shard: {name}")
            for name in sorted(observed_files - expected_files):
                problems.append(f"extra shard: {name}")
            for model_id in model_ids:
                entry = shard_meta.get(model_id)
                if not isinstance(entry, dict):
                    problems.append(f"{model_id}: manifest shard metadata missing")
                    continue
                expected_path = f"{_SHARD_DIR}/{model_id}.npz"
                if entry.get("path") != expected_path:
                    problems.append(f"{model_id}: manifest shard path mismatch")
                path = root / expected_path
                if not path.is_file():
                    problems.append(f"{model_id}: shard path is not a file")
                    continue
                if entry.get("sha256") != _sha256_file(path):
                    problems.append(f"{model_id}: shard SHA-256 mismatch")
                try:
                    with np.load(path, allow_pickle=False) as loaded:
                        hvg_mask = loaded["hvg_panel_mask"]
                    if hvg_mask.shape == (len(genes),) and hvg_mask.dtype == np.dtype(
                        bool
                    ):
                        if checkpoint_hvg_mask is None:
                            checkpoint_hvg_mask = hvg_mask.copy()
                            checkpoint_hvg_model_id = model_id
                        elif not np.array_equal(checkpoint_hvg_mask, hvg_mask):
                            problems.append(
                                f"{model_id}: hvg_panel_mask differs from "
                                f"checkpoint-fixed mask in {checkpoint_hvg_model_id}"
                            )
                except Exception as exc:
                    problems.append(f"{model_id}: cannot compare hvg_panel_mask: {exc}")
                source_hash = entry.get("source_sha256")
                checkpoint_hash = entry.get("model_checkpoint_sha256")
                if isinstance(checkpoint_hash, str):
                    observed_checkpoint_hashes.add(checkpoint_hash)
                if not _is_sha256(source_hash):
                    problems.append(f"{model_id}: source_sha256 is invalid")
                if not _is_sha256(checkpoint_hash):
                    problems.append(f"{model_id}: model_checkpoint_sha256 is invalid")
                if (
                    expected_checkpoint_sha256 is not None
                    and checkpoint_hash != expected_checkpoint_sha256
                ):
                    problems.append(f"{model_id}: stale model checkpoint hash")
                if (
                    expected_source_sha256 is not None
                    and source_hash != expected_source_sha256.get(model_id)
                ):
                    problems.append(f"{model_id}: stale source hash")
                if all(
                    isinstance(value, str)
                    for value in (
                        source_hash,
                        checkpoint_hash,
                        schema_hash,
                        projection_hash,
                    )
                ):
                    problems.extend(
                        _check_shard(
                            path,
                            model_id=model_id,
                            gene_symbols=genes,
                            source_sha256=source_hash,
                            checkpoint_sha256=checkpoint_hash,
                            schema_sha256=schema_hash,
                            projection_sha256=projection_hash,
                        )
                    )
        if len(observed_checkpoint_hashes) > 1:
            problems.append("store mixes model checkpoint hashes")
    except Exception as exc:
        problems.append(f"malformed store: {exc}")
    return {
        "status": "passed" if not problems else "failed",
        "discrepancies": problems,
        "manifest": manifest,
    }


@dataclass(frozen=True)
class LoadedFeatureBatch:
    model_id: str
    gene_symbols: tuple[str, ...]
    features: PrecomputedFeatureBatch


class GeneEffectFrozenFeatureCache:
    """Device-resident feature tensors for an explicit Stage-2 context scope."""

    def __init__(
        self,
        *,
        gene_symbols: tuple[str, ...],
        model_ids: tuple[str, ...],
        selected_model_ids: tuple[str, ...],
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self.gene_symbols = gene_symbols
        self.model_ids = model_ids
        self.selected_model_ids = selected_model_ids
        self._context_positions = {
            self.model_ids.index(model_id): position
            for position, model_id in enumerate(selected_model_ids)
        }
        self._tensors = tensors
        self._closed = False

    @property
    def tensor_nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self._tensors.values()
        )

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        selected_model_ids: Sequence[str],
        expected_gene_symbols: Sequence[str],
        expected_model_ids: Sequence[str],
        expected_stage: str,
        device: torch.device | str,
    ) -> GeneEffectFrozenFeatureCache:
        """Load exactly ``selected_model_ids`` into device-resident tensors."""
        root = Path(root)
        genes = _ordered_unique(expected_gene_symbols, "expected_gene_symbols")
        model_ids = _ordered_unique(expected_model_ids, "expected_model_ids")
        selected = _ordered_unique(selected_model_ids, "selected_model_ids")
        unknown = [model_id for model_id in selected if model_id not in model_ids]
        if unknown:
            raise ValueError(
                f"selected_model_ids are outside expected_model_ids: {unknown}"
            )
        if expected_stage not in VALID_STAGES:
            raise ValueError(f"expected_stage must be one of {sorted(VALID_STAGES)}")

        try:
            manifest = json.loads((root / _MANIFEST_FILE).read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"feature store manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("feature store manifest root must be an object")
        manifest_models = _ordered_unique(
            manifest.get("model_ids", ()), "manifest model_ids"
        )
        manifest_genes = _ordered_unique(
            manifest.get("gene_symbols", ()), "manifest gene_symbols"
        )
        identity_checks = {
            "schema_version": (manifest.get("schema_version"), SCHEMA_VERSION),
            "stage": (manifest.get("stage"), expected_stage),
            "model order": (manifest_models, model_ids),
            "gene order": (manifest_genes, genes),
        }
        for name, (observed, expected) in identity_checks.items():
            if observed != expected:
                raise ValueError(
                    f"feature store {name} mismatch: expected {expected!r}, "
                    f"got {observed!r}"
                )
        shard_meta = manifest.get("shards")
        if not isinstance(shard_meta, dict) or set(shard_meta) != set(model_ids):
            raise ValueError("feature store shard metadata is inconsistent")

        target = torch.device(device)
        with np.load(root / _GENE_FILE, allow_pickle=False) as loaded:
            if set(loaded.files) != {"gene_symbols", "e_g", "source_sha256"}:
                raise ValueError("genes.npz keys mismatch")
            stored_genes = loaded["gene_symbols"]
            if (
                stored_genes.ndim != 1
                or stored_genes.dtype.kind not in {"U", "S"}
                or tuple(stored_genes.astype(str).tolist()) != genes
            ):
                raise ValueError("genes.npz gene order mismatch")
            e_g = _float32("e_g", loaded["e_g"], (len(genes), GENE_WIDTH)).copy()
        with np.load(root / _CONTEXT_FILE, allow_pickle=False) as loaded:
            if set(loaded.files) != {"model_ids", "z_c"}:
                raise ValueError("contexts.npz keys mismatch")
            stored_models = loaded["model_ids"]
            if (
                stored_models.ndim != 1
                or stored_models.dtype.kind not in {"U", "S"}
                or tuple(stored_models.astype(str).tolist()) != model_ids
            ):
                raise ValueError("contexts.npz model order mismatch")
            z_c = _float32("z_c", loaded["z_c"], (len(model_ids), CONTEXT_WIDTH)).copy()

        context_count = len(selected)
        gene_count = len(genes)
        tensors = {
            "delta_proj": torch.empty(
                (context_count, gene_count, DELTA_PROJ_WIDTH),
                dtype=torch.float32,
                device=target,
            ),
            "s": torch.empty(
                (context_count, gene_count, SUMMARY_WIDTH),
                dtype=torch.float32,
                device=target,
            ),
            "q_sc": torch.empty(
                (context_count, gene_count, Q_SC_WIDTH),
                dtype=torch.float32,
                device=target,
            ),
            "q_sc_mask": torch.empty(
                (context_count, gene_count), dtype=torch.bool, device=target
            ),
            "hvg_panel_mask": torch.empty(
                (context_count, gene_count), dtype=torch.bool, device=target
            ),
            "own_gene_shift_mask": torch.empty(
                (context_count, gene_count), dtype=torch.bool, device=target
            ),
            "e_g": torch.as_tensor(e_g, dtype=torch.float32, device=target),
            "z_c": torch.as_tensor(
                z_c[[model_ids.index(model_id) for model_id in selected]],
                dtype=torch.float32,
                device=target,
            ),
        }
        checkpoint_hvg_mask: np.ndarray | None = None
        for position, model_id in enumerate(selected):
            entry = shard_meta[model_id]
            expected_path = f"{_SHARD_DIR}/{model_id}.npz"
            if not isinstance(entry, dict) or entry.get("path") != expected_path:
                raise ValueError(f"{model_id}: manifest shard path mismatch")
            path = root / expected_path
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != _SHARD_ARRAYS:
                    raise ValueError(f"{model_id}: shard keys mismatch")
                if _scalar_text(loaded, "model_id") != model_id:
                    raise ValueError(f"{model_id}: embedded model_id mismatch")
                shard_genes = loaded["gene_symbols"]
                if (
                    shard_genes.ndim != 1
                    or shard_genes.dtype.kind not in {"U", "S"}
                    or tuple(shard_genes.astype(str).tolist()) != genes
                ):
                    raise ValueError(f"{model_id}: gene order mismatch")
                arrays = {
                    "delta_proj": _float32(
                        "delta_proj",
                        loaded["delta_proj"],
                        (gene_count, DELTA_PROJ_WIDTH),
                    ),
                    "s": _float32("s", loaded["s"], (gene_count, SUMMARY_WIDTH)),
                    "q_sc": _float32("q_sc", loaded["q_sc"], (gene_count, Q_SC_WIDTH)),
                    "q_sc_mask": _boolean("q_sc_mask", loaded["q_sc_mask"], gene_count),
                    "hvg_panel_mask": _boolean(
                        "hvg_panel_mask", loaded["hvg_panel_mask"], gene_count
                    ),
                    "own_gene_shift_mask": _boolean(
                        "own_gene_shift_mask",
                        loaded["own_gene_shift_mask"],
                        gene_count,
                    ),
                }
                if np.any(arrays["own_gene_shift_mask"] & ~arrays["hvg_panel_mask"]):
                    raise ValueError(
                        f"{model_id}: own_gene_shift_mask requires hvg_panel_mask"
                    )
                if checkpoint_hvg_mask is None:
                    checkpoint_hvg_mask = arrays["hvg_panel_mask"].copy()
                elif not np.array_equal(checkpoint_hvg_mask, arrays["hvg_panel_mask"]):
                    raise ValueError(
                        f"{model_id}: hvg_panel_mask differs across selected contexts"
                    )
                for name, array in arrays.items():
                    tensors[name][position].copy_(torch.as_tensor(array, device=target))
            del arrays

        return cls(
            gene_symbols=genes,
            model_ids=model_ids,
            selected_model_ids=selected,
            tensors=tensors,
        )

    def gather(self, pairs: Sequence[tuple[int, int]]) -> PrecomputedFeatureBatch:
        """Gather arbitrary global-index pairs without changing their order."""
        if self._closed:
            raise RuntimeError("frozen feature cache is closed")
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise ValueError("pairs must be a sequence of (gene, context) indices")
        gene_indices: list[int] = []
        context_positions: list[int] = []
        gene_symbols: list[str] = []
        model_ids: list[str] = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each pair must contain exactly two indices")
            gene, context = pair
            if (
                isinstance(gene, bool)
                or isinstance(context, bool)
                or not isinstance(gene, (int, np.integer))
                or not isinstance(context, (int, np.integer))
            ):
                raise ValueError("pair indices must be integers")
            gene = int(gene)
            context = int(context)
            if gene < 0 or gene >= len(self.gene_symbols):
                raise IndexError(f"gene index is out of range: {gene}")
            if context < 0 or context >= len(self.model_ids):
                raise IndexError(f"context index is out of range: {context}")
            if context not in self._context_positions:
                raise ValueError(
                    f"context is outside the frozen cache scope: "
                    f"{self.model_ids[context]}"
                )
            gene_indices.append(gene)
            context_positions.append(self._context_positions[context])
            gene_symbols.append(self.gene_symbols[gene])
            model_ids.append(self.model_ids[context])
        device = self._tensors["e_g"].device
        genes = torch.tensor(gene_indices, dtype=torch.long, device=device)
        contexts = torch.tensor(context_positions, dtype=torch.long, device=device)
        features = PrecomputedFeatureBatch(
            delta_proj=self._tensors["delta_proj"][contexts, genes],
            s=self._tensors["s"][contexts, genes],
            q_sc=self._tensors["q_sc"][contexts, genes],
            e_g=self._tensors["e_g"][genes],
            z_c=self._tensors["z_c"][contexts],
            q_sc_mask=self._tensors["q_sc_mask"][contexts, genes],
            hvg_panel_mask=self._tensors["hvg_panel_mask"][contexts, genes],
            own_gene_shift_mask=self._tensors["own_gene_shift_mask"][contexts, genes],
            gene_symbols=tuple(gene_symbols),
            model_ids=tuple(model_ids),
        )
        features.validate()
        return features

    def close(self) -> None:
        """Release all cache-owned tensor references."""
        self._tensors.clear()
        self._closed = True

    def __enter__(self) -> GeneEffectFrozenFeatureCache:
        if self._closed:
            raise RuntimeError("frozen feature cache is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_geneeffect_feature_batch(
    root: Path,
    model_id: str,
    *,
    expected_gene_symbols: Sequence[str],
    expected_model_ids: Sequence[str],
    expected_stage: str,
    expected_checkpoint_sha256: str,
    expected_feature_schema_sha256: str,
    expected_projection_sha256: str,
    expected_source_sha256: Mapping[str, str],
    expected_gene_embedding_source_sha256: str,
) -> LoadedFeatureBatch:
    """Load one complete context in stored gene order, never by implicit join."""
    report = verify_geneeffect_feature_store(
        root,
        expected_stage=expected_stage,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_feature_schema_sha256=expected_feature_schema_sha256,
        expected_projection_sha256=expected_projection_sha256,
        expected_source_sha256=expected_source_sha256,
        expected_gene_embedding_source_sha256=expected_gene_embedding_source_sha256,
        expected_model_ids=expected_model_ids,
        expected_gene_symbols=expected_gene_symbols,
    )
    if report["status"] != "passed":
        raise ValueError(
            f"feature store verification failed: {report['discrepancies']}"
        )
    manifest = report["manifest"]
    genes = tuple(manifest["gene_symbols"])
    if tuple(expected_gene_symbols) != genes:
        raise ValueError(
            "requested gene order does not exactly match the feature store"
        )
    model_ids = tuple(manifest["model_ids"])
    if model_id not in model_ids:
        raise KeyError(f"model_id is absent from feature store: {model_id}")
    root = Path(root)
    with np.load(root / _GENE_FILE, allow_pickle=False) as gene_file:
        e_g = gene_file["e_g"].copy()
    with np.load(root / _CONTEXT_FILE, allow_pickle=False) as context_file:
        z_row = context_file["z_c"][model_ids.index(model_id)].copy()
    with np.load(root / _SHARD_DIR / f"{model_id}.npz", allow_pickle=False) as shard:
        arrays = {name: shard[name].copy() for name in shard.files}
    size = len(genes)
    features = PrecomputedFeatureBatch(
        delta_proj=torch.from_numpy(arrays["delta_proj"]),
        s=torch.from_numpy(arrays["s"]),
        q_sc=torch.from_numpy(arrays["q_sc"]),
        e_g=torch.from_numpy(e_g),
        z_c=torch.from_numpy(np.broadcast_to(z_row, (size, CONTEXT_WIDTH)).copy()),
        q_sc_mask=torch.from_numpy(arrays["q_sc_mask"]),
        hvg_panel_mask=torch.from_numpy(arrays["hvg_panel_mask"]),
        own_gene_shift_mask=torch.from_numpy(arrays["own_gene_shift_mask"]),
        gene_symbols=genes,
        model_ids=(model_id,) * size,
    )
    features.validate()
    return LoadedFeatureBatch(model_id=model_id, gene_symbols=genes, features=features)
