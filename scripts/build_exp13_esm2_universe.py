"""Build separate Exp13 scored and Stage-1-inclusive ESM2 universes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

import numpy as np
import pandas as pd
import torch

from aivc_model.geneeffect_data import (
    Exp13Split,
    PINNED_COPY_PRIOR_SHA256,
    load_exp13_split,
    load_geneeffect_long,
)
from aivc_model.esm2_provenance import load_and_authenticate_esm2_provenance
from aivc_model.stage1_artifact import Stage1ArtifactManifest
from aivc_model.state_core import sha256_strings

EXPECTED_COVERAGE_QUALIFIED_COUNT = 17_931
EXPECTED_COPY_PRIOR_ELIGIBLE_COUNT = 17_787
SCHEMA_VERSION = "exp13-esm2-universes-v2"
COVERAGE_THRESHOLDS = {"train": 5, "val": 3, "test": 3}
PINNED_GENE_EFFECT_SHA256 = (
    "e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e"
)


@dataclass(frozen=True)
class NpzCoverage:
    required_count: int
    resolved_count: int
    missing: tuple[str, ...]
    vector_width: int
    artifact_sha256: str


@dataclass(frozen=True)
class CoverageUniverse:
    symbols: tuple[str, ...]
    dropped: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class AuthenticatedStage1Vocabulary:
    symbols: tuple[str, ...]
    vocabulary_sha256: str
    authentication_kind: str
    authentication_source: str
    authentication_source_sha256: str
    checkpoint_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symbols_sha256(symbols: Sequence[str]) -> str:
    payload = "".join(f"{symbol}\n" for symbol in symbols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def require_pinned_gene_effect(path: Path) -> str:
    observed = _sha256(path)
    if observed != PINNED_GENE_EFFECT_SHA256:
        raise ValueError(
            "DepMap GeneEffect SHA-256 mismatch: "
            f"{observed} != {PINNED_GENE_EFFECT_SHA256}"
        )
    return observed


def _unique_symbols(symbols: Sequence[object], label: str) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError(f"{label} must contain nonempty strings")
    if any(symbol != symbol.upper() for symbol in symbols):
        raise ValueError(f"{label} must contain uppercase symbols")
    duplicates = sorted(
        symbol for symbol, count in Counter(symbols).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate symbols: {duplicates[:10]}")
    return tuple(symbols)


def build_coverage_universe(
    labels: pd.DataFrame, split: Exp13Split
) -> CoverageUniverse:
    """Return coverage-qualified scored candidates and an explicit drop report."""
    required = {"model_id", "gene_symbol", "gene_effect"}
    missing_columns = sorted(required - set(labels.columns))
    if missing_columns:
        raise ValueError(f"labels is missing columns: {missing_columns}")
    if labels[["model_id", "gene_symbol"]].duplicated().any():
        raise ValueError("labels contains duplicate (model_id, gene_symbol) rows")
    unknown = sorted(set(labels["model_id"]) - set(split.all_model_ids))
    if unknown:
        raise ValueError(f"labels contains ModelIDs outside the split: {unknown[:10]}")
    if labels["gene_symbol"].isna().any():
        raise ValueError("labels contains missing gene symbols")
    symbols = tuple(sorted(set(labels["gene_symbol"].astype(str))))
    _unique_symbols(symbols, "GeneEffect symbols")
    counts = {
        name: (
            labels.loc[labels["model_id"].isin(set(getattr(split, name)))]
            .groupby("gene_symbol", sort=False)["gene_effect"]
            .count()
        )
        for name in COVERAGE_THRESHOLDS
    }
    qualified: list[str] = []
    dropped: list[dict[str, object]] = []
    for symbol in symbols:
        observed = {name: int(counts[name].get(symbol, 0)) for name in counts}
        reasons = [
            f"{name}_finite_lt{minimum}"
            for name, minimum in COVERAGE_THRESHOLDS.items()
            if observed[name] < minimum
        ]
        if reasons:
            dropped.append(
                {"gene_symbol": symbol, "finite_counts": observed, "reasons": reasons}
            )
        else:
            qualified.append(symbol)
    if not qualified:
        raise ValueError("no genes satisfy the Exp13 coverage thresholds")
    return CoverageUniverse(tuple(qualified), tuple(dropped))


def restrict_coverage_universe_to_copy_prior(
    universe: CoverageUniverse, copy_prior_symbols: Sequence[str]
) -> CoverageUniverse:
    """Apply the global finite-donor gate before freezing the ESM2 universe."""
    eligible = set(_unique_symbols(copy_prior_symbols, "copy-prior symbols"))
    kept = tuple(symbol for symbol in universe.symbols if symbol in eligible)
    dropped = list(universe.dropped)
    dropped.extend(
        {"gene_symbol": symbol, "reasons": ["copy_prior_missing"]}
        for symbol in universe.symbols
        if symbol not in eligible
    )
    if not kept:
        raise ValueError("copy-prior coverage produced an empty scored universe")
    return CoverageUniverse(kept, tuple(dropped))


def load_authenticated_copy_prior_symbols(
    copy_prior_path: Path,
    manifest_path: Path,
    labels: pd.DataFrame,
    *,
    split_path: Path,
    gene_effect_path: Path,
) -> tuple[str, ...]:
    """Authenticate and reconstruct the finite K562 donor vocabulary."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        "exp13-copy-prior-v1"
    ):
        raise ValueError("copy-prior manifest schema mismatch")
    if manifest.get("donor") != {
        "model_id": "ACH-000551",
        "split": "train",
        "unlabeled": False,
    }:
        raise ValueError("copy-prior donor identity mismatch")
    source = manifest.get("source")
    split_record = manifest.get("split")
    output_record = manifest.get("output")
    if not all(
        isinstance(value, dict) for value in (source, split_record, output_record)
    ):
        raise ValueError("copy-prior manifest hash metadata is missing")
    if source.get("sha256") != _sha256(gene_effect_path):
        raise ValueError("copy-prior source SHA-256 mismatch")
    if split_record.get("sha256") != _sha256(split_path):
        raise ValueError("copy-prior split SHA-256 mismatch")
    if output_record.get("sha256") != _sha256(copy_prior_path):
        raise ValueError("copy-prior output SHA-256 mismatch")
    frame = pd.read_csv(copy_prior_path)
    if tuple(frame.columns) != ("gene_symbol", "gene_effect"):
        raise ValueError("copy-prior CSV columns mismatch")
    symbols = tuple(frame["gene_symbol"].astype(str))
    values = pd.to_numeric(frame["gene_effect"], errors="coerce")
    if len(symbols) != len(set(symbols)) or values.isna().any():
        raise ValueError("copy-prior CSV values are invalid")
    donor = labels.loc[
        (labels["model_id"] == "ACH-000551") & labels["gene_effect"].notna()
    ]
    if symbols != tuple(donor["gene_symbol"].astype(str)) or (
        _sha256(copy_prior_path) != PINNED_COPY_PRIOR_SHA256
    ):
        raise ValueError("copy-prior CSV does not match pinned K562 donor row")
    if output_record.get("gene_symbols_sha256") != _symbols_sha256(symbols):
        raise ValueError("copy-prior symbol coverage hash mismatch")
    return symbols


def coverage_qualified_symbols(
    labels: pd.DataFrame, split: Exp13Split
) -> tuple[str, ...]:
    return build_coverage_universe(labels, split).symbols


def authenticate_stage1_manifest(
    path: Path, expected_manifest_sha256: str
) -> AuthenticatedStage1Vocabulary:
    """Read a sealed manifest and authenticate its ordered vocabulary hash."""
    path = Path(path)
    _require_sha256(expected_manifest_sha256, "Stage-1 manifest SHA-256")
    actual_manifest_sha256 = _sha256(path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Stage-1 manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    manifest = Stage1ArtifactManifest.read(path)
    return AuthenticatedStage1Vocabulary(
        symbols=_unique_symbols(manifest.stage1_genes, "sealed Stage-1 vocabulary"),
        vocabulary_sha256=manifest.stage1_gene_vocabulary_sha256,
        authentication_kind="sealed_stage1_manifest",
        authentication_source=str(path),
        authentication_source_sha256=actual_manifest_sha256,
        checkpoint_sha256=manifest.checkpoint_sha256,
    )


def authenticate_vocabulary(
    symbols: Sequence[str],
    expected_sha256: str,
    *,
    source: str,
    source_sha256: str,
    checkpoint_path: Path,
) -> AuthenticatedStage1Vocabulary:
    """Authenticate an ordered Stage-1 vocabulary supplied without a seal."""
    ordered = _unique_symbols(symbols, "Stage-1 vocabulary")
    _require_sha256(expected_sha256, "Stage-1 vocabulary SHA-256")
    actual = sha256_strings(np.asarray(ordered, dtype=object))
    if actual != expected_sha256:
        raise ValueError(
            f"Stage-1 vocabulary SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    _require_sha256(source_sha256, "Stage-1 vocabulary source SHA-256")
    checkpoint_path = Path(checkpoint_path)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"cannot read Stage-1 checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    key = "perturbations.gene_vocabulary_sha256"
    if not isinstance(state, dict) or key not in state:
        raise ValueError("Stage-1 checkpoint lacks authenticated vocabulary hash")
    value = state[key]
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.numel() != 32
    ):
        raise ValueError("Stage-1 checkpoint vocabulary hash buffer is malformed")
    checkpoint_vocabulary_sha256 = bytes(value.cpu().tolist()).hex()
    if checkpoint_vocabulary_sha256 != actual:
        raise ValueError("Stage-1 checkpoint vocabulary SHA-256 mismatch")
    return AuthenticatedStage1Vocabulary(
        ordered,
        actual,
        "authenticated_vocabulary",
        source,
        source_sha256,
        _sha256(checkpoint_path),
    )


def load_authenticated_vocabulary(
    path: Path, expected_sha256: str, checkpoint_path: Path
) -> AuthenticatedStage1Vocabulary:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Stage-1 vocabulary {path}: {exc}") from exc
    return authenticate_vocabulary(
        payload,
        expected_sha256,
        source=str(path),
        source_sha256=_sha256(path),
        checkpoint_path=checkpoint_path,
    )


def _validate_stage1_vocabulary(
    stage1: AuthenticatedStage1Vocabulary,
) -> tuple[str, ...]:
    symbols = _unique_symbols(stage1.symbols, "authenticated Stage-1 vocabulary")
    actual = sha256_strings(np.asarray(symbols, dtype=object))
    if actual != stage1.vocabulary_sha256:
        raise ValueError("authenticated Stage-1 vocabulary SHA-256 is stale")
    if stage1.authentication_kind not in {
        "sealed_stage1_manifest",
        "authenticated_vocabulary",
    }:
        raise ValueError("unknown Stage-1 vocabulary authentication kind")
    _require_sha256(
        stage1.authentication_source_sha256,
        "Stage-1 authentication source SHA-256",
    )
    if stage1.checkpoint_sha256 is None:
        raise ValueError("authenticated Stage-1 vocabulary lacks checkpoint identity")
    _require_sha256(stage1.checkpoint_sha256, "Stage-1 checkpoint SHA-256")
    return symbols


def build_embedding_union(
    scored_symbols: Sequence[str], stage1_symbols: Sequence[str]
) -> tuple[str, ...]:
    """Return the deterministic union consumed by the sorting precompute script."""
    scored = _unique_symbols(scored_symbols, "scored GeneEffect universe")
    stage1 = _unique_symbols(stage1_symbols, "Stage-1 vocabulary")
    return tuple(sorted(set(scored) | set(stage1)))


def inspect_npz_coverage(
    npz_path: Path, required_symbols: Sequence[str]
) -> NpzCoverage:
    required = _unique_symbols(required_symbols, "required universe")
    try:
        with np.load(npz_path, allow_pickle=True) as payload:
            expected_keys = {"symbols", "vectors", "resolved"}
            if set(payload.files) != expected_keys:
                raise ValueError(
                    f"ESM2 NPZ keys {sorted(payload.files)} != {sorted(expected_keys)}"
                )
            symbols = tuple(str(symbol) for symbol in payload["symbols"].tolist())
            vectors = np.asarray(payload["vectors"])
            resolved = np.asarray(payload["resolved"])
    except (OSError, EOFError, KeyError) as exc:
        raise ValueError(f"cannot read ESM2 NPZ {npz_path}: {exc}") from exc
    _unique_symbols(symbols, "ESM2 NPZ")
    if vectors.ndim != 2 or vectors.shape[0] != len(symbols):
        raise ValueError("ESM2 NPZ vectors shape does not match symbols")
    if resolved.dtype != np.dtype(bool) or resolved.shape != (len(symbols),):
        raise ValueError("ESM2 NPZ resolved must be a one-dimensional bool array")
    if not np.issubdtype(vectors.dtype, np.number):
        raise ValueError("ESM2 NPZ vectors must be numeric")
    if resolved.any() and not np.isfinite(vectors[resolved]).all():
        raise ValueError("ESM2 NPZ contains non-finite resolved vectors")
    resolved_symbols = {
        symbol
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if is_resolved
    }
    missing = tuple(symbol for symbol in required if symbol not in resolved_symbols)
    return NpzCoverage(
        len(required),
        len(required) - len(missing),
        missing,
        int(vectors.shape[1]),
        _sha256(npz_path),
    )


def require_npz_coverage(
    npz_path: Path,
    required_symbols: Sequence[str],
    *,
    must_resolve_symbols: Sequence[str] | None = None,
    expected_width: int = 1_280,
) -> NpzCoverage:
    report = inspect_npz_coverage(npz_path, required_symbols)
    required = _unique_symbols(required_symbols, "required universe")
    must_resolve = _unique_symbols(
        required if must_resolve_symbols is None else must_resolve_symbols,
        "must-resolve ESM2 universe",
    )
    if not set(must_resolve).issubset(required):
        raise ValueError("must-resolve ESM2 symbols must belong to the NPZ universe")
    missing_required = tuple(
        symbol for symbol in must_resolve if symbol in set(report.missing)
    )
    if missing_required:
        raise ValueError(
            "ESM2 NPZ must-resolve coverage incomplete: "
            f"missing {len(missing_required)}/{len(must_resolve)} symbols; "
            f"examples={list(missing_required[:20])}"
        )
    with np.load(npz_path, allow_pickle=True) as payload:
        observed = tuple(str(symbol) for symbol in payload["symbols"].tolist())
    if observed != required:
        raise ValueError("ESM2 NPZ symbol order/universe does not exactly match target")
    if report.vector_width != expected_width:
        raise ValueError(
            f"ESM2 NPZ vector width {report.vector_width} != {expected_width}"
        )
    with np.load(npz_path, allow_pickle=True) as payload:
        if payload["vectors"].dtype != np.dtype(np.float32):
            raise ValueError("ESM2 NPZ vectors must have dtype float32")
    return report


def build_precompute_command(
    embedding_union_csv: Path,
    esm_out: Path,
    sequence_cache: Path,
    *,
    provenance_out: Path | None = None,
    model: str = "facebook/esm2_t33_650M_UR50D",
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "scripts/precompute_esm2_embeddings.py",
        "--benchmark-csv",
        str(embedding_union_csv),
        "--symbol-column",
        "gene_symbol",
        "--out",
        str(esm_out),
        "--seq-cache",
        str(sequence_cache),
        "--provenance-out",
        str(provenance_out or esm_out.with_suffix(esm_out.suffix + ".provenance.json")),
        "--model",
        model,
    ]
    if cache_dir is not None:
        command.extend(("--cache-dir", str(cache_dir)))
    if local_files_only:
        command.append("--local-files-only")
    return shlex.join(command)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _universe_record(symbols: tuple[str, ...], path: Path) -> dict[str, object]:
    return {
        "symbols": list(symbols),
        "count": len(symbols),
        "symbols_sha256": _symbols_sha256(symbols),
        "csv": str(path),
        "csv_sha256": _sha256(path),
    }


def write_universe_artifacts(
    coverage_qualified_upper_bound: CoverageUniverse,
    copy_prior_eligible_candidates: CoverageUniverse,
    stage1: AuthenticatedStage1Vocabulary,
    scored_output_csv: Path,
    embedding_union_output_csv: Path,
    output_manifest: Path,
    *,
    split_path: Path,
    gene_effect_path: Path,
    copy_prior_path: Path,
    copy_prior_manifest_path: Path,
    expected_upper_bound_count: int = EXPECTED_COVERAGE_QUALIFIED_COUNT,
    expected_candidate_count: int = EXPECTED_COPY_PRIOR_ELIGIBLE_COUNT,
    verified_npz_path: Path | None = None,
    expected_npz_sha256: str | None = None,
    esm2_provenance_path: Path | None = None,
    esm2_model: str = "facebook/esm2_t33_650M_UR50D",
) -> dict[str, object]:
    """Fresh-write separate scored and embedding-union artifacts."""
    upper_symbols = _unique_symbols(
        coverage_qualified_upper_bound.symbols,
        "coverage-qualified GeneEffect upper bound",
    )
    scored_symbols = _unique_symbols(
        copy_prior_eligible_candidates.symbols,
        "copy-prior-eligible GeneEffect candidates",
    )
    stage1_symbols = _validate_stage1_vocabulary(stage1)
    if tuple(sorted(scored_symbols)) != scored_symbols:
        raise ValueError("scored GeneEffect universe must be sorted")
    if tuple(sorted(upper_symbols)) != upper_symbols:
        raise ValueError("coverage-qualified GeneEffect upper bound must be sorted")
    if len(upper_symbols) != expected_upper_bound_count:
        raise ValueError(
            f"coverage-qualified GeneEffect upper bound has {len(upper_symbols)} "
            f"genes; expected {expected_upper_bound_count}"
        )
    if not set(scored_symbols).issubset(upper_symbols):
        raise ValueError("copy-prior candidates must be a subset of the upper bound")
    if len(scored_symbols) != expected_candidate_count:
        raise ValueError(
            f"copy-prior-eligible candidate universe has {len(scored_symbols)} "
            f"genes; expected {expected_candidate_count}"
        )
    union = build_embedding_union(scored_symbols, stage1_symbols)
    outputs = (scored_output_csv, embedding_union_output_csv, output_manifest)
    if len({Path(path).resolve() for path in outputs}) != len(outputs):
        raise ValueError(
            "scored CSV, embedding-union CSV, and manifest must be distinct"
        )
    verification_inputs = (
        verified_npz_path,
        expected_npz_sha256,
        esm2_provenance_path,
    )
    if any(value is not None for value in verification_inputs) and any(
        value is None for value in verification_inputs
    ):
        raise ValueError(
            "verified NPZ, expected SHA-256, and ESM2 provenance must be "
            "provided together"
        )
    verified_npz = None
    esm2_provenance = None
    if verified_npz_path is not None:
        _require_sha256(expected_npz_sha256, "expected ESM2 NPZ SHA-256")
        verified_npz = require_npz_coverage(
            verified_npz_path,
            union,
            must_resolve_symbols=stage1_symbols,
        )
        if verified_npz.artifact_sha256 != expected_npz_sha256:
            raise ValueError("ESM2 NPZ SHA-256 mismatch")
        esm2_provenance = load_and_authenticate_esm2_provenance(
            esm2_provenance_path,
            verified_npz_path,
            expected_width=verified_npz.vector_width,
        )
        if esm2_provenance["requested_model_id"] != esm2_model:
            raise ValueError("ESM2 requested model differs from builder contract")
        sequence_source = esm2_provenance.get("sequence_source")
        expected_union_csv = "gene_symbol\n" + "".join(
            f"{symbol}\n" for symbol in union
        )
        if (
            not isinstance(sequence_source, dict)
            or sequence_source.get("symbol_columns") != ["gene_symbol"]
            or sequence_source.get("benchmark_csv_sha256")
            != hashlib.sha256(expected_union_csv.encode("utf-8")).hexdigest()
        ):
            raise ValueError("ESM2 provenance does not bind the embedding-union CSV")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    csv_texts = {
        scored_output_csv: "gene_symbol\n"
        + "".join(f"{symbol}\n" for symbol in scored_symbols),
        embedding_union_output_csv: "gene_symbol\n"
        + "".join(f"{symbol}\n" for symbol in union),
    }
    written: list[Path] = []
    try:
        for path, text in csv_texts.items():
            _atomic_write_text(path, text)
            written.append(path)
        scored_set = set(scored_symbols)
        stage1_only = tuple(
            symbol for symbol in stage1.symbols if symbol not in scored_set
        )
        stage1_record = asdict(stage1)
        stage1_record["symbols"] = list(stage1.symbols)
        missing_esm2 = set(verified_npz.missing) if verified_npz else set()
        unresolved_candidates = tuple(
            symbol for symbol in scored_symbols if symbol in missing_esm2
        )
        final_symbols = tuple(
            symbol for symbol in scored_symbols if symbol not in missing_esm2
        )
        if esm2_provenance is not None and not final_symbols:
            raise ValueError("ESM2 resolution produced an empty evaluated universe")
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "authenticated_complete"
                if esm2_provenance is not None
                else "coverage_qualified_pre_esm2"
            ),
            "metrics_membership": "copy_prior_candidates_intersect_esm2_resolved",
            "scored_symbols": (
                list(final_symbols) if esm2_provenance is not None else None
            ),
            "scored_gene_count": (
                len(final_symbols) if esm2_provenance is not None else None
            ),
            "coverage_thresholds": COVERAGE_THRESHOLDS,
            "input_sha256": {
                "split": _sha256(split_path),
                "gene_effect": _sha256(gene_effect_path),
                "copy_prior": _sha256(copy_prior_path),
                "copy_prior_manifest": _sha256(copy_prior_manifest_path),
            },
            "stage1_vocabulary": stage1_record,
            "coverage_qualified_upper_bound": {
                "symbols": list(upper_symbols),
                "count": len(upper_symbols),
                "symbols_sha256": _symbols_sha256(upper_symbols),
                "drop_report": list(coverage_qualified_upper_bound.dropped),
            },
            "copy_prior_eligible_candidates": {
                **_universe_record(scored_symbols, scored_output_csv),
                "drop_report": list(copy_prior_eligible_candidates.dropped),
            },
            "final_evaluated_universe": (
                None
                if esm2_provenance is None
                else {
                    "symbols": list(final_symbols),
                    "count": len(final_symbols),
                    "symbols_sha256": _symbols_sha256(final_symbols),
                    "unresolved_candidate_symbols": list(unresolved_candidates),
                    "unresolved_candidate_count": len(unresolved_candidates),
                }
            ),
            "embedding_union": {
                **_universe_record(union, embedding_union_output_csv),
                "stage1_only_symbols": list(stage1_only),
                "stage1_only_count": len(stage1_only),
                "drop_report": [],
                "requested_precompute_model": esm2_model,
                "model_identity_status": (
                    "recorded_from_loaded_runtime_state"
                    if esm2_provenance is not None
                    else "pending"
                ),
                "provenance_manifest": (
                    None
                    if esm2_provenance is None
                    else {
                        "path": str(esm2_provenance_path),
                        "sha256": _sha256(esm2_provenance_path),
                        "payload": esm2_provenance,
                    }
                ),
                "uniprot_mapping": (
                    None
                    if esm2_provenance is None
                    else {
                        "isoform_policy": "canonical_reviewed_top_hit",
                        "json_sha256": esm2_provenance["sequence_source"][
                            "uniprot_mapping_json_sha256"
                        ],
                        "csv_sha256": esm2_provenance["sequence_source"][
                            "uniprot_mapping_csv_sha256"
                        ],
                    }
                ),
                "verified_npz": (
                    None
                    if verified_npz is None
                    else {
                        "path": str(verified_npz_path),
                        "artifact_sha256": verified_npz.artifact_sha256,
                        "resolved_count": verified_npz.resolved_count,
                        "vector_width": verified_npz.vector_width,
                    }
                ),
            },
        }
        _atomic_write_text(
            output_manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build separate Exp13 scored and ESM2 embedding universes."
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--copy-prior", type=Path, required=True)
    parser.add_argument("--copy-prior-manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage1-manifest", type=Path)
    source.add_argument("--stage1-vocabulary-json", type=Path)
    parser.add_argument("--stage1-manifest-sha256")
    parser.add_argument("--stage1-vocabulary-sha256")
    parser.add_argument("--stage1-vocabulary-checkpoint", type=Path)
    parser.add_argument("--scored-out-csv", type=Path, required=True)
    parser.add_argument("--embedding-union-out-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-npz", type=Path)
    parser.add_argument("--verify-npz-sha256")
    parser.add_argument("--esm-provenance", type=Path)
    parser.add_argument("--esm-out", type=Path)
    parser.add_argument("--seq-cache", type=Path)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.esm_out is None) != (args.seq_cache is None):
        parser.error("--esm-out and --seq-cache must be provided together")
    if args.stage1_vocabulary_json is not None:
        if args.stage1_manifest_sha256 is not None:
            parser.error(
                "--stage1-manifest-sha256 is only valid with --stage1-manifest"
            )
        if args.stage1_vocabulary_sha256 is None:
            parser.error(
                "--stage1-vocabulary-sha256 is required with --stage1-vocabulary-json"
            )
        if args.stage1_vocabulary_checkpoint is None:
            parser.error(
                "--stage1-vocabulary-checkpoint is required with "
                "--stage1-vocabulary-json"
            )
        stage1 = load_authenticated_vocabulary(
            args.stage1_vocabulary_json,
            args.stage1_vocabulary_sha256,
            args.stage1_vocabulary_checkpoint,
        )
    else:
        if args.stage1_vocabulary_checkpoint is not None:
            parser.error(
                "--stage1-vocabulary-checkpoint is only valid with "
                "--stage1-vocabulary-json"
            )
        if args.stage1_manifest_sha256 is None:
            parser.error("--stage1-manifest-sha256 is required with --stage1-manifest")
        if args.stage1_vocabulary_sha256 is not None:
            parser.error(
                "--stage1-vocabulary-sha256 is only valid with --stage1-vocabulary-json"
            )
        stage1 = authenticate_stage1_manifest(
            args.stage1_manifest, args.stage1_manifest_sha256
        )
    split = load_exp13_split(args.split)
    require_pinned_gene_effect(args.gene_effect)
    labels = load_geneeffect_long(args.gene_effect, split)
    upper_bound = build_coverage_universe(labels, split)
    copy_prior_symbols = load_authenticated_copy_prior_symbols(
        args.copy_prior,
        args.copy_prior_manifest,
        labels,
        split_path=args.split,
        gene_effect_path=args.gene_effect,
    )
    candidates = restrict_coverage_universe_to_copy_prior(
        upper_bound, copy_prior_symbols
    )
    union = build_embedding_union(candidates.symbols, stage1.symbols)
    verification_inputs = (
        args.verify_npz,
        args.verify_npz_sha256,
        args.esm_provenance,
    )
    if any(value is not None for value in verification_inputs) and any(
        value is None for value in verification_inputs
    ):
        parser.error(
            "--verify-npz, --verify-npz-sha256, and --esm-provenance must be "
            "provided together"
        )
    npz_report = (
        require_npz_coverage(
            args.verify_npz,
            union,
            must_resolve_symbols=stage1.symbols,
        )
        if args.verify_npz is not None
        else None
    )
    manifest = write_universe_artifacts(
        upper_bound,
        candidates,
        stage1,
        args.scored_out_csv,
        args.embedding_union_out_csv,
        args.manifest,
        split_path=args.split,
        gene_effect_path=args.gene_effect,
        copy_prior_path=args.copy_prior,
        copy_prior_manifest_path=args.copy_prior_manifest,
        verified_npz_path=args.verify_npz,
        expected_npz_sha256=args.verify_npz_sha256,
        esm2_provenance_path=args.esm_provenance,
        esm2_model=args.model,
    )
    scored_count = manifest["copy_prior_eligible_candidates"]["count"]
    union_count = manifest["embedding_union"]["count"]
    print(f"wrote {scored_count} scored genes and {union_count} embedding genes")
    if npz_report is not None:
        print(
            f"verified {npz_report.resolved_count}/{npz_report.required_count} "
            "ESM2 genes"
        )
    if args.esm_out is not None:
        print("next command:", file=sys.stdout)
        print(
            build_precompute_command(
                args.embedding_union_out_csv,
                args.esm_out,
                args.seq_cache,
                model=args.model,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            ),
            file=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
