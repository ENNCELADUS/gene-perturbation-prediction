#!/usr/bin/env python3
"""Build or fully verify the Exp13 raw-UMI q_sc cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aivc_model.esm2_provenance import load_and_authenticate_esm2_provenance
from aivc_model.gene_embeddings import load_esm2_embeddings
from aivc_model.geneeffect_data import (
    PINNED_COPY_PRIOR_SHA256,
    build_q_sc_shards,
    build_scored_universe,
    load_exp13_split,
    load_geneeffect_long,
    load_source_registry,
    restrict_scored_universe_to_copy_prior,
    verify_q_sc_shards,
)


SCHEMA_VERSION = "exp13-q-sc-cache-report-v1"
COVERAGE_THRESHOLDS = {"train": 5, "val": 3, "test": 3}
PINNED_GENE_EFFECT_SHA256 = (
    "e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symbols_sha256(symbols: Sequence[str]) -> str:
    payload = "".join(f"{symbol}\n" for symbol in symbols).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_resolved_esm2_symbols(path: Path) -> tuple[str, ...]:
    try:
        with np.load(path, allow_pickle=True) as payload:
            if set(payload.files) != {"symbols", "vectors", "resolved"}:
                raise ValueError(
                    "target ESM2 NPZ must contain exactly symbols, vectors, resolved"
                )
            symbols = np.asarray(payload["symbols"])
            vectors = np.asarray(payload["vectors"])
            resolved = np.asarray(payload["resolved"])
    except (OSError, EOFError, KeyError) as exc:
        raise ValueError(f"cannot read target ESM2 NPZ {path}: {exc}") from exc
    if symbols.ndim != 1 or vectors.ndim != 2:
        raise ValueError("target ESM2 symbols/vectors must be one/two-dimensional")
    if vectors.shape[0] != len(symbols):
        raise ValueError("target ESM2 vectors row count does not match symbols")
    if resolved.dtype != np.dtype(bool) or resolved.shape != (len(symbols),):
        raise ValueError("target ESM2 resolved must be a one-dimensional bool array")

    table = load_esm2_embeddings(path)
    ordered = tuple(table.vectors_by_symbol)
    if not ordered:
        raise ValueError("target ESM2 NPZ has no resolved genes")
    expected = tuple(
        str(symbol).upper()
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if is_resolved
    )
    if ordered != expected:
        raise ValueError("target ESM2 resolved symbols are empty or duplicated")
    return ordered


def build_or_verify_q_sc_cache(
    *,
    split_path: Path,
    gene_effect_path: Path,
    esm2_path: Path,
    esm2_universe_manifest_path: Path,
    esm2_provenance_manifest_path: Path,
    copy_prior_path: Path,
    copy_prior_manifest_path: Path,
    registry_path: Path,
    output_dir: Path,
    resume: bool = False,
    verify_only: bool = False,
    reader: Callable[[Path], Any] | None = None,
) -> dict[str, object]:
    """Build when requested, then run the unrestricted cache verifier."""
    if resume and verify_only:
        raise ValueError("resume and verify_only are mutually exclusive")

    split = load_exp13_split(split_path)
    labels = load_geneeffect_long(gene_effect_path, split)
    if _sha256(gene_effect_path) != PINNED_GENE_EFFECT_SHA256:
        raise ValueError("DepMap GeneEffect does not match pinned Public 26Q1")
    esm2_symbols = _load_resolved_esm2_symbols(esm2_path)
    universe = build_scored_universe(labels, split, esm2_symbols)
    copy_prior = pd.read_csv(copy_prior_path)
    if tuple(copy_prior.columns) != ("gene_symbol", "gene_effect"):
        raise ValueError("copy-prior CSV columns must be gene_symbol,gene_effect")
    if (
        copy_prior["gene_symbol"].isna().any()
        or copy_prior["gene_symbol"].duplicated().any()
    ):
        raise ValueError("copy-prior gene symbols must be nonmissing and unique")
    try:
        copy_manifest = json.loads(copy_prior_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read copy-prior manifest: {exc}") from exc
    if (
        not isinstance(copy_manifest, dict)
        or copy_manifest.get("schema_version") != "exp13-copy-prior-v1"
        or copy_manifest.get("donor")
        != {"model_id": "ACH-000551", "split": "train", "unlabeled": False}
        or not isinstance(copy_manifest.get("source"), dict)
        or copy_manifest["source"].get("sha256") != _sha256(gene_effect_path)
        or not isinstance(copy_manifest.get("split"), dict)
        or copy_manifest["split"].get("sha256") != _sha256(split_path)
        or not isinstance(copy_manifest.get("output"), dict)
        or copy_manifest["output"].get("sha256") != _sha256(copy_prior_path)
    ):
        raise ValueError("copy-prior manifest does not authenticate its inputs and CSV")
    copy_values = pd.to_numeric(copy_prior["gene_effect"], errors="coerce")
    if copy_values.isna().any() or not np.isfinite(copy_values.to_numpy()).all():
        raise ValueError("copy-prior GeneEffect values must be finite numeric values")
    donor = labels.loc[
        (labels["model_id"] == "ACH-000551") & labels["gene_effect"].notna()
    ]
    copy_symbols = tuple(copy_prior["gene_symbol"].astype(str))
    if (
        copy_symbols != tuple(donor["gene_symbol"].astype(str))
        or _sha256(copy_prior_path) != PINNED_COPY_PRIOR_SHA256
    ):
        raise ValueError("copy-prior CSV does not match the pinned K562 donor row")
    universe = restrict_scored_universe_to_copy_prior(universe, copy_symbols)
    if not universe.symbols:
        raise ValueError("coverage qualification produced an empty gene universe")
    esm2_table = load_esm2_embeddings(esm2_path)
    provenance = load_and_authenticate_esm2_provenance(
        esm2_provenance_manifest_path,
        esm2_path,
        expected_width=esm2_table.dim,
    )
    try:
        esm2_universe = json.loads(
            esm2_universe_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ESM2 universe manifest: {exc}") from exc
    if not isinstance(esm2_universe, dict):
        raise ValueError("ESM2 universe manifest root must be an object")
    candidate_record = esm2_universe.get("copy_prior_eligible_candidates")
    final_record = esm2_universe.get("final_evaluated_universe")
    union_record = esm2_universe.get("embedding_union")
    input_hashes = esm2_universe.get("input_sha256")
    candidates = (
        candidate_record.get("symbols") if isinstance(candidate_record, dict) else None
    )
    resolved = set(esm2_symbols)
    expected_final = (
        tuple(symbol for symbol in candidates if symbol in resolved)
        if isinstance(candidates, list)
        else ()
    )
    unresolved = (
        tuple(symbol for symbol in candidates if symbol not in resolved)
        if isinstance(candidates, list)
        else ()
    )
    expected_final_record = {
        "symbols": list(expected_final),
        "count": len(expected_final),
        "symbols_sha256": _symbols_sha256(expected_final),
        "unresolved_candidate_symbols": list(unresolved),
        "unresolved_candidate_count": len(unresolved),
    }
    if (
        esm2_universe.get("schema_version") != "exp13-esm2-universes-v2"
        or esm2_universe.get("status") != "authenticated_complete"
        or not isinstance(candidate_record, dict)
        or final_record != expected_final_record
        or expected_final != universe.symbols
        or esm2_universe.get("scored_symbols") != list(expected_final)
        or esm2_universe.get("scored_gene_count") != len(expected_final)
        or esm2_universe.get("coverage_thresholds") != COVERAGE_THRESHOLDS
        or not isinstance(union_record, dict)
        or not isinstance(union_record.get("verified_npz"), dict)
        or union_record["verified_npz"].get("artifact_sha256") != _sha256(esm2_path)
        or not isinstance(union_record.get("provenance_manifest"), dict)
        or union_record["provenance_manifest"].get("sha256")
        != _sha256(esm2_provenance_manifest_path)
        or union_record["provenance_manifest"].get("payload") != provenance
        or input_hashes
        != {
            "split": _sha256(split_path),
            "gene_effect": _sha256(gene_effect_path),
            "copy_prior": _sha256(copy_prior_path),
            "copy_prior_manifest": _sha256(copy_prior_manifest_path),
        }
    ):
        raise ValueError("ESM2 universe manifest does not authenticate final symbols")
    registry = load_source_registry(registry_path, split)

    build_manifest = None
    if not verify_only:
        build_manifest = build_q_sc_shards(
            registry,
            output_dir,
            universe.symbols,
            reader=reader,
            resume=resume,
        )
    verification = verify_q_sc_shards(registry, output_dir, universe.symbols)
    coverage = universe.coverage
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": verification["status"],
        "mode": "verify_only" if verify_only else "resume" if resume else "fresh",
        "input_sha256": {
            "split": _sha256(split_path),
            "gene_effect": _sha256(gene_effect_path),
            "target_esm2": _sha256(esm2_path),
            "esm2_universe_manifest": _sha256(esm2_universe_manifest_path),
            "esm2_provenance_manifest": _sha256(esm2_provenance_manifest_path),
            "copy_prior": _sha256(copy_prior_path),
            "copy_prior_manifest": _sha256(copy_prior_manifest_path),
            "source_registry": _sha256(registry_path),
        },
        "gene_universe": {
            **universe.manifest,
            "symbols_sha256": _symbols_sha256(universe.symbols),
        },
        "coverage_qualification": {
            "thresholds": COVERAGE_THRESHOLDS,
            "included_gene_count": int(coverage["included"].sum()),
            "excluded_gene_count": int((~coverage["included"]).sum()),
            "test_labels_role": "coverage_qualification_only",
            "test_labels_used_for_fit": False,
            "fit_operations": [],
        },
        "registry": {
            "expected_line_count": len(split.all_model_ids),
            "validated_line_count": len(registry),
        },
        "build_manifest": build_manifest,
        "verification": verification,
    }
    return report


def _atomic_write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--esm2", type=Path, required=True)
    parser.add_argument("--esm2-universe-manifest", type=Path, required=True)
    parser.add_argument("--esm2-provenance-manifest", type=Path, required=True)
    parser.add_argument("--copy-prior", type=Path, required=True)
    parser.add_argument("--copy-prior-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume", action="store_true", help="resume only hash-valid existing shards"
    )
    mode.add_argument(
        "--verify-only", action="store_true", help="do not create or replace shards"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_or_verify_q_sc_cache(
        split_path=args.split,
        gene_effect_path=args.gene_effect,
        esm2_path=args.esm2,
        esm2_universe_manifest_path=args.esm2_universe_manifest,
        esm2_provenance_manifest_path=args.esm2_provenance_manifest,
        copy_prior_path=args.copy_prior,
        copy_prior_manifest_path=args.copy_prior_manifest,
        registry_path=args.registry,
        output_dir=args.output_dir,
        resume=args.resume,
        verify_only=args.verify_only,
    )
    _atomic_write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
