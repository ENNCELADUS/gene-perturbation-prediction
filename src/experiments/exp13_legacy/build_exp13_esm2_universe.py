"""experiments / exp13 legacy / build exp13 esm2 universe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from src.data.geneeffect import load_exp13_split, load_geneeffect_long
from src.data.esm2_provenance import load_and_authenticate_esm2_provenance
from src.experiments.exp13_legacy.stage1_artifact import Stage1ArtifactManifest
from src.data.gene_order import sha256_strings
from src.data.prepare.build_exp13_esm2_universe import COVERAGE_THRESHOLDS
from src.data.prepare.build_exp13_esm2_universe import CoverageUniverse
from src.data.prepare.build_exp13_esm2_universe import (
    EXPECTED_COPY_PRIOR_ELIGIBLE_COUNT,
)
from src.data.prepare.build_exp13_esm2_universe import EXPECTED_COVERAGE_QUALIFIED_COUNT
from src.data.prepare.build_exp13_esm2_universe import SCHEMA_VERSION
from src.data.prepare.build_exp13_esm2_universe import _atomic_write_text
from src.data.prepare.build_exp13_esm2_universe import _require_sha256
from src.data.prepare.build_exp13_esm2_universe import _sha256
from src.data.prepare.build_exp13_esm2_universe import _symbols_sha256
from src.data.prepare.build_exp13_esm2_universe import _unique_symbols
from src.data.prepare.build_exp13_esm2_universe import _universe_record
from src.data.prepare.build_exp13_esm2_universe import build_coverage_universe
from src.data.prepare.build_exp13_esm2_universe import build_embedding_union
from src.data.prepare.build_exp13_esm2_universe import build_precompute_command
from src.data.prepare.build_exp13_esm2_universe import (
    load_authenticated_copy_prior_symbols,
)
from src.data.prepare.build_exp13_esm2_universe import require_npz_coverage
from src.data.prepare.build_exp13_esm2_universe import require_pinned_gene_effect
from src.data.prepare.build_exp13_esm2_universe import (
    restrict_coverage_universe_to_copy_prior,
)


@dataclass(frozen=True)
class AuthenticatedStage1Vocabulary:
    symbols: tuple[str, ...]
    vocabulary_sha256: str
    authentication_kind: str
    authentication_source: str
    authentication_source_sha256: str
    checkpoint_sha256: str | None


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
