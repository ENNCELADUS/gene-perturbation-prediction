#!/usr/bin/env python3
"""Build or verify the 226-line Exp13 Tx1 cache from raw-UMI h5ad sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from aivc_model.gene_splits import sha256_file  # noqa: E402
from aivc_model.geneeffect_data import (  # noqa: E402
    RAW_UMI_SEMANTICS,
    load_exp13_split,
    load_source_registry,
)
from aivc_model.state_core import sha256_strings  # noqa: E402
from aivc_model.tx1_embed_cache import (  # noqa: E402
    MODEL_LABEL,
    authenticate_tx1_registration,
    embed_registry_lines,
    load_hvg_gene_order,
    verify_cache,
    write_run_manifest,
)
from scripts.build_tx1_basal_embeddings import (  # noqa: E402
    _build_tx1_encoder,
    _verify_model_dir_matches_source_manifest,
)


def _source_hashes(registry: object) -> dict[str, str]:
    return {
        str(model_id): sha256_file(Path(str(row["source_path"])))
        for model_id, row in registry.iterrows()
    }


def _verify(args: argparse.Namespace, registry: object) -> dict[str, object]:
    source_manifest, _ = authenticate_tx1_registration(args.tx1_registration)
    return verify_cache(
        args.cache_dir,
        expected_model_ids=tuple(registry.index.astype(str)),
        expected_source_sha256=_source_hashes(registry),
        expected_matrix_semantics=RAW_UMI_SEMANTICS,
        expected_tx1_source_manifest=source_manifest,
    )


def _require_build_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--model-dir", args.model_dir),
            ("--hvg-state-model-dir", args.hvg_state_model_dir),
            ("--tx1-registration", args.tx1_registration),
            ("--var-ensembl-col", args.var_ensembl_col),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"missing required build arguments: {missing}")


def run(args: argparse.Namespace) -> dict[str, object]:
    split = load_exp13_split(args.split)
    registry = load_source_registry(args.source_registry, split)
    if args.verify_only:
        return _verify(args, registry)
    _require_build_args(args)
    source_manifest, source_manifest_sha256 = authenticate_tx1_registration(
        args.tx1_registration
    )
    _verify_model_dir_matches_source_manifest(args.model_dir, source_manifest)
    encoder, load_report = _build_tx1_encoder(
        args.model_dir, args.batch_size, args.max_length
    )
    only_lines = list(args.only_line) or None
    entries = embed_registry_lines(
        registry,
        args.cache_dir,
        encoder=encoder,
        hvg_state_model_dir=args.hvg_state_model_dir,
        var_ensembl_col=args.var_ensembl_col,
        hvg_gene_symbol_col=args.hvg_gene_symbol_col,
        max_cells_per_line=args.max_cells_per_line,
        seed=args.seed,
        only_lines=only_lines,
    )
    hvg_order = load_hvg_gene_order(args.hvg_state_model_dir)
    write_run_manifest(
        args.cache_dir,
        model_label=MODEL_LABEL,
        source_manifest=source_manifest,
        line_entries=entries,
        config_snapshot={
            "source_registry_path": str(args.source_registry),
            "source_registry_sha256": sha256_file(args.source_registry),
            "tx1_registration_path": str(args.tx1_registration),
            "tx1_registration_sha256": sha256_file(args.tx1_registration),
            "tx1_source_manifest_sha256": source_manifest_sha256,
            "matrix_semantics": RAW_UMI_SEMANTICS,
            "max_cells_per_line": args.max_cells_per_line,
            "seed": args.seed,
            "selection_algorithm": "sha256_model_id_cell_id_v1",
            "hvg_gene_symbol_col": args.hvg_gene_symbol_col,
            "var_ensembl_col": args.var_ensembl_col,
            "hvg_gene_order_sha256": sha256_strings(hvg_order),
            "checkpoint_load_report": load_report,
        },
    )
    if only_lines is not None:
        return verify_cache(
            args.cache_dir,
            only_lines=only_lines,
            expected_model_ids=tuple(registry.index.astype(str)),
            expected_source_sha256=_source_hashes(registry),
            expected_matrix_semantics=RAW_UMI_SEMANTICS,
            expected_tx1_source_manifest=source_manifest,
        )
    return _verify(args, registry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/benchmarks/cell_line_geneeffect_226_split.json"),
    )
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--hvg-state-model-dir", type=Path)
    parser.add_argument(
        "--tx1-registration",
        type=Path,
        default=Path("results/phase_a_tx1_20260724/phase_a_registration.json"),
    )
    parser.add_argument("--var-ensembl-col")
    parser.add_argument("--hvg-gene-symbol-col", default="gene_symbol")
    parser.add_argument("--max-cells-per-line", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--only-line", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
