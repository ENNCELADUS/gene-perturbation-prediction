#!/usr/bin/env python3
"""Build the complete per-line raw-UMI source registry for Exp13."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from aivc_model.geneeffect_data import RAW_UMI_SEMANTICS, load_exp13_split


REGISTRY_COLUMNS = (
    "model_id",
    "source_path",
    "source_kind",
    "matrix_semantics",
)
EXPECTED_SOURCE_COUNTS = {
    "kinker_umi_152": 152,
    "raw_umi_27": 27,
    "original_47": 47,
}


def _scan_h5ad_dir(path: Path, source_label: str) -> list[dict[str, str]]:
    if not path.is_dir():
        raise FileNotFoundError(f"{source_label} directory does not exist: {path}")
    return [
        {
            "model_id": source.stem,
            "source_path": str(source.resolve()),
            "source_kind": "h5ad",
            "matrix_semantics": RAW_UMI_SEMANTICS,
        }
        for source in sorted(path.glob("*.h5ad"))
    ]


def _load_original_registry(path: Path) -> list[dict[str, str]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != REGISTRY_COLUMNS:
        raise ValueError(
            "original-47 registry columns must be exactly "
            f"{list(REGISTRY_COLUMNS)}; observed {list(frame.columns)}"
        )
    if (frame == "").any().any():
        raise ValueError("original-47 registry contains empty values")
    bad_kind = sorted(frame.loc[frame["source_kind"] != "h5ad", "model_id"])
    if bad_kind:
        raise ValueError(f"original-47 registry has non-h5ad sources: {bad_kind}")
    bad_semantics = sorted(
        frame.loc[frame["matrix_semantics"] != RAW_UMI_SEMANTICS, "model_id"]
    )
    if bad_semantics:
        raise ValueError(
            f"original-47 registry has non-raw-UMI semantics: {bad_semantics}"
        )
    rows = frame.to_dict("records")
    for row in rows:
        source = Path(row["source_path"])
        if not source.is_absolute():
            source = path.parent / source
        row["source_path"] = str(source.resolve())
    return rows


def _verify_obs_model_id(
    rows: list[dict[str, str]], reader: Callable[[Path], Any]
) -> list[str]:
    problems = []
    for row in rows:
        model_id = row["model_id"]
        adata = reader(Path(row["source_path"]))
        try:
            if "model_id" not in adata.obs.columns:
                problems.append(f"{model_id}: AnnData obs is missing model_id")
                continue
            observed = set(adata.obs["model_id"].astype(str))
            if observed != {model_id}:
                problems.append(
                    f"{model_id}: AnnData obs.model_id values are {sorted(observed)}"
                )
        finally:
            backing = getattr(adata, "file", None)
            if backing is not None and hasattr(backing, "close"):
                backing.close()
    return problems


def _audit(
    rows_by_source: dict[str, list[dict[str, str]]], expected_ids: tuple[str, ...]
) -> dict[str, object]:
    rows = [row for source_rows in rows_by_source.values() for row in source_rows]
    observed_ids = [row["model_id"] for row in rows]
    observed_paths = [row["source_path"] for row in rows]
    expected = set(expected_ids)
    observed = set(observed_ids)
    duplicate_ids = sorted(
        model_id for model_id, count in Counter(observed_ids).items() if count > 1
    )
    duplicate_paths = sorted(
        source_path
        for source_path, count in Counter(observed_paths).items()
        if count > 1
    )
    missing_sources = sorted(
        row["source_path"] for row in rows if not Path(row["source_path"]).is_file()
    )
    source_counts = {
        name: len(source_rows) for name, source_rows in rows_by_source.items()
    }
    count_mismatches = {
        name: {"expected": expected_count, "observed": source_counts.get(name, 0)}
        for name, expected_count in EXPECTED_SOURCE_COUNTS.items()
        if source_counts.get(name, 0) != expected_count
    }
    report: dict[str, object] = {
        "status": "passed",
        "expected_model_id_count": len(expected_ids),
        "observed_row_count": len(rows),
        "source_counts": source_counts,
        "missing_model_ids": sorted(expected - observed),
        "extra_model_ids": sorted(observed - expected),
        "duplicate_model_ids": duplicate_ids,
        "duplicate_source_paths": duplicate_paths,
        "missing_source_paths": missing_sources,
        "source_count_mismatches": count_mismatches,
        "obs_model_id_discrepancies": [],
    }
    if any(
        report[key]
        for key in (
            "missing_model_ids",
            "extra_model_ids",
            "duplicate_model_ids",
            "duplicate_source_paths",
            "missing_source_paths",
            "source_count_mismatches",
        )
    ):
        report["status"] = "failed"
    return report


def build_registry(
    split_path: Path,
    kinker_umi_dir: Path,
    raw_umi_27_dir: Path,
    original_registry_path: Path,
    output_path: Path,
    *,
    dry_run: bool = False,
    verify_obs_model_id: bool = False,
    reader: Callable[[Path], Any] | None = None,
) -> dict[str, object]:
    """Validate all sources and atomically write the exact 226-line registry."""
    split = load_exp13_split(split_path)
    rows_by_source = {
        "kinker_umi_152": _scan_h5ad_dir(kinker_umi_dir, "kinker_umi_152"),
        "raw_umi_27": _scan_h5ad_dir(raw_umi_27_dir, "raw_umi_27"),
        "original_47": _load_original_registry(original_registry_path),
    }
    report = _audit(rows_by_source, split.all_model_ids)
    rows = [row for source_rows in rows_by_source.values() for row in source_rows]
    if verify_obs_model_id and report["status"] == "passed":
        if reader is None:
            import anndata as ad

            def reader(path: Path) -> Any:
                return ad.read_h5ad(path, backed="r")

        discrepancies = _verify_obs_model_id(rows, reader)
        report["obs_model_id_discrepancies"] = discrepancies
        if discrepancies:
            report["status"] = "failed"
    if dry_run:
        return report
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if report["status"] != "passed":
        details = json.dumps(report, sort_keys=True)
        raise ValueError(f"source registry audit failed: {details}")
    by_id = {row["model_id"]: row for row in rows}
    ordered = pd.DataFrame([by_id[model_id] for model_id in split.all_model_ids])
    ordered = ordered.loc[:, REGISTRY_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        ordered.to_csv(temporary, index=False)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report["output_path"] = str(output_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/benchmarks/cell_line_geneeffect_226_split.json"),
    )
    parser.add_argument("--kinker-umi-dir", type=Path, required=True)
    parser.add_argument("--raw-umi-27-dir", type=Path, required=True)
    parser.add_argument("--original-47-registry", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/experiments/13_geneeffect_226/basal_source_registry.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-obs-model-id", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_registry(
        args.split,
        args.kinker_umi_dir,
        args.raw_umi_27_dir,
        args.original_47_registry,
        args.output,
        dry_run=args.dry_run,
        verify_obs_model_id=args.verify_obs_model_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
