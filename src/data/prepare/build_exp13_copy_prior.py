#!/usr/bin/env python3
"""Materialize the authenticated K562 copy-prior baseline for Exp13."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.geneeffect import (
    PINNED_COPY_PRIOR_SHA256,
    PINNED_SPLIT_SHA256,
    load_exp13_split,
    parse_gene_symbol,
)


SCHEMA_VERSION = "exp13-copy-prior-v1"
DONOR_MODEL_ID = "ACH-000551"
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
    return hashlib.sha256(
        "".join(f"{symbol}\n" for symbol in symbols).encode()
    ).hexdigest()


def materialize_copy_prior(
    gene_effect_path: Path,
    split_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Fresh-write the pinned K562 GeneEffect row and its provenance manifest."""
    gene_effect_path = Path(gene_effect_path)
    split_path = Path(split_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("copy-prior CSV and manifest paths must be distinct")
    source_sha256 = _sha256(gene_effect_path)
    if source_sha256 != PINNED_GENE_EFFECT_SHA256:
        raise ValueError(
            "DepMap GeneEffect SHA-256 mismatch: "
            f"{source_sha256} != {PINNED_GENE_EFFECT_SHA256}"
        )
    split = load_exp13_split(split_path)
    split_sha256 = _sha256(split_path)
    if split_sha256 != PINNED_SPLIT_SHA256:
        raise ValueError("Exp13 split SHA-256 mismatch")
    if DONOR_MODEL_ID not in split.train:
        raise ValueError(f"copy-prior donor {DONOR_MODEL_ID} is not in train")
    if DONOR_MODEL_ID in split.unlabeled_train:
        raise ValueError(f"copy-prior donor {DONOR_MODEL_ID} is unlabeled")
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {manifest_path}"
        )

    wide = pd.read_csv(gene_effect_path, index_col=0)
    wide.index = wide.index.astype(str)
    if not wide.index.is_unique:
        raise ValueError("DepMap GeneEffect contains duplicate ModelIDs")
    if DONOR_MODEL_ID not in wide.index:
        raise ValueError(f"copy-prior donor {DONOR_MODEL_ID} is absent from GeneEffect")
    symbols = tuple(parse_gene_symbol(column) for column in wide.columns)
    if len(symbols) != len(set(symbols)):
        raise ValueError("DepMap GeneEffect columns map to duplicate gene symbols")
    raw = wide.loc[DONOR_MODEL_ID]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if invalid.any():
        bad_columns = [str(column) for column in raw.index[invalid][:10]]
        raise ValueError(f"copy-prior donor contains nonnumeric values: {bad_columns}")
    infinite = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
    if infinite.any():
        raise ValueError(
            f"copy-prior donor contains {int(infinite.sum())} infinite values"
        )
    finite = numeric.notna()
    output = pd.DataFrame(
        {
            "gene_symbol": np.asarray(symbols, dtype=object)[finite.to_numpy()],
            "gene_effect": numeric.loc[finite].to_numpy(dtype=float),
        }
    )
    if output.empty:
        raise ValueError("copy-prior donor has no finite GeneEffect values")
    missing_symbols = tuple(np.asarray(symbols, dtype=object)[(~finite).to_numpy()])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    output_installed = False
    try:
        output.to_csv(temporary_output, index=False)
        output_sha256 = _sha256(temporary_output)
        if output_sha256 != PINNED_COPY_PRIOR_SHA256:
            raise ValueError(
                "materialized copy-prior does not match the pinned artifact SHA-256"
            )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "donor": {
                "model_id": DONOR_MODEL_ID,
                "split": "train",
                "unlabeled": False,
            },
            "source": {
                "path": str(gene_effect_path),
                "sha256": source_sha256,
            },
            "split": {
                "path": str(split_path),
                "sha256": split_sha256,
            },
            "output": {
                "path": str(output_path),
                "sha256": output_sha256,
                "gene_symbols_sha256": _symbols_sha256(
                    output["gene_symbol"].astype(str)
                ),
            },
            "counts": {
                "source_gene_count": len(symbols),
                "output_gene_count": len(output),
                "dropped_gene_count": int((~finite).sum()),
            },
            "drop_reason_counts": {
                "missing_gene_effect": int((~finite).sum()),
            },
            "donor_missing": {
                "count": len(missing_symbols),
                "symbols": list(missing_symbols),
                "symbols_sha256": _symbols_sha256(missing_symbols),
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_path.exists():
            if _sha256(output_path) != output_sha256:
                raise FileExistsError(
                    "refusing to recover mismatched output-only artifact: "
                    f"{output_path}"
                )
        else:
            os.replace(temporary_output, output_path)
            output_installed = True
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        if output_installed:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/benchmarks/cell_line_geneeffect_226_split.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_copy_prior(
        args.gene_effect, args.split, args.output, args.manifest
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
