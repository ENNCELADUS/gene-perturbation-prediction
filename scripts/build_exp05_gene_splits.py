"""Build and freeze the canonical exp05 five-fold outer manifest."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from aivc_model.gene_splits import build_canonical_outer_manifest, sha256_file

DEFAULT_LABELS_CSV = Path("data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv")
DEFAULT_MANIFEST_CSV = Path(
    "data/sl_dependency_v0/splits/k562_gwps_depmap_outer5_seed42.csv"
)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_canonical_outer_manifest(
    labels_csv: Path,
    output_csv: Path,
    n_splits: int,
    seed: int,
) -> str:
    """Write an immutable canonical manifest and its SHA-256 sidecar."""
    labels = pd.read_csv(labels_csv)
    manifest = build_canonical_outer_manifest(labels, n_splits=n_splits, seed=seed)
    content = manifest.to_csv(index=False).encode("utf-8")
    if output_csv.exists() and output_csv.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite non-identical {output_csv}")
    if not output_csv.exists():
        _atomic_write(output_csv, content)
    digest = sha256_file(output_csv)
    _atomic_write(Path(f"{output_csv}.sha256"), f"{digest}\n".encode("ascii"))
    return digest


def main() -> None:
    """Build the fixed exp05 outer-five-fold manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_canonical_outer_manifest(
        args.labels_csv, args.out, n_splits=args.n_splits, seed=args.seed
    )


if __name__ == "__main__":
    main()
