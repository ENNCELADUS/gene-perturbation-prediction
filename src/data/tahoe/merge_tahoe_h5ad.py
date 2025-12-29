#!/usr/bin/env python
"""Merge Tahoe H5AD shards into a single file on disk.

Requires each input AnnData to share the same var/obs schema and layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anndata.experimental import concat_on_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Tahoe *.h5ad files into one tahoe.h5ad using on-disk concat.",
    )
    parser.add_argument(
        "--input-glob",
        default="data/tahoe/*.h5ad",
        help="Glob for input H5AD shards.",
    )
    parser.add_argument(
        "--output",
        default="data/tahoe/tahoe.h5ad",
        help="Output H5AD path.",
    )
    parser.add_argument(
        "--max-loaded-elems",
        type=int,
        default=20_000_000,
        help="Max elements loaded into memory at once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(Path().glob(args.input_glob))
    if not input_paths:
        raise SystemExit(f"No input files found for glob: {args.input_glob}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concat_on_disk(
        [str(p) for p in input_paths],
        str(output_path),
        axis=0,
        join="inner",
        merge="same",
        uns_merge="same",
        max_loaded_elems=args.max_loaded_elems,
    )


if __name__ == "__main__":
    main()
