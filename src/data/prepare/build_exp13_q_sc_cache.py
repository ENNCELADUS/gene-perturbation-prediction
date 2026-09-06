"""Prepare raw-UMI q_sc shards against one explicit common gene panel."""

import argparse
from pathlib import Path


def build_q_sc_cache(*, split_path, registry_path, panel_path, output_dir, reader=None):
    import pandas as pd
    from src.data.geneeffect import load_exp13_split, load_source_registry
    from src.data.q_sc import build_q_sc_shards

    split = load_exp13_split(Path(split_path))
    registry = load_source_registry(Path(registry_path), split)
    panel = pd.read_csv(panel_path)
    if list(panel.columns) != ["gene_symbol"]:
        raise ValueError("common panel must have exactly the gene_symbol column")
    return build_q_sc_shards(
        registry, Path(output_dir), tuple(panel.gene_symbol), reader=reader, resume=True
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_q_sc_cache(
        split_path=args.split,
        registry_path=args.registry,
        panel_path=args.panel,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
