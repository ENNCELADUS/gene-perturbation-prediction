"""CellNet embeddings of every cell of a split's lines, as a cell-embedding cache
for the gene-knowledge, cell-set model (SL protocol §18): ``<out>/<ModelID>/
embeddings.npy`` (cells x 768, float32) in the order of the line's raw-UMI h5ad, the
cells and order of the Tx1 cache, and ``report.json`` (genes mapped to CellNet's
layout). CellNet (arXiv:2610.21943, github.com/daydayup-xq/CellNet, Apache-2.0) is
frozen: its converter normalises the raw counts to 10,000, takes log1p and draws each
cell as an image; the embedding is its native readout. Genes are matched by Ensembl
ID without its version suffix; columns with the same ID are summed first.

Run from the repo root with the project environment, CellNet's clone and ``timm`` on
``PYTHONPATH`` (hpc/README.md), one process for each GPU:

    hpc/cellnet/embed_lines.py --release data/models/cellnet/CellNet \
        --split configs/benchmarks/cell_line_sl_split.json \
        --raw data/sl_dependency_v0/processed/basal_raw_umi \
        --out data/cellnet_basal_embeddings/v1 --shard 0/4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

WIDTH = 768


def combine_duplicates(x: sparse.spmatrix, ids: np.ndarray):
    """Sum the columns that share an ID: (cells x unique IDs, the unique IDs)."""
    codes, unique = pd.factorize(ids)
    if len(unique) == len(ids):
        return sparse.csr_matrix(x), list(unique)
    merge = sparse.csr_matrix(
        (np.ones(len(ids), np.float32), (np.arange(len(ids)), codes)),
        shape=(len(ids), len(unique)),
    )
    return sparse.csr_matrix(x @ merge), list(unique)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ensembl-column", default="ensembl_id")
    args = parser.parse_args()
    from cellnet import CellNet, CellNetImageConverter

    from src.data.splits import load_geneeffect_split
    from src.data.tx1_cache import read_basal_source

    index, count = (int(part) for part in args.shard.split("/"))
    lines = sorted(load_geneeffect_split(args.split).all_model_ids)[index::count]
    model = CellNet.from_pretrained(args.release, args.device)
    converter = CellNetImageConverter(args.release, "Human")
    for model_id in lines:
        target = args.out / model_id / "embeddings.npy"
        if target.is_file():
            continue
        source = read_basal_source(
            args.raw / f"{model_id}.h5ad",
            model_id=model_id,
            var_ensembl_col=args.ensembl_column,
        )
        # Ensembl IDs without their version suffix (some deposits list ENSG...5)
        ids = source.var_names.str.split(".").str[0].to_numpy()
        x, genes = combine_duplicates(source.X, ids)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name("embeddings.npy.partial")
        out = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(x.shape[0], WIDTH)
        )
        start = time.monotonic()
        with torch.inference_mode():
            for first in range(0, x.shape[0], args.batch_size):
                batch = converter.transform(
                    x[first : first + args.batch_size],
                    genes,
                    input_scale="counts",
                    allow_unknown=True,
                )
                encoded = model.encode(
                    **{key: value.to(args.device) for key, value in batch.items()}
                )
                out[first : first + args.batch_size] = (
                    encoded["cls_deep"][0].float().cpu().numpy()
                )
        if not np.isfinite(out).all():
            raise ValueError(f"{model_id}: a CellNet embedding is not finite")
        out.flush()
        del out
        report = converter.last_report
        (target.parent / "report.json").write_text(
            json.dumps(
                {"cells": x.shape[0], "input_genes": len(genes),
                 "mapped_genes": report["mapped_genes"],
                 "unknown_genes": len(report["unknown_genes"])},
                indent=2,
            )
            + "\n"
        )  # fmt: skip
        partial.replace(target)
        seconds = time.monotonic() - start
        print(
            f"{model_id}: {x.shape[0]} cells, {report['mapped_genes']} of "
            f"{len(genes)} genes mapped, {x.shape[0] / seconds:.0f} cells/s",
            flush=True,
        )
    print("CELLNET_SHARD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
