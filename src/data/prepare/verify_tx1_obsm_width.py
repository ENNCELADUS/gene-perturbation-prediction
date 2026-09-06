"""data / prepare / verify tx1 obsm width."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from src.model.tx1 import load_local_safetensors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tiny_dmso_adata(
    shard_dir: Path, gene_metadata_path: Path, cells: int
) -> ad.AnnData:
    selected: list[tuple[np.ndarray, np.ndarray, str]] = []
    for path in sorted(shard_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["genes", "expressions", "cell_line_id"])
        for row in frame.itertuples(index=False):
            genes = np.asarray(row.genes, dtype=np.int32)
            values = np.asarray(row.expressions, dtype=np.float32)
            valid = (genes >= 3) & (values > 0)
            selected.append((genes[valid], values[valid], str(row.cell_line_id)))
            if len(selected) == cells:
                break
        if len(selected) == cells:
            break
    if len(selected) != cells:
        raise ValueError(f"Requested {cells} DMSO cells, found {len(selected)}")

    metadata = pd.read_parquet(gene_metadata_path).set_index("token_id")
    tokens = sorted(set(np.concatenate([item[0] for item in selected]).tolist()))
    tokens = [token for token in tokens if token in metadata.index]
    positions = {token: index for index, token in enumerate(tokens)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, (genes, counts, _) in enumerate(selected):
        for token, count in zip(genes, counts):
            if int(token) in positions:
                rows.append(row_index)
                columns.append(positions[int(token)])
                values.append(float(count))
    matrix = csr_matrix((values, (rows, columns)), shape=(cells, len(tokens)))
    obs = pd.DataFrame(
        {"cell_type": [item[2] for item in selected]},
        index=[f"width_probe_{index}" for index in range(cells)],
    )
    var = metadata.loc[tokens, ["ensembl_id", "gene_symbol"]].copy()
    var.index = var["ensembl_id"].astype(str)
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    if np.any(result.X.data < 0):
        raise ValueError("Width probe must contain raw non-negative counts")
    return result


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the Tahoe-x1 3B width probe")
    adata = tiny_dmso_adata(args.dmso_shards, args.gene_metadata, args.cells)
    model, vocab, collator_config, load_report = load_local_safetensors(args.model_dir)
    genes = adata.var["ensembl_id"].astype(str).tolist()
    gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
    loader = loader_from_adata(
        adata=adata,
        collator_cfg=collator_config,
        vocab=vocab,
        batch_size=args.cells,
        max_length=args.max_length,
        gene_ids=gene_ids,
        num_workers=0,
        prefetch_factor=None,
    )
    trainer = Trainer(model=model, device="gpu")
    predictions = trainer.predict(loader, return_outputs=True)
    embeddings = torch.cat(
        [output["cell_emb"].detach().float().cpu() for output in predictions], dim=0
    ).numpy()
    adata.obsm[args.obsm_key] = embeddings
    actual_width = int(adata.obsm[args.obsm_key].shape[1])
    expected_width = 2560
    if actual_width != expected_width:
        raise ValueError(f"Expected width {expected_width}, observed {actual_width}")
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)
    result = {
        "status": "verified",
        "model_label": "tahoe_x1_3b",
        "rejected_label": "tx-70m-merged",
        "obsm_key": args.obsm_key,
        "obsm_shape": list(adata.obsm[args.obsm_key].shape),
        "actual_obsm_width": actual_width,
        "expected_obsm_width": expected_width,
        "input_contract": {
            "X": "raw non-negative counts",
            "var_key": "ensembl_id",
            "obs_key": "cell_type",
            "cells": args.cells,
            "genes": int(adata.n_vars),
        },
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "attention_implementation": (
            "torch portability fallback for the width probe; weights and output "
            "shape are unchanged"
        ),
        "padding_metadata_implementation": (
            "pure-Torch equivalent of flash_attn.bert_padding.unpad_input"
        ),
        "torch_version": torch.__version__,
        "checkpoint_load": load_report,
        "source_sha256": {
            "model_safetensors": sha256_file(args.model_dir / "model.safetensors"),
            "model_config": sha256_file(args.model_dir / "model_config.yml"),
            "collator_config": sha256_file(args.model_dir / "collator_config.yml"),
            "vocab": sha256_file(args.model_dir / "vocab.json"),
            "probe_h5ad": sha256_file(args.output_h5ad),
        },
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dmso-shards", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--obsm-key", default="tahoe_x1_3b")
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
