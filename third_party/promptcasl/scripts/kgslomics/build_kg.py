"""Build the PrimeKG tensors the KG-SLomics baseline runs on.

Writes cache/primekg.pt (edge_index, edge_type, num_nodes, rel_to_id) and
cache/kg_emb128.npy (ConvE entity embeddings, row order = entity id).

    python build_kg.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from common import CACHE, CONVE_128, TRIPLES_TSV, load_entity_to_id

CHUNK = 2_000_000


def main():
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, "primekg.pt")
    if os.path.exists(out):
        print(f"{out} exists, nothing to do")
        return

    ent = load_entity_to_id()
    print(f"entity_to_id: {len(ent):,} entities")

    src_parts, dst_parts, rel_parts = [], [], []
    rel_to_id, n_unmapped, n_total = {}, 0, 0
    reader = pd.read_csv(TRIPLES_TSV, sep="\t", header=None, chunksize=CHUNK,
                         names=["h", "r", "t"], dtype=str, na_filter=False)
    for i, chunk in enumerate(reader):
        n_total += len(chunk)
        h = chunk["h"].map(ent)
        t = chunk["t"].map(ent)
        keep = h.notna() & t.notna()
        n_unmapped += int((~keep).sum())
        chunk, h, t = chunk[keep], h[keep], t[keep]
        for r in chunk["r"].unique():
            rel_to_id.setdefault(r, len(rel_to_id))
        src_parts.append(h.to_numpy(dtype=np.int32))
        dst_parts.append(t.to_numpy(dtype=np.int32))
        rel_parts.append(chunk["r"].map(rel_to_id).to_numpy(dtype=np.int16))
        print(f"  chunk {i}: {n_total:,} rows read, {len(rel_to_id)} relations")

    src = torch.from_numpy(np.concatenate(src_parts)).long()
    dst = torch.from_numpy(np.concatenate(dst_parts)).long()
    rel = torch.from_numpy(np.concatenate(rel_parts)).long()
    edge_index = torch.stack([src, dst])

    emb = np.load(CONVE_128)
    if emb.shape[0] != len(ent):
        sys.exit(f"ConvE rows {emb.shape[0]} != entities {len(ent)}")

    torch.save({"edge_index": edge_index, "edge_type": rel,
                "num_nodes": len(ent), "rel_to_id": rel_to_id}, out)
    np.save(os.path.join(CACHE, "kg_emb128.npy"), emb.astype(np.float32))

    deg = torch.bincount(edge_index.reshape(-1), minlength=len(ent))
    print(f"\nnodes      {len(ent):,}")
    print(f"edges      {edge_index.shape[1]:,}  (dropped {n_unmapped:,} unmapped)")
    print(f"relations  {len(rel_to_id)}")
    print(f"degree     mean {deg.float().mean():.1f}  max {deg.max()}  "
          f"isolated {(deg == 0).sum()}")
    print(f"kg_emb     {emb.shape} {emb.dtype}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
