"""An X-Atlas-shaped Perturb-seq shard and the naive per-gene capped cell selection
the streaming reader must reproduce."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.fixtures.prepare import SOURCE_GENES

#: The world's per-perturbation response cap.
CAPPED = {"response_max_cells_per_gene": 80}
RESPONSE_GENES = tuple(f"P{i:02d}" for i in range(40))


LABEL_VARIANTS = {"kif11": 7, "KIF11": 9, " Tp53 ": 5, "TP53": 4}


def response_labels(rng: np.random.Generator, control: str) -> list[str]:
    labels = [control] * 2500
    for gene in RESPONSE_GENES:
        labels += [gene] * int(rng.integers(20, 200))
    for label, count in LABEL_VARIANTS.items():
        labels += [label] * count
    return [labels[i] for i in rng.permutation(len(labels))]


def write_xatlas(root: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    symbols = [*SOURCE_GENES[1:], "H3"]  # no H1; two tokens named H3
    tokens = [100 + 7 * i for i in range(len(symbols))]
    metadata = root / "xatlas_genes.parquet"
    pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{8000 + i:011d}" for i in range(len(symbols))],
            "gene_name": symbols,
            "gene_token_id": tokens,
        }
    ).to_parquet(metadata)
    vocabulary = np.asarray([*tokens, 5555, 6666])  # two tokens absent from metadata
    labels = response_labels(rng, "Non-Targeting")
    shards = root / "xatlas"
    shards.mkdir()
    for shard, rows in enumerate(np.array_split(np.arange(len(labels)), 5)):
        records = []
        for row in rows:
            n_tokens = int(rng.integers(3, len(vocabulary)))
            cell_tokens = rng.choice(vocabulary, size=n_tokens, replace=False)
            if row % 3 == 0:
                cell_tokens = np.append(cell_tokens, cell_tokens[0])  # duplicate token
            records.append(
                {
                    "gene_token_id": cell_tokens.astype(np.int64),
                    "gene_expression": rng.integers(0, 6, size=len(cell_tokens)).astype(
                        np.float32
                    ),
                    "cell_barcode": f"bc{row}",
                    "sample": f"s{shard}",
                    "gene_target": labels[row],
                    "pass_guide_filter": int(rng.random() < 0.8),
                }
            )
        pd.DataFrame(records).to_parquet(shards / f"HCT116_Batch{shard}.parquet")
    pd.DataFrame(records[:5]).to_parquet(shards / "Other_Batch0.parquet")
    return {
        "source_type": "xatlas_orion_parquet",
        "shard_dir": str(shards),
        "shard_glob": "HCT116_Batch*.parquet",
        "gene_metadata_path": str(metadata),
        "control_label": "Non-Targeting",
        "pass_guide_filter_value": 1,
        "target_gene_symbol_col": "gene_name",
    }


def naive_xatlas_cells(source: dict, hvg: tuple[str, ...], caps: dict, seed: int):
    """``(labels, library sizes, dense HVG counts)`` of X-Atlas-Orion shards."""
    cap = caps["response_max_cells_per_gene"]
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list] = {}
    seen: dict[str, int] = {}
    for path in sorted(Path(source["shard_dir"]).glob(source["shard_glob"])):
        frame = pd.read_parquet(path)
        frame = frame[frame["gene_target"].astype(str) != source["control_label"]]
        frame = frame[
            frame["pass_guide_filter"].astype(int) == source["pass_guide_filter_value"]
        ]
        for row in frame.itertuples(index=False):
            gene = str(row.gene_target)
            cell = (np.asarray(row.gene_token_id), np.asarray(row.gene_expression))
            bucket = reservoirs.setdefault(gene, [])
            seen[gene] = seen.get(gene, 0) + 1
            if len(bucket) < cap:
                bucket.append(cell)
                continue
            replacement = int(rng.integers(0, seen[gene]))
            if replacement < cap:
                bucket[replacement] = cell
    cells = [(gene, cell) for gene in sorted(reservoirs) for cell in reservoirs[gene]]
    metadata = pd.read_parquet(source["gene_metadata_path"])
    symbol = dict(zip(metadata["gene_token_id"], metadata["gene_name"].astype(str)))
    labels, sizes, aligned = [], [], []
    for gene, (tokens, values) in cells:
        hvg_counts = np.zeros(len(hvg))
        size = 0.0
        for token, value in zip(tokens.tolist(), values.tolist(), strict=True):
            if value > 0 and token in symbol:
                size += value
                if symbol[token] in hvg:
                    hvg_counts[hvg.index(symbol[token])] += value
        labels.append(gene)
        sizes.append(size)
        aligned.append(hvg_counts)
    return np.asarray(labels), np.asarray(sizes), np.asarray(aligned)
