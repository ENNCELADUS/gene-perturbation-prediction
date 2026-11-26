"""Inputs of the official ESM4SL code (third_party/esm4sl) from the SL benchmark: the
protein sequence of each benchmark gene, the id table, and the three pair files with
the official columns 0, 1, 2 (gene id, gene id, label).

A gene symbol takes the longest reviewed human UniProt sequence, cut to 2,000 residues
(the model reads 2,000 at most); a gene without one is not covered, and a pair with
such a gene is left out of every file (its row ties at the median in the score table).
Gene id: the position of the symbol among the sorted covered symbols. Train and
validation files: covered pairs, balanced by drawing non-SL pairs (seed 0). Test file:
every covered pair of the training, validation and test lines; its label column is not
used for any choice.

    .venv-esm4sl/bin/python hpc/esm4sl/prepare.py --out data/esm4sl
"""

import argparse
import hashlib
import json
import pickle
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
import sl_common as common  # noqa: E402

UNIPROT = (
    "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true"
    "&format=tsv&fields=accession,gene_primary,sequence"
)
SYMBOL = "Gene Names (primary)"


def gene_sequences(
    table: pd.DataFrame, symbols: pd.Index, limit: int = 2000
) -> pd.Series:
    """Symbol to the first ``limit`` residues of its longest sequence."""
    table = table.dropna(subset=[SYMBOL, "Sequence"])
    longest = table.loc[table["Sequence"].str.len().sort_values(ascending=False).index]
    best = longest.drop_duplicates(SYMBOL).set_index(SYMBOL)["Sequence"].str[:limit]
    return best[best.index.isin(symbols)].rename("sequence")


def pair_file(pairs: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Official columns "0", "1", "2"; a pair with a gene outside ``ids`` is dropped."""
    first, second = pairs["gene_1"].map(ids), pairs["gene_2"].map(ids)
    keep = first.notna() & second.notna()
    return pd.DataFrame(
        {
            "0": first[keep].astype(int),
            "1": second[keep].astype(int),
            "2": pairs["label"][keep].astype(int),
        }
    ).reset_index(drop=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/sl_studies/benchmark_v2/labels.parquet",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "configs/benchmarks/cell_line_sl_split.json",
    )
    args = parser.parse_args()
    out = args.out
    (out / "mapping").mkdir(parents=True, exist_ok=True)
    (out / "benchmark").mkdir(exist_ok=True)

    table_path = out / "uniprot_human_reviewed.tsv"
    if not table_path.exists():
        urllib.request.urlretrieve(UNIPROT, table_path)
    table = pd.read_csv(table_path, sep="\t")

    rows = common.benchmark_rows(args.labels, args.split)
    symbols = pd.Index(sorted(set(rows["gene_1"]) | set(rows["gene_2"])))
    seqs = gene_sequences(table, symbols).sort_index()
    ids = pd.Series(np.arange(len(seqs)), index=seqs.index)
    with open(out / "mapping/all_id_seq.pkl", "wb") as f:
        pickle.dump([(int(i), s) for i, s in zip(ids, seqs)], f)
    pd.DataFrame({"id": ids.to_numpy(), "symbol": ids.index}).to_csv(
        out / "genes.csv", index=False
    )

    def covered(pairs: pd.DataFrame) -> pd.DataFrame:
        return pairs[pairs["gene_1"].isin(ids.index) & pairs["gene_2"].isin(ids.index)]

    def drawn(sides: tuple[str, ...]) -> pd.DataFrame:
        pairs = covered(common.pair_labels(rows, sides))
        pairs = common.balanced(pairs, np.random.default_rng(0))
        return pairs.sample(frac=1, random_state=0)

    files = {
        "sl_train.csv": drawn(common.FIT_SIDES),
        "sl_val.csv": drawn(("val",)),
    }
    scored = rows[rows["side"].isin(common.SCORED_SIDES)]
    test = covered(scored[["gene_1", "gene_2"]].drop_duplicates())
    label = common.pair_labels(rows, common.SCORED_SIDES)
    files["sl_test.csv"] = test.merge(label, how="left").fillna({"label": 0})
    counts = {}
    for name, pairs in files.items():
        frame = pair_file(pairs, ids)
        frame.to_csv(out / "benchmark" / name, index=False)
        counts[name] = {"rows": len(frame), "sl": int(frame["2"].sum())}

    held = rows[rows["side"].isin(common.HELD_SIDES)]
    metrics = {
        "uniprot_sha256": sha256(table_path),
        "uniprot_retrieved": date.fromtimestamp(table_path.stat().st_mtime).isoformat(),
        "sequence_limit": 2000,
        "benchmark_genes": len(symbols),
        "genes_with_sequence": len(ids),
        "held_rows_both_genes_covered": float(
            (held["gene_1"].isin(ids.index) & held["gene_2"].isin(ids.index)).mean()
        ),
        "files": counts,
        "labels_sha256": sha256(args.labels),
        "split_sha256": sha256(args.split),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
