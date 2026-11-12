"""De Kegel et al. 2021 (Cell Systems) paralog classifier on the multi-study SL
benchmark: the released prediction score of each paralog pair (Table S8, 36,648 pairs,
from cancergd.org/static/paralogs/supp_tables/). The classifier was trained on DepMap
dependencies of 2020 and no screen labels, so it reads no benchmark line; it is not
refitted. A pair is matched by Entrez ID, named by DepMap 26Q1's symbol for that ID
(Table S8's 2020 symbol when DepMap lacks the ID), in either order. A held-out pair
that is not in the table gets the table's median score (``covered`` false), so these
pairs tie in the middle of the released scores (0.01-0.43; 0.5 would rank them first).

    .venv/bin/python hpc/paralog/dekegel_scores.py --out outputs/sl/dekegel
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
#: SHA-256 of the downloaded Table S8 (2026-11-12).
TABLE_S8_SHA256 = "8925be5cf847b201f986b2326507e6ec8204f174a03dcfd627e8d08879e9ca6f"


def entrez_symbols(path: Path) -> dict[str, str]:
    """Entrez ID -> upper-case symbol from a DepMap ``SYMBOL (Entrez)`` header."""
    header = pd.read_csv(path, nrows=0).columns[1:]
    parts = header.str.extract(r"^(?P<symbol>\S+) \((?P<entrez>\d+)\)$")
    return dict(zip(parts["entrez"], parts["symbol"].str.upper()))


def table_s8(path: Path, symbols: dict[str, str]) -> pd.DataFrame:
    """Columns gene_1, gene_2 (sorted) and the released score."""
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TABLE_S8_SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {TABLE_S8_SHA256}")
    table = pd.read_csv(path, dtype={"A1_entrez": str, "A2_entrez": str})
    a = table["A1_entrez"].map(symbols).fillna(table["A1"].str.upper())
    b = table["A2_entrez"].map(symbols).fillna(table["A2"].str.upper())
    out = pd.DataFrame(
        {
            "gene_1": np.minimum(a, b),
            "gene_2": np.maximum(a, b),
            "score": table["prediction_score"],
        }
    )
    return out.drop_duplicates(["gene_1", "gene_2"])


def held_pairs(labels: Path, split: Path) -> pd.DataFrame:
    sides = json.loads(split.read_text())
    held = set(sides["val"]) | set(sides["test"])
    rows = pd.read_parquet(labels, columns=["model_id", "gene_a", "gene_b", "label"])
    rows = rows[rows["label"].notna() & rows["model_id"].isin(held)]
    return rows[["gene_a", "gene_b"]].drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--table", type=Path, default=ROOT / "data/sl_baselines/dekegel/Table_S8.csv"
    )
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
    parser.add_argument(
        "--gene-effect",
        type=Path,
        default=ROOT / "data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv",
    )
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    table = table_s8(cli.table, entrez_symbols(cli.gene_effect))
    pairs = held_pairs(cli.labels, cli.split)
    key = pd.MultiIndex.from_arrays(
        [
            np.minimum(pairs["gene_a"], pairs["gene_b"]),
            np.maximum(pairs["gene_a"], pairs["gene_b"]),
        ]
    )
    score = table.set_index(["gene_1", "gene_2"])["score"].reindex(key).to_numpy()
    covered = np.isfinite(score)
    fill = float(table["score"].median())
    out = pairs.assign(score=np.where(covered, score, fill), covered=covered)
    cli.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cli.out / "scores.parquet", index=False)
    metrics = {
        "table_pairs": len(table),
        "uncovered_score": fill,
        "held_pairs": len(out),
        "held_pairs_covered": int(covered.sum()),
        "held_pairs_covered_share": float(covered.mean()),
    }
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
