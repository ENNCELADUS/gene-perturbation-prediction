"""Build K562-mappable SL_benchmark splits from cached Rand 1:1 folds."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


GENE_COLUMN_RE = re.compile(r"^(?P<symbol>.+?) \((?P<entrez>\d+)\)$")


@dataclass(frozen=True)
class DepMapGene:
    """K562 DepMap GeneEffect value for one gene symbol."""

    symbol: str
    entrez_id: str
    gene_effect: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter SL_benchmark CV1/CV2/CV3 Rand 1:1 split caches to pairs "
            "whose two genes have numeric K562 DepMap GeneEffect values."
        )
    )
    parser.add_argument(
        "--sl-root",
        type=Path,
        default=Path("data/SL_benchmark"),
        help="Local SL_benchmark checkout root.",
    )
    parser.add_argument(
        "--depmap-root",
        type=Path,
        default=Path("data/sl_dependency_v0/raw/depmap"),
        help="Directory containing CRISPRGeneEffect.csv and Model.csv.",
    )
    parser.add_argument("--model-id", default="ACH-000551", help="DepMap model id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/SL_benchmark/derived/k562_depmap_rand_1to1"),
        help="Output directory for filtered split tables.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="Deterministic seed for post-filter 1:1 balancing.",
    )
    return parser.parse_args()


def _load_k562_gene_effects(depmap_root: Path, model_id: str) -> dict[str, DepMapGene]:
    gene_effect = pd.read_csv(depmap_root / "CRISPRGeneEffect.csv", index_col=0)
    if model_id not in gene_effect.index:
        msg = f"{model_id} not found in {depmap_root / 'CRISPRGeneEffect.csv'}"
        raise ValueError(msg)
    row = pd.to_numeric(gene_effect.loc[model_id], errors="coerce")
    genes: dict[str, DepMapGene] = {}
    for column, value in row.dropna().items():
        match = GENE_COLUMN_RE.match(str(column))
        if match is None:
            continue
        symbol = match.group("symbol").upper()
        genes[symbol] = DepMapGene(
            symbol=symbol,
            entrez_id=match.group("entrez"),
            gene_effect=float(value),
        )
    return genes


def _load_gene_entities(sl_root: Path) -> dict[int, str]:
    entities = pd.read_csv(sl_root / "data" / "fin_entities.csv")
    genes = entities[entities["entity_type_name"] == "Gene"].copy()
    genes["unified_id"] = genes["unified_id"].astype(int)
    genes["symbol"] = genes["entity_name"].astype(str).str.upper()
    return dict(zip(genes["unified_id"], genes["symbol"], strict=False))


def _pair_index_table(pair_index: np.ndarray | sparse.spmatrix) -> pd.DataFrame:
    if sparse.isspmatrix(pair_index):
        coo = pair_index.tocoo()
        row = coo.row.astype(int)
        col = coo.col.astype(int)
    else:
        pair_array = np.asarray(pair_index)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            msg = f"Expected pair index with shape (n, 2), got {pair_array.shape}"
            raise ValueError(msg)
        row = pair_array[:, 0].astype(int)
        col = pair_array[:, 1].astype(int)
    pairs = pd.DataFrame(
        {
            "gene_a_unified_id": row,
            "gene_b_unified_id": col,
        }
    )
    return pairs[pairs["gene_a_unified_id"] != pairs["gene_b_unified_id"]]


def _annotate_and_filter_pairs(
    pairs: pd.DataFrame,
    id_to_symbol: dict[int, str],
    k562_genes: dict[str, DepMapGene],
) -> pd.DataFrame:
    annotated = pairs.copy()
    annotated["gene_a_symbol"] = annotated["gene_a_unified_id"].map(id_to_symbol)
    annotated["gene_b_symbol"] = annotated["gene_b_unified_id"].map(id_to_symbol)
    has_symbols = (
        annotated["gene_a_symbol"].notna() & annotated["gene_b_symbol"].notna()
    )
    annotated = annotated.loc[has_symbols].copy()
    in_k562 = annotated["gene_a_symbol"].isin(k562_genes) & annotated[
        "gene_b_symbol"
    ].isin(k562_genes)
    annotated = annotated.loc[in_k562].copy()
    annotated["gene_a_entrez_id"] = annotated["gene_a_symbol"].map(
        lambda symbol: k562_genes[symbol].entrez_id
    )
    annotated["gene_b_entrez_id"] = annotated["gene_b_symbol"].map(
        lambda symbol: k562_genes[symbol].entrez_id
    )
    annotated["gene_a_k562_gene_effect"] = annotated["gene_a_symbol"].map(
        lambda symbol: k562_genes[symbol].gene_effect
    )
    annotated["gene_b_k562_gene_effect"] = annotated["gene_b_symbol"].map(
        lambda symbol: k562_genes[symbol].gene_effect
    )
    return annotated


def _split_table(
    samples: np.ndarray,
    split_type: str,
    label_name: str,
    label_value: int,
    id_to_symbol: dict[int, str],
    k562_genes: dict[str, DepMapGene],
) -> pd.DataFrame:
    split_roles = {"train": 2, "test": 3}
    frames = []
    for split_role, matrix_index in split_roles.items():
        for fold_id, matrix in enumerate(samples[matrix_index]):
            pairs = _pair_index_table(matrix)
            filtered = _annotate_and_filter_pairs(pairs, id_to_symbol, k562_genes)
            filtered.insert(0, "fold_id", fold_id)
            filtered.insert(0, "split_role", split_role)
            frames.append(filtered)
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "sl_label", label_value)
    table.insert(0, "label_name", label_name)
    table.insert(0, "positive_negative_ratio", "1:1")
    table.insert(0, "negative_sampling_method", "Rand")
    table.insert(0, "split_type", split_type)
    table["pair_id"] = (
        table["split_type"]
        + "_fold"
        + table["fold_id"].astype(str)
        + "_"
        + table["split_role"]
        + "_"
        + table["label_name"]
        + "_"
        + table["gene_a_unified_id"].astype(str)
        + "_"
        + table["gene_b_unified_id"].astype(str)
    )
    return table


def _build_split(
    sl_root: Path,
    split_type: str,
    id_to_symbol: dict[int, str],
    k562_genes: dict[str, DepMapGene],
) -> pd.DataFrame:
    split_path = sl_root / "data" / "data_split" / f"{split_type}_1.npy"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    pos_samples, neg_samples = np.load(split_path, allow_pickle=True)
    pos_table = _split_table(
        pos_samples, split_type, "positive", 1, id_to_symbol, k562_genes
    )
    neg_table = _split_table(
        neg_samples, split_type, "negative", 0, id_to_symbol, k562_genes
    )
    return pd.concat([pos_table, neg_table], ignore_index=True)


def _summarize(table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        table.groupby(["split_type", "fold_id", "split_role", "label_name"])
        .agg(
            n_pairs=("pair_id", "size"),
            n_gene_a=("gene_a_symbol", "nunique"),
            n_gene_b=("gene_b_symbol", "nunique"),
            n_any_gene=(
                "gene_a_symbol",
                lambda values: len(
                    set(values) | set(table.loc[values.index, "gene_b_symbol"])
                ),
            ),
        )
        .reset_index()
    )
    return summary.sort_values(["split_type", "fold_id", "split_role", "label_name"])


def _balance_one_to_one(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = ["split_type", "fold_id", "split_role"]
    frames = []
    for _, group in table.groupby(groups, sort=False):
        positive = group[group["sl_label"] == 1]
        negative = group[group["sl_label"] == 0]
        n_keep = min(len(positive), len(negative))
        frames.append(positive.sample(n=n_keep, random_state=seed))
        frames.append(negative.sample(n=n_keep, random_state=seed))
    balanced = pd.concat(frames, ignore_index=True)
    return balanced.sort_values(
        ["split_type", "fold_id", "split_role", "sl_label", "pair_id"],
        ascending=[True, True, True, False, True],
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k562_genes = _load_k562_gene_effects(args.depmap_root, args.model_id)
    id_to_symbol = _load_gene_entities(args.sl_root)
    split_tables = []
    for split_type in ["CV1", "CV2", "CV3"]:
        table = _build_split(args.sl_root, split_type, id_to_symbol, k562_genes)
        split_tables.append(table)
        split_path = output_dir / f"{split_type}_Rand_1to1_k562_depmap_pairs.csv"
        table.to_csv(split_path, index=False)

    all_pairs = pd.concat(split_tables, ignore_index=True)
    all_pairs.to_csv(output_dir / "all_CV_Rand_1to1_k562_depmap_pairs.csv", index=False)
    summary = _summarize(all_pairs)
    summary.to_csv(output_dir / "summary.csv", index=False)

    balanced = _balance_one_to_one(all_pairs, args.balance_seed)
    balanced.to_csv(
        output_dir / "all_CV_Rand_1to1_k562_depmap_pairs_balanced.csv",
        index=False,
    )
    for split_type, split_table in balanced.groupby("split_type", sort=False):
        split_table.to_csv(
            output_dir / f"{split_type}_Rand_1to1_k562_depmap_pairs_balanced.csv",
            index=False,
        )
    _summarize(balanced).to_csv(output_dir / "summary_balanced.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "model_id": args.model_id,
                "n_k562_depmap_genes": len(k562_genes),
                "n_sl_gene_entities": len(id_to_symbol),
                "n_sl_genes_with_k562_depmap": len(
                    set(id_to_symbol.values()) & set(k562_genes)
                ),
                "output_dir": str(output_dir),
            }
        ]
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
