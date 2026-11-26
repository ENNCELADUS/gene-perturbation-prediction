"""Inputs of the MVGCN-iSL harness: the node index, the four gene graphs, the node
features of every line, and the labelled rows (data/mvgcn_isl/benchmark/).

A held-out line (val, test) reads stage one's predicted GeneEffect, never its measured
GeneEffect, in its node features; the co-essentiality graph is computed on the split's
train and bulk_labelled lines (labelled or not), never a val or test line.
Run from the repo root with the vendored code on the path:

    PYTHONPATH=third_party/mvgcn_isl/code MPLBACKEND=Agg \
        .venv-mvgcn/bin/python hpc/mvgcn_isl/prepare.py \
        --out data/mvgcn_isl/benchmark \
        --held-gene-effect outputs/sl/conditional/conditional_seed0/predictions.parquet
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hpc" / "common"))

import sl_common  # noqa: E402

DEPMAP = ROOT / "data/sl_dependency_v0/raw/depmap"
CODE = ROOT / "third_party/mvgcn_isl/code"
#: The official node-feature order (main.py: raw_omics) and graph order (--indep_graph).
FEATURES = ("exp", "mut", "cnv", "ess")
GRAPHS = ("PPI-genetic", "PPI-physical", "co-exp", "co-ess")


def assert_no_held_gene_effect(measured: pd.DataFrame, held_lines) -> None:
    """Fail when a measured GeneEffect profile of a held-out line is in ``measured``."""
    leaked = sorted(measured.index.intersection(pd.Index(held_lines)))
    if leaked:
        raise ValueError(
            f"measured GeneEffect of held-out lines {leaked[:5]} would enter"
        )


def gene_effect_input(
    measured: pd.DataFrame, predicted: pd.DataFrame, held_lines
) -> pd.DataFrame:
    """Measured GeneEffect for every line but the held-out ones, which take the
    prediction; both on the predicted genes. A held-out line without a prediction is
    absent."""
    genes = measured.columns.intersection(predicted.columns)
    held = predicted.index.intersection(pd.Index(held_lines))
    table = pd.concat(
        [
            measured.loc[~measured.index.isin(held_lines), genes],
            predicted.loc[held, genes],
        ]
    )
    assert np.array_equal(
        table.loc[held].to_numpy(),
        predicted.loc[held, genes].to_numpy(),
        equal_nan=True,
    )
    return table


def node_features(
    tables: dict[str, pd.DataFrame], line: str, genes: pd.Index
) -> np.ndarray:
    """Unscaled genes x 4 features of one line in the official order. A gene or a line
    that a table lacks reads 0 (the official zero fill)."""
    columns = []
    for name in FEATURES:
        table = tables[name]
        if line in table.index:
            columns.append(table.loc[line].reindex(genes).fillna(0.0).to_numpy(float))
        else:
            columns.append(np.zeros(len(genes)))
    return np.stack(columns, axis=1)


def line_edges(edges, expressed):
    """The edges (2 x E, numpy or torch) whose two genes are expressed in the line (the
    official filter)."""
    return edges[:, expressed[edges[0]] & expressed[edges[1]]]


def graph_edges(
    edges: pd.DataFrame, rows: pd.DataFrame, index: dict[str, int]
) -> np.ndarray:
    """One graph as a symmetric 2 x E node-index array, without the edges between the
    two genes of a benchmark pair (they can carry the label)."""
    edges = edges[["gene1", "gene2"]].astype(str).apply(lambda s: s.str.upper())
    edges = sl_common.drop_pair_edges(edges, rows)
    pairs = np.stack(
        [edges["gene1"].map(index).to_numpy(), edges["gene2"].map(index).to_numpy()]
    )
    return np.unique(np.concatenate([pairs, pairs[::-1]], axis=1), axis=1)


def pseudo_bulk_cpm(path: Path) -> pd.Series:
    """Counts per million of a line's summed single cells, by upper-case symbol."""
    import anndata

    cells = anndata.read_h5ad(path)
    counts = pd.Series(
        np.asarray(cells.X.sum(axis=0)).ravel(),
        index=cells.var["gene_symbol"].astype(str).str.upper().to_numpy(),
    )
    counts = counts.groupby(level=0).sum()
    return counts / counts.sum() * 1e6


def coessentiality_input(measured: pd.DataFrame, split: dict) -> pd.DataFrame:
    """Measured GeneEffect of every line the split lists in ``train`` or
    ``bulk_labelled`` (labelled or not), all genes. Fails when such a line is also a
    val or test line, or when a val or test profile would be in the result."""
    held = set(split["val"]) | set(split["test"])
    listed = set(split["train"]) | set(split["bulk_labelled"])
    if listed & held:
        raise ValueError(f"split lists {sorted(listed & held)[:5]} as fit and held-out")
    table = measured.loc[measured.index.isin(listed)]
    assert_no_held_gene_effect(table, held)
    return table


def official_graphs(coess_effect: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The four official graphs. The co-essentiality graph is built again by the
    official ``calculate_coexpression`` from ``coess_effect`` (the released file reads
    the held-out lines' GeneEffect); the loaders read ../data from code/."""
    sys.path.insert(0, str(CODE))
    # pandas 2 dropped DataFrame.append, which the official load_graph_data calls
    pd.DataFrame.append = lambda self, other: pd.concat([self, other])
    import utils

    data = CODE.parent / "data"
    (data / "CCLE").mkdir(parents=True, exist_ok=True)
    coess_effect.to_csv(data / "CCLE" / "CCLE_ess.csv")
    here = os.getcwd()
    os.chdir(CODE)
    try:
        utils.calculate_coexpression("ess", 0.2)
        return {name: utils.load_graph_data(name) for name in GRAPHS}
    finally:
        os.chdir(here)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-gene-effect", type=Path, required=True)
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
    parser.add_argument("--depmap", type=Path, default=DEPMAP)
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=ROOT / "data/sl_dependency_v0/processed/basal_raw_umi",
    )
    cli = parser.parse_args()
    # imported here: train.py imports this file without the project's packages
    from src.data.depmap import read_depmap_matrix
    from src.experiments.sl.slidr import read_copy_number

    rows = sl_common.benchmark_rows(cli.labels, cli.split)
    fit_lines = set(rows.loc[rows["side"].isin(sl_common.FIT_SIDES), "model_id"])
    held_lines = sorted(
        rows.loc[rows["side"].isin(sl_common.HELD_SIDES), "model_id"].unique()
    )
    all_lines = sorted(rows["model_id"].unique())

    measured = read_depmap_matrix(cli.depmap / "CRISPRGeneEffect.csv")
    predicted = pd.read_parquet(cli.held_gene_effect)
    ess = gene_effect_input(measured, predicted, held_lines)
    ess = ess.loc[ess.index.isin(all_lines)]
    no_effect = sorted(
        fit_lines - set(ess.index)
    )  # fit lines without measured GeneEffect
    no_prediction = sorted(set(held_lines) - set(ess.index))

    # co-essentiality reads every train and bulk_labelled line of the split, never a
    # val or test line (the official graph reads all CCLE lines)
    coess_effect = coessentiality_input(measured, json.loads(cli.split.read_text()))
    graphs = official_graphs(coess_effect)

    exp = (
        2.0
        ** read_depmap_matrix(
            cli.depmap / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
        )
        - 1
    )
    damaging = read_depmap_matrix(
        cli.depmap / "OmicsSomaticMutationsMatrixDamaging.csv"
    )
    hotspot = read_depmap_matrix(cli.depmap / "OmicsSomaticMutationsMatrixHotspot.csv")
    lines_m = damaging.index.union(hotspot.index)
    genes_m = damaging.columns.union(hotspot.columns)
    mut = (
        (damaging.reindex(index=lines_m, columns=genes_m).fillna(0) > 0)
        | (hotspot.reindex(index=lines_m, columns=genes_m).fillna(0) > 0)
    ).astype(float)
    cnv = 2.0 ** read_copy_number(cli.depmap / "PortalOmicsCNGeneLog2.csv") - 1

    # lines the expression table lacks read their pseudo-bulk (counts per million)
    source = {}
    extra = {}
    for line in all_lines:
        if line in exp.index:
            source[line] = "depmap_tpm"
        elif (cli.h5ad / f"{line}.h5ad").exists():
            extra[line] = pseudo_bulk_cpm(cli.h5ad / f"{line}.h5ad")
            source[line] = "pseudo_bulk_cpm"
        else:
            source[line] = "none"
    if extra:
        exp = pd.concat(
            [exp, pd.DataFrame(extra).T]
        )  # columns align; a missing gene reads 0
    tables = {"exp": exp, "mut": mut, "cnv": cnv, "ess": ess}

    # node index: the benchmark genes and the genes of the four graphs
    edge_genes = [
        pd.unique(g[["gene1", "gene2"]].astype(str).to_numpy().ravel())
        for g in graphs.values()
    ]
    genes = pd.Index(
        sorted(
            set(rows["gene_1"])
            | set(rows["gene_2"])
            | {s.upper() for e in edge_genes for s in e}
        )
    )
    index = {g: i for i, g in enumerate(genes)}
    edges = {name: graph_edges(graphs[name], rows, index) for name in GRAPHS}

    kept = [line for line in all_lines if line not in no_effect]
    features = {
        line: node_features(tables, line, genes).astype(np.float32) for line in kept
    }
    features["mean_input"] = np.mean(
        [features[line] for line in kept if line in fit_lines], axis=0
    ).astype(np.float32)
    # a held-out line's GeneEffect feature is the prediction, nothing measured
    for line in set(held_lines) & set(ess.index):
        want = predicted.loc[line].reindex(genes).fillna(0.0).to_numpy(np.float32)
        assert np.array_equal(features[line][:, FEATURES.index("ess")], want)

    cli.out.mkdir(parents=True, exist_ok=True)
    (cli.out / "genes.json").write_text(json.dumps(list(genes)))
    np.savez(cli.out / "graphs.npz", **edges)
    np.savez(cli.out / "features.npz", **features)
    rows.to_parquet(cli.out / "rows.parquet", index=False)
    report = {
        "genes": len(genes),
        "edges_per_graph": {name: int(e.shape[1]) for name, e in edges.items()},
        "fit_lines": len(fit_lines),
        "coessentiality_lines": len(coess_effect),
        "held_lines": len(held_lines),
        "fit_lines_without_gene_effect": no_effect,
        "held_lines_without_prediction": no_prediction,
        "expression_source": source,
        "expression_source_counts": pd.Series(source).value_counts().to_dict(),
        "lines_without_copy_number": sorted(set(kept) - set(cnv.index)),
        "lines_without_mutation": sorted(set(kept) - set(mut.index)),
        "input_sha256": {
            "held_gene_effect": sha256(cli.held_gene_effect),
            "CRISPRGeneEffect.csv": sha256(cli.depmap / "CRISPRGeneEffect.csv"),
            "labels": sha256(cli.labels),
            "split": sha256(cli.split),
        },
    }
    (cli.out / "prepare.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "expression_source"}, indent=1
        )
    )


if __name__ == "__main__":
    main()
