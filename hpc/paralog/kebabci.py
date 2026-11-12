"""Kebabci et al. 2026 context-specific paralog SL classifier on the multi-study SL
benchmark: the official contextualised random forest (16 features; the settings of
notebooks/04_model_evaluation/02_cross_validation.ipynb), refitted on the paralog rows
of the SL split's training-side lines. The released model trained on Ito 2021 lines,
which are benchmark lines, so it is not used.

The features transcribe the official notebooks at commit f966551 of
github.com/cancergenetics/context_specific_paralog_SL (the repository's scripts/ only
merge their precomputed tables). For a line and a paralog pair (A1, A2), on DepMap 26Q1:

- rMaxExp/rMinExp_A1A2: the larger and smaller log2(TPM + 1) of the two genes
  (03_feature_annotation/01_annotate_DepMap).
- max/min_ranked_A1A2: the genes' ranks of GeneEffect in the line, 1 = most essential
  (02_feature_calculation/01_ranked_essentiality, 03_annotate_ranked_essentiality).
- max_cn/min_cn: gene copy number; Protein_Altering/Damaging: how many of the two genes
  have a somatic mutation / a damaging one (01_annotate_DepMap).
- min_sequence_identity, prediction_score: De Kegel 2021 Table S8.
- weighted_PPI_essentiality/expression: the mean rank (expression) of the pair's shared
  STRING v11.5 interactors, weighted by the product of the two interaction scores
  (03a, 03b, 03c).
- smallest_BP/CC_GO_essentiality/expression: the mean rank (expression) of the genes of
  the smallest GO term that annotates both genes, terms counted over paralog genes and
  kept at 5 < size < 100; ties take the most essential (most expressed) term (04, 05).

Rows without expression or ranks are dropped (not covered); missing PPI and GO values
are 0 for expression and 18,000 for ranks, as the notebooks do. Pairs with a DepMap
common essential gene are dropped (01_annotate_DepMap). Deviations: DepMap 26Q1, not
22Q4; mutations and common essentials from DepMap 24Q4 (26Q1's portal links need a
browser); a line absent from the mutation table has missing mutation features, not 0;
STRING and GO are named by symbol, not by Ensembl ID; GO is GOA human release 218
(2023-07-12), not an Ensembl BioMart export; no PC9 or HeLa screen fills a line without
GeneEffect.

The features read the held-out line's measured GeneEffect, so this is a reference, like
the measured GeneEffect row. A held-out row is covered when its pair is in Table S8, has
no common essential gene, and its line has DepMap expression and GeneEffect; the other
rows get the median score of the covered held-out rows, so they tie.

    .venv-tx1/bin/python hpc/paralog/kebabci.py --out outputs/sl/kebabci/seed0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hpc" / "paralog"))

from dekegel_scores import entrez_symbols, table_s8  # noqa: E402

from src.data.depmap import read_depmap_matrix  # noqa: E402
from src.experiments.sl.slidr import (  # noqa: E402
    FIT_SIDES,
    HELD_SIDES,
    read_copy_number,
)

DEPMAP = ROOT / "data/sl_dependency_v0/raw/depmap"
BASELINES = ROOT / "data/sl_baselines"
#: The contextualised model's features (02_cross_validation, feature_columns_1).
FEATURES = [
    "rMaxExp_A1A2", "rMinExp_A1A2", "max_ranked_A1A2", "min_ranked_A1A2",
    "max_cn", "min_cn", "Protein_Altering", "Damaging", "min_sequence_identity",
    "prediction_score", "weighted_PPI_essentiality", "weighted_PPI_expression",
    "smallest_BP_GO_essentiality", "smallest_CC_GO_essentiality",
    "smallest_BP_GO_expression", "go_CC_expression",
]  # fmt: skip
#: 01_preprocess_training_dataset: rows missing these are dropped; the others filled.
REQUIRED = ["rMaxExp_A1A2", "rMinExp_A1A2", "max_ranked_A1A2", "min_ranked_A1A2"]
FILL_ZERO = ["weighted_PPI_expression", "smallest_BP_GO_expression", "go_CC_expression"]
FILL_LARGE = [
    "weighted_PPI_essentiality",
    "smallest_BP_GO_essentiality",
    "smallest_CC_GO_essentiality",
]
LARGE = 18_000
#: RandomForestClassifier of 02_cross_validation.
FOREST = dict(
    n_estimators=600,
    random_state=42,
    max_features=0.2,
    max_depth=20,
    min_samples_leaf=4,
)


def string_scores(links: Path, info: Path) -> dict[str, dict[str, float]]:
    """Symbol -> {interactor symbol: STRING combined score}, both directions; a symbol
    with several proteins keeps the largest score."""
    names = pd.read_csv(info, sep="\t", usecols=[0, 1])
    names = dict(zip(names.iloc[:, 0], names.iloc[:, 1].str.upper()))
    edges = pd.read_csv(links, sep=" ")
    a, b = edges["protein1"].map(names), edges["protein2"].map(names)
    edges = pd.DataFrame({"a": a, "b": b, "s": edges["combined_score"]}).dropna()
    edges = edges[edges["a"] != edges["b"]].groupby(["a", "b"])["s"].max()
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (x, y), s in edges.items():
        out[x][y] = float(s)
    return out


def weighted_ppi(
    pairs: pd.DataFrame, interactors: dict, values: pd.DataFrame
) -> np.ndarray:
    """pairs x lines: sum_i w_i values[line, i] / sum_i w_i over the pair's shared
    interactors i with a column in ``values``, w_i = s(A1, i) s(A2, i); NaN values add
    nothing to the sum (pandas ``sum``); NaN when no interactor is shared."""
    column = {g: j for j, g in enumerate(values.columns)}
    data = np.nan_to_num(values.to_numpy(np.float64)).T  # genes x lines
    out = np.full((len(pairs), values.shape[0]), np.nan)
    for k, (a, b) in enumerate(zip(pairs["gene_1"], pairs["gene_2"])):
        pa, pb = interactors.get(a, {}), interactors.get(b, {})
        shared = [i for i in set(pa) & set(pb) if i in column]
        if not shared:
            continue
        w = np.array([pa[i] * pb[i] for i in shared])
        out[k] = w @ data[[column[i] for i in shared]] / w.sum()
    return out


def go_terms(gaf: Path, paralog_genes: set[str]) -> dict[str, dict[str, set[str]]]:
    """Aspect (P, C) -> GO term -> its paralog genes, terms with 5 < size < 100."""
    columns = ["db", "id", "symbol", "qualifier", "go", "aspect"]
    rows = pd.read_csv(
        gaf, sep="\t", comment="!", header=None, usecols=[0, 1, 2, 3, 4, 8],
        names=columns, dtype=str,
    )  # fmt: skip
    rows = rows[~rows["qualifier"].str.contains("NOT", na=False)]
    rows = rows.assign(symbol=rows["symbol"].str.upper())
    rows = rows[rows["symbol"].isin(paralog_genes) & rows["aspect"].isin(["P", "C"])]
    out = {}
    for aspect, group in rows.groupby("aspect"):
        sets = group.groupby("go")["symbol"].agg(set)
        out[aspect] = {t: s for t, s in sets.items() if 5 < len(s) < 100}
    return out


def smallest_go(
    pairs: pd.DataFrame, terms: dict[str, set[str]], values: pd.DataFrame, pick: str
) -> np.ndarray:
    """pairs x lines: the mean of ``values`` over the genes of the smallest term that
    annotates both genes (terms with at least two genes in ``values``), the ``pick``
    (max or min) over equally small terms; NaN when no term is shared."""
    present = set(values.columns)
    valid = {t: s for t, s in terms.items() if len(s & present) >= 2}
    by_gene: dict[str, set[str]] = defaultdict(set)
    for t, s in valid.items():
        for g in s:
            by_gene[g].add(t)
    data = values.to_numpy(np.float64)
    means: dict[str, np.ndarray] = {}
    out = np.full((len(pairs), values.shape[0]), np.nan)
    for k, (a, b) in enumerate(zip(pairs["gene_1"], pairs["gene_2"])):
        shared = by_gene.get(a, set()) & by_gene.get(b, set())
        if not shared:
            continue
        size = min(len(valid[t]) for t in shared)
        for t in (t for t in shared if len(valid[t]) == size):
            if t not in means:
                genes = values.columns.get_indexer(sorted(valid[t] & present))
                means[t] = np.nanmean(data[:, genes], axis=1)
        stack = np.array([means[t] for t in shared if len(valid[t]) == size])
        out[k] = np.nanmax(stack, axis=0) if pick == "max" else np.nanmin(stack, axis=0)
    return out


def mutation_counts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(line, gene) -> has a somatic mutation, has a damaging one (DepMap's LikelyLoF,
    or CCLEDeleterious where the release has it), as 0/1 wide tables over the lines
    the table has."""
    table = pd.read_csv(path, low_memory=False)
    line = "ModelID" if "ModelID" in table else "DepMap_ID"
    flags = [c for c in ("LikelyLoF", "CCLEDeleterious") if c in table]
    damaging = np.zeros(len(table), bool)
    for c in flags:
        damaging |= table[c].astype(str).str.upper().isin(["TRUE", "Y", "YES", "1"])
    rows = pd.DataFrame(
        {"line": table[line].astype(str), "gene": table["HugoSymbol"].str.upper(),
         "mut": 1, "damaging": damaging.astype(int)}
    ).dropna(subset=["gene"])  # fmt: skip
    agg = rows.groupby(["line", "gene"])[["mut", "damaging"]].max()
    return agg["mut"].unstack(fill_value=0), agg["damaging"].unstack(fill_value=0)


def features(
    rows: pd.DataFrame, pair_table: pd.DataFrame, inputs: dict
) -> pd.DataFrame:
    """The 16 features of each (model_id, gene_1, gene_2) row."""
    expression, ranks = inputs["expression"], inputs["ranks"]
    lines = sorted(set(rows["model_id"]))
    pairs = rows[["gene_1", "gene_2"]].drop_duplicates().reset_index(drop=True)
    p = pd.MultiIndex.from_frame(pairs).get_indexer(
        pd.MultiIndex.from_frame(rows[["gene_1", "gene_2"]])
    )
    out = pd.DataFrame(index=rows.index)

    def gene_values(table: pd.DataFrame, gene: str, absent: float) -> np.ndarray:
        i = table.index.get_indexer(rows["model_id"])
        j = table.columns.get_indexer(rows[gene])
        values = table.to_numpy(np.float64)
        value = np.where(j >= 0, values[i, np.maximum(j, 0)], absent)
        return np.where(i >= 0, value, np.nan)

    for name, table, absent in (
        ("Exp", expression, np.nan), ("ranked", ranks, np.nan),
        ("cn", inputs["copy_number"], np.nan),
    ):  # fmt: skip
        a, b = (
            gene_values(table, "gene_1", absent),
            gene_values(table, "gene_2", absent),
        )
        high, low = np.maximum(a, b), np.minimum(a, b)  # NaN if either is NaN
        if name == "Exp":
            out["rMaxExp_A1A2"], out["rMinExp_A1A2"] = high, low
        elif name == "ranked":
            out["max_ranked_A1A2"], out["min_ranked_A1A2"] = high, low
        else:
            out["max_cn"], out["min_cn"] = high, low
    # a gene without a call in a profiled line is not mutated
    for name, table in (("Protein_Altering", inputs["mut"]),
                        ("Damaging", inputs["damaging"])):  # fmt: skip
        out[name] = gene_values(table, "gene_1", 0.0) + gene_values(
            table, "gene_2", 0.0
        )
    released = pair_table.set_index(["gene_1", "gene_2"])
    key = pd.MultiIndex.from_frame(rows[["gene_1", "gene_2"]])
    for name in ("min_sequence_identity", "prediction_score"):
        out[name] = released[name].reindex(key).to_numpy()
    line_index = pd.Index(lines)
    columns = line_index.get_indexer(rows["model_id"])
    on = {"expression": expression.reindex(lines), "ranks": ranks.reindex(lines)}
    computed = {
        "weighted_PPI_essentiality": weighted_ppi(pairs, inputs["string"], on["ranks"]),
        "weighted_PPI_expression": weighted_ppi(
            pairs, inputs["string"], on["expression"]
        ),
        "smallest_BP_GO_essentiality": smallest_go(
            pairs, inputs["go"]["P"], on["ranks"], "min"
        ),
        "smallest_CC_GO_essentiality": smallest_go(
            pairs, inputs["go"]["C"], on["ranks"], "min"
        ),
        "smallest_BP_GO_expression": smallest_go(
            pairs, inputs["go"]["P"], on["expression"], "max"
        ),
        "go_CC_expression": smallest_go(
            pairs, inputs["go"]["C"], on["expression"], "max"
        ),
    }
    for name, matrix in computed.items():
        out[name] = matrix[p, columns]
    out[FILL_ZERO] = out[FILL_ZERO].fillna(0.0)
    out[FILL_LARGE] = out[FILL_LARGE].fillna(LARGE)
    return out[FEATURES]


def benchmark_rows(labels: Path, split: Path) -> pd.DataFrame:
    sides = json.loads(split.read_text())
    side = {m: s for s in FIT_SIDES + HELD_SIDES for m in sides[s]}
    rows = pd.read_parquet(
        labels,
        columns=["model_id", "gene_a", "gene_b", "label", "condition", "study_id"],
    )
    rows = rows[rows["label"].notna() & rows["model_id"].isin(side)]
    return rows.assign(
        side=rows["model_id"].map(side),
        label=rows["label"].astype(int),
        gene_1=np.minimum(rows["gene_a"], rows["gene_b"]),
        gene_2=np.maximum(rows["gene_a"], rows["gene_b"]),
    )


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
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
    cli = parser.parse_args()
    symbols = entrez_symbols(DEPMAP / "CRISPRGeneEffect.csv")
    pair_table = table_s8(BASELINES / "dekegel/Table_S8.csv", symbols).rename(
        columns={"score": "prediction_score"}
    )
    essentials = pd.read_csv(
        BASELINES / "depmap24q4/CRISPRInferredCommonEssentials.csv"
    )
    essential = set(essentials.iloc[:, 0].str.split(" ").str[0].str.upper())
    effect = read_depmap_matrix(DEPMAP / "CRISPRGeneEffect.csv")
    expression = read_depmap_matrix(
        DEPMAP / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    mut, damaging = mutation_counts(BASELINES / "depmap24q4/OmicsSomaticMutations.csv")
    # the notebooks' paralog gene list: Table S8 genes with expression
    paralog = set(pair_table["gene_1"]) | set(pair_table["gene_2"])
    inputs = {
        "expression": expression,
        "ranks": effect.rank(axis=1),
        "copy_number": read_copy_number(DEPMAP / "PortalOmicsCNGeneLog2.csv"),
        "mut": mut,
        "damaging": damaging,
        "string": string_scores(
            BASELINES / "string/9606.protein.links.v11.5.txt.gz",
            BASELINES / "string/9606.protein.info.v11.5.txt.gz",
        ),
        "go": go_terms(
            BASELINES / "go/goa_human.gaf.218.gz", paralog & set(expression.columns)
        ),
    }
    rows = benchmark_rows(cli.labels, cli.split)
    released = pd.MultiIndex.from_frame(pair_table[["gene_1", "gene_2"]])
    key = pd.MultiIndex.from_frame(rows[["gene_1", "gene_2"]])
    eligible = key.isin(released) & ~(
        rows["gene_1"].isin(essential) | rows["gene_2"].isin(essential)
    )
    # one training sample per (line, pair): SL when half or more of its rows are SL
    fit = rows[
        eligible & rows["side"].isin(FIT_SIDES) & (rows["condition"] == "genetic")
    ]
    fit = (
        fit.groupby(["model_id", "gene_1", "gene_2"])["label"].mean().ge(0.5)
        .astype(int).rename("label").reset_index()
    )  # fmt: skip
    held = rows[rows["side"].isin(HELD_SIDES)].drop_duplicates(
        ["model_id", "gene_a", "gene_b"]
    )
    held_eligible = held[eligible[held.index]]
    x_fit = features(fit, pair_table, inputs)
    keep = x_fit[REQUIRED].notna().all(axis=1).to_numpy()
    x_fit, y_fit = x_fit[keep], fit.loc[keep, "label"].to_numpy()
    forest = RandomForestClassifier(**FOREST, n_jobs=32).fit(x_fit, y_fit)
    x_held = features(held_eligible, pair_table, inputs)
    covered = x_held[REQUIRED].notna().all(axis=1)
    score = pd.Series(np.nan, index=held.index)
    score[covered.index[covered]] = forest.predict_proba(x_held[covered])[:, 1]
    fill = float(score.median())
    out = held[["model_id", "gene_a", "gene_b"]].assign(
        score=score.fillna(fill).to_numpy(), covered=score.notna().to_numpy()
    )
    cli.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cli.out / "scores.parquet", index=False)
    scored = held.assign(score=score).dropna(subset=["score"])
    scored = scored[scored["condition"] == "genetic"]

    def auroc(frame: pd.DataFrame) -> float | None:
        if frame["label"].nunique() < 2:
            return None
        return float(roc_auc_score(frame["label"], frame["score"]))

    metrics = {
        "training_rows": int(len(x_fit)),
        "training_sl_share": float(y_fit.mean()),
        "training_lines": int(fit.loc[keep, "model_id"].nunique()),
        "held_rows": len(out),
        "held_rows_covered_share": float(out["covered"].mean()),
        "uncovered_score": fill,
        "auroc_covered_genetic": {
            side: auroc(scored[scored["side"] == side]) for side in HELD_SIDES
        },
        "auroc_covered_genetic_by_study": {
            f"{side}|{study}": auroc(group)
            for (side, study), group in scored.groupby(["side", "study_id"])
        },
        "feature_importance": dict(
            zip(FEATURES, map(float, forest.feature_importances_))
        ),
    }
    (cli.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importance"}))


if __name__ == "__main__":
    main()
