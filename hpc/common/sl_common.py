"""Shared pieces of the published-baseline harnesses (hpc/<model>/): the benchmark rows,
pair-level labels and the score tables that src/experiments/sl/benchmark_baselines.py
reads. pandas, numpy and pyarrow only, so that every harness environment imports it."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Training lines with single cells and the labelled lines with bulk data only.
FIT_SIDES = ("train", "bulk_labelled")
HELD_SIDES = ("val", "test")
#: Lines whose rows a score table holds (the training lines in sample).
SCORED_SIDES = ("train", *HELD_SIDES)


def benchmark_rows(labels: Path, split: Path) -> pd.DataFrame:
    """Labelled rows of the fit and held-out lines, with the pair in sorted order."""
    sides = json.loads(Path(split).read_text())
    side = {m: s for s in FIT_SIDES + HELD_SIDES for m in sides[s]}
    rows = pd.read_parquet(
        labels, columns=["model_id", "gene_a", "gene_b", "label", "condition"]
    )
    rows = rows[rows["label"].notna() & rows["model_id"].isin(side)].reset_index(
        drop=True
    )
    return rows.assign(
        label=rows["label"].astype(int),
        side=rows["model_id"].map(side),
        gene_1=np.minimum(rows["gene_a"], rows["gene_b"]),
        gene_2=np.maximum(rows["gene_a"], rows["gene_b"]),
    )


def pair_labels(rows: pd.DataFrame, sides: tuple[str, ...]) -> pd.DataFrame:
    """One row for each pair of the sides' genetic rows: SL when at least half of its
    rows are SL (the rule of hpc/kg4sl/train.py)."""
    chosen = rows[rows["side"].isin(sides) & (rows["condition"] == "genetic")]
    rate = chosen.groupby(["gene_1", "gene_2"])["label"].mean()
    return (rate >= 0.5).astype(int).rename("label").reset_index()


def balanced(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """All SL rows and as many non-SL rows, drawn without replacement."""
    positive, negative = pairs[pairs["label"] == 1], pairs[pairs["label"] == 0]
    drawn = rng.choice(len(negative), min(len(positive), len(negative)), replace=False)
    return pd.concat([positive, negative.iloc[np.sort(drawn)]], ignore_index=True)


def drop_pair_edges(edges: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """``edges`` (gene1, gene2) without the edges between the two genes of a benchmark
    pair, in either order: such an edge can carry the label."""
    pairs = pd.MultiIndex.from_frame(rows[["gene_1", "gene_2"]].drop_duplicates())
    low = np.minimum(edges["gene1"], edges["gene2"])
    high = np.maximum(edges["gene1"], edges["gene2"])
    return edges[~pd.MultiIndex.from_arrays([low, high]).isin(pairs)]


def _write(
    out: Path, name: str, rows: pd.DataFrame, scores: pd.Series, keys: list[str]
) -> dict:
    """One table row for each stored orientation of each scored row. A row without a
    score ties at the median of the covered held-out rows (``covered`` false)."""
    scored = rows[rows["side"].isin(SCORED_SIDES)]
    lookup = [
        k if k == "model_id" else {"gene_a": "gene_1", "gene_b": "gene_2"}[k]
        for k in keys
    ]
    value = scores.reindex(pd.MultiIndex.from_frame(scored[lookup])).to_numpy(float)
    covered = ~np.isnan(value)
    held = scored["side"].isin(HELD_SIDES).to_numpy()
    if not (covered & held).any():
        raise ValueError("no held-out row has a score")
    fill = float(np.median(value[covered & held]))
    table = scored[keys].assign(score=np.where(covered, value, fill), covered=covered)
    table = table.drop_duplicates(keys)
    Path(out).mkdir(parents=True, exist_ok=True)
    table.to_parquet(Path(out) / f"{name}.parquet", index=False)
    return {
        "scored_pairs" if "model_id" not in keys else "scored_rows": len(table),
        "scored_pairs_covered"
        if "model_id" not in keys
        else "scored_rows_covered": int(table["covered"].sum()),
        "held_rows_covered_share": float(covered[held].mean()),
        "uncovered_score": fill,
    }


def write_pair_scores(out: Path, rows: pd.DataFrame, scores: pd.Series) -> dict:
    """``scores`` indexed by (gene_1, gene_2): ``out/scores.parquet`` with gene_a,
    gene_b, score, covered."""
    return _write(out, "scores", rows, scores, ["gene_a", "gene_b"])


def write_line_scores(
    out: Path, rows: pd.DataFrame, scores: pd.Series, name: str = "scores"
) -> dict:
    """``scores`` indexed by (model_id, gene_1, gene_2): ``out/<name>.parquet`` with
    model_id, gene_a, gene_b, score, covered."""
    return _write(out, name, rows, scores, ["model_id", "gene_a", "gene_b"])
