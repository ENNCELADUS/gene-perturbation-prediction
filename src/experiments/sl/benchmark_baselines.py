"""Baselines on the multi-study SL benchmark (docs/specs/2026-10-29-multi-study-sl-
benchmark-design.md). A list's negatives are tested and not called, not true negatives,
so the metrics are positive-unlabelled ones. Headline: recall in the top 10% of each
list (a random ranking gives 0.10), with each known positive ranked inside its own list
and the positives of a line's lists pooled, then the mean over a split's lines, and
the per-line difference from pair identity. Then AUROC (the mean percentile of the
known positives) and AUPRC over the positive rate. Scorers: the pair's hit rate in the
same study's training lines (pair identity), each gene's mean GeneEffect over the
training lines (pan-essentiality), stage one's production prediction of GeneEffect in
the line, and the measured GeneEffect as a reference, never a bar (only lines with
GeneEffect have it). Every split is reported for all lines, the lines with and without
GeneEffect, and the cancer and non-cancer lines."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.depmap import read_line_identities
from src.data.io import write_json, write_parquet, write_text
from src.data.labels import parse_gene_symbol
from src.data.splits import FixedSplit, assert_fit_eligible, load_geneeffect_split
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind
from src.experiments.sl.stage_two import ranking_metrics

SIDES = ("train", "val", "test")
LIST_KEYS = ["split", "design", "study_id", "list_id", "model_id"]
#: The headline's top fraction of each list.
TOP = 0.10


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "labels", "split", "gene_effect",
         "model", "stage_one", "top_k", "bootstrap"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_benchmark_baselines":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def read_gene_effect(path: Path) -> pd.DataFrame:
    """DepMap GeneEffect, lines x symbols."""
    wide = pd.read_csv(path, index_col=0)
    wide.index = wide.index.astype(str)
    wide.columns = [parse_gene_symbol(column) for column in wide.columns]
    return wide


def non_cancer_lines(model: Path) -> set[str]:
    """Lines outside the cancer lines that train stage one."""
    frame = pd.read_csv(model, usecols=["ModelID", "OncotreePrimaryDisease"], dtype=str)
    depmap = frame.loc[frame["OncotreePrimaryDisease"] == "Non-Cancerous", "ModelID"]
    identities = read_line_identities()
    return set(depmap) | set(identities.index[identities["non_cancer"] == "true"])


def benchmark_rows(
    labels: pd.DataFrame, split: FixedSplit, genes: set[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Labelled rows of split lines whose genes every scorer can read: both genes of a
    pair, gene B of an anchor list (its anchor may be a drug)."""
    side = {m: s for s in SIDES for m in getattr(split, s)}
    labelled = labels[labels["label"].notna()]
    in_split = labelled["model_id"].isin(side)
    known = labelled["gene_b"].isin(genes) & (
        (labelled["design"] == "anchor") | labelled["gene_a"].isin(genes)
    )
    rows = labelled[in_split & known].assign(
        split=lambda r: r["model_id"].map(side), label=lambda r: r["label"].astype(int)
    )
    dropped = {
        "rows_line_not_in_split": int((~in_split).sum()),
        "rows_gene_without_score": int((in_split & ~known).sum()),
    }
    return rows.reset_index(drop=True), dropped


def pair_identity(training: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """The pair's hit rate over the same study's training lines; a pair never
    labelled there scores the study's training hit rate, so it ties."""
    keys = ["study_id", "gene_a", "gene_b"]
    rate = training.groupby(keys)["label"].mean()
    fill = training.groupby("study_id")["label"].mean()
    scored = rate.reindex(pd.MultiIndex.from_frame(rows[keys])).to_numpy()
    return np.where(np.isnan(scored), rows["study_id"].map(fill).to_numpy(), scored)


def pair_sum(rows: pd.DataFrame, value: Callable[[str], np.ndarray]) -> np.ndarray:
    """value("gene_a") + value("gene_b"). An anchor list's gene A is one gene or drug
    for the whole list, so its term is 0: the list's ranking is unchanged."""
    # a copy: pandas may return a read-only array
    a = np.array(value("gene_a"), dtype=np.float64)
    a[(rows["design"] == "anchor").to_numpy()] = 0.0
    return a + value("gene_b")


def line_values(effects: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """effects[line, gene_a] + effects[line, gene_b] (pair_sum); NaN where a needed
    value is missing."""
    lines = effects.index.get_indexer(rows["model_id"])
    values = effects.to_numpy(dtype=np.float64)

    def value(gene: str) -> np.ndarray:
        column = effects.columns.get_indexer(rows[gene])
        ok = (lines >= 0) & (column >= 0)
        out = np.full(len(rows), np.nan)
        out[ok] = values[lines[ok], column[ok]]
        return out

    return pair_sum(rows, value)


def positive_ranks(label: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per known positive of one list (finite scores only): its top-10% credit, the
    expected share of its tie block that falls in the top ``TOP`` of the list (a
    random ranking gives ``TOP``), and its percentile among the list's negatives
    (ties count half; NaN when the list has no negative)."""
    keep = np.isfinite(score)
    label, score = label[keep].astype(bool), score[keep]
    ranked = np.sort(score)
    positive = score[label]
    above = len(score) - np.searchsorted(ranked, positive, side="right")
    ties = np.searchsorted(ranked, positive, side="right") - np.searchsorted(
        ranked, positive, side="left"
    )
    credit = np.clip(TOP * len(score) - above, 0, ties) / ties
    negative = np.sort(score[~label])
    if not len(negative):
        return credit, np.full(len(positive), np.nan)
    below = np.searchsorted(negative, positive, side="left")
    tied = np.searchsorted(negative, positive, side="right") - below
    return credit, (below + 0.5 * tied) / len(negative)


def list_records(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per held-out list and scorer: the known positives' summed top-10% credit and
    percentile, and the ranking metrics (AUPRC over the positive rate)."""
    records = []
    for keys, group in scores.groupby([*LIST_KEYS, "scorer"], sort=True):
        label, score = group["label"].to_numpy(), group["score"].to_numpy()
        credit, percentile = positive_ranks(label, score)
        defined = percentile[np.isfinite(percentile)]
        records.append(
            dict(zip([*LIST_KEYS, "scorer"], keys))
            | ranking_metrics(label, score, k)
            | {"top_credit": float(credit.sum()), "ranked_positives": len(credit),
               "percentile_sum": float(defined.sum()), "percentiles": len(defined)}
        )  # fmt: skip
    return pd.DataFrame(records)


def line_records(lists: pd.DataFrame) -> pd.DataFrame:
    """Per split, scorer and line, the line's positives pooled over its lists:
    recall in the top 10%, AUROC (mean percentile of the positives) and the mean
    AUPRC over the positive rate of its lists with both classes."""
    keys = ["split", "scorer", "model_id"]
    pooled = lists.groupby(keys).agg(
        positives=("ranked_positives", "sum"),
        top_credit=("top_credit", "sum"),
        percentile_sum=("percentile_sum", "sum"),
        percentiles=("percentiles", "sum"),
        auprc_lift=("auprc_lift", "mean"),
        lists=("list_id", "size"),
    )
    pooled = pooled[pooled["positives"] > 0]
    pooled = pooled.assign(
        recall_top10=pooled["top_credit"] / pooled["positives"],
        auroc=pooled["percentile_sum"]
        / pooled["percentiles"].where(pooled["percentiles"] > 0),
    )
    columns = [*keys, "positives", "lists", "recall_top10", "auroc", "auprc_lift"]
    return pooled.reset_index()[columns]


def summary(
    lines: pd.DataFrame, groups: Mapping[str, set[str]], reps: int, seed: int
) -> list[dict[str, Any]]:
    """Per split, group of lines and scorer: the mean over lines of recall in the
    top 10%, the mean per-line difference from pair identity with a 95% interval
    from resampling lines, AUROC and AUPRC over the positive rate; each with its
    line count and positive count."""
    rng = np.random.default_rng(seed)
    out = []
    for split in ("val", "test"):
        side = lines[lines["split"] == split]
        control = side[side["scorer"] == "pair_identity"].set_index("model_id")
        for name, members in groups.items():
            chosen = side[side["model_id"].isin(members)]
            for scorer, rows in chosen.groupby("scorer"):
                rows = rows.set_index("model_id")
                gap = (rows["recall_top10"] - control["recall_top10"]).dropna()
                values = gap.to_numpy()
                draws = [rng.choice(values, len(values)).mean() for _ in range(reps)]
                low, high = (
                    np.quantile(draws, [0.025, 0.975]) if len(values) else (np.nan,) * 2
                )
                out.append(
                    {"split": split, "group": name, "scorer": scorer,
                     "lines": len(rows), "positives": int(rows["positives"].sum()),
                     "recall_top10": float(rows["recall_top10"].mean()),
                     "minus_pair_identity": (
                         float(values.mean()) if len(values) else None),
                     "interval": [float(low), float(high)],
                     "auroc": float(rows["auroc"].mean()),
                     "auprc_lift": float(rows["auprc_lift"].mean())}
                )  # fmt: skip
    return out


def line_groups(split_table: pd.DataFrame, non_cancer: set[str]) -> dict[str, set[str]]:
    """The report's groups of lines: all, each GeneEffect group, cancer and
    non-cancer."""
    lines = set(split_table["model_id"])
    by = split_table.groupby("group")["model_id"].agg(set)
    return {
        "all": lines,
        "with_geneeffect": by.get("with_geneeffect", set()),
        "without_geneeffect": by.get("without_geneeffect", set()),
        "cancer": lines - non_cancer,
        "non_cancer": lines & non_cancer,
    }


def run(config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split = load_geneeffect_split(Path(config["split"]))
    effects = read_gene_effect(Path(config["gene_effect"]))
    training_lines = [m for m in split.supervised_train if m in effects.index]
    for model_id in training_lines:
        assert_fit_eligible(model_id, split)
    mean = effects.loc[training_lines].mean()
    stage_one = config["stage_one"]
    predicted = None if stage_one is None else pd.read_parquet(stage_one)
    genes = set(effects.columns)
    if predicted is not None:
        genes &= set(predicted.columns)
    rows, dropped = benchmark_rows(pd.read_parquet(config["labels"]), split, genes)
    training = rows[rows["split"] == "train"]
    held = rows[rows["split"].isin(["val", "test"])].reset_index(drop=True)
    scorers = {
        "pair_identity": pair_identity(training, held),
        "pan_essentiality": -pair_sum(
            held, lambda gene: mean.reindex(held[gene]).to_numpy(dtype=np.float64)
        ),
        "measured_geneeffect": -line_values(effects, held),
    }
    if predicted is not None:
        scorers["stage_one_production"] = -line_values(predicted, held)
    base = held[[*LIST_KEYS, "gene_a", "gene_b", "label"]]
    scores = pd.concat(
        [base.assign(scorer=name, score=values) for name, values in scorers.items()],
        ignore_index=True,
    )
    # the measured control exists only on lines with GeneEffect
    without = (scores["scorer"] == "measured_geneeffect") & ~scores["model_id"].isin(
        effects.index
    )
    scores = scores[~without].reset_index(drop=True)
    lists = list_records(scores, int(config["top_k"]))
    lines = line_records(lists)
    split_table = pd.read_csv(Path(config["split"]).with_suffix(".csv"), dtype=str)
    groups = line_groups(split_table, non_cancer_lines(Path(config["model"])))
    lines["group"] = lines["model_id"].map(split_table.set_index("model_id")["group"])
    metrics = {
        "run_id": run_id,
        "rows": {"training": len(training), "held": len(held), **dropped,
                 "measured_rows_without_geneeffect": int(without.sum())},
        "lists_without_both_classes": int(
            lists.loc[lists["scorer"] == "pair_identity", "auprc_lift"].isna().sum()
        ),
        "summary": summary(
            lines, groups, int(config["bootstrap"]), int(config["seed"])),
    }  # fmt: skip
    write_parquet(run_dir / "scores.parquet", scores)
    lists.to_csv(run_dir / "lists.csv", index=False)
    lines.to_csv(run_dir / "lines.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        f"# SL benchmark baselines: {metrics['run_id']}",
        "",
        "Recall in the top 10% of each list (a random ranking gives 0.10): the known "
        "positives of a line's lists, pooled, then the mean over lines; the difference "
        "from pair identity per line, with a 95% interval from resampling lines. AUROC "
        "is the mean percentile of the known positives; AUPRC lift is AUPRC over the "
        "positive rate. A list's negatives are tested and not called, not true "
        "negatives. The measured GeneEffect is scored only on lines with GeneEffect. "
        "Stage two trains mostly on in-sample stage-one inputs, so the number for "
        "lines without GeneEffect is a low estimate; that group is small and mostly "
        "non-cancer.",
        "",
        "| Split | Group | Scorer | Lines | Positives | Recall top 10% "
        "| − pair identity | 95% interval | AUROC | AUPRC lift |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in metrics["summary"]:
        low, high = row["interval"]
        gap = row["minus_pair_identity"]
        lines.append(
            f"| {row['split']} | {row['group']} | {row['scorer']} | {row['lines']} | "
            f"{row['positives']} | {row['recall_top10']:.3f} | "
            f"{'—' if gap is None else f'{gap:+.3f}'} | [{low:.3f}, {high:.3f}] | "
            f"{row['auroc']:.3f} | {row['auprc_lift']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
