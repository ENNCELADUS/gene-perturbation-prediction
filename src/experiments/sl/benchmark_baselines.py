"""Baselines on the multi-study SL benchmark (docs/specs/2026-10-29-multi-study-sl-
benchmark-design.md). A list's negatives are tested and not called, not true negatives,
so the metrics are positive-unlabelled ones. Headline: recall in the top 10% of each
genetic screen with both classes (a random ranking gives 0.10), then the mean over a
study's screens, over a line's studies and over a split's lines, and the per-line
difference from pair identity. AUROC (the mean percentile of the known positives) and
AUPRC over the positive rate are averaged the same way. Screens under a treatment, an
activating anchor or a group-level call are scored but kept out of the headline.
Scorers: the pair's hit rate in the same study's training lines (pair identity), each
gene's mean GeneEffect over the training lines (pan-essentiality), stage one's
prediction of GeneEffect in the line (the SL backbone's reader, fitted without the
SL held-out lines, ``hpc/run.sh sl-conditional``), and the measured GeneEffect as a
reference, never a bar (only lines with GeneEffect have it). Two checks of the
benchmark code: the labels as the score (the ceiling, min(1, 0.1 n / positives) per
screen) and a seeded random score (about 0.10). Every split is reported for all
lines, the lines with and without GeneEffect, and the cancer and non-cancer lines,
and for each stratum of held-out pairs (STRATA), ranked inside it in each screen."""

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
LIST_KEYS = ["split", "design", "condition", "study_id", "list_id", "model_id"]
#: The only condition the headline reads (the rules' ``condition`` column).
HEADLINE = "genetic"
METRICS = ["recall_top10", "auroc", "auprc_lift"]
NAMES = {
    "stage_one": "Stage one's predicted GeneEffect (linear expression model)",
    "conditional": "Conditional score (backbone: B's predicted change when A is lost)",
    "conditional_ablated": "Conditional score, context-ablated (training mean profile)",
    "head": "Backbone head (logistic, on the conditional terms)",
    "head_ablated": "Backbone head, context-ablated (training mean profile)",
    "pair_identity": "Pair identity (hit rate in the study's training lines)",
    "pan_essentiality": "Pan-essentiality (gene mean GeneEffect)",
    "measured_geneeffect": "Measured GeneEffect (reference, lines with GeneEffect)",
    "labels": "Labels as the score (check: the ceiling)",
    "random": "Random score (check: about 0.10)",
    "kg4sl": "KG4SL (knowledge graph, official code, trained on the training lines)",
    "cilantro_sl": "Cilantro-SL (Geneformer deletions, official code, refitted)",
    "paris": "PARIS (random-forest dependency pairs, Python port of the R code)",
    "slidr": "SLIdR (mutation-dependency test, Python port of the R code)",
    "dekegel_paralog": "De Kegel 2021 paralog classifier (released scores)",
    "kebabci_paralog": "Kebabci 2026 paralog classifier (reads measured GeneEffect, "
    "reference, official code, refitted)",
    "mit4sl": "MiT4SL (cell-line-specific graph model, official code, refitted)",
}
#: Scorers that check the code or give a reference; never the best row.
NOT_MODELS = {"measured_geneeffect", "kebabci_paralog", "labels", "random"}
#: Scorers ranked beside each partial scorer on the rows it covers (Table 4).
COVERED_WITH = ("pair_identity", "labels", "random")
#: Strata of held-out pairs. A pair is seen when the same study labelled it in a
#: training line; it agrees when its label equals the majority of those labels. A new
#: pair with a gene that no training row has is not reported (too few rows).
STRATA = {
    "all": "All pairs",
    "seen_agree": "Seen pair, label agrees with the training lines",
    "seen_flip": "Seen pair, label differs from the training lines",
    "new_pair": "New pair, both genes seen",
}
#: The headline's top fraction of each list.
TOP = 0.10


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "labels", "split", "gene_effect",
         "model", "stage_one", "extra_scores", "top_k", "bootstrap"},
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


def strata(training: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """The stratum of each row (STRATA keys; ``unseen_gene`` is not reported). Gene A
    of an anchor list is one gene or drug for the whole list, so only gene B counts."""
    keys = ["study_id", "gene_a", "gene_b"]
    rate = training.groupby(keys)["label"].mean()
    rate = rate.reindex(pd.MultiIndex.from_frame(rows[keys])).to_numpy()
    genes = set(training["gene_a"]) | set(training["gene_b"])
    seen_genes = rows["gene_b"].isin(genes) & (
        (rows["design"] == "anchor") | rows["gene_a"].isin(genes)
    )
    agree = (rate >= 0.5) == rows["label"].to_numpy().astype(bool)
    return np.select(
        [~np.isnan(rate) & agree, ~np.isnan(rate), seen_genes.to_numpy()],
        ["seen_agree", "seen_flip", "new_pair"],
        "unseen_gene",
    )


def label_agreement(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """The stage-two ceiling, from the benchmark's own rows (every side): how often
    two labels of one pair agree, within a line across lists (``same_line``: another
    study or screen of the line) and within a study across lines (``same_study``).
    ``agreement`` is over all label pairs, ``positive_agreement`` the chance that a
    second label is SL when the first is."""

    def pairs(frame: pd.DataFrame, key: str) -> dict[str, float | int | None]:
        counts = frame.groupby([key, "gene_a", "gene_b"])["label"].agg(["size", "sum"])
        counts = counts[counts["size"] >= 2]
        n, k = counts["size"].to_numpy(float), counts["sum"].to_numpy(float)
        both, total = k * (k - 1) + (n - k) * (n - k - 1), n * (n - 1)
        return {
            "pairs": len(counts),
            "agreement": float(both.sum() / total.sum()) if len(counts) else None,
            "positive_agreement": (
                float((k * (k - 1)).sum() / (k * (n - 1)).sum()) if k.sum() else None
            ),
        }

    # one label per line and study, so ``same_study`` compares lines only
    per_line = rows.drop_duplicates(["study_id", "model_id", "gene_a", "gene_b"])
    return {
        "same_line": pairs(rows, "model_id"),
        "same_study": pairs(per_line, "study_id"),
    }


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
    """Per split, scorer and line: each headline screen's (genetic, both classes
    ranked) recall in the top 10%, AUROC (mean percentile of its positives) and AUPRC
    over the positive rate, averaged over a study's screens, then over the line's
    studies, so neither a large screen nor a study with many screens dominates."""
    keys = ["split", "scorer", "model_id"]
    columns = [*keys, "positives", "studies", "screens", *METRICS]
    two_class = (lists["ranked_positives"] > 0) & (lists["percentiles"] > 0)
    if not (two_class & (lists["condition"] == HEADLINE)).any():
        return pd.DataFrame(columns=columns)  # e.g. a stratum with one class per screen
    screens = lists[two_class & (lists["condition"] == HEADLINE)].assign(
        recall_top10=lambda x: x["top_credit"] / x["ranked_positives"],
        auroc=lambda x: x["percentile_sum"] / x["percentiles"],
    )
    studies = screens.groupby([*keys, "study_id"]).agg(
        **{m: (m, "mean") for m in METRICS},
        positives=("ranked_positives", "sum"),
        screens=("list_id", "size"),
    )
    lines = studies.groupby(keys).agg(
        **{m: (m, "mean") for m in METRICS},
        positives=("positives", "sum"),
        screens=("screens", "sum"),
        studies=("screens", "size"),
    )
    return lines.reset_index()[columns]


def screen_counts(lists: pd.DataFrame) -> list[dict[str, Any]]:
    """Per split and condition: screens, those with both classes, and positives."""
    one = lists[lists["scorer"] == "pair_identity"]
    one = one.assign(two_class=(one["ranked_positives"] > 0) & (one["percentiles"] > 0))
    counts = one.groupby(["split", "condition"]).agg(
        screens=("list_id", "size"),
        two_class=("two_class", "sum"),
        positives=("ranked_positives", "sum"),
    )
    return [
        {k: (v if isinstance(v, str) else int(v)) for k, v in row.items()}
        for row in counts.reset_index().to_dict("records")
    ]


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
        "labels": held["label"].to_numpy(dtype=np.float64),
        "random": np.random.default_rng(int(config["seed"])).random(len(held)),
    }
    if predicted is not None:
        scorers["stage_one"] = -line_values(predicted, held)
    # scores made outside this module (hpc/kg4sl, hpc/cilantro, sl-conditional, ...):
    # one per (gene_a, gene_b), with model_id and design when the table has them; a
    # held row the table lacks raises, a NaN score leaves the row unranked by that
    # scorer; an optional ``covered`` column marks the rows the model really scores
    covered = {}
    for name, path in config["extra_scores"].items():
        table = pd.read_parquet(path)
        keys = [k for k in ("model_id", "design", "gene_a", "gene_b") if k in table]
        index = pd.MultiIndex.from_frame(held[keys])
        missing = int((~index.isin(pd.MultiIndex.from_frame(table[keys]))).sum())
        if missing:
            raise ValueError(f"{path} has no row for {missing} held rows")
        values = table.set_index(keys).reindex(index)
        scorers[name] = values["score"].to_numpy(dtype=np.float64)
        if "covered" in values and not values["covered"].all():
            covered[name] = values["covered"].to_numpy(dtype=bool)
    held["stratum"] = strata(training, held)
    base = held[[*LIST_KEYS, "gene_a", "gene_b", "label", "stratum"]].assign(
        row=np.arange(len(held))
    )
    scores = pd.concat(
        [base.assign(scorer=name, score=values) for name, values in scorers.items()],
        ignore_index=True,
    )
    # the measured control exists only on lines with GeneEffect
    without = (scores["scorer"] == "measured_geneeffect") & ~scores["model_id"].isin(
        effects.index
    )
    scores = scores[~without].reset_index(drop=True)
    split_table = pd.read_csv(Path(config["split"]).with_suffix(".csv"), dtype=str)
    groups = line_groups(split_table, non_cancer_lines(Path(config["model"])))
    by_stratum, results = [], []
    for stratum in STRATA:
        chosen = scores if stratum == "all" else scores[scores["stratum"] == stratum]
        if chosen.empty:
            continue
        stratum_lists = list_records(chosen, int(config["top_k"]))
        stratum_lines = line_records(stratum_lists).assign(stratum=stratum)
        results += [
            row | {"stratum": stratum}
            for row in summary(
                stratum_lines, groups, int(config["bootstrap"]), int(config["seed"])
            )
        ]
        if not stratum_lines.empty:
            by_stratum.append(stratum_lines)
        if stratum == "all":
            lists = stratum_lists
    lines = pd.concat(by_stratum, ignore_index=True)
    # each partial scorer beside the training SL fraction on the rows it covers
    on_covered = {}
    for name, mask in covered.items():
        chosen = scores[
            scores["row"].isin(np.flatnonzero(mask))
            & scores["scorer"].isin([name, *COVERED_WITH])
        ]
        covered_lines = line_records(list_records(chosen, int(config["top_k"])))
        on_covered[name] = {
            "rows": int(mask.sum()),
            "share": float(mask.mean()),
            "summary": summary(
                covered_lines,
                {"all": groups["all"]},
                int(config["bootstrap"]),
                int(config["seed"]),
            ),
        }
    lines["group"] = lines["model_id"].map(split_table.set_index("model_id")["group"])
    metrics = {
        "run_id": run_id,
        "rows": {"training": len(training), "held": len(held), **dropped,
                 "measured_rows_without_geneeffect": int(without.sum())},
        "screens": screen_counts(lists),
        "strata": {k: int(v) for k, v in held["stratum"].value_counts().items()},
        "summary": results,
        "covered": on_covered,
        "label_agreement": label_agreement(rows),
    }  # fmt: skip
    write_parquet(run_dir / "scores.parquet", scores.drop(columns="row"))
    lists.to_csv(run_dir / "lists.csv", index=False)
    lines.to_csv(run_dir / "lines.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    """Table 1, recall in the top 10% per scorer on validation and test (all lines);
    Table 2, the difference from pair identity; Table 3, each stratum; Table 4, each
    partial scorer on the rows it covers. The other line groups, AUROC, AUPRC
    and each screen are in metrics.json, lines.csv and lists.csv."""
    rows = {
        (r["split"], r["scorer"]): r
        for r in metrics["summary"]
        if r["group"] == "all" and r["stratum"] == "all"
    }
    strata = {
        (r["stratum"], r["split"], r["scorer"]): r["recall_top10"]
        for r in metrics["summary"]
        if r["group"] == "all"
    }
    scorers = [s for s in NAMES if (("test", s) in rows)]
    best = {
        split: max(
            rows[(split, s)]["recall_top10"]
            for s in scorers
            if (split, s) in rows and s not in NOT_MODELS
        )
        for split in ("val", "test")
    }

    def cell(split: str, scorer: str) -> str:
        row = rows.get((split, scorer))
        if row is None:
            return "—"
        text = f"{row['recall_top10']:.3f}"
        return f"**{text}**" if row["recall_top10"] == best[split] else text

    def gap(split: str, scorer: str) -> str:
        row = rows.get((split, scorer))
        if row is None or row["minus_pair_identity"] is None:
            return "—"
        low, high = row["interval"]
        return f"{row['minus_pair_identity']:+.3f} [{low:+.3f}, {high:+.3f}]"

    n = {split: rows[(split, "pair_identity")]["lines"] for split in ("val", "test")}
    lines = [
        f"# SL benchmark baselines: {metrics['run_id']}",
        "",
        "**Table 1.** Recall in the top 10% on the multi-study SL benchmark, mean over "
        "held-out lines (each line: the mean over its studies of the mean over their "
        "genetic screens with both classes). Higher is better; a random ranking gives "
        "0.10. Best in bold.",
        "",
        f"| Model | Validation ({n['val']} lines) | Test ({n['test']} lines) |",
        "| --- | ---: | ---: |",
        *(f"| {NAMES[s]} | {cell('val', s)} | {cell('test', s)} |" for s in scorers),
        "",
        "**Table 2.** Difference from pair identity in recall in the top 10%, mean "
        "over lines, with a 95% bootstrap interval over lines. Higher is better.",
        "",
        "| Model − pair identity | Validation | Test |",
        "| --- | ---: | ---: |",
        *(f"| {NAMES[s]} | {gap('val', s)} | {gap('test', s)} |"
          for s in scorers if s not in {"pair_identity", "labels", "random"}),
        "",
        "**Table 3.** Recall in the top 10% for each stratum of held-out pairs, "
        "ranked inside the stratum of each screen, mean over lines (validation / "
        "test). Higher is better.",
        "",
        "| Model | " + " | ".join(STRATA[k] for k in STRATA if k != "all") + " |",
        "| --- |" + " ---: |" * (len(STRATA) - 1),
        *("| " + NAMES[s] + " | " + " | ".join(
            " / ".join(
                f"{strata[(k, split, s)]:.3f}" if (k, split, s) in strata else "—"
                for split in ("val", "test"))
            for k in STRATA if k != "all") + " |"
          for s in scorers),
        "",
        *covered_table(metrics["covered"]),
        *agreement_lines(metrics["label_agreement"]),
        "Other line groups, AUROC and AUPRC are in `metrics.json`; each line in "
        "`lines.csv`; each screen, the non-genetic ones included, in `lists.csv`.",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def agreement_lines(agreement: Mapping[str, Any]) -> list[str]:
    """The stage-two ceiling (``label_agreement``) as one sentence."""

    def text(entry: Mapping[str, Any]) -> str:
        if not entry["pairs"]:
            return "no pair labelled twice"
        positive = entry["positive_agreement"]
        return (
            f"{entry['pairs']:,} pairs, agreement {entry['agreement']:.3f}, SL given "
            "SL " + ("—" if positive is None else f"{positive:.3f}")
        )

    return [
        "Label agreement (the stage-two ceiling), within a line across lists: "
        f"{text(agreement['same_line'])}; within a study across lines: "
        f"{text(agreement['same_study'])}.",
        "",
    ]


def covered_table(covered: Mapping[str, Any]) -> list[str]:
    """Table 4: each scorer that scores only part of the rows, beside the training SL
    fraction and the labels on those rows (recall in the top 10%, AUROC)."""
    if not covered:
        return []

    def cell(summary: list[dict[str, Any]], split: str, scorer: str) -> str:
        row = next(
            (r for r in summary if (r["split"], r["scorer"]) == (split, scorer)), None
        )
        return "—" if row is None else f"{row['recall_top10']:.3f} ({row['auroc']:.2f})"

    columns = [(s, c) for s in ("val", "test") for c in ("model", *COVERED_WITH[:2])]
    head = {"model": "model", "pair_identity": "training SL fraction",
            "labels": "labels"}  # fmt: skip
    split_name = {"val": "Validation", "test": "Test"}
    return [
        "**Table 4.** Recall in the top 10% (AUROC) of each model that scores only "
        "part of the held-out rows, on those rows only, beside the training SL "
        "fraction and the labels (the ceiling) on the same rows; mean over lines. "
        "Higher is better.",
        "",
        "| Model | Rows covered | "
        + " | ".join(f"{split_name[s]}, {head[c]}" for s, c in columns) + " |",
        "| --- | ---: |" + " ---: |" * len(columns),
        *(f"| {NAMES.get(name, name)} | {entry['rows']:,} ({entry['share']:.1%}) | "
          + " | ".join(cell(entry["summary"], s, name if c == "model" else c)
                       for s, c in columns) + " |"
          for name, entry in covered.items()),
        "",
    ]  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
