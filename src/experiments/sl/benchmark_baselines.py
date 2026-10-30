"""Baselines on the multi-study SL benchmark (docs/specs/2026-10-29-multi-study-sl-
benchmark-design.md): each held-out (study, line) list is ranked by a scorer and scored
as AUPRC over the list's positive rate; the headline is the mean over a split's lines
of the mean over each line's lists. Scorers: the pair's hit rate in the same study's
training lines (pair identity), each gene's mean GeneEffect over the training lines
(pan-essentiality), the linear context prior's predicted GeneEffect in the line, and
the measured GeneEffect in the line as a reference, never a bar."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json, write_parquet, write_text
from src.data.labels import parse_gene_symbol
from src.data.prior_offsets import read_prior_offsets
from src.data.splits import FixedSplit, assert_fit_eligible, load_geneeffect_split
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind
from src.experiments.sl.stage_two import ranking_metrics

SIDES = ("train", "val", "test")
LIST_KEYS = ["split", "design", "study_id", "list_id", "model_id"]


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "labels", "split", "gene_effect",
         "prior", "top_k", "bootstrap"},
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
    a = value("gene_a")
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


def prior_effects(path: Path, mean: pd.Series) -> pd.DataFrame:
    """The prior export as GeneEffect: each gene's training mean plus the residual
    prediction times the residual SD it was fitted with."""
    prior = read_prior_offsets(path)
    genes = list(prior.values.columns)
    return prior.values * prior.residual_scale[genes] + mean.reindex(genes)


def list_records(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per held-out list and scorer, the ranking metrics."""
    records = []
    for keys, group in scores.groupby([*LIST_KEYS, "scorer"], sort=True):
        record = dict(zip([*LIST_KEYS, "scorer"], keys))
        metrics = ranking_metrics(
            group["label"].to_numpy(), group["score"].to_numpy(), k
        )
        records.append(record | metrics)
    return pd.DataFrame(records)


def summary(lists: pd.DataFrame, reps: int, seed: int) -> list[dict[str, Any]]:
    """Per split and scorer: the mean over lines of each line's mean AUPRC lift, with
    a 95% interval from resampling lines."""
    rng = np.random.default_rng(seed)
    scored = lists[lists["auprc_lift"].notna()]
    out = []
    for (split, scorer), group in scored.groupby(["split", "scorer"], sort=True):
        per_line = group.groupby("model_id")["auprc_lift"].mean().to_numpy()
        draws = [rng.choice(per_line, len(per_line)).mean() for _ in range(reps)]
        low, high = np.quantile(draws, [0.025, 0.975])
        out.append(
            {"split": split, "scorer": scorer, "lines": len(per_line),
             "lists": len(group), "auprc_lift": float(per_line.mean()),
             "interval": [float(low), float(high)],
             "auroc": float(group.groupby("model_id")["auroc"].mean().mean())}
        )  # fmt: skip
    return out


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
    predicted = prior_effects(Path(config["prior"]), mean) if config["prior"] else None
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
        scorers["prior_a_blind"] = -line_values(predicted, held)
    base = held[[*LIST_KEYS, "gene_a", "gene_b", "label"]]
    scores = pd.concat(
        [base.assign(scorer=name, score=values) for name, values in scorers.items()],
        ignore_index=True,
    )
    lists = list_records(scores, int(config["top_k"]))
    metrics = {
        "run_id": run_id,
        "rows": {"training": len(training), "held": len(held), **dropped},
        "lists_without_both_classes": int(
            lists.loc[lists["scorer"] == "pair_identity", "auprc_lift"].isna().sum()
        ),
        "summary": summary(lists, int(config["bootstrap"]), int(config["seed"])),
    }
    write_parquet(run_dir / "scores.parquet", scores)
    lists.to_csv(run_dir / "lists.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        f"# SL benchmark baselines: {metrics['run_id']}",
        "",
        "AUPRC over the list's positive rate (1 is a random ranking), averaged within "
        "each line, then over lines; 95% interval from resampling lines.",
        "",
        "| Scorer | Split | Lines | Lists | AUPRC lift | 95% interval | AUROC |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in metrics["summary"]:
        low, high = row["interval"]
        lines.append(
            f"| {row['scorer']} | {row['split']} | {row['lines']} | {row['lists']} | "
            f"{row['auprc_lift']:.3f} | [{low:.3f}, {high:.3f}] | {row['auroc']:.3f} |"
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
