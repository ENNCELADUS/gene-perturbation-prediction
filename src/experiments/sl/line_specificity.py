"""Line-specific signal in the multi-study SL benchmark's labels, before any model:
does the difference between lines in a pair's screen score repeat? Rows are the
genetic screens with both classes on the split's lines (every side: nothing is
fitted). A screen's score becomes a percentile rank inside the screen, turned so that
a study's SL calls are on the high side, and the lists of one study and line are
averaged. Table 1, for each study with ``min_lines`` lines: the share of the variance
that lies between pairs (one-way intraclass correlation), for the rank and for the
call. Table 2, for each two studies that share a line: the correlation over pairs of
one study's line with the same line and with another line of the other study, and
the repeat of the line-specific part (a pair's deviation from its mean over the lines
that both studies screened it in), with a bootstrap over pairs. The square root of
that repeat estimates the top score of a perfect model against one screen. Study
pairs are compared in parallel processes."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json, write_text
from src.data.splits import load_geneeffect_split
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind, number
from src.experiments.sl.benchmark_baselines import HEADLINE

PAIR = ["gene_a", "gene_b"]


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "labels", "split", "bootstrap",
         "min_pairs", "min_lines"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_line_specificity":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def study_cells(
    labels: pd.DataFrame, lines: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (study, line, unordered gene pair): ``value``, the mean rank of
    the pair in the study's screens of the line (high = SL), and ``label``, SL when
    any of them calls it. Second frame: each study's screens, lines and score side."""
    rows = labels[
        labels["label"].notna()
        & (labels["condition"] == HEADLINE)
        & labels["model_id"].isin(lines)
        & labels["score"].notna()
    ]
    rows = rows.assign(label=rows["label"].astype(int))
    rows = rows[rows.groupby("list_id")["label"].transform("nunique") == 2]
    rows = rows.assign(rank=rows.groupby("list_id")["score"].rank(pct=True))
    by_class = rows.groupby(["study_id", "list_id", "label"])["rank"].mean().unstack()
    # a study's score direction: the side its SL calls are on, mean over its screens
    gap = (by_class[1] - by_class[0]).groupby("study_id").mean()
    low = rows["study_id"].map(gap).to_numpy() < 0
    first = rows["gene_a"] < rows["gene_b"]
    rows = rows.assign(
        value=np.where(low, 1 - rows["rank"], rows["rank"]),
        gene_a=np.where(first, rows["gene_a"], rows["gene_b"]),
        gene_b=np.where(first, rows["gene_b"], rows["gene_a"]),
    )
    cells = rows.groupby(["study_id", "model_id", *PAIR], sort=False).agg(
        value=("value", "mean"), label=("label", "max")
    )
    studies = rows.groupby("study_id").agg(
        design=("design", lambda x: "/".join(sorted(set(x)))),
        screens=("list_id", "nunique"),
        lines=("model_id", "nunique"),
    )
    studies["rank_gap"], studies["sl_on_low_side"] = gap.abs(), gap < 0
    return cells.reset_index(), studies.reset_index()


def pair_share(matrix: np.ndarray) -> float:
    """Share of the variance between pairs: the one-way intraclass correlation of a
    lines x pairs matrix (NaN = not screened), over the pairs in two or more lines."""
    seen = np.isfinite(matrix)
    keep = seen.sum(0) >= 2
    seen = seen[:, keep]
    x = np.where(seen, matrix[:, keep], 0.0)
    n, pairs = seen.sum(0), int(keep.sum())
    total = n.sum()
    if pairs < 2 or total == pairs:
        return float("nan")
    mean = x.sum(0) / n
    between = (n * (mean - x.sum() / total) ** 2).sum() / (pairs - 1)
    within = (((x - mean) * seen) ** 2).sum() / (total - pairs)
    size = (total - (n**2).sum() / total) / (pairs - 1)
    return float((between - within) / (between + (size - 1) * within))


def correlations(
    a: np.ndarray, b: np.ndarray, weight: np.ndarray, min_pairs: int
) -> np.ndarray:
    """Pearson correlation of each row of ``a`` with each row of ``b`` over the
    columns (pairs) both have, each column with a weight; NaN under ``min_pairs``
    common columns (counted without the weights)."""
    seen_a, seen_b = np.isfinite(a), np.isfinite(b)
    a, b = np.where(seen_a, a, 0.0), np.where(seen_b, b, 0.0)
    weighted, count = seen_a * weight, seen_b.astype(float)
    n = weighted @ count.T
    sum_a, sum_b = (a * weight) @ count.T, weighted @ b.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (a * weight) @ b.T - sum_a * sum_b / n
        var_a = (a**2 * weight) @ count.T - sum_a**2 / n
        var_b = weighted @ (b**2).T - sum_b**2 / n
        r = cov / np.sqrt(var_a * var_b)
    r[seen_a.astype(float) @ count.T < min_pairs] = np.nan
    return r


def mean_of(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def line_part(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Two studies on the same lines (rows): each pair's sums of products of the two
    studies' deviations from the pair's mean over the lines both screened it in (two
    or more), and the two means themselves."""
    both = np.isfinite(a) & np.isfinite(b)
    keep = both.sum(0) >= 2
    both = both[:, keep]
    n = both.sum(0)

    def parts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.where(both, matrix[:, keep], 0.0)
        mean = matrix.sum(0) / n
        return (matrix - mean) * both, mean

    (da, mean_a), (db, mean_b) = parts(a), parts(b)
    return {"ab": (da * db).sum(0), "aa": (da**2).sum(0), "bb": (db**2).sum(0),
            "mean_a": mean_a, "mean_b": mean_b, "keep": keep}  # fmt: skip


def ratio(parts: Mapping[str, np.ndarray], weight: np.ndarray) -> float:
    """The correlation of the two studies' line-specific parts, pairs weighted."""
    scale = (weight @ parts["aa"]) * (weight @ parts["bb"])
    return float(weight @ parts["ab"] / np.sqrt(scale)) if scale > 0 else float("nan")


def positive_agreement(
    a: np.ndarray, b: np.ndarray, match: np.ndarray
) -> tuple[float, float]:
    """The chance that a second call is SL when the first is (2 x both SL over the SL
    calls of each side that the other side tested), for the same line and for another
    line of the other study."""
    seen_a, seen_b = np.isfinite(a).astype(float), np.isfinite(b).astype(float)
    sl_a, sl_b = (a == 1).astype(float), (b == 1).astype(float)
    both, calls = sl_a @ sl_b.T, sl_a @ seen_b.T + seen_a @ sl_b.T
    return tuple(
        float(2 * both[m].sum() / calls[m].sum()) if calls[m].sum() else float("nan")
        for m in (match, ~match)
    )


def compare(task: Mapping[str, Any]) -> dict[str, Any]:
    """One pair of studies (lines x common pairs of each): Table 2's row and its
    bootstrap draws."""
    a, b = task["value_a"], task["value_b"]
    lines_a, lines_b = task["lines_a"], task["lines_b"]
    match = np.array(lines_a)[:, None] == np.array(lines_b)[None, :]
    pairs, min_pairs = a.shape[1], task["min_pairs"]

    def same_other(weight: np.ndarray) -> tuple[float, float]:
        r = correlations(a, b, weight, min_pairs)
        return mean_of(r[match]), mean_of(r[~match])

    shared = [line for line in lines_a if line in lines_b]
    parts = line_part(
        a[[lines_a.index(line) for line in shared]],
        b[[lines_b.index(line) for line in shared]],
    )
    ones = np.ones(pairs)
    same, other = same_other(ones)
    valid = np.isfinite(correlations(a, b, ones, min_pairs))
    rng = np.random.default_rng(task["seed"])
    draws = np.empty((task["bootstrap"], 2))
    for draw in draws:
        weight = rng.multinomial(pairs, np.full(pairs, 1 / pairs)).astype(float)
        s, o = same_other(weight)
        draw[:] = s - o, ratio(parts, weight[parts["keep"]])
    kept = int(parts["keep"].sum())
    agreement = positive_agreement(task["label_a"], task["label_b"], match)
    return {
        "first": task["first"],
        "second": task["second"],
        "shared_lines": len(shared),
        "pairs": pairs,
        "same_line_comparisons": int(valid[match].sum()),
        "other_line_comparisons": int(valid[~match].sum()),
        "same_line": same,
        "other_line": other,
        "difference": same - other,
        "line_part_pairs": kept,
        "line_part_repeat": ratio(parts, np.ones(kept)),
        "pair_mean_repeat": (
            float(np.corrcoef(parts["mean_a"], parts["mean_b"])[0, 1])
            if kept > 2
            else float("nan")
        ),  # fmt: skip
        "positive_agreement_same_line": agreement[0],
        "positive_agreement_other_line": agreement[1],
        "draws": draws,
    }


def interval(draws: np.ndarray) -> list[float] | None:
    draws = draws[np.isfinite(draws)]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])] if len(draws) else None


def run(config: Mapping[str, Any], run_id: str, workers: int) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split = load_geneeffect_split(Path(config["split"]))
    columns = ["study_id", "list_id", "model_id", "design", "condition", *PAIR,
               "score", "label"]  # fmt: skip
    labels = pd.read_parquet(config["labels"], columns=columns)
    cells, studies = study_cells(labels, set(split.all_model_ids))
    wide = {
        study: (
            group.pivot(index="model_id", columns=PAIR, values="value"),
            group.pivot(index="model_id", columns=PAIR, values="label"),
        )
        for study, group in cells.groupby("study_id")
    }
    within = [
        {"study_id": study, "lines": len(value),
         "pairs_in_two_lines": int((value.notna().sum() >= 2).sum()),
         "pair_share_rank": pair_share(value.to_numpy(float)),
         "pair_share_call": pair_share(label.to_numpy(float))}
        for study, (value, label) in wide.items()
        if len(value) >= config["min_lines"]
    ]  # fmt: skip
    tasks = []
    for first, second in combinations(sorted(wide), 2):
        (value_a, label_a), (value_b, label_b) = wide[first], wide[second]
        common = value_a.columns.intersection(value_b.columns)
        if len(common) < config["min_pairs"] or not set(value_a.index) & set(
            value_b.index
        ):
            continue
        tasks.append(
            {"first": first, "second": second, "min_pairs": config["min_pairs"],
             "bootstrap": config["bootstrap"], "seed": [config["seed"], len(tasks)],
             "lines_a": list(value_a.index), "lines_b": list(value_b.index),
             "value_a": value_a.reindex(columns=common).to_numpy(float),
             "value_b": value_b.reindex(columns=common).to_numpy(float),
             "label_a": label_a.reindex(columns=common).to_numpy(float),
             "label_b": label_b.reindex(columns=common).to_numpy(float)}
        )  # fmt: skip
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        compared = [r for r in pool.map(compare, tasks) if np.isfinite(r["same_line"])]
    draws = np.stack([r.pop("draws") for r in compared]) if compared else None
    pooled = {}
    for r, d in zip(compared, draws if compared else []):
        r["difference_interval"] = interval(d[:, 0])
        r["line_part_repeat_interval"] = interval(d[:, 1])
    if compared:
        with np.errstate(invalid="ignore"):
            mean_draws = np.nanmean(draws, axis=0)
        repeat = mean_of(np.array([r["line_part_repeat"] for r in compared]))
        pooled = {
            "study_pairs": len(compared),
            "same_line": mean_of(np.array([r["same_line"] for r in compared])),
            "other_line": mean_of(np.array([r["other_line"] for r in compared])),
            "difference": mean_of(np.array([r["difference"] for r in compared])),
            "difference_interval": interval(mean_draws[:, 0]),
            "line_part_repeat": repeat,
            "line_part_repeat_interval": interval(mean_draws[:, 1]),
            "ceiling": float(np.sqrt(repeat)) if repeat > 0 else None,
        }
    metrics = {
        "run_id": run_id,
        "workers": workers,
        "rows": {"cells": len(cells), "studies": len(studies),
                 "lines": int(cells["model_id"].nunique())},
        "studies": studies.to_dict("records"),
        "within_study": within,
        "study_pairs": compared,
        "pooled": pooled,
    }  # fmt: skip
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    def ranged(row: Mapping[str, Any], key: str) -> str:
        bounds = row[f"{key}_interval"]
        if bounds is None:
            return number(row[key])
        return f"{number(row[key], 3)} [{number(bounds[0], 3)}, {number(bounds[1], 3)}]"

    rows = metrics["rows"]
    lines = [
        f"# Line-specific signal in the SL labels ({metrics['run_id']})",
        "",
        f"{rows['studies']} studies, {rows['lines']} lines, {rows['cells']} (study, "
        "line, pair) values: genetic screens with both classes and a score, each "
        "score a percentile rank inside its screen (high = SL).",
        "",
        "**Table 1.** Share of the variance between pairs (intraclass correlation "
        "over the pairs that two or more lines of the study screened), for the rank "
        "and for the SL call. The remainder is the line-specific part plus noise.",
        "",
        "| Study | Lines | Pairs in two or more lines | Rank | SL call |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {r['study_id']} | {r['lines']} | {r['pairs_in_two_lines']} | "
            f"{number(r['pair_share_rank'], 3)} | {number(r['pair_share_call'], 3)} |"
            for r in sorted(metrics["within_study"], key=lambda r: -r["lines"])
        ),
        "",
        "**Table 2.** Two studies that share a line. Same line / another line: mean "
        "correlation over the common pairs between a line of one study and the same "
        "line, or another line, of the other study. Repeat of the line-specific "
        "part: correlation between the two studies of a pair's deviation from its "
        "mean over the shared lines (needs two shared lines). 95% bootstrap over "
        "pairs. Positive agreement: the chance that a second call is SL when the "
        "first is, same line / another line. Higher is better in every column.",
        "",
        "| Studies | Shared lines | Common pairs | Same line | Another line | "
        "Same − another | Repeat of the line-specific part | Positive agreement |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        *(
            f"| {r['first']} × {r['second']} | {r['shared_lines']} | {r['pairs']} | "
            f"{number(r['same_line'], 3)} | {number(r['other_line'], 3)} | "
            f"{ranged(r, 'difference')} | {ranged(r, 'line_part_repeat')} | "
            f"{number(r['positive_agreement_same_line'], 3)} / "
            f"{number(r['positive_agreement_other_line'], 3)} |"
            for r in sorted(metrics["study_pairs"], key=lambda r: -r["pairs"])
        ),
    ]
    pooled = metrics["pooled"]
    if pooled:
        lines += [
            f"| **Mean over the {pooled['study_pairs']} study pairs** | | | "
            f"{number(pooled['same_line'], 3)} | {number(pooled['other_line'], 3)} | "
            f"{ranged(pooled, 'difference')} | "
            f"{ranged(pooled, 'line_part_repeat')} | |",
            "",
            "Ceiling of a line-specific metric (square root of the mean repeat): "
            f"{number(pooled['ceiling'], 3)}.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
