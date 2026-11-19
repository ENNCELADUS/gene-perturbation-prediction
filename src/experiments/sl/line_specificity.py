"""Line-specific signal in the multi-study SL benchmark's labels, before any model:
does the difference between lines in a pair's screen score repeat? Rows are the
genetic screens with both classes on the split's lines (every side: nothing is
fitted). A screen's score becomes a percentile rank inside the screen, turned so that
a study's SL calls are on the high side, and the lists of one study and line are
averaged. Table 1, for each study with ``min_lines`` lines: the share of the variance
that lies between pairs (one-way intraclass correlation), for the rank and for the
call. Tables 2 (rank) and 3 (SL call), for each two studies that share a line: the
correlation over pairs of one study's line with the same line and with another line
of the other study, the same-line advantage (that contrast with each screen's own
level removed, so a screen of low quality does not decide it), and the repeat of the
line-specific part (a pair's deviation from its mean over the lines that both studies
screened it in), with a bootstrap over pairs. The square root of that repeat
estimates the top score of a perfect model against one screen. Study pairs are
compared in parallel processes."""

from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
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
from src.experiments.sl.benchmark_baselines import HEADLINE, TOP

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


def oriented_rank(rows: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """Each row's percentile rank inside its screen (``list_id``), turned so that the
    study's SL calls are on the high side (NaN without a score), and each study's
    gap: the mean rank of its SL calls minus that of its other rows, mean over its
    screens, before the turn."""
    rank = rows.groupby("list_id")["score"].rank(pct=True)
    by_class = rows.assign(rank=rank).groupby(["study_id", "list_id", "label"])["rank"]
    by_class = by_class.mean().unstack()
    gap = (by_class[1] - by_class[0]).groupby("study_id").mean()
    low = rows["study_id"].map(gap).to_numpy() < 0
    # a turned screen ranks from the other end, so its top row also reads 1
    turned = rows.groupby("list_id")["score"].rank(pct=True, ascending=False)
    return np.where(low, turned, rank), gap


def ramp(value: np.ndarray) -> np.ndarray:
    """The top 10% of a rank as a ramp from 0 to 1; 0 below it."""
    return np.clip((value - (1 - TOP)) / TOP, 0, 1)


def study_cells(
    labels: pd.DataFrame, lines: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (study, line, unordered gene pair): ``value``, the mean rank of
    the pair in the study's screens of the line (high = SL), ``tail``, the mean of the
    rank's top 10% as a ramp, and ``label``, SL when any of them calls it. Second
    frame: each study's screens, lines and score side."""
    rows = labels[
        labels["label"].notna()
        & (labels["condition"] == HEADLINE)
        & labels["model_id"].isin(lines)
        & labels["score"].notna()
    ]
    rows = rows.assign(label=rows["label"].astype(int))
    rows = rows[rows.groupby("list_id")["label"].transform("nunique") == 2]
    value, gap = oriented_rank(rows)
    first = rows["gene_a"] < rows["gene_b"]
    rows = rows.assign(
        value=value,
        tail=ramp(value),
        gene_a=np.where(first, rows["gene_a"], rows["gene_b"]),
        gene_b=np.where(first, rows["gene_b"], rows["gene_a"]),
    )
    cells = rows.groupby(["study_id", "model_id", *PAIR], sort=False).agg(
        value=("value", "mean"), tail=("tail", "mean"), label=("label", "max")
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


#: What is compared: the rank, the rank's top 10% as a ramp from 0 to 1 (0 below it),
#: and the SL call.
KINDS = {"rank": "value", "tail": "tail", "call": "label"}


def advantage(r: np.ndarray, match: np.ndarray) -> float:
    """Same line against another line with each screen's level removed: for each
    matched cell of a lines x lines correlation matrix, the cell minus the mean of
    its row's unmatched cells, minus that of its column's, plus that of the unmatched
    cells in neither. It equals same minus other when every line is shared."""
    other = np.where(match, np.nan, r)
    rows, columns = np.arange(r.shape[0]), np.arange(r.shape[1])
    values = [
        r[i, j]
        - mean_of(other[i])
        - mean_of(other[:, j])
        + mean_of(other[np.ix_(rows != i, columns != j)])
        for i, j in zip(*np.nonzero(match))
    ]
    return mean_of(np.array(values))


def compare(task: Mapping[str, Any]) -> dict[str, Any]:
    """One pair of studies (lines x common pairs of each, for the rank and for the
    call): its row of Tables 2 and 3 and its bootstrap draws."""
    lines_a, lines_b = task["lines_a"], task["lines_b"]
    match = np.array(lines_a)[:, None] == np.array(lines_b)[None, :]
    shared = [line for line in lines_a if line in lines_b]
    rows_a = [lines_a.index(line) for line in shared]
    rows_b = [lines_b.index(line) for line in shared]
    pairs, min_pairs = task["value_a"].shape[1], task["min_pairs"]
    ones = task["weight"]  # each pair's weight: 1, or the caller's
    out = {"first": task["first"], "second": task["second"],
           "shared_lines": len(shared), "pairs": pairs}  # fmt: skip
    parts = {}
    for kind, column in KINDS.items():
        a, b = task[f"{column}_a"], task[f"{column}_b"]
        parts[kind] = line_part(a[rows_a], b[rows_b])
        r = correlations(a, b, ones, min_pairs)
        kept = int(parts[kind]["keep"].sum())
        out[kind] = {
            "same_line_comparisons": int(np.isfinite(r[match]).sum()),
            "other_line_comparisons": int(np.isfinite(r[~match]).sum()),
            "same_line": mean_of(r[match]),
            "other_line": mean_of(r[~match]),
            "advantage": advantage(r, match),
            "line_part_pairs": kept,
            "line_part_repeat": ratio(parts[kind], ones[parts[kind]["keep"]]),
        }
    agreement = positive_agreement(task["label_a"], task["label_b"], match)
    out["call"]["positive_agreement_same_line"] = agreement[0]
    out["call"]["positive_agreement_other_line"] = agreement[1]
    rng = np.random.default_rng(task["seed"])
    draws = np.empty((task["bootstrap"], len(KINDS), 2))
    for draw in draws:
        weight = ones * rng.multinomial(pairs, np.full(pairs, 1 / pairs))
        for k, (kind, column) in enumerate(KINDS.items()):
            r = correlations(
                task[f"{column}_a"], task[f"{column}_b"], weight, min_pairs
            )
            draw[k] = (
                advantage(r, match),
                ratio(parts[kind], weight[parts[kind]["keep"]]),
            )
    return out | {"draws": draws}


def interval(draws: np.ndarray) -> list[float] | None:
    draws = draws[np.isfinite(draws)]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])] if len(draws) else None


def analyse(
    cells: pd.DataFrame,
    config: Mapping[str, Any],
    workers: int,
    keep: Callable[[str, str], bool] = lambda first, second: True,
    weight: pd.Series | None = None,
) -> dict[str, Any]:
    """Tables 1 to 3 from the (study, line, pair) values: ``within_study``,
    ``study_pairs`` and ``pooled``. ``keep`` chooses the study pairs; ``weight`` gives
    a pair's weight in every correlation (index: gene_a, gene_b; default 1)."""
    wide = {
        study: {
            column: group.pivot(index="model_id", columns=PAIR, values=column)
            for column in KINDS.values()
        }
        for study, group in cells.groupby("study_id")
    }
    within = [
        {"study_id": study, "lines": len(tables["value"]),
         "pairs_in_two_lines": int((tables["value"].notna().sum() >= 2).sum()),
         "pair_share_rank": pair_share(tables["value"].to_numpy(float)),
         "pair_share_call": pair_share(tables["label"].to_numpy(float))}
        for study, tables in wide.items()
        if len(tables["value"]) >= config["min_lines"]
    ]  # fmt: skip
    tasks = []
    for first, second in combinations(sorted(wide), 2):
        a, b = wide[first], wide[second]
        common = a["value"].columns.intersection(b["value"].columns)
        if (
            not keep(first, second)
            or len(common) < config["min_pairs"]
            or not set(a["value"].index) & set(b["value"].index)
        ):
            continue
        base = np.ones(len(common)) if weight is None else weight.reindex(common)
        tasks.append(
            {"first": first, "second": second, "min_pairs": config["min_pairs"],
             "bootstrap": config["bootstrap"], "seed": [config["seed"], len(tasks)],
             "weight": np.nan_to_num(np.asarray(base, float)),
             "lines_a": list(a["value"].index), "lines_b": list(b["value"].index)}
            | {f"{column}_{side}": t[column].reindex(columns=common).to_numpy(float)
               for column in KINDS.values() for side, t in (("a", a), ("b", b))}
        )  # fmt: skip
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        compared = [
            r for r in pool.map(compare, tasks) if np.isfinite(r["rank"]["same_line"])
        ]
    pooled = {}
    if compared:
        draws = np.stack([r.pop("draws") for r in compared])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a column of NaN
            mean_draws = np.nanmean(draws, axis=0)
        for k, kind in enumerate(KINDS):
            for r, d in zip(compared, draws):
                r[kind]["advantage_interval"] = interval(d[:, k, 0])
                r[kind]["line_part_repeat_interval"] = interval(d[:, k, 1])
            entries = pd.DataFrame([r[kind] for r in compared])
            repeat = mean_of(entries["line_part_repeat"].to_numpy(float))
            pooled[kind] = {
                "study_pairs": len(compared),
                "study_pairs_with_advantage": int(entries["advantage"].count()),
                "study_pairs_with_repeat": int(entries["line_part_repeat"].count()),
                "same_line": mean_of(entries["same_line"].to_numpy(float)),
                "other_line": mean_of(entries["other_line"].to_numpy(float)),
                "advantage": mean_of(entries["advantage"].to_numpy(float)),
                "advantage_interval": interval(mean_draws[:, k, 0]),
                "line_part_repeat": repeat,
                "line_part_repeat_interval": interval(mean_draws[:, k, 1]),
                "ceiling": float(np.sqrt(repeat)) if repeat > 0 else None,
            }  # fmt: skip
    return {"within_study": within, "study_pairs": compared, "pooled": pooled}


COLUMNS = ["study_id", "list_id", "model_id", "design", "condition", *PAIR,
           "score", "label"]  # fmt: skip


def run(config: Mapping[str, Any], run_id: str, workers: int) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split = load_geneeffect_split(Path(config["split"]))
    labels = pd.read_parquet(config["labels"], columns=COLUMNS)
    cells, studies = study_cells(labels, set(split.all_model_ids))
    metrics = {
        "run_id": run_id,
        "workers": workers,
        "rows": {"cells": len(cells), "studies": len(studies),
                 "lines": int(cells["model_id"].nunique())},
        "studies": studies.to_dict("records"),
        **analyse(cells, config, workers),
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

    def table(kind: str, caption: str) -> list[str]:
        agreement = kind == "call"
        out = [
            caption,
            "",
            "| Studies | Shared lines | Common pairs | Same line | Another line | "
            "Same-line advantage | Repeat of the line-specific part |"
            + (" Positive agreement |" if agreement else ""),
            "| --- | ---: | ---: | ---: | ---: | --- | --- |"
            + (" --- |" if agreement else ""),
        ]
        for r in sorted(metrics["study_pairs"], key=lambda r: -r["pairs"]):
            e = r[kind]
            out.append(
                f"| {r['first']} × {r['second']} | {r['shared_lines']} | {r['pairs']} "
                f"| {number(e['same_line'], 3)} | {number(e['other_line'], 3)} | "
                f"{ranged(e, 'advantage')} | {ranged(e, 'line_part_repeat')} |"
                + (
                    f" {number(e['positive_agreement_same_line'], 3)} / "
                    f"{number(e['positive_agreement_other_line'], 3)} |"
                    if agreement
                    else ""
                )
            )
        pooled = metrics["pooled"].get(kind)
        if pooled:
            out += [
                f"| **Mean over the study pairs** ({pooled['study_pairs']}; advantage "
                f"{pooled['study_pairs_with_advantage']}; repeat "
                f"{pooled['study_pairs_with_repeat']}) | | | "
                f"{number(pooled['same_line'], 3)} | {number(pooled['other_line'], 3)} "
                f"| {ranged(pooled, 'advantage')} | "
                f"{ranged(pooled, 'line_part_repeat')} |" + (" |" if agreement else ""),
                "",
                "Ceiling of a line-specific metric (square root of the mean repeat): "
                f"{number(pooled['ceiling'], 3)}.",
            ]
        return [*out, ""]

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
        *table(
            "rank",
            "**Table 2.** The rank, for two studies that share a line. Same line / "
            "another line: mean correlation over the common pairs between a line of "
            "one study and the same line, or another line, of the other study. "
            "Same-line advantage: that contrast with each screen's own level removed "
            "(needs a second line in each study). Repeat of the line-specific part: "
            "correlation between the two studies of a pair's deviation from its mean "
            "over the shared lines (needs two shared lines). 95% bootstrap over "
            "pairs. Higher is better in every column.",
        ),
        *table(
            "call",
            "**Table 3.** The SL call (1 or 0) in place of the rank, same columns. "
            "Positive agreement: the chance that a second call is SL when the first "
            "is, same line / another line.",
        ),
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
