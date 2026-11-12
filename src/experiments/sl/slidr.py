"""SLIdR (Srivatsa et al. 2022, Nature Communications), ported from the official R
code (``third_party/slidr/R``, ``identifySLHits`` and ``prepareDataObjects``) and run
pan-cancer on the SL split's training-side lines. For a mutated (driver) gene, each
line's viabilities are ranked over all genes; a partner's ranks are summed over the
mutated lines and divided by the gene count, and the Irwin-Hall CDF of that sum is the
mutated-lines p-value (small: the partner is more essential where the driver is
mutated). The same test on the wild-type lines, two-sided, filters pairs that are
essential everywhere (R's ``WT_pval_thresh`` 0.1).

A benchmark pair is tested in both orientations (each gene as the driver). Its score is
-log10 of the smaller mutated-lines p-value among the orientations that pass the
wild-type filter, 0 (p = 1) when every tested orientation fails it, and -log10 0.5
(``covered`` false) when neither gene has enough mutated and wild-type lines or the
partner has no viability. The score reads no held-out line. Deviations from R are in
``third_party/slidr/PROVENANCE.md``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from math import comb, factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.depmap import read_depmap_matrix
from src.data.io import write_json, write_parquet
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind

#: SL split sides whose lines are fitted, and the held-out sides that are scored.
FIT_SIDES = ("train", "bulk_labelled")
HELD_SIDES = ("val", "test")
#: R's normal approximation starts above this number of lines (``IH_CDF``).
EXACT_MAX = 20
#: Score of a pair that no orientation tests: p = 0.5, no evidence either way.
UNTESTED = -np.log10(0.5)
#: Smallest p-value kept, so that -log10 stays finite.
TINY = 1e-300


def irwin_hall_cdf(x: np.ndarray, n: int) -> np.ndarray:
    """R ``IH_CDF``: the CDF at ``x`` of the sum of n uniforms, by its alternating sum
    for n <= 20, else by the normal approximation N(n/2, n/12)."""
    from scipy.stats import norm

    x = np.asarray(x, np.float64)
    if n > EXACT_MAX:
        return norm.cdf(x, n / 2, np.sqrt(n / 12))
    total = np.zeros_like(x)
    for k in range(n + 1):
        on = k <= np.floor(x)
        total += np.where(on, (-1.0) ** k * comb(n, k) * np.maximum(x - k, 0) ** n, 0)
    return np.clip(total / factorial(n), 0.0, 1.0)


def mutation_matrix(
    lines: Sequence[str],
    genes: Sequence[str],
    mutated: Sequence[pd.DataFrame],
    log2_cn: pd.DataFrame,
    min_mutated: int,
) -> pd.DataFrame:
    """``prepareDataObjects`` with ``CN_Thr = 1``: a gene is altered in a line when a
    mutation table calls it (count > 0) or its copy number rounds to 0 copies; lines
    with no alteration drop out, then genes with fewer than ``min_mutated`` altered or
    wild-type lines. DepMap's copy number is log2(relative CN + 1)."""
    altered = pd.DataFrame(False, index=list(lines), columns=list(genes))
    for table in mutated:
        # NaN (no call) compares False
        altered |= table.reindex(index=altered.index, columns=altered.columns) > 0
    relative = 2.0 ** log2_cn.reindex(index=altered.index, columns=altered.columns) - 1
    altered |= np.round(2 * relative) == 0
    altered = altered.loc[altered.any(axis=1)]
    count = altered.sum(axis=0)
    keep = (count >= min_mutated) & (len(altered) - count >= min_mutated)
    return altered.loc[:, keep]


def pair_pvalues(
    viability: pd.DataFrame, altered: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """For each (driver, partner) row of ``pairs`` with both in the data: R's
    mutated-lines p-value and two-sided wild-type p-value. ``viability`` is lines x
    genes (NaN filled with the gene's mean, as R does); ``altered`` lines x drivers."""
    from scipy.stats import rankdata

    lines = altered.index
    values = viability.loc[lines]
    values = values.fillna(values.mean())
    # a line's genes ranked by viability (1 = most essential), / the gene count
    ranks = rankdata(values.to_numpy(np.float64), axis=1) / values.shape[1]
    total = ranks.sum(axis=0)
    partner = values.columns.get_indexer(pairs["partner"])
    driver = altered.columns.get_indexer(pairs["driver"])
    ok = (partner >= 0) & (driver >= 0)
    chosen = pairs.loc[ok].assign(p=partner[ok], d=driver[ok])
    mask = altered.to_numpy(np.float64)
    out = []
    for d, group in chosen.groupby("d", sort=False):
        mutated = mask[:, d].astype(bool)
        n = int(mutated.sum())
        columns = group["p"].to_numpy()
        rank_mut = ranks[np.ix_(mutated, columns)].sum(axis=0)
        rank_wt = total[columns] - rank_mut
        mut_p = irwin_hall_cdf(rank_mut, n)
        wt_cdf = irwin_hall_cdf(rank_wt, len(lines) - n)
        out.append(
            group[["driver", "partner"]].assign(
                mutated_lines=n,
                mut_pvalue=mut_p,
                wt_pvalue=2 * np.minimum(wt_cdf, 1 - wt_cdf),
            )
        )
    columns = ["driver", "partner", "mutated_lines", "mut_pvalue", "wt_pvalue"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def pair_scores(
    pairs: pd.DataFrame, tests: pd.DataFrame, wt_threshold: float
) -> pd.DataFrame:
    """One score per (gene_a, gene_b): -log10 of the best mutated-lines p-value over
    the orientations that pass the wild-type filter (see the module docstring)."""
    passed = tests.assign(
        value=np.where(
            tests["wt_pvalue"] < wt_threshold,
            0.0,
            -np.log10(np.maximum(tests["mut_pvalue"].astype(float), TINY)),
        )
    )
    best = passed.groupby(["driver", "partner"])["value"].max()
    orient = [
        best.reindex(pd.MultiIndex.from_arrays([pairs[a], pairs[b]])).to_numpy()
        for a, b in (("gene_a", "gene_b"), ("gene_b", "gene_a"))
    ]
    value = np.fmax(*orient)
    covered = np.isfinite(value)
    return pairs[["gene_a", "gene_b"]].assign(
        score=np.where(covered, value, UNTESTED), covered=covered
    )


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "labels", "split", "gene_effect", "mutations", "copy_number",
         "output_root", "task", "min_mutated", "wt_pvalue"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_slidr":
        raise ValueError(f"{path} is a {config['kind']} config")
    return config


def run(config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    split = json.loads(Path(config["split"]).read_text())
    held_lines = {m for s in HELD_SIDES for m in split[s]}
    fit_lines = [m for s in FIT_SIDES for m in split[s]]
    if held_lines & set(fit_lines):
        raise ValueError("a validation or test line is among the fit lines")
    labels = pd.read_parquet(
        config["labels"], columns=["model_id", "gene_a", "gene_b", "label"]
    )
    held = labels[labels["label"].notna() & labels["model_id"].isin(held_lines)]
    pairs = held[["gene_a", "gene_b"]].drop_duplicates().reset_index(drop=True)
    effect = read_depmap_matrix(Path(config["gene_effect"]))
    mutated = [read_depmap_matrix(Path(p)) for p in config["mutations"]]
    lines = [m for m in fit_lines if m in effect.index and m in mutated[0].index]
    genes = sorted((set(pairs["gene_a"]) | set(pairs["gene_b"])) & set(effect.columns))
    altered = mutation_matrix(
        lines,
        genes,
        mutated,
        read_depmap_matrix(Path(config["copy_number"])),
        int(config["min_mutated"]),
    )
    orientations = pd.concat(
        [
            pairs.rename(columns={"gene_a": "driver", "gene_b": "partner"}),
            pairs.rename(columns={"gene_b": "driver", "gene_a": "partner"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    tests = pair_pvalues(effect, altered, orientations)
    scores = pair_scores(pairs, tests, float(config["wt_pvalue"]))
    write_parquet(run_dir / "tests.parquet", tests)
    write_parquet(run_dir / "scores.parquet", scores)
    write_json(
        run_dir / "metrics.json",
        {
            "run_id": run_id,
            "fit_lines": len(lines),
            "lines_with_an_alteration": len(altered),
            "drivers": altered.shape[1],
            "held_pairs": len(pairs),
            "tests": len(tests),
            "held_pairs_covered_share": float(scores["covered"].mean()),
            "wt_filtered_share": float(
                (tests["wt_pvalue"] < config["wt_pvalue"]).mean()
            ),
        },
    )
    print(f"{run_dir}: done", flush=True)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
