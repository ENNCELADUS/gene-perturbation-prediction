"""PARIS (Benfatto et al. 2021, Molecular Cancer), ported from the official R code
(``third_party/paris/R``, ``RF_exp``). For each dependency gene B a random forest
predicts B's GeneEffect across cell lines from the expression of the other genes; the
features A the forest finds important are reported with the Pearson r between B's
dependency and A's expression. Stage two reads these pairs as SL evidence.

``raw``: Boruta (Kursa & Rudnicki) with ranger's raw out-of-bag permutation importance.
``Ginicorrected``: ranger ``importance = "impurity_corrected"`` (Nembrini et al. 2018)
with ``importance_pvalues`` (Janitza), keeping p-value 0. Deviations from R (RNG, trees,
the OOB estimator) are listed in ``third_party/paris/PROVENANCE.md``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import write_json, write_parquet
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind

MODES = ("raw", "Ginicorrected")
COLUMNS = ["dependency_gene", "feature_gene", "importance", "r", "r_pvalue", "mode"]
TREES = 500  # ranger's default num.trees
MAX_RUNS = 500  # run_PARIS: Boruta(maxRuns = 500)
P_VALUE = 0.01  # Boruta's default
MIN_NODE_SIZE = 5  # ranger's default for regression
MIN_LINES = 10  # fewer labelled lines than this cannot run Boruta; the gene is skipped


# --- forests --------------------------------------------------------------------


def _trees(
    x: np.ndarray,
    y: np.ndarray,
    mtry: int,
    rng: np.random.Generator,
    copy: bool = False,
) -> Iterator[tuple[Any, np.ndarray]]:
    """Bootstrap trees ``(tree, in-bag counts)``: ranger draws n rows with replacement,
    and a node of at most ``MIN_NODE_SIZE`` rows is not split. ``copy`` grows each tree
    on the features plus a copy with the rows permuted (one permutation per tree)."""
    from sklearn.tree import DecisionTreeRegressor

    n = len(y)
    for _ in range(TREES):
        counts = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        grown = np.concatenate([x, x[rng.permutation(n)]], axis=1) if copy else x
        tree = DecisionTreeRegressor(
            max_features=mtry,
            min_samples_split=MIN_NODE_SIZE + 1,  # deviation: distinct rows, not draws
            random_state=int(rng.integers(2**31)),
        )
        yield tree.fit(grown, y, sample_weight=counts), counts


def permutation_importance(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """ranger ``importance = "permutation"``, ``scale.permutation.importance = FALSE``:
    per tree, the rise in out-of-bag MSE when one feature's out-of-bag values are
    permuted, averaged over all trees (a feature a tree never splits on adds 0)."""
    m = x.shape[1]
    mtry = max(1, int(np.sqrt(m)))
    importance = np.zeros(m)
    for tree, counts in _trees(x, y, mtry, rng):
        oob = np.flatnonzero(counts == 0)
        t = tree.tree_
        split = t.children_left != -1
        if len(oob) == 0 or not split.any():
            continue
        used = np.unique(t.feature[split])
        k = len(used)
        column = np.full(t.node_count, -1)
        column[split] = np.searchsorted(used, t.feature[split])
        xo, yo = x[np.ix_(oob, used)], y[oob]
        # Variant v < k permutes used feature v among the OOB rows; variant k is the
        # unpermuted baseline. One permutation per variant, as ranger draws per feature.
        rows = np.arange(len(oob))[None, :]
        source = np.argsort(rng.random((k + 1, len(oob))), axis=1)
        variant = np.arange(k + 1)[:, None]
        node = np.zeros((k + 1, len(oob)), dtype=np.intp)
        while True:
            left = t.children_left[node]
            internal = left != -1
            if not internal.any():
                break
            col = np.where(internal, column[node], 0)
            row = np.where(col == variant, source, rows)
            go_left = xo[row, col] <= t.threshold[node]
            node = np.where(
                internal, np.where(go_left, left, t.children_right[node]), node
            )
        error = ((yo[None, :] - t.value[node, 0, 0]) ** 2).mean(axis=1)
        importance[used] += error[:k] - error[k]
    return importance / TREES


def corrected_impurity(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """ranger ``importance = "impurity_corrected"``: every tree grows on the features
    plus a copy of them with the rows permuted (one permutation per tree), ``mtry``
    drawn from both; a feature's importance is the SSE decrease of its splits minus
    its copy's, averaged over trees."""
    m = x.shape[1]
    mtry = max(1, int(np.sqrt(m)))
    importance = np.zeros(m)
    for tree, _ in _trees(x, y, mtry, rng, copy=True):
        t = tree.tree_
        split = np.flatnonzero(t.children_left != -1)
        sse = t.weighted_n_node_samples * t.impurity
        gain = sse[split] - sse[t.children_left[split]] - sse[t.children_right[split]]
        feature = t.feature[split]
        np.add.at(importance, feature % m, np.where(feature < m, gain, -gain))
    return importance / TREES


# --- selection ------------------------------------------------------------------


def boruta(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """R ``Boruta(maxRuns = 500, getImp = getImpRfRaw)`` (pValue 0.01, Bonferroni):
    each run permutes the not-yet-rejected features into shadows (at least five),
    scores a feature a hit when its importance beats the best shadow, and tests the
    hits against Binomial(runs, 1/2). Returns the confirmed mask and every feature's
    median importance over the runs it took part in (``attStats``' ``medianImp``)."""
    from scipy.stats import binom

    m = x.shape[1]
    tentative, confirmed, rejected = 0, 1, 2
    decision = np.zeros(m, dtype=np.int8)
    hits = np.zeros(m)
    history: list[np.ndarray] = []
    runs = 0
    while (decision == tentative).any() and runs + 1 < MAX_RUNS:
        runs += 1
        active = np.flatnonzero(decision != rejected)
        real = x[:, active]
        source = real
        while source.shape[1] < 5:
            source = np.concatenate([source, source], axis=1)
        shadow = rng.permuted(source, axis=0)  # every column permuted on its own
        imp = permutation_importance(np.concatenate([real, shadow], axis=1), y, rng)
        full = np.full(m, -np.inf)
        full[active] = imp[: len(active)]
        history.append(full)
        hits[full > imp[len(active) :].max()] += 1
        accept = binom.sf(hits - 1, runs, 0.5) * m < P_VALUE
        reject = binom.cdf(hits, runs, 0.5) * m < P_VALUE
        open_ = decision == tentative
        decision[open_ & accept] = confirmed
        decision[open_ & reject] = rejected
    stack = np.array(history)
    stack[~np.isfinite(stack)] = np.nan
    return decision == confirmed, np.nanmedian(stack, axis=0)


def janitza_selected(importance: np.ndarray) -> np.ndarray | None:
    """ranger ``importance_pvalues(method = "janitza")``: the null mirrors the
    negative importances (zeros included); p-value 0 is a value above all of it.
    ``None`` when no importance is negative (ranger raises, R skips the gene)."""
    negative = importance[importance < 0]
    if len(negative) == 0:
        return None
    return importance > (-negative).max()


# --- the method -----------------------------------------------------------------


def _gene(
    gene: str,
    y: np.ndarray,
    x: np.ndarray,
    names: np.ndarray,
    mode: str,
    seed: int,
) -> tuple[str, pd.DataFrame, float]:
    """One dependency gene: its NaN lines dropped (R ``na.omit``), its own expression
    never a feature."""
    from threadpoolctl import threadpool_limits

    start = time.perf_counter()
    keep = np.isfinite(y)
    y, x = y[keep], x[keep][:, names != gene]
    names = names[names != gene]
    rng = np.random.default_rng(seed)  # R: set.seed(43) before every gene
    with threadpool_limits(1):
        if len(y) < MIN_LINES:
            selected, importance = np.zeros(len(names), bool), np.zeros(len(names))
        elif mode == "raw":
            selected, importance = boruta(x, y, rng)
        else:
            importance = corrected_impurity(x, y, rng)
            selected = janitza_selected(importance)
            if selected is None:
                selected = np.zeros(len(names), bool)
    from scipy.stats import pearsonr

    rows = []
    for j in np.flatnonzero(selected):
        # constant features give NaN, as cor.test errors there
        result = pearsonr(y, x[:, j]) if x[:, j].std() > 0 else (np.nan, np.nan)
        rows.append((gene, names[j], importance[j], result[0], result[1], mode))
    return gene, pd.DataFrame(rows, columns=COLUMNS), time.perf_counter() - start


def iter_paris(
    dependency: pd.DataFrame,
    expression: pd.DataFrame,
    mode: str,
    seed: int,
    n_jobs: int,
) -> Iterator[tuple[str, pd.DataFrame, float]]:
    """``(dependency gene, its selected pairs, seconds)`` as each gene finishes. The
    genes run on ``n_jobs`` processes, each forest on one thread."""
    from joblib import Parallel, delayed

    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if not dependency.index.equals(expression.index):
        raise ValueError("dependency and expression must be on the same lines")
    if not np.isfinite(expression.to_numpy()).all():
        raise ValueError("expression has missing values")
    x = np.ascontiguousarray(expression.to_numpy(dtype=np.float32))
    names = expression.columns.to_numpy(dtype=object)
    y = dependency.to_numpy(dtype=np.float64)
    jobs = (
        delayed(_gene)(g, y[:, i], x, names, mode, seed)
        for i, g in enumerate(dependency.columns)
    )
    yield from Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs)


def stack(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    full = [p for p in parts if len(p)] or [pd.DataFrame(columns=COLUMNS)]
    out = pd.concat(full, ignore_index=True)
    return out.sort_values(["dependency_gene", "feature_gene"], ignore_index=True)


def paris(
    dependency: pd.DataFrame,
    expression: pd.DataFrame,
    mode: str,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Selected ``(dependency_gene, feature_gene)`` pairs with their importance and
    the Pearson r (and p-value) between the gene's dependency and the feature's
    expression. Both inputs are lines x genes on the same lines."""
    results = iter_paris(dependency, expression, mode, seed, n_jobs)
    parts = [rows for _, rows, _ in results]
    return stack(parts)


# --- the command ----------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"ladder_config", "benchmark", "output_root", "task", "mode", "seed", "n_jobs"},
        "config",
    )
    if config["mode"] not in MODES:
        raise ValueError(f"{path}: mode {config['mode']!r} is not one of {MODES}")
    return config


def inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dependency (GeneEffect units) and expression on the reference lines: the
    stage-one frame minus every line of the SL benchmark, so no validation or test
    line, and no line an SL row is scored on, enters."""
    from src.data.ladder_reader import mean_channel
    from src.experiments.stage_one.setup import load_setup

    setup = load_setup(config["ladder_config"])
    ladder = setup.ladder
    pairs = pd.read_parquet(config["benchmark"])
    screened = set(pairs["model_id"].dropna())
    lines = [m for m in ladder.frame if m not in screened]
    if set(lines) & set(ladder.role_lines("val", "test")):
        raise ValueError("a validation or test line is among the reference lines")
    labelled = pairs[pairs["label"].notna()]
    genes = sorted(set(labelled["gene_a"]) | set(labelled["gene_b"]))
    genes = [g for g in genes if g in ladder.genes.index]
    units = setup.truth(lines) * ladder.genes["sigma_g"] + ladder.genes["mu_g"]
    return units[genes], mean_channel(ladder).loc[lines]


def run(config: Mapping[str, Any], run_id: str, genes: int | None = None) -> Path:
    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, {**config, "genes": genes})
    dependency, expression = inputs(config)
    if genes is not None:
        dependency = dependency.iloc[:, :genes]
    start = time.perf_counter()
    parts, seconds = [], []
    results = iter_paris(
        dependency, expression, config["mode"], config["seed"], config["n_jobs"]
    )
    for done, (gene, rows, took) in enumerate(results, 1):
        parts.append(rows)
        seconds.append(took)
        print(
            f"{done}/{dependency.shape[1]} {gene}: {len(rows)} pairs, {took:.1f}s",
            flush=True,
        )
    wall = time.perf_counter() - start
    pairs = stack(parts)
    write_parquet(run_dir / "pairs.parquet", pairs)
    write_json(
        run_dir / "run.json",
        {
            "mode": config["mode"],
            "n_jobs": config["n_jobs"],
            "lines": len(dependency),
            "dependency_genes": dependency.shape[1],
            "feature_genes": expression.shape[1],
            "selected_pairs": len(pairs),
            "genes_with_selection": int(pairs["dependency_gene"].nunique()),
            "wall_seconds": wall,
            "wall_seconds_per_gene": wall / dependency.shape[1],
            "mean_gene_seconds": float(np.mean(seconds)),
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
    parser.add_argument(
        "--genes", type=int, help="the first N dependency genes (pilot)"
    )
    parser.add_argument("--mode", choices=MODES, help="overrides the config's mode")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    run(config, args.run_id, args.genes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
