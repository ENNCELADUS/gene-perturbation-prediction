"""Inputs of MuSL (Feng et al.; official code third_party/musl, commit f8021cf) on the
multi-study SL benchmark. For each labelled `train`, `val` and `test` line: at most
2,000 of its single cells (raw UMI) go through the notebook's steps (library-size
filter, library-size normalization, sqrt, MAGIC) into a genes x cells matrix. The rows
the model reads (fit and validation samples of a line, every row of a held-out line)
get the 35 official statistical features, and the held-out pairs get them again on one
pooled matrix of training-line cells (the context-ablated control). Writes under --out:
nodes.csv, edges.npy, lines/<ModelID>.npy, rows/<ModelID>.parquet,
features/<ModelID>.npy (raw features, same row order), pooled.npy, pooled_rows.parquet,
pooled_features.npy, manifest.json.

    OMP_NUM_THREADS=1 .venv-musl/bin/python hpc/musl/prepare.py \
        --out data/musl/benchmark --workers 64
"""

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpc"))
import sl_common as sl  # noqa: E402

VENDORED = ROOT / "third_party" / "musl"
MAX_CELLS = 2000
# torch.quantile refuses an input above 16 million values
QUANTILE_LIMIT = 16_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _mutual_info(exp_a, exp_b, n_neighbors=3, max_workers=None):
    """The official get_mutual_info (sklearn's mutual_info_regression of each row pair),
    in this process: the official one opens 48 processes, and prepare.py already runs
    one worker for each core. The noise seed is fixed."""
    import torch
    from sklearn.feature_selection import mutual_info_regression

    a, b = exp_a.numpy(), exp_b.numpy()
    return torch.tensor(
        [
            mutual_info_regression(
                a[i].reshape(-1, 1), b[i], n_neighbors=n_neighbors, random_state=0
            )[0]
            for i in range(len(a))
        ]
    )


@functools.cache
def _package():
    """The vendored package under the name `musl_src`: the project has its own `src`."""
    package = VENDORED / "src"
    spec = importlib.util.spec_from_file_location(
        "musl_src",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["musl_src"] = module
    spec.loader.exec_module(module)
    return module


def official(name: str):
    """`utils` or `module` of the official code."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    _package()
    loaded = importlib.import_module(f"musl_src.{name}")
    if name == "utils":
        loaded.get_mutual_info = _mutual_info
    return loaded


def _draw(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def impute(
    counts: pd.DataFrame,
    rng: np.random.Generator,
    max_cells: int = MAX_CELLS,
    min_cells: int = 10,
) -> pd.DataFrame:
    """Cells x genes raw counts to MAGIC-imputed values: at most `max_cells` cells, the
    notebook's library-size filter (skipped when fewer than `min_cells` cells pass),
    library-size normalization, sqrt, MAGIC. MAGIC's 100-component PCA needs more than
    100 cells: a smaller line runs without it. `attrs["library_filter_skipped"]` says
    whether the filter was skipped."""
    import magic
    import scprep

    counts = counts.iloc[_draw(len(counts), max_cells, rng)]
    kept = scprep.filter.filter_library_size(counts, cutoff=1500)
    skipped = len(kept) < min_cells
    data = scprep.normalize.library_size_normalize(counts if skipped else kept)
    data = scprep.transform.sqrt(data)
    n_pca = None if len(data) <= 100 else 100
    out = magic.MAGIC(n_pca=n_pca, random_state=0, n_jobs=1, verbose=0).fit_transform(
        data
    )
    out.attrs["library_filter_skipped"] = skipped
    return out


SYMBOL_COLUMNS = ("gene_symbol", "gene_symbols", "gene_name")


def counts_frame(x: np.ndarray, var: pd.DataFrame) -> pd.DataFrame:
    """Cells x gene symbols. The symbol column is the first of SYMBOL_COLUMNS the h5ad
    has (the lines differ); counts of genes with one symbol are summed."""
    column = next(c for c in SYMBOL_COLUMNS if c in var.columns)
    counts = pd.DataFrame(
        x.astype(np.float32), columns=var[column].astype(str).to_numpy()
    )
    if counts.columns.duplicated().any():
        counts = counts.T.groupby(level=0, sort=False).sum().T
    return counts


def line_matrix(imputed: pd.DataFrame, genes: pd.Index) -> np.ndarray:
    """Genes x cells float32; a gene the line lacks is a zero row."""
    values = imputed.reindex(columns=genes, fill_value=0.0).to_numpy(np.float32)
    return np.ascontiguousarray(values.T)


def line_samples(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """For each line, one row for each pair (SL when at least half of its rows are SL):
    all SL pairs and as many non-SL pairs (all of them when there are fewer)."""
    pairs = rows.groupby(["model_id", "gene_1", "gene_2"], as_index=False)[
        "label"
    ].mean()
    pairs["label"] = (pairs["label"] >= 0.5).astype(int)
    return pd.concat(
        [sl.balanced(g, rng) for _, g in pairs.groupby("model_id")], ignore_index=True
    )


def pooled_cells(
    sizes: dict[str, int], n: int, rng: np.random.Generator
) -> list[tuple[str, int]]:
    """`n` (line, cell) picks, evenly over the lines: one random cell of each line in
    turn until `n`; a line with no cells left drops out."""
    picks = [
        (rank, rng.random(), line, int(cell))
        for line, size in sizes.items()
        for rank, cell in enumerate(rng.permutation(size))
    ]
    return [(line, cell) for _, _, line, cell in sorted(picks)[:n]]


def pair_features(
    matrix: np.ndarray, pairs: np.ndarray, chunk_rows: int | None = None
) -> np.ndarray:
    """The 35 official statistical features of node pairs (n x 2) on a genes x cells
    matrix: `calculate_features` in chunks, as `vector_state` builds its inputs, with
    NaN as 0."""
    import torch

    utils = official("utils")
    exp = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    log = torch.log1p(exp)
    std = torch.std(log, dim=1, keepdim=True)
    std = torch.where(std < 1e-10, torch.ones_like(std), std)
    norm = (log - torch.mean(log, dim=1, keepdim=True)) / std
    chunk = chunk_rows or min(20000, QUANTILE_LIMIT // exp.shape[1])
    out = []
    for start in range(0, len(pairs), chunk):
        block = torch.from_numpy(np.ascontiguousarray(pairs[start : start + chunk].T))
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            features = utils.calculate_features(exp, log, norm, block.long())
        out.append(torch.nan_to_num(features, nan=0.0).float())
    return torch.cat(out).numpy() if out else np.zeros((0, 35), np.float32)


# ---- worker functions (one thread each) ----


def _init_worker():
    import torch

    torch.set_num_threads(1)
    official("utils")


def _impute_line(model_id: str, raw: Path, out: Path, nodes: list[str]) -> dict:
    import anndata as ad

    start = time.perf_counter()
    adata = ad.read_h5ad(raw / f"{model_id}.h5ad")
    rows = _draw(adata.n_obs, MAX_CELLS, np.random.default_rng(0))
    x = adata.X[rows]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    counts = counts_frame(x, adata.var)
    imputed = impute(counts, np.random.default_rng(0))
    matrix = line_matrix(imputed, pd.Index(nodes))
    np.save(out / "lines" / f"{model_id}.npy", matrix)
    return {
        "model_id": model_id,
        "cells_in_file": int(adata.n_obs),
        "cells_drawn": len(rows),
        "cells_used": int(matrix.shape[1]),
        "library_filter_skipped": bool(imputed.attrs["library_filter_skipped"]),
        "magic_without_pca": bool(matrix.shape[1] <= 100),
        "genes_with_expression": int(matrix.any(axis=1).sum()),
        "seconds": time.perf_counter() - start,
    }


def _features(
    path: Path, pairs: np.ndarray, start: int, stop: int
) -> tuple[int, int, np.ndarray, float]:
    t0 = time.perf_counter()
    matrix = np.load(path)
    return (
        start,
        stop,
        pair_features(matrix, pairs[start:stop]),
        time.perf_counter() - t0,
    )


# ---- main ----


def smoke_lines(side: pd.Series, cells: pd.Series, n: int) -> list[str]:
    """The line with the fewest cells, the smallest lines that give at least two
    training lines and one validation line, then the smallest others up to `n`."""
    order = list(cells.sort_values().index)
    chosen = [order[0]]
    for name, need in (("train", 2), ("val", 1)):
        while sum(side[c] == name for c in chosen) < need:
            chosen.append(next(c for c in order if c not in chosen and side[c] == name))
    return chosen + [c for c in order if c not in chosen][: max(0, n - len(chosen))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lines", type=int, help="smoke run: this many small lines")
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/sl_dependency_v0/processed/basal_raw_umi",
    )
    parser.add_argument(
        "--ppi",
        type=Path,
        default=ROOT / "data/musl/official/merged_ppi_gene_names.csv",
    )
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
    t_start = time.perf_counter()
    out = cli.out
    for sub in ("lines", "rows", "features"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = sl.benchmark_rows(cli.labels, cli.split)
    scored = rows[rows["side"].isin(sl.SCORED_SIDES)].copy()
    nodes = sorted(set(scored["gene_1"]) | set(scored["gene_2"]))
    node_id = {g: i for i, g in enumerate(nodes)}
    scored["i"] = scored["gene_1"].map(node_id)
    scored["j"] = scored["gene_2"].map(node_id)
    side = scored.drop_duplicates("model_id").set_index("model_id")["side"].sort_index()
    if cli.lines:
        import anndata as ad

        cells = pd.Series(
            {
                m: ad.read_h5ad(cli.raw / f"{m}.h5ad", backed="r").n_obs
                for m in side.index
            }
        )
        side = side[smoke_lines(side, cells, cli.lines)]
        scored = scored[scored["model_id"].isin(side.index)]
    pd.Series(nodes, name="gene").to_csv(out / "nodes.csv", index=False)

    # the released PPI graph without the edges between the two genes of a benchmark pair
    ppi = pd.read_csv(cli.ppi, usecols=["gene1", "gene2"])
    kept = sl.drop_pair_edges(ppi, rows)
    edges = np.stack([kept["gene1"].map(node_id), kept["gene2"].map(node_id)])
    edges = edges[:, ~np.isnan(edges).any(axis=0)].astype(np.int64)
    np.save(out / "edges.npy", edges)

    # 1. imputed line matrices
    t0 = time.perf_counter()
    with ProcessPoolExecutor(cli.workers, initializer=_init_worker) as pool:
        stats = list(
            pool.map(
                _impute_line,
                side.index,
                [cli.raw] * len(side),
                [out] * len(side),
                [nodes] * len(side),
            )
        )
    impute_seconds = time.perf_counter() - t0
    line_info = {s["model_id"]: {**s, "side": side[s["model_id"]]} for s in stats}

    # the pooled matrix: 2,000 cells drawn evenly from the training lines
    train = [m for m in side.index if side[m] == "train"]
    picks = pooled_cells(
        {m: line_info[m]["cells_used"] for m in train},
        MAX_CELLS,
        np.random.default_rng(0),
    )
    lines = {m: np.load(out / "lines" / f"{m}.npy", mmap_mode="r") for m in train}
    pooled = np.stack([lines[m][:, c] for m, c in picks], axis=1)
    np.save(out / "pooled.npy", pooled)

    # 2. the rows each line contributes. A gene without expression in a line is a
    # zero row (the official get_cell_expression); its pairs stay in.
    held_all = scored[scored["side"].isin(sl.HELD_SIDES)].drop_duplicates(
        ["model_id", "gene_1", "gene_2"]
    )
    usable = scored[scored["condition"] == "genetic"]
    rng = np.random.default_rng(0)
    fit = line_samples(usable[usable["side"] == "train"], rng).assign(fit=True)
    val = line_samples(usable[usable["side"] == "val"], rng).assign(val=True)
    held = (
        scored[scored["side"].isin(sl.HELD_SIDES)]
        .groupby(["model_id", "gene_1", "gene_2"], as_index=False)["label"]
        .mean()
        .assign(score=True, label=lambda d: (d["label"] >= 0.5).astype(int))
    )
    flags = ["fit", "val", "score"]
    table = pd.concat(
        [fit, val, held]
    )  # a pair's label: the genetic one when it has one
    table[flags] = table[flags].fillna(False).astype(bool)
    table = table.groupby(["model_id", "gene_1", "gene_2"], as_index=False).agg(
        {"label": "first", **dict.fromkeys(flags, "any")}
    )
    table["i"] = table["gene_1"].map(node_id)
    table["j"] = table["gene_2"].map(node_id)
    per_line = {  # a line without a usable row gets an empty table
        m: table[table["model_id"] == m].reset_index(drop=True) for m in side.index
    }
    pooled_pairs = (
        table[table["score"]][["gene_1", "gene_2", "i", "j"]]
        .drop_duplicates(["i", "j"])
        .reset_index(drop=True)
    )
    pooled_pairs.to_parquet(out / "pooled_rows.parquet", index=False)

    # 3. features, in chunks over the workers; the largest first
    chunk = 20000
    jobs = []  # (name, matrix path, pairs, array to fill)
    arrays = {}
    for m, g in per_line.items():
        g.to_parquet(out / "rows" / f"{m}.parquet", index=False)
        arrays[m] = np.zeros((len(g), 35), np.float32)
        jobs.append((m, out / "lines" / f"{m}.npy", g[["i", "j"]].to_numpy()))
    arrays["pooled"] = np.zeros((len(pooled_pairs), 35), np.float32)
    jobs.append(("pooled", out / "pooled.npy", pooled_pairs[["i", "j"]].to_numpy()))
    tasks = [
        (name, path, pairs, a, min(a + chunk, len(pairs)))
        for name, path, pairs in jobs
        for a in range(0, len(pairs), chunk)
    ]

    def cost(task):
        cells = line_info[task[0]]["cells_used"] if task[0] in line_info else MAX_CELLS
        return -(task[4] - task[3]) * cells

    tasks.sort(key=cost)
    t0 = time.perf_counter()
    worker_seconds = 0.0
    with ProcessPoolExecutor(cli.workers, initializer=_init_worker) as pool:
        futures = [
            (name, pool.submit(_features, path, pairs, a, b))
            for name, path, pairs, a, b in tasks
        ]
        for name, future in futures:
            a, b, values, seconds = future.result()
            arrays[name][a:b] = values
            worker_seconds += seconds
    feature_seconds = time.perf_counter() - t0
    for name, values in arrays.items():
        path = out / (
            "pooled_features.npy" if name == "pooled" else f"features/{name}.npy"
        )
        np.save(path, values)

    n_feature_rows = sum(len(a) for a in arrays.values())
    manifest = {
        "lines": line_info,
        "nodes": len(nodes),
        "edges_released": len(ppi),
        "edges_after_pair_drop": len(kept),
        "edges_in_nodes": int(edges.shape[1]),
        "fit_rows": int(len(fit)),
        "val_rows": int(len(val)),
        "held_rows": int(len(held_all)),
        "held_rows_scored": int(len(held)),
        "held_rows_share": float(len(held) / max(len(held_all), 1)),
        "pooled_cells": len(picks),
        "pooled_pairs": len(pooled_pairs),
        "magic_without_pca_lines": int(
            sum(v["magic_without_pca"] for v in line_info.values())
        ),
        "library_filter_skipped_lines": int(
            sum(v["library_filter_skipped"] for v in line_info.values())
        ),
        "feature_rows": int(n_feature_rows),
        "feature_values_not_finite": int(
            sum((~np.isfinite(a)).sum() for a in arrays.values())
        ),
        "impute_seconds_wall": impute_seconds,
        "impute_seconds_mean_per_line": float(np.mean([v["seconds"] for v in stats])),
        "feature_seconds_wall": feature_seconds,
        "feature_rows_per_second_per_worker": n_feature_rows
        / max(worker_seconds, 1e-9),
        "workers": cli.workers,
        "seconds_total": time.perf_counter() - t_start,
        "ppi_sha256": sha256(cli.ppi),
        "labels_sha256": sha256(cli.labels),
        "split_sha256": sha256(cli.split),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "lines"}, indent=2))


if __name__ == "__main__":
    main()
