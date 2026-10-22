"""Historical pre-GPU audit of the context screen's coverage by Cilantro-SL.

Per line and SL context, the rows whose two genes Geneformer's deletion operator can
reach (both among the input's top 2048 tokens, with a Gene2vec embedding), in
pseudo-bulk and in per-cell bags, as
``docs/specs/2026-10-08-cilantro-sl-reproduction-design.md`` §5.2 sets out; it reports,
the reader decides."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

MODEL_INPUT_SIZE = 2048
BAG_CELLS = 128
BAG_SEED = 0
MIN_PRESENT_CELLS = 10
HVG_COUNT = 500

GENEFORMER = Path("data/cilantro/geneformer/geneformer/gene_dictionaries_30m")
DEFAULTS = {
    "split": Path("configs/benchmarks/cell_line_geneeffect_226_split.json"),
    "basal_raw_umi": Path("data/sl_dependency_v0/processed/basal_raw_umi"),
    "sl_split": Path("configs/benchmarks/context_screen_v2_split.json"),
    "sl_pairs": Path(
        "data/SL_Benchmark_Formal/derived/context_screen_v2/sl_context_pairs.csv"
    ),
    "gene_effect": Path("data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv"),
    "gene2vec": Path("data/cilantro/official/gene2vec_embs.pt"),
    "token_dictionary": GENEFORMER / "token_dictionary_gc30M.pkl",
    "gene_medians": GENEFORMER / "gene_median_dictionary_gc30M.pkl",
    "ensembl_mapping": GENEFORMER / "ensembl_mapping_dict_gc30M.pkl",
    "gene_names": GENEFORMER / "gene_name_id_dict_gc30M.pkl",
}


def top_tokens(values: np.ndarray, medians: np.ndarray, size: int) -> np.ndarray:
    """Per row, the genes among the top ``size`` tokens: nonzero, ranked by value
    over median (Geneformer's rank-value encoding, truncated at its input size)."""
    ranked = np.asarray(values, dtype=np.float64) / medians
    present = np.zeros(ranked.shape, dtype=bool)
    for row, scores in enumerate(ranked):
        nonzero = np.flatnonzero(scores > 0)
        if len(nonzero) > size:
            nonzero = nonzero[np.argpartition(-scores[nonzero], size - 1)[:size]]
        present[row, nonzero] = True
    return present


def bag_indices(n_cells: int, size: int, seed: int) -> np.ndarray:
    if n_cells <= size:
        return np.arange(n_cells)
    return np.sort(np.random.default_rng(seed).choice(n_cells, size, replace=False))


def unversioned(gene_id: str) -> str:
    """``ENSG00000223972.5`` -> ``ENSG00000223972``; the dictionaries hold bare IDs."""
    return gene_id.split(".")[0] if gene_id.startswith("ENS") else gene_id


def covered_rows(pairs: pd.DataFrame, covered: set[str]) -> pd.Series:
    return pairs["ensembl_a"].isin(covered) & pairs["ensembl_b"].isin(covered)


def line_summary(
    model_id: str,
    source_path: str,
    *,
    var_ensembl_col: str,
    vocabulary: tuple[str, ...],
    mapping: dict[str, str],
    medians: np.ndarray,
) -> tuple[str, int, np.ndarray, np.ndarray]:
    """One line: its cell count, pseudo-bulk (log1p CPM, ``vocabulary`` order) and
    the number of bag cells in which each vocabulary gene is a top token."""
    from src.data.basal import align_columns
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        Path(source_path), model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    ids = [mapping.get(unversioned(str(gene)), "") for gene in source.var_names]
    counts, _ = align_columns(source.X, ids, vocabulary)
    library = float(source.X.sum())
    totals = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    pseudobulk = np.log1p(totals * 1e6 / library).astype(np.float32)
    bag = counts[bag_indices(counts.shape[0], BAG_CELLS, BAG_SEED)].toarray()
    present = top_tokens(bag, medians, MODEL_INPUT_SIZE).sum(axis=0)
    return model_id, int(counts.shape[0]), pseudobulk, present.astype(np.int32)


def hvgs(pseudobulk: pd.DataFrame, count: int) -> list[str]:
    """Notebook 1's HVG call (scanpy default flavour on log values)."""
    import anndata as ad
    import scanpy as sc

    adata = ad.AnnData(pseudobulk.to_numpy(dtype=np.float64))
    adata.var_names = list(pseudobulk.columns)
    sc.pp.highly_variable_genes(adata, n_top_genes=count, inplace=True)
    return list(adata.var_names[adata.var["highly_variable"].to_numpy()])


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def context_rows(
    pairs: pd.DataFrame, by_line: dict[str, set[str]], arm: str
) -> list[dict]:
    rows = []
    for (context, model_id, split), group in pairs.groupby(
        ["context", "model_id", "split"], sort=True
    ):
        hit = covered_rows(group, by_line[model_id])
        labels = group["sl_label"].astype(int)
        rows.append(
            {
                "arm": arm,
                "split": split,
                "context": context,
                "model_id": model_id,
                "rows": int(len(group)),
                "positives": int(labels.sum()),
                "negatives": int((labels == 0).sum()),
                "covered_rows": int(hit.sum()),
                "covered_positives": int((hit & (labels == 1)).sum()),
                "covered_negatives": int((hit & (labels == 0)).sum()),
            }
        )
    return rows


def run(paths: dict[str, Path], out: Path, processes: int) -> dict:
    import torch

    from src.data.splits import load_geneeffect_split

    split = load_geneeffect_split(paths["split"])
    sl_split = json.loads(paths["sl_split"].read_text())
    held_out = {
        sl_split["contexts"][name]["model_id"]
        for name, side in sl_split["assignments"].items()
        if side != "train"
    }
    gene_effect = pd.read_csv(paths["gene_effect"], index_col=0, nrows=0)
    names = load_pickle(paths["gene_names"])
    labelled_genes = {
        names[column.split(" (")[0]]
        for column in gene_effect.columns
        if column.split(" (")[0] in names
    }
    labelled_lines = set(pd.read_csv(paths["gene_effect"], usecols=[0]).iloc[:, 0])
    stage_one = sorted(
        m for m in split.all_model_ids if m in labelled_lines and m not in held_out
    )

    tokens = load_pickle(paths["token_dictionary"])
    median_of = load_pickle(paths["gene_medians"])
    mapping = load_pickle(paths["ensembl_mapping"])
    vocabulary = tuple(sorted(g for g in tokens if g in median_of))
    medians = np.asarray([median_of[g] for g in vocabulary], dtype=np.float64)

    work = partial(
        line_summary,
        var_ensembl_col="ensembl_id",
        vocabulary=vocabulary,
        mapping=mapping,
        medians=medians,
    )
    cells, pseudobulk, present = {}, {}, {}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = [
            pool.submit(work, m, str(paths["basal_raw_umi"] / f"{m}.h5ad"))
            for m in split.all_model_ids
        ]
        for done, future in enumerate(futures, start=1):
            model_id, n_cells, bulk, bag = future.result()
            cells[model_id], pseudobulk[model_id], present[model_id] = (
                n_cells,
                bulk,
                bag,
            )
            _LOGGER.info("line %d of %d: %s", done, len(futures), model_id)
    bulk = pd.DataFrame(pseudobulk, index=list(vocabulary)).T

    pairs = pd.read_csv(paths["sl_pairs"])
    for side in ("a", "b"):
        pairs[f"ensembl_{side}"] = pairs[f"gene_{side}"].map(names).fillna("")
    gene2vec = set(torch.load(paths["gene2vec"], weights_only=True))
    screen_genes = set(pairs["ensembl_a"]) | set(pairs["ensembl_b"])
    paper_genes = (screen_genes | set(hvgs(bulk.loc[stage_one], HVG_COUNT))) & set(
        vocabulary
    )
    paper_columns = [g for g in vocabulary if g in paper_genes]
    paper_medians = medians[[vocabulary.index(g) for g in paper_columns]]
    paper_present = top_tokens(
        bulk[paper_columns].to_numpy(), paper_medians, MODEL_INPUT_SIZE
    )

    usable = gene2vec & set(vocabulary)
    by_arm = {
        "pseudo-bulk, paper vocabulary": {
            m: set(np.asarray(paper_columns)[paper_present[i]]) & usable
            for i, m in enumerate(bulk.index)
        },
        "per-cell bags, native vocabulary": {
            m: set(np.asarray(vocabulary)[present[m] >= MIN_PRESENT_CELLS]) & usable
            for m in bulk.index
        },
    }
    contexts = pd.DataFrame(
        [
            row
            for arm, by_line in by_arm.items()
            for row in context_rows(pairs, by_line, arm)
        ]
        + context_rows(pairs, {m: usable for m in bulk.index}, "ceiling: mapped")
    )
    lines = pd.DataFrame(
        {
            "cells": pd.Series(cells),
            "stage_one": [m in stage_one for m in bulk.index],
            **{
                f"covered: {arm}": [len(by_line[m]) for m in bulk.index]
                for arm, by_line in by_arm.items()
            },
            **{
                f"covered with label: {arm}": [
                    len(by_line[m] & labelled_genes) for m in bulk.index
                ]
                for arm, by_line in by_arm.items()
            },
        },
        index=bulk.index,
    )
    out.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(out / "contexts.csv", index=False)
    lines.rename_axis("model_id").to_csv(out / "lines.csv")
    summary = {
        "settings": {
            "model_input_size": MODEL_INPUT_SIZE,
            "bag_cells": BAG_CELLS,
            "bag_seed": BAG_SEED,
            "min_present_cells": MIN_PRESENT_CELLS,
            "hvg_count": HVG_COUNT,
            "dictionaries": "gc30M",
        },
        "lines": len(bulk),
        "stage_one_lines": len(stage_one),
        "native_vocabulary": len(vocabulary),
        "paper_vocabulary": len(paper_columns),
        "screen_genes": len(screen_genes - {""}),
        "screen_genes_unmapped": int(
            (pairs[["ensembl_a", "ensembl_b"]] == "").any(axis=1).sum()
        ),
        "stage_one_median_covered_with_label": {
            arm: float(lines.loc[stage_one, f"covered with label: {arm}"].median())
            for arm in by_arm
        },
        "inputs": {name: str(path) for name, path in paths.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = run(DEFAULTS, args.out, args.processes)
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs"}, indent=1))
    print(pd.read_csv(args.out / "contexts.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
