"""Program and scVI fits of the cell-state vocabularies (PIE stage-one design
§3.5, Task 8), run in ``.venv-states`` (``hpc/states/setup.sh``: cNMF, starCAT,
scvi-tools).

``python -m src.experiments.states.programs --plan <ladder>/states/fit/plan.json``
(called by ``src.experiments.states.state_telemetry``'s ``fit`` step through
``STATES_PYTHON``). It exchanges files only: it reads the plan, the fit samples
and the raw h5ads, and writes under the plan's root, per fitting seed, skipping
what exists:

- ``programs/seed<s>/``. Consensus NMF (cNMF) per training line on its sampled
  cells (each distinct cell once) over the panel genes, at every K of
  ``programs.ks``, factorising the line's own overdispersed genes; a cell with
  no count in them is left out of that line's factorisation. Per line, a
  program is **robust** when its top genes share ``robust_overlap`` with a
  program at another K, and the robust programs are kept greedily (most
  matches first) unless they share ``redundant_overlap`` with a kept one. A
  kept program sharing ``cycle_overlap`` top genes with Tirosh's S and G2/M
  genes is a **cycle** program (S or G2/M by the larger overlap) and is set
  aside; the others are clustered across lines (average linkage on the share of
  top genes, cut at ``cluster_overlap``) and a cluster with programs from
  ``min_lines`` lines is a **recurrent** program. The **reference**
  (``reference.npz``, and ``reference.tsv`` in starCAT's format, programs x
  genes): every recurrent program, the two cycle phases and a **background**
  program, each the mean over its lines of the line's mean member spectrum (cNMF's
  TPM spectra, renormalised to 10^6), on the pooled sample's overdispersed genes
  and the programs' top genes, divided by each gene's TPM SD over the pooled
  sample (starCAT's variance normalisation, with the SD fixed by the fit). The
  background program is the pooled sample's mean TPM profile. **Usages**
  (``usages/<ModelID>.npy``, ``usages_second/``): starCAT's non-negative least
  squares of each cell's TPM over the reference genes, divided by the same SD,
  on the fixed spectra, solved exactly cell by cell, so a cell's usages do not
  depend on the cells projected with it (starCAT itself rescales genes by the
  query's own SD). ``manifest.json`` records these rules, the counts, each
  recurrent program's member lines and leading genes, and starCAT's own
  ``fit_transform`` on one training line beside ours.
- ``scvi/seed<s>/``. scVI on the pooled sample (each line's distinct sampled
  cells) over STATE's HVGs: each cell's chemistry (resolved by deposit, line and
  arm) as the categorical nuisance (``batch_key``), the whole-library UMI as the
  size factor, the S and G2/M
  scores as continuous covariates; the encoder reads neither (scVI's default),
  so a cell's latent mean needs no category the fit saw. Latent means of every
  cell of the plan's lines and second deposits (``latent/``,
  ``latent_second/``) and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse
from src.data.io import write_json, write_npy, write_npz

_LOGGER = logging.getLogger(__name__)

#: cNMF's TPM scale.
TPM: Final[float] = 1e6
USAGE_RULE: Final[str] = (
    "a cell's memberships are its usages of the recurrent programs and the "
    "background program, cycle usages dropped, normalised to sum to one; a cell "
    "with none of them is all background (src.data.cell_states.program_memberships)"
)


def _workers() -> int:
    return max(
        1,
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1,
    )


def read_counts(
    path: str | Path, model_id: str, plan: Mapping[str, Any], genes: Sequence[str]
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """A line's raw UMI on ``genes`` (upper-case symbols, duplicates summed), its
    whole-library sizes and its cell ids, read as Task 2 reads them."""
    from src.data.basal import align_columns, symbol_column
    from src.data.ladder_cells import library_sizes, upper_symbols
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        Path(path), model_id=model_id, var_ensembl_col=plan["var_ensembl_col"]
    )
    symbols = source.var[symbol_column(source.var)]
    matrix = sparse.csr_matrix(source.X, dtype=np.float64)
    aligned, _ = align_columns(matrix, upper_symbols(symbols.astype(str)), list(genes))
    return (
        aligned.tocsr(),
        library_sizes(matrix),
        np.asarray(source.obs_names, dtype=str),
    )


def _tpm(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    total = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.zeros_like(total)
    scale[total > 0] = TPM / total[total > 0]
    return sparse.csr_matrix(sparse.diags(scale) @ counts)


# --- consensus NMF per line -----------------------------------------------------------


def program_spectra(
    spectra_tpm: pd.DataFrame, top: pd.DataFrame, panel: Sequence[str]
) -> pd.DataFrame:
    """cNMF's TPM spectra as programs x panel genes, rows in ``top``'s program
    order. cNMF 1.7.1's ``load_results`` returns them genes x programs (it reads
    its programs x genes file transposed, against its docstring); rows are taken
    by program label, so they align with ``top``. Fails closed on an empty
    spectrum. (A top-gene ranking check is not made: cNMF ranks top genes by
    its z-scored gene scores, and a program split off at a larger K can rank
    genes there that sit low in its TPM spectrum.)"""
    programs = list(top.columns)
    if set(spectra_tpm.columns) != set(programs):
        raise ValueError("cNMF's spectra do not list the programs as columns")
    spectra = spectra_tpm.T.loc[programs].reindex(columns=list(panel), fill_value=0.0)
    if not (spectra.to_numpy(dtype=np.float64).sum(axis=1) > 0).all():
        raise ValueError("cNMF returned an empty program spectrum")
    return spectra


def line_programs(
    model_id: str,
    path: str,
    rows: np.ndarray,
    plan: Mapping[str, Any],
    seed: int,
    output: Path,
) -> str:
    """cNMF on one training line's sampled cells: per K, the programs' top genes
    and TPM spectra over the panel (``cnmf/<ModelID>.npz``)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import anndata as ad
    from cnmf import cNMF
    from cnmf.cnmf import compute_tpm, get_highvar_genes_sparse

    settings = plan["programs"]
    counts, _, ids = read_counts(path, model_id, plan, plan["panel"])
    rows = np.unique(rows)
    adata = ad.AnnData(
        X=counts[rows].astype(np.float32),
        obs=pd.DataFrame(index=ids[rows]),
        var=pd.DataFrame(index=list(plan["panel"])),
    )
    adata = adata[:, np.asarray(adata.X.sum(axis=0)).ravel() > 0].copy()
    tpm = compute_tpm(adata)
    stats, _ = get_highvar_genes_sparse(
        tpm.X, numgenes=int(settings["highly_variable"])
    )
    hvg = list(tpm.var.index[stats["high_var"].to_numpy()])
    expressed = np.asarray(adata[:, hvg].X.sum(axis=1)).ravel() > 0
    adata = adata[expressed].copy()
    ks = [int(k) for k in settings["ks"]]
    workdir = Path(tempfile.mkdtemp(prefix=f"cnmf_{model_id}_", dir=output.parent))
    try:
        counts_path = workdir / "counts.h5ad"
        adata.write_h5ad(counts_path)
        genes_path = workdir / "genes.txt"
        genes_path.write_text("\n".join(hvg))
        model = cNMF(output_dir=str(workdir), name="line")
        model.prepare(
            counts_fn=str(counts_path),
            components=ks,
            n_iter=int(settings["iterations"]),
            seed=seed,
            genes_file=str(genes_path),
        )
        model.factorize(worker_i=0, total_workers=1)
        model.combine()
        arrays: dict[str, np.ndarray] = {}
        threshold = float(settings["density_threshold"])
        for k in ks:
            model.consensus(
                k,
                density_threshold=threshold,
                show_clustering=False,
                close_clustergram_fig=True,
                build_ref=False,
            )
            _, scores, spectra, top = model.load_results(
                K=k, density_threshold=threshold, n_top_genes=int(settings["top_genes"])
            )
            spectra = program_spectra(spectra, top, list(plan["panel"]))
            arrays[f"top_k{k}"] = top.T.to_numpy(dtype=str)
            arrays[f"tpm_k{k}"] = spectra.to_numpy(dtype=np.float32)
        write_npz(
            output,
            **arrays,
            cells=np.asarray(adata.n_obs),
            cells_left_out=np.asarray(int((~expressed).sum())),
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return model_id


def _overlap(left: Sequence[str], right: Sequence[str]) -> int:
    return len(set(left) & set(right))


def kept_programs(
    line: Mapping[str, np.ndarray], ks: Sequence[int], robust: int, redundant: int
) -> list[tuple[int, int]]:
    """A line's robust, non-redundant programs as (K, index)."""
    programs = [(k, j) for k in ks for j in range(int(k))]
    top = {(k, j): list(line[f"top_k{k}"][j]) for k, j in programs}
    matches = {
        p: sum(
            any(_overlap(top[p], top[(k2, j2)]) >= robust for j2 in range(int(k2)))
            for k2 in ks
            if k2 != p[0]
        )
        for p in programs
    }
    candidates = [p for p in programs if len(ks) == 1 or matches[p] > 0]
    candidates.sort(key=lambda p: (-matches[p], p[0], p[1]))
    kept: list[tuple[int, int]] = []
    for p in candidates:
        if all(_overlap(top[p], top[q]) < redundant for q in kept):
            kept.append(p)
    return kept


def _renormalised(spectrum: np.ndarray) -> np.ndarray:
    total = spectrum.sum()
    return spectrum * (TPM / total) if total > 0 else spectrum


def recurrent_programs(
    lines: Mapping[str, Mapping[str, np.ndarray]],
    settings: Mapping[str, Any],
    cycle_genes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Robust programs of every line, cycle programs set aside by phase, the rest
    clustered across lines; recurrent clusters and their members."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    ks = [int(k) for k in settings["ks"]]
    top_n = int(settings["top_genes"])
    s_genes = {g.upper() for g in cycle_genes["S"]}
    g2m_genes = {g.upper() for g in cycle_genes["G2M"]}
    programs, cycle = [], {"S": [], "G2M": []}
    for model_id, line in lines.items():
        for k, j in kept_programs(
            line,
            ks,
            int(settings["robust_overlap"]),
            int(settings["redundant_overlap"]),
        ):
            top = [str(g) for g in line[f"top_k{k}"][j]]
            entry = {
                "line": model_id,
                "top": top,
                "spectrum": _renormalised(line[f"tpm_k{k}"][j].astype(np.float64)),
            }
            s, g2m = len(s_genes & set(top)), len(g2m_genes & set(top))
            if max(s, g2m) >= int(settings["cycle_overlap"]):
                cycle["S" if s >= g2m else "G2M"].append(entry)
            else:
                programs.append(entry)
    clusters: list[list[dict]] = []
    if len(programs) > 1:
        n = len(programs)
        distance = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                shared = _overlap(programs[a]["top"], programs[b]["top"]) / top_n
                distance[a, b] = distance[b, a] = 1.0 - shared
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(
            tree, t=1.0 - int(settings["cluster_overlap"]) / top_n, criterion="distance"
        )
        for label in sorted(set(labels)):
            clusters.append([programs[i] for i in np.flatnonzero(labels == label)])
    recurrent = [
        c for c in clusters if len({p["line"] for p in c}) >= int(settings["min_lines"])
    ]
    recurrent.sort(key=lambda c: -len({p["line"] for p in c}))
    return {
        "recurrent": recurrent,
        "cycle": cycle,
        "programs": len(programs),
        "clusters": len(clusters),
    }


def _line_mean_spectrum(members: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The mean over lines of each line's mean member spectrum."""
    by_line: dict[str, list[np.ndarray]] = {}
    for member in members:
        by_line.setdefault(member["line"], []).append(member["spectrum"])
    return np.mean([np.mean(v, axis=0) for v in by_line.values()], axis=0)


def _leading_genes(members: Sequence[Mapping[str, Any]], count: int) -> list[str]:
    frequency: dict[str, int] = {}
    for member in members:
        for gene in member["top"]:
            frequency[gene] = frequency.get(gene, 0) + 1
    return [
        g for g, _ in sorted(frequency.items(), key=lambda x: (-x[1], x[0]))[:count]
    ]


def pooled_tpm(plan: Mapping[str, Any], sample: pd.DataFrame) -> sparse.csr_matrix:
    """TPM over the panel of the pooled fit sample (each line's rows as drawn)."""
    parts = []
    for model_id, group in sample.groupby("model_id", sort=False):
        counts, _, _ = read_counts(
            plan["encode"][model_id], model_id, plan, plan["panel"]
        )
        parts.append(_tpm(counts[group["row"].to_numpy()]))
    return sparse.vstack(parts).tocsr()


def build_reference(
    plan: Mapping[str, Any], seed: int, folder: Path, sample: pd.DataFrame
) -> dict[str, Any]:
    """The program vocabulary's reference spectra and gene scale."""
    from cnmf.cnmf import get_highvar_genes_sparse

    settings = plan["programs"]
    lines = {
        m: dict(np.load(folder / "cnmf" / f"{m}.npz", allow_pickle=False))
        for m in plan["train"]
    }
    found = recurrent_programs(lines, settings, plan["cycle_genes"])
    panel = list(plan["panel"])
    pooled = pooled_tpm(plan, sample)
    stats, _ = get_highvar_genes_sparse(
        pooled, numgenes=int(settings["highly_variable"])
    )
    chosen = set(np.asarray(panel)[stats["high_var"].to_numpy()])
    for members in [*found["recurrent"], *found["cycle"].values()]:
        for member in members:
            chosen |= set(member["top"])
    mean = np.asarray(pooled.mean(axis=0)).ravel()
    square = np.asarray(pooled.multiply(pooled).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(square - mean**2, 0.0))
    genes = [g for g in panel if g in chosen and sd[panel.index(g)] > 0]
    columns = np.asarray([panel.index(g) for g in genes])
    names, kinds, rows, recorded = [], [], [], []
    for i, members in enumerate(found["recurrent"]):
        names.append(f"program{i}")
        kinds.append("recurrent")
        rows.append(_line_mean_spectrum(members))
        recorded.append(
            {
                "name": f"program{i}",
                "lines": sorted({m["line"] for m in members}),
                "programs": len(members),
                "leading_genes": _leading_genes(members, int(settings["top_genes"])),
            }
        )
    for phase, members in found["cycle"].items():
        if members:
            names.append(f"cycle_{phase.lower()}")
            kinds.append("cycle")
            rows.append(_line_mean_spectrum(members))
    background = np.mean(
        [
            np.asarray(pooled[group.index.to_numpy()].mean(axis=0)).ravel()
            for _, group in sample.reset_index(drop=True).groupby(
                "model_id", sort=False
            )
        ],
        axis=0,
    )
    names.append("background")
    kinds.append("background")
    rows.append(_renormalised(background))
    spectra = np.vstack(rows)[:, columns] / sd[columns]
    if not (spectra.sum(axis=1) > 0).all():
        raise ValueError("a reference program has an empty spectrum")
    write_npz(
        folder / "reference.npz",
        programs=np.asarray(names),
        kinds=np.asarray(kinds),
        genes=np.asarray(genes),
        spectra=spectra,
        scale=sd[columns],
    )
    pd.DataFrame(spectra, index=names, columns=genes).to_csv(
        folder / "reference.tsv", sep="\t"
    )
    return {
        "programs": names,
        "kinds": kinds,
        "genes": len(genes),
        "robust_non_cycle_programs": found["programs"],
        "clusters": found["clusters"],
        "cycle_programs": {k: len(v) for k, v in found["cycle"].items()},
        "recurrent": recorded,
    }


def nnls_usages(tpm: sparse.csr_matrix, spectra: np.ndarray) -> np.ndarray:
    """Each cell's non-negative least-squares usages of the fixed spectra
    (programs x genes), one cell at a time."""
    from scipy.optimize import nnls

    design = np.asarray(spectra, dtype=np.float64).T
    dense = tpm.toarray() if sparse.issparse(tpm) else np.asarray(tpm)
    return np.vstack([nnls(design, row)[0] for row in dense]).astype(np.float32)


def line_usages(
    model_id: str,
    path: str,
    plan: Mapping[str, Any],
    reference: Mapping[str, np.ndarray],
    output: Path,
) -> str:
    counts, _, _ = read_counts(path, model_id, plan, plan["panel"])
    panel = list(plan["panel"])
    columns = [panel.index(str(g)) for g in reference["genes"]]
    tpm = _tpm(counts)[:, columns]
    scaled = sparse.csr_matrix(tpm @ sparse.diags(1.0 / reference["scale"]))
    write_npy(output, nnls_usages(scaled, reference["spectra"]))
    return model_id


def starcat_check(
    plan: Mapping[str, Any], folder: Path, reference: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """starCAT's own ``fit_transform`` on the first training line beside our
    usages of the same cells: the median per-cell Pearson of normalised usages."""
    import anndata as ad
    from starcat import starCAT

    model_id = plan["train"][0]
    counts, _, ids = read_counts(
        plan["encode"][model_id], model_id, plan, plan["panel"]
    )
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(index=list(plan["panel"])),
    )
    theirs, _ = starCAT(reference=str(folder / "reference.tsv")).fit_transform(adata)
    ours = np.load(folder / "usages" / f"{model_id}.npy").astype(np.float64)
    ours = ours / np.maximum(ours.sum(axis=1, keepdims=True), 1e-12)
    theirs = theirs.loc[:, list(reference["programs"])].to_numpy()
    pearson = [
        float(np.corrcoef(a, b)[0, 1])
        for a, b in zip(ours, theirs, strict=True)
        if a.std() > 0 and b.std() > 0
    ]
    return {
        "line": model_id,
        "cells": int(len(ours)),
        "median_cell_pearson": float(np.median(pearson)) if pearson else None,
    }


def programs_seed(plan: Mapping[str, Any], seed: int) -> None:
    folder = Path(plan["root"]) / "programs" / f"seed{seed}"
    sample = pd.read_parquet(plan["samples"][str(seed)])
    todo = [m for m in plan["train"] if not (folder / "cnmf" / f"{m}.npz").is_file()]
    (folder / "cnmf").mkdir(parents=True, exist_ok=True)
    rows = {m: g["row"].to_numpy() for m, g in sample.groupby("model_id", sort=False)}
    calls = [
        partial(
            line_programs,
            m,
            plan["encode"][m],
            rows[m],
            plan,
            seed,
            folder / "cnmf" / f"{m}.npz",
        )
        for m in todo
    ]
    _in_processes(calls, f"cNMF seed {seed}")
    if not (folder / "reference.npz").is_file():
        summary = build_reference(plan, seed, folder, sample)
        write_json(folder / "reference.json", summary, keep_nan=True)
    reference = dict(np.load(folder / "reference.npz", allow_pickle=False))
    calls = [
        partial(line_usages, m, p, plan, reference, folder / sub / f"{m}.npy")
        for sub, sources in (
            ("usages", plan["encode"]),
            ("usages_second", plan["second"]),
        )
        for m, p in sources.items()
        if not (folder / sub / f"{m}.npy").is_file()
    ]
    _in_processes(calls, f"usages seed {seed}")
    if not (folder / "manifest.json").is_file():
        write_json(
            folder / "manifest.json",
            {
                "settings": dict(plan["programs"]),
                "seed": seed,
                "spectra_normalisation": (
                    "cNMF TPM spectra renormalised to 1e6, averaged per line then over "
                    "lines, divided by each gene's TPM SD over the pooled fit sample "
                    "(starCAT's variance normalisation with the SD fixed by the fit)"
                ),
                "background_program": (
                    "the pooled fit sample's mean TPM profile (each line's mean, then "
                    "the mean over lines), renormalised to 1e6 and scaled likewise"
                ),
                "projection": (
                    "per cell, non-negative least squares of its TPM over the "
                    "reference genes divided by the same SD on the fixed spectra "
                    "(scipy nnls)"
                ),
                "usage_to_weight": USAGE_RULE,
                "reference": json.loads((folder / "reference.json").read_text()),
                "starcat_check": starcat_check(plan, folder, reference),
            },
            keep_nan=True,
        )


# --- scVI ---------------------------------------------------------------------------


def _scvi_adata(
    counts: sparse.csr_matrix,
    library: np.ndarray,
    names: Sequence[str],
    hvg: Sequence[str],
    chemistry: Sequence[str],
    cycle: np.ndarray,
) -> Any:
    import anndata as ad

    obs = pd.DataFrame(
        {
            "chemistry": pd.Categorical(list(chemistry)),
            "library": np.asarray(library, dtype=np.float64),
            "s_score": cycle[:, 0],
            "g2m_score": cycle[:, 1],
        },
        index=list(names),
    )
    return ad.AnnData(
        X=sparse.csr_matrix(counts, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=list(hvg)),
    )


def scvi_seed(plan: Mapping[str, Any], seed: int) -> None:
    import scvi
    import torch

    folder = Path(plan["root"]) / "scvi" / f"seed{seed}"
    settings = plan["scvi"]
    hvg = [str(g).upper() for g in plan["hvg"]]
    model_dir = folder / "model"
    if not (model_dir / "model.pt").is_file():
        sample = pd.read_parquet(plan["samples"][str(seed)])
        parts, libraries, names, chemistry, cycle = [], [], [], [], []
        for model_id, group in sample.groupby("model_id", sort=False):
            counts, library, _ = read_counts(
                plan["encode"][model_id], model_id, plan, hvg
            )
            rows = group["row"].to_numpy()
            parts.append(counts[rows])
            libraries.append(library[rows])
            names += [f"{model_id}:{i}" for i in range(len(rows))]
            chemistry += list(group["chemistry"].astype(str))
            cycle.append(group[["s_score", "g2m_score"]].to_numpy(dtype=np.float64))
        adata = _scvi_adata(
            sparse.vstack(parts),
            np.concatenate(libraries),
            names,
            hvg,
            chemistry,
            np.vstack(cycle),
        )
        scvi.settings.seed = seed
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key="chemistry",
            size_factor_key="library",
            continuous_covariate_keys=["s_score", "g2m_score"],
        )
        model = scvi.model.SCVI(
            adata,
            n_latent=int(settings["latent"]),
            n_hidden=int(settings["hidden"]),
            n_layers=int(settings["layers"]),
        )
        if model.module.encode_covariates:
            raise ValueError(
                "scVI's encoder reads the covariates; the latent would need them"
            )
        model.train(
            max_epochs=int(settings["max_epochs"]),
            batch_size=int(settings["batch_size"]),
            early_stopping=True,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )
        model.save(str(model_dir), overwrite=True, save_anndata=True)
        history = {
            key: float(value.iloc[-1, 0])
            for key, value in model.history.items()
            if key in ("elbo_train", "elbo_validation") and len(value)
        }
        write_json(
            folder / "fit.json",
            {
                "cells": int(adata.n_obs),
                "chemistry": sorted(set(chemistry)),
                "epochs": int(len(model.history["elbo_train"])),
                "history": history,
            },
            keep_nan=True,
        )
    # Saved with its training AnnData, so a resumed run reloads the model whole.
    model = scvi.model.SCVI.load(str(model_dir))
    if model.adata is None:
        raise ValueError(f"{model_dir} holds no training AnnData; refit the model")
    batch = model.registry["field_registries"]["batch"]["state_registry"]
    category = str(batch["categorical_mapping"][0])
    for sub, sources in (("latent", plan["encode"]), ("latent_second", plan["second"])):
        for model_id, path in sources.items():
            out = folder / sub / f"{model_id}.npy"
            if out.is_file():
                continue
            counts, library, ids = read_counts(path, model_id, plan, hvg)
            # The encoder reads neither the chemistry nor the cycle covariates
            # (checked at the fit), so any registered category and zeros serve.
            adata = _scvi_adata(
                counts,
                library,
                ids,
                hvg,
                [category] * len(ids),
                np.zeros((len(ids), 2)),
            )
            latent = model.get_latent_representation(adata, give_mean=True)
            write_npy(out, np.asarray(latent, dtype=np.float32))
    if not (folder / "manifest.json").is_file():
        write_json(
            folder / "manifest.json",
            {
                "settings": dict(settings),
                "seed": seed,
                "genes": "STATE's HVGs (plan hvg), raw UMI",
                "nuisance": (
                    "chemistry (batch_key), from "
                    "configs/benchmarks/single_cell_source_chemistry.csv"
                ),
                "library": "whole-library UMI as scVI's size factor",
                "continuous_covariates": ["s_score", "g2m_score"],
                "encoder_reads_covariates": False,
                "fit": json.loads((folder / "fit.json").read_text()),
            },
            keep_nan=True,
        )


# --- entry point ------------------------------------------------------------------


def _in_processes(calls: Sequence[Any], label: str) -> None:
    """The calls in a pool of spawned workers. Not forked: after the first seed's
    scVI fit the process holds torch's threads and a CUDA context, and workers
    forked from it deadlock on locks those threads held (the H20's seed-1 cNMF hung
    for three days, every worker in a futex wait)."""
    if not calls:
        return
    _LOGGER.info("%s: %d jobs", label, len(calls))
    with ProcessPoolExecutor(
        max_workers=min(_workers(), len(calls)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for done, _ in enumerate(pool.map(_call, calls), start=1):
            if done % 25 == 0 or done == len(calls):
                _LOGGER.info("%s: %d of %d", label, done, len(calls))


def _call(function: Any) -> Any:
    return function()


def run(plan: Mapping[str, Any]) -> None:
    for seed in plan["seeds"]:
        programs_seed(plan, int(seed))
        scvi_seed(plan, int(seed))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(json.loads(args.plan.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
