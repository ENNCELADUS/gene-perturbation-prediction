"""Stage-one inputs (PIE stage-one design §5 Task 2) under ``ladder_root``.

``python -m src.experiments.ladder_inputs CONFIG --steps lines,cells,mean,summaries``
(``hpc/run.sh ladder-inputs``), and ``--steps tx1 [--shard I/N]`` on a GPU. Steps,
each skipping the outputs that exist, so rerunning resumes:

- ``lines``: the **frame** of 981 labelled lines every arm trains on (the split's
  labelled single-cell training lines, then the bulk-only labelled lines the
  default prior fits, outside its dropped lineages), the held-out lines, five
  patient-grouped folds of the frame; per gene the base ``m_g`` (mean GeneEffect
  over the frame), the protocol's definitions (377-line mean, residual SD,
  selective and variable genes) and its ``gene_text`` key: ``lines.parquet``,
  ``genes.parquet``. The decode panel is the definitions' genes with a key.
- ``cells``: per split line, from all of its cells (``src.data.ladder_cells``):
  PIE's input, pseudo-bulk and per-gene spread over the panel and the bulk genes,
  per-cell QC and cycle values (the noise diagnosis's S and G2/M program scores,
  a Scrublet doublet score), HVG cells, their shuffle and the mean as one cell:
  ``cells/<ModelID>.npz``.
- ``mean``: the mean channel in the default prior's space (its genes, a gap
  filled with the training mean, quantile-normalised to the training side's bulk
  profile). A line with cells reads its cells through an affine bridge refitted
  on the paired lines, a training line's outside its fold; a bulk-only line reads
  its bulk. PIE's input for every line (a bulk-only line's bulk rescaled to
  CP10K; an unlisted gene the frame mean). ``mean/``.
- ``summaries``: the noise diagnosis's per-line features from all cells: the
  fraction above threshold and mean of each program, cluster shares at its
  cluster counts (fitted as it fits them, on the prepared 128-cell bags of the
  training lines, then assigned every cell): ``summaries.parquet``.
- ``tx1``: Tx1-3B embeddings of every cell and of every count-shuffled cell of
  the lines with cells, under ``paths.tx1_cache`` (``real/``, ``shuffled/``), in
  the existing cache format; ``--shard I/N`` takes every N-th line.

``manifest.json`` records the settings (a root written under others is refused),
the expression spaces and each step's counts. It decides nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("lines", "cells", "mean", "summaries", "tx1")
MANIFEST: Final[str] = "manifest.json"
#: Scrublet needs enough cells for its 30 principal components and neighbours.
SCRUBLET_MIN_CELLS: Final[int] = 100
CYCLE_PROGRAMS: Final[tuple[str, str]] = ("S", "G2M")
QC_COLUMNS: Final[tuple[str, ...]] = (
    "log_umi",
    "genes_detected",
    "mito_fraction",
    "ribo_fraction",
    "s_score",
    "g2m_score",
    "doublet_score",
)


# --- configuration ------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """The ladder config and the configs it names, loaded once."""

    config: Mapping[str, Any]
    prior: Mapping[str, Any]
    joint: Mapping[str, Any]
    noise: Mapping[str, Any]

    @property
    def root(self) -> Path:
        return Path(self.config["ladder_root"])

    @property
    def prepared(self) -> Path:
        return Path(self.joint["prepared_root"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


def load_setup(config: Mapping[str, Any]) -> Setup:
    from src.experiments.config import (
        load_config,
        load_noise_config,
        load_prior_config,
        validate_ladder_config,
    )

    validate_ladder_config(config)
    prior = load_prior_config(Path(config["prior_config"]))
    return Setup(
        config=config,
        prior=prior,
        joint=load_config(Path(prior["joint_config"])),
        noise=load_noise_config(Path(config["noise_config"])),
    )


def settings(setup: Setup) -> dict[str, Any]:
    """Everything that changes what the steps write."""
    from src.data.prepared import read_manifest

    prepared = read_manifest(setup.prepared)
    return {
        "config": dict(setup.config),
        "split": setup.joint["paths"]["split"],
        "gene_effect": setup.joint["paths"]["gene_effect"],
        "basal_raw_umi": setup.joint["paths"]["basal_raw_umi"],
        "features": dict(setup.joint["features"]),
        "model": setup.prior["paths"]["model"],
        "bulk_expression": setup.prior["paths"]["bulk_expression"],
        "reference": setup.prior["paths"]["reference"],
        "exclude_lineages": list(setup.prior["training_side"]["exclude_lineages"]),
        "cell_states": dict(setup.noise["cell_states"]),
        "prepared_target_sum": float(prepared["expression_space"]["target_sum"]),
        "hvg_order": list(prepared["hvg_order"]),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def open_manifest(setup: Setup) -> dict[str, Any]:
    """The root's manifest, written now if new; refuses a root of other settings."""
    setup.root.mkdir(parents=True, exist_ok=True)
    path = setup.root / MANIFEST
    current = settings(setup)
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != current:
            raise ValueError(
                f"{setup.root} was written under other settings; choose a new "
                "ladder_root for these"
            )
        return manifest
    manifest = {
        "settings": current,
        "expression_spaces": {
            "pie_input": "mean over cells of log1p(CP10K); bulk-only lines "
            "log1p(TPM / 100)",
            "pseudobulk": "log1p(mean over cells of CP10K), then the default "
            "prior's space (quantile-normalised to the training side's bulk)",
            "hvg_cells": "log1p(x * T / L) over STATE's HVGs, "
            f"T = {current['prepared_target_sum']}",
            "tx1": "raw UMI",
        },
        "steps": {},
    }
    _write_json(path, manifest)
    return manifest


def record_step(setup: Setup, step: str, record: Mapping[str, Any]) -> None:
    """Add a step's record to the manifest, re-read so concurrent shards keep theirs."""
    path = setup.root / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["steps"][step] = {
        **record,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(path, manifest)


# --- the frame ------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The lines of the ladder and their folds."""

    split: Any
    models: pd.DataFrame
    single_cell_train: tuple[str, ...]
    bulk_only: tuple[str, ...]
    side: tuple[str, ...]
    paired: tuple[str, ...]
    folds: Mapping[str, int]

    @property
    def lines(self) -> tuple[str, ...]:
        return (*self.single_cell_train, *self.bulk_only)

    @property
    def held_out(self) -> tuple[str, ...]:
        return (*self.split.val, *self.split.test)

    @property
    def cell_lines(self) -> tuple[str, ...]:
        """Every split line (every one has cells)."""
        return tuple(self.split.all_model_ids)

    def role(self, model_id: str) -> str:
        if model_id in self.split.val:
            return "val"
        if model_id in self.split.test:
            return "test"
        return "train_cells" if model_id in self.single_cell_train else "train_bulk"


def load_frame(setup: Setup) -> Frame:
    """The frame: single-cell training lines admitted by ``assert_fit_eligible``,
    then bulk-only labelled lines with bulk RNA outside the dropped lineages
    (``load_bulk_lines`` refuses any overlap with the split); the training side
    the prior quantile-normalises against, and its paired lines."""
    from src.context_prior.folds import patient_folds
    from src.data.depmap import read_model_ids, read_models
    from src.data.extra_lines import load_bulk_lines
    from src.data.splits import assert_fit_eligible, load_geneeffect_split

    split_path = Path(setup.joint["paths"]["split"])
    split = load_geneeffect_split(split_path)
    extra = load_bulk_lines(split_path, split)
    models = read_models(Path(setup.prior["paths"]["model"]))
    bulk = read_model_ids(Path(setup.prior["paths"]["bulk_expression"]))
    dropped = set(setup.prior["training_side"]["exclude_lineages"])

    def kept(ids: Sequence[str]) -> list[str]:
        return [m for m in ids if models.at[m, "lineage"] not in dropped]

    single_cell = tuple(split.supervised_train)
    for model_id in single_cell:
        assert_fit_eligible(model_id, split)
    bulk_only = tuple(m for m in kept(extra.labelled) if m in bulk)
    side = tuple(
        m for m in kept([*split.train, *extra.labelled, *extra.unlabelled]) if m in bulk
    )
    on_side = set(side)
    patients = models["patient_id"].to_dict()
    held_patients = {patients[m] for m in (*split.val, *split.test)}
    siblings = [m for m in (*single_cell, *bulk_only) if patients[m] in held_patients]
    if siblings:
        raise ValueError(f"frame lines share a patient with held-out lines: {siblings}")
    return Frame(
        split=split,
        models=models,
        single_cell_train=single_cell,
        bulk_only=bulk_only,
        side=side,
        paired=tuple(m for m in single_cell if m in on_side),
        folds=patient_folds(
            [*single_cell, *bulk_only],
            patients,
            n_folds=int(setup.config["frame"]["folds"]),
            seed=setup.seed,
        ),
    )


def gene_text_keys(setup: Setup) -> list[str]:
    meta = json.loads(
        (Path(setup.config["paths"]["gene_text"]) / "meta.json").read_text()
    )
    return [str(key) for key in meta["keys"]]


def build_lines(setup: Setup, frame: Frame) -> dict[str, Any]:
    """``lines.parquet`` and ``genes.parquet``."""
    from src.context_prior.targets import fit_definitions
    from src.data.depmap import read_depmap_matrix
    from src.data.ladder_cells import symbol_key
    from src.data.prepared import read_manifest

    gene_effect = read_depmap_matrix(Path(setup.joint["paths"]["gene_effect"]))
    panel = list(read_manifest(setup.prepared)["common_gene_panel"])
    definitions = fit_definitions(
        gene_effect, frame.split, panel, setup.joint["features"]
    )
    genes = list(definitions.genes)
    labels = gene_effect.reindex(index=list(frame.lines), columns=genes)
    keys = symbol_key(genes, gene_text_keys(setup))
    table = pd.DataFrame(
        {
            "gene_text_key": [keys.get(g, "") for g in genes],
            "in_panel": [g in keys for g in genes],
            "m_g": labels.mean(axis=0, skipna=True).to_numpy(),
            "frame_labels": labels.notna().sum(axis=0).to_numpy(),
            "mu_g": definitions.gene_means.loc[genes].to_numpy(),
            "sigma_g": definitions.residual_scale.loc[genes].to_numpy(),
            "selective": [g in set(definitions.selective) for g in genes],
            "variable": [g in set(definitions.variable) for g in genes],
        },
        index=pd.Index(genes, name="gene"),
    )
    lines = [*frame.lines, *frame.held_out]
    rows = pd.DataFrame(
        {
            "role": [frame.role(m) for m in lines],
            "fold": [frame.folds.get(m, -1) for m in lines],
            "lineage": [frame.models.at[m, "lineage"] for m in lines],
            "patient_id": [frame.models.at[m, "patient_id"] for m in lines],
            "paired": [m in set(frame.paired) for m in lines],
        },
        index=pd.Index(lines, name="model_id"),
    )
    _write_parquet(setup.root / "genes.parquet", table)
    _write_parquet(setup.root / "lines.parquet", rows)
    return {
        "frame": len(frame.lines),
        "single_cell_train": len(frame.single_cell_train),
        "single_cell_train_lymphoid_myeloid": int(
            sum(
                frame.models.at[m, "lineage"]
                in set(setup.prior["training_side"]["exclude_lineages"])
                for m in frame.single_cell_train
            )
        ),
        "bulk_only": len(frame.bulk_only),
        "paired": len(frame.paired),
        "val": len(frame.split.val),
        "test": len(frame.split.test),
        "genes": len(genes),
        "panel": int(table["in_panel"].sum()),
        "selective": int(table["selective"].sum()),
        "selective_in_panel": int((table["selective"] & table["in_panel"]).sum()),
        "fold_sizes": {
            str(k): int(v) for k, v in pd.Series(frame.folds).value_counts().items()
        },
    }


def read_genes(root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "genes.parquet")


def read_lines(root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "lines.parquet")


# --- cells ----------------------------------------------------------------------


def cell_genes(panel: Sequence[str], bulk_genes: Sequence[str]) -> list[str]:
    """The genes a line's cells are summarised over: the panel, then the bulk
    genes outside it."""
    inside = set(panel)
    return [*panel, *(g for g in bulk_genes if g not in inside)]


def doublet_scores(counts: object) -> np.ndarray:
    """Scrublet's doublet score per cell; NaN for a line below
    ``SCRUBLET_MIN_CELLS``."""
    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    matrix = sparse.csr_matrix(counts, dtype=np.float32)
    if matrix.shape[0] < SCRUBLET_MIN_CELLS:
        return np.full(matrix.shape[0], np.nan, dtype=np.float32)
    adata = ad.AnnData(matrix)
    # Only the scores are kept; a fixed threshold spares Scrublet's automatic
    # one, which needs scikit-image.
    sc.pp.scrublet(adata, threshold=0.25, random_state=0, verbose=False)
    return adata.obs["doublet_score"].to_numpy(dtype=np.float32)


def _cells_line(
    model_id: str,
    source_path: Path,
    programs_path: Path,
    output_path: Path,
    *,
    genes: tuple[str, ...],
    hvg_order: tuple[str, ...],
    target_sum: float,
    cycle_columns: tuple[int, int],
    var_ensembl_col: str,
    hvg_gene_symbol_col: str,
    seed: int,
) -> int:
    """Write one line's ``cells/<ModelID>.npz``; return its cell count."""
    from src.data.basal import symbol_column
    from src.data.ladder_cells import cell_qc, gene_summaries, hvg_cells, line_rng
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        source_path, model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    symbols = source.var[symbol_column(source.var, hvg_gene_symbol_col)]
    symbols = symbols.astype(str).tolist()
    with np.load(programs_path) as payload:
        programs = payload["scores"]
    if programs.shape[0] != source.n_obs:
        raise ValueError(
            f"{model_id}: {programs.shape[0]} program-score rows, "
            f"{source.n_obs} cells"
        )
    summaries = gene_summaries(
        source.X, symbols, genes, rng=line_rng(model_id, seed, "poisson")
    )
    hvg = hvg_cells(
        source.X,
        symbols,
        hvg_order,
        target_sum,
        rng=line_rng(model_id, seed, "hvg_shuffle"),
    )
    qc = np.column_stack(
        [
            cell_qc(source.X, symbols),
            programs[:, list(cycle_columns)],
            doublet_scores(source.X),
        ]
    ).astype(np.float32)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            cell_ids=np.asarray(source.obs_names, dtype=str),
            pie_input=summaries.pie_input,
            pseudobulk=summaries.pseudobulk,
            detected=summaries.detected,
            sd=summaries.sd,
            detected_expected=summaries.detected_expected,
            sd_expected=summaries.sd_expected,
            listed=summaries.listed,
            qc=qc,
            hvg=hvg.cells,
            hvg_shuffled=hvg.shuffled,
            hvg_mean=hvg.mean_cell,
            hvg_listed=hvg.listed,
        )
    os.replace(temporary, output_path)
    return int(source.n_obs)


def build_cells(setup: Setup, frame: Frame) -> dict[str, Any]:
    """``cells/<ModelID>.npz`` for every split line lacking one."""
    from src.experiments.prepare import LINE_PROCESSES, NOISE_DIR, _in_processes

    genes = read_genes(setup.root)
    panel = list(genes.index[genes["in_panel"]])
    union = cell_genes(
        panel, bulk_gene_columns(Path(setup.prior["paths"]["bulk_expression"]))
    )
    programs = list(setup.noise["cell_states"]["programs"])
    cycle = tuple(programs.index(name) for name in CYCLE_PROGRAMS)
    out = setup.root / "cells"
    out.mkdir(exist_ok=True)
    noise_programs = setup.prepared / NOISE_DIR / "programs"
    recorded = settings(setup)
    preparation = setup.joint["preparation"]
    directory = Path(setup.joint["paths"]["basal_raw_umi"])
    missing = [m for m in frame.cell_lines if not (out / f"{m}.npz").is_file()]
    calls = [
        partial(
            _cells_line,
            model_id,
            directory / f"{model_id}.h5ad",
            noise_programs / f"{model_id}.npz",
            out / f"{model_id}.npz",
            genes=tuple(union),
            hvg_order=tuple(recorded["hvg_order"]),
            target_sum=recorded["prepared_target_sum"],
            cycle_columns=cycle,
            var_ensembl_col=preparation["var_ensembl_col"],
            hvg_gene_symbol_col=preparation["hvg_gene_symbol_col"],
            seed=setup.seed,
        )
        for model_id in missing
    ]
    _LOGGER.info("cells: %d of %d lines to write", len(calls), len(frame.cell_lines))
    for done, _ in enumerate(_in_processes(calls, LINE_PROCESSES), start=1):
        if done % 25 == 0:
            _LOGGER.info("cells: line %d of %d", done, len(calls))
    _write_json(out / "genes.json", {"genes": union, "panel": len(panel)})
    counts = {}
    for model_id in frame.cell_lines:
        with np.load(out / f"{model_id}.npz") as payload:
            counts[model_id] = int(payload["cell_ids"].shape[0])
    return {
        "lines": len(frame.cell_lines),
        "cells": int(sum(counts.values())),
        "genes": len(union),
        "panel": len(panel),
        "qc_columns": list(QC_COLUMNS),
        "lines_without_doublet_score": sorted(
            m for m, n in counts.items() if n < SCRUBLET_MIN_CELLS
        ),
    }


def bulk_gene_columns(path: Path) -> list[str]:
    """The upper-case gene symbols of a DepMap matrix's header, in file order, as
    ``read_depmap_matrix`` names its columns."""
    from src.data.depmap import _OMICS_METADATA, depmap_symbol

    header = list(pd.read_csv(path, nrows=0).columns)
    if "ModelID" in header:
        columns = [
            c
            for c in header
            if c not in _OMICS_METADATA and not c.startswith("Unnamed")
        ]
    else:
        columns = header[1:]
    return [depmap_symbol(c) for c in columns]


# --- the mean channel and PIE's input ----------------------------------------------


def _quartiles(values: pd.Series) -> list[float]:
    finite = values.dropna().to_numpy(dtype=np.float64)
    return [float(q) for q in np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])]


def read_cells(root: Path, model_id: str) -> dict[str, np.ndarray]:
    with np.load(Path(root) / "cells" / f"{model_id}.npz") as payload:
        return {key: payload[key] for key in payload.files}


def build_mean(setup: Setup, frame: Frame) -> dict[str, Any]:
    """``mean/``: the mean channel, the quantile-normalised pseudo-bulk of every
    split line, the paired lines' bulk, and PIE's input with its listed mask."""
    from src.context_prior.bridge import bridge_quality, fit_bridge
    from src.context_prior.bridging import oof_bridged
    from src.context_prior.space import (
        fill_unmeasured,
        measured_genes,
        quantile_normalize,
        quantile_reference,
    )
    from src.data.depmap import read_depmap_matrix
    from src.data.ladder_cells import bulk_to_pie_input

    genes = read_genes(setup.root)
    panel = list(genes.index[genes["in_panel"]])
    union = json.loads((setup.root / "cells" / "genes.json").read_text())["genes"]
    position = {g: i for i, g in enumerate(union)}
    # Only training-side bulk is read: no validation or test line's bulk RNA.
    bulk = read_depmap_matrix(Path(setup.prior["paths"]["bulk_expression"]))
    bulk = bulk.loc[[m for m in bulk.index if m in set(frame.side)]]
    bulk_genes = list(bulk.columns)
    cells = {m: read_cells(setup.root, m) for m in frame.cell_lines}
    pseudo = pd.DataFrame(
        [cells[m]["pseudobulk"][[position[g] for g in bulk_genes]] for m in cells],
        index=list(cells),
        columns=bulk_genes,
        dtype=np.float64,
    )
    space = measured_genes(pseudo, bulk_genes, frame.held_out)
    pseudo, filled = fill_unmeasured(pseudo.loc[:, space], frame.split.train)
    side_bulk = bulk.loc[list(frame.side), space]
    profile = quantile_reference(side_bulk)
    bulk_qn = quantile_normalize(side_bulk, profile)
    pseudo_qn = quantile_normalize(pseudo, profile)
    paired = list(frame.paired)
    trained = oof_bridged(
        pseudo_qn, bulk_qn, paired, frame.single_cell_train, frame.folds
    )
    bridge = fit_bridge(pseudo_qn.loc[paired], bulk_qn.loc[paired])
    held = bridge.apply(pseudo_qn.loc[list(frame.held_out)])
    channel = pd.concat([trained, bulk_qn.loc[list(frame.bulk_only)], held])
    channel = channel.loc[[*frame.lines, *frame.held_out]]

    lines = [*frame.lines, *frame.held_out]
    values = pd.DataFrame(np.nan, index=lines, columns=panel, dtype=np.float64)
    listed = pd.DataFrame(False, index=lines, columns=panel)
    columns = [position[g] for g in panel]
    for m in lines:
        if m in cells:
            values.loc[m] = cells[m]["pie_input"][columns]
            listed.loc[m] = cells[m]["listed"][columns]
    in_bulk = [g for g in panel if g in set(bulk_genes)]
    raw = bulk.loc[list(frame.bulk_only), in_bulk]
    values.loc[list(frame.bulk_only), in_bulk] = bulk_to_pie_input(raw.to_numpy())
    listed.loc[list(frame.bulk_only), in_bulk] = True
    fill = values.loc[list(frame.lines)].where(listed.loc[list(frame.lines)]).mean()
    if fill.isna().any():
        raise ValueError(
            f"panel genes no frame line lists: {list(fill.index[fill.isna()])[:10]}"
        )
    values = values.where(listed, fill, axis=1)

    out = setup.root / "mean"
    out.mkdir(exist_ok=True)
    _write_parquet(out / "mean_channel.parquet", channel.astype(np.float32))
    _write_parquet(out / "pseudobulk.parquet", pseudo_qn.astype(np.float32))
    _write_parquet(out / "bulk_paired.parquet", bulk_qn.loc[paired].astype(np.float32))
    _write_parquet(out / "pie_listed.parquet", listed)
    _write_parquet(out / "pie_input.parquet", values.astype(np.float32))
    quality = bridge_quality(trained.loc[paired], bulk_qn.loc[paired])
    return {
        "space_genes": len(space),
        "lines": len(lines),
        "paired": len(paired),
        "pseudobulk_filled_per_line": _quartiles(filled.astype(float)),
        "oof_bridge_pearson": _quartiles(quality),
        "pie_listed_share": _quartiles(listed.mean(axis=1)),
    }


# --- summaries ------------------------------------------------------------------


def build_summaries(setup: Setup, frame: Frame) -> dict[str, Any]:
    """``summaries.parquet``: per split line, program fractions and means and
    cluster shares, from all cells."""
    from src.context_prior.noise import cluster_proportions, program_features
    from src.data.prepared import read_manifest, read_prepared_line
    from src.experiments.prepare import NOISE_DIR

    states = setup.noise["cell_states"]
    programs = list(states["programs"])
    noise_programs = setup.prepared / NOISE_DIR / "programs"
    scores = {}
    for model_id in frame.cell_lines:
        with np.load(noise_programs / f"{model_id}.npz") as payload:
            scores[model_id] = payload["scores"].astype(np.float64)
    train = list(frame.single_cell_train)
    fractions, means, thresholds = program_features(
        scores, train, programs, float(states["threshold_percentile"])
    )
    panel = tuple(read_manifest(setup.prepared)["common_gene_panel"])
    bags = {
        m: read_prepared_line(
            setup.prepared / "lines" / f"{m}.npz", panel
        ).basal_hvg.astype(np.float32)
        for m in train
    }
    every = {m: read_cells(setup.root, m)["hvg"] for m in frame.cell_lines}
    clusters = cluster_proportions(
        bags,
        train,
        components=int(states["cluster_components"]),
        counts=states["cluster_counts"],
        seed=int(setup.noise["seed"]),
        assigned=every,
    )
    parts = [
        fractions.add_prefix("fraction|"),
        means.add_prefix("mean|"),
        *(frame.add_prefix(f"clusters_{k}|") for k, frame in clusters.items()),
    ]
    table = pd.concat(parts, axis=1).loc[list(frame.cell_lines)]
    table.index.name = "model_id"
    _write_parquet(setup.root / "summaries.parquet", table.astype(np.float32))
    return {
        "lines": len(table),
        "features": int(table.shape[1]),
        "thresholds": dict(zip(programs, map(float, thresholds), strict=True)),
    }


# --- Tx1 ------------------------------------------------------------------------


def tx1_lines(frame: Frame) -> list[str]:
    """The labelled lines with cells, in a fixed order the shards divide."""
    return sorted([*frame.single_cell_train, *frame.held_out])


def build_tx1(setup: Setup, frame: Frame, shard: tuple[int, int]) -> dict[str, Any]:
    """Tx1-3B embeddings of every cell (``real/``) and every count-shuffled cell
    (``shuffled/``) of this shard's lines lacking them."""
    import time

    from src.data.basal import align_columns, symbol_column
    from src.data.ladder_cells import line_rng, shuffle_counts
    from src.data.tx1_cache import missing_lines, read_basal_source, write_line_cache
    from src.model.tx1 import _build_tx1_encoder

    index, count = shard
    lines = tx1_lines(frame)[index::count]
    root = Path(setup.config["paths"]["tx1_cache"])
    todo = {kind: missing_lines(root / kind, lines) for kind in ("real", "shuffled")}
    if not any(todo.values()):
        return {"lines": len(lines), "encoded": 0}
    encoder, _ = _build_tx1_encoder(
        Path(setup.joint["paths"]["tx1_model_dir"]),
        int(setup.config["tx1"]["batch_size"]),
        int(setup.config["tx1"]["max_length"]),
    )
    preparation = setup.joint["preparation"]
    hvg_order = settings(setup)["hvg_order"]
    directory = Path(setup.joint["paths"]["basal_raw_umi"])
    encoded, cells, seconds = 0, 0, 0.0
    for model_id in lines:
        for kind in ("real", "shuffled"):
            if model_id not in todo[kind]:
                continue
            source = read_basal_source(
                directory / f"{model_id}.h5ad",
                model_id=model_id,
                var_ensembl_col=preparation["var_ensembl_col"],
            )
            if kind == "shuffled":
                source.X = shuffle_counts(
                    source.X, line_rng(model_id, setup.seed, "count_shuffle")
                ).astype(np.float32)
            symbols = source.var[
                symbol_column(source.var, preparation["hvg_gene_symbol_col"])
            ].astype(str)
            hvg, _ = align_columns(source.X, symbols, hvg_order)
            start = time.monotonic()
            embeddings = encoder(source)
            elapsed = time.monotonic() - start
            write_line_cache(
                root / kind, model_id, embeddings, hvg.toarray(), source.obs
            )
            encoded += 1
            cells += source.n_obs
            seconds += elapsed
            _LOGGER.info(
                "tx1 %s %s: %d cells in %.0f s (%.0f cells/s)",
                kind,
                model_id,
                source.n_obs,
                elapsed,
                source.n_obs / max(elapsed, 1e-9),
            )
    return {
        "lines": len(lines),
        "encoded": encoded,
        "cells": cells,
        "cells_per_second": cells / max(seconds, 1e-9),
    }


# --- entry point ----------------------------------------------------------------


def _step_done(setup: Setup, step: str) -> bool:
    outputs = {
        "lines": [setup.root / "lines.parquet", setup.root / "genes.parquet"],
        "mean": [setup.root / "mean" / "pie_input.parquet"],
        "summaries": [setup.root / "summaries.parquet"],
    }
    return step in outputs and all(path.is_file() for path in outputs[step])


def run(
    config: Mapping[str, Any], steps: Sequence[str], shard: tuple[int, int]
) -> None:
    setup = load_setup(config)
    open_manifest(setup)
    frame = load_frame(setup)
    builders = {
        "lines": build_lines,
        "cells": build_cells,
        "mean": build_mean,
        "summaries": build_summaries,
    }
    for step in steps:
        if step == "tx1":
            record = build_tx1(setup, frame, shard)
            record_step(setup, f"tx1_shard_{shard[0]}_of_{shard[1]}", record)
            continue
        if _step_done(setup, step):
            _LOGGER.info("%s: written, skipping", step)
            continue
        record_step(setup, step, builders[step](setup, frame))
        _LOGGER.info("%s: done", step)


def _shard(value: str) -> tuple[int, int]:
    index, count = (int(part) for part in value.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"shard {value}: need 0 <= I < N")
    return index, count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", default="lines,cells,mean,summaries")
    parser.add_argument("--shard", type=_shard, default=(0, 1))
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS}")
    from src.experiments.config import load_ladder_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(load_ladder_config(args.config), steps, args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
