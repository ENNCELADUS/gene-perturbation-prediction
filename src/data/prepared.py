"""Prepared joint-training inputs: per-line arrays, response targets, manifest.

Layout under ``config["prepared_root"]``:

- ``prepared_inputs.json`` (written last): ``expression_space``,
  ``common_gene_panel``, ``hvg_order``, ``response_anchors``;
- ``lines/<ModelID>.npz``: ``controls_tx1``, ``basal_hvg``, ``q_sc_values``,
  ``q_sc_available``;
- ``response/``: log-space response targets (``src.data.response_cache``);
- ``common_gene_panel.csv``, ``embedding_union.{csv,json}``.

Every expression quantity is in STATE's space: whole-library normalize_total
to ``target_sum``, then log1p. Only Tx1 embeddings come from raw counts.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.basal import align_columns
from src.data.embeddings import load_esm2_embeddings
from src.data.expression import library_sizes, log_normalize
from src.data.geneeffect import (
    fit_residual_scale,
    fit_selective_genes,
    fit_variable_gene_membership,
    load_geneeffect_long,
)
from src.data.q_sc import QScFeatures, compute_q_sc
from src.data.residual_target import fit_gene_means
from src.data.response_cache import ResponseTargetsCache, open_response_targets
from src.data.splits import FixedSplit, assert_fit_eligible, load_geneeffect_226_split

PREPARED_METADATA_FILENAME: Final[str] = "prepared_inputs.json"
EXPRESSION_TRANSFORM: Final[str] = "log1p_normalize_total"


@dataclass(frozen=True)
class PreparedLine:
    """One line's selected basal cells: Tx1 embeddings, log-space HVG, q_sc."""

    controls_tx1: np.ndarray
    basal_hvg: np.ndarray
    q_sc: QScFeatures


@dataclass(frozen=True)
class PreparedInputs:
    """Fixed labels, train-fit preprocessing, feature orders and prepared arrays.

    Every expression quantity is in STATE's space: whole-library normalize_total
    to ``target_sum``, then log1p. Only Tx1 embeddings come from raw counts.
    """

    split: FixedSplit
    labels: pd.DataFrame
    genes: tuple[str, ...]
    train_gene_means: pd.Series
    variable_genes: frozenset[str]
    selective_genes: frozenset[str]
    residual_scale: pd.Series
    hvg_order: tuple[str, ...]
    esm2_symbols: tuple[str, ...]
    esm2_vectors: np.ndarray = field(repr=False, compare=False)
    lines: Mapping[str, PreparedLine] = field(repr=False, compare=False)
    response_targets: ResponseTargetsCache = field(repr=False, compare=False)
    response_anchors: tuple[str, ...]
    target_sum: float

    def preprocessing_state(self) -> dict[str, object]:
        """Return checkpoint-ready fitted state, including actual ESM2 vectors."""
        import torch  # deferred: basal-line preparation workers never need it

        return {
            "target_sum": float(self.target_sum),
            "gene_means": {
                "symbols": list(self.genes),
                "values": [float(self.train_gene_means[gene]) for gene in self.genes],
            },
            "variable_genes": [
                gene for gene in self.genes if gene in self.variable_genes
            ],
            "selective_genes": [
                gene for gene in self.genes if gene in self.selective_genes
            ],
            "residual_scale": {
                "symbols": list(self.genes),
                "values": [float(self.residual_scale[gene]) for gene in self.genes],
            },
            "esm2_symbols": list(self.esm2_symbols),
            "esm2_vectors": torch.from_numpy(
                np.asarray(self.esm2_vectors, dtype=np.float32)
            ).clone(),
        }


# --- one basal line --------------------------------------------------------------


def select_context_cells(
    model_id: str, cell_ids: Sequence[object], count: int
) -> np.ndarray:
    """``count`` cell positions ranked by ``sha256(ModelID|cell)``.

    Repeats the ranking when the line has fewer cells than ``count``.
    """
    identifiers = tuple(str(value) for value in cell_ids)
    ranked = sorted(
        range(len(identifiers)),
        key=lambda index: hashlib.sha256(
            f"{model_id}|{identifiers[index]}".encode()
        ).digest(),
    )
    selected = ranked[: min(count, len(ranked))]
    while len(selected) < count:
        selected.append(selected[len(selected) % len(ranked)])
    return np.asarray(selected, dtype=np.int64)


def prepare_line(
    counts: object,
    cell_ids: Sequence[str],
    symbols: Sequence[str],
    embeddings: np.ndarray,
    cached_cells: Sequence[str],
    *,
    model_id: str,
    hvg_order: Sequence[str],
    genes: Sequence[str],
    target_sum: float,
    cells_per_context: int,
) -> PreparedLine:
    """Log-space HVG and q_sc of one line from its raw source matrix.

    ``counts`` holds raw UMI over every gene of the source (``cell_ids`` rows,
    ``symbols`` columns); the library size of each cell is its total over
    all of them. The context cells are the Tx1-cached cells, in
    :func:`select_context_cells` order; q_sc summarises every source cell.
    """
    sizes = library_sizes(counts)
    order = select_context_cells(model_id, cached_cells, cells_per_context)
    rows = pd.Index(cell_ids).get_indexer([cached_cells[i] for i in order])
    if (rows < 0).any():
        raise ValueError(f"{model_id}: Tx1-cached cells are missing from the source")
    hvg, _ = align_columns(counts[rows], symbols, hvg_order)
    panel_symbols = [str(symbol).strip().upper() for symbol in symbols]
    return PreparedLine(
        controls_tx1=np.array(embeddings[order], dtype=np.float32),
        basal_hvg=log_normalize(hvg, sizes[rows], target_sum),
        q_sc=compute_q_sc(counts, panel_symbols, genes, sizes, target_sum),
    )


def write_prepared_line(path: Path, line: PreparedLine) -> None:
    """Atomically write one line's ``.npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            controls_tx1=line.controls_tx1,
            basal_hvg=line.basal_hvg,
            q_sc_values=line.q_sc.values,
            q_sc_available=line.q_sc.available,
        )
    os.replace(temporary, path)


def read_prepared_line(path: Path, genes: tuple[str, ...]) -> PreparedLine:
    with np.load(path) as payload:
        return PreparedLine(
            controls_tx1=payload["controls_tx1"],
            basal_hvg=payload["basal_hvg"],
            q_sc=QScFeatures(
                symbols=genes,
                values=payload["q_sc_values"],
                available=payload["q_sc_available"],
            ),
        )


# --- opening the prepared root ---------------------------------------------------


def read_manifest(root: Path) -> dict[str, Any]:
    """The prepared manifest; refuses one prepared before the log expression space."""
    path = Path(root) / PREPARED_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run preparation first")
    manifest = json.loads(path.read_text())
    if "expression_space" not in manifest:
        raise ValueError(
            f"{path} has no expression_space: it was prepared in raw-count space by "
            "older code and STATE must not read it; prepare a new prepared_root"
        )
    return manifest


def _restored_target_sum(preprocessing: Mapping[str, Any], target_sum: float) -> None:
    if "target_sum" not in preprocessing:
        raise ValueError(
            "checkpoint preprocessing has no target_sum: the model was trained on "
            "raw-count inputs that predate the log expression_space; retrain it"
        )
    if float(preprocessing["target_sum"]) != target_sum:
        raise ValueError(
            f"checkpoint target_sum {preprocessing['target_sum']} differs from the "
            f"prepared expression_space target_sum {target_sum}"
        )


def _restored_keys(preprocessing: Mapping[str, Any]) -> None:
    missing = [
        key for key in ("selective_genes", "residual_scale") if key not in preprocessing
    ]
    if missing:
        raise ValueError(
            f"checkpoint preprocessing has no {', '.join(missing)}: it predates the "
            "selective-gene revision; retrain it"
        )


def load_inputs(
    config: Mapping[str, Any],
    *,
    preprocessing: Mapping[str, Any] | None = None,
    include_test: bool = False,
) -> PreparedInputs:
    """Open prepared inputs; fit (or restore) gene means, variable and selective
    genes and the residual scale.

    Fitting uses labeled training lines only. ``preprocessing`` restores a
    checkpoint's fitted state instead. Test labels and lines are opened only
    with ``include_test``.
    """
    root = Path(config["prepared_root"])
    manifest = read_manifest(root)
    target_sum = float(manifest["expression_space"]["target_sum"])
    if preprocessing is not None:
        _restored_target_sum(preprocessing, target_sum)
        _restored_keys(preprocessing)
    paths, features = config["paths"], config["features"]
    split = load_geneeffect_226_split(Path(paths["split"]))
    genes = tuple(manifest["common_gene_panel"])
    labels = load_geneeffect_long(Path(paths["gene_effect"]), split)
    labels = labels.loc[labels["gene_symbol"].isin(genes)].copy()
    train = split.supervised_train

    if preprocessing is None:
        for model_id in train:
            assert_fit_eligible(model_id, split)
        gene_means = fit_gene_means(labels, train).loc[list(genes)]
    else:
        state = preprocessing["gene_means"]
        gene_means = pd.Series(
            np.asarray(state["values"], dtype=np.float64),
            index=list(state["symbols"]),
            name="gene_mean",
        ).loc[list(genes)]
    gene_means.index.name = "gene_symbol"
    labels["residual"] = labels["gene_effect"] - labels["gene_symbol"].map(gene_means)

    if preprocessing is None:
        variable_genes = fit_variable_gene_membership(
            labels,
            train,
            genes,
            min_observations=int(features["variable_gene_min_observations"]),
            percentile=float(features["variable_gene_percentile"]),
        )
        selective_genes = fit_selective_genes(
            labels,
            train,
            genes,
            min_lines=int(features["selective_min_lines"]),
            max_fraction=float(features["selective_max_fraction"]),
        )
        residual_scale = fit_residual_scale(
            labels,
            train,
            genes,
            floor_percentile=float(features["residual_sd_floor_percentile"]),
        )
        table = load_esm2_embeddings(Path(paths["esm2_embeddings"]))
        esm2_symbols = tuple(table.vectors_by_symbol)
        esm2_vectors = np.stack([table.vectors_by_symbol[s] for s in esm2_symbols])
    else:
        variable_genes = frozenset(preprocessing["variable_genes"])
        selective_genes = frozenset(preprocessing["selective_genes"])
        state = preprocessing["residual_scale"]
        residual_scale = pd.Series(
            np.asarray(state["values"], dtype=np.float64),
            index=list(state["symbols"]),
            name="residual_scale",
        ).loc[list(genes)]
        residual_scale.index.name = "gene_symbol"
        values = residual_scale.to_numpy()
        if not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError("restored residual_scale must be finite and positive")
        esm2_symbols = tuple(preprocessing["esm2_symbols"])
        vectors = preprocessing["esm2_vectors"]
        import torch

        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
        esm2_vectors = np.asarray(vectors)

    exposed = {*split.train, *split.val, *(split.test if include_test else ())}
    exposed -= set(split.unlabeled_train)
    labels = labels.loc[
        labels["model_id"].isin(exposed) & np.isfinite(labels["gene_effect"]),
        ["model_id", "gene_symbol", "gene_effect", "residual"],
    ].reset_index(drop=True)
    lines = {
        model_id: read_prepared_line(root / "lines" / f"{model_id}.npz", genes)
        for model_id in split.all_model_ids
        if model_id in exposed
    }
    return PreparedInputs(
        split=split,
        labels=labels,
        genes=genes,
        train_gene_means=gene_means,
        variable_genes=variable_genes,
        selective_genes=selective_genes,
        residual_scale=residual_scale,
        hvg_order=tuple(manifest["hvg_order"]),
        esm2_symbols=esm2_symbols,
        esm2_vectors=np.asarray(esm2_vectors, dtype=np.float32),
        lines=lines,
        response_targets=open_response_targets(root / "response"),
        response_anchors=tuple(manifest["response_anchors"]),
        target_sum=target_sum,
    )


__all__ = [
    "EXPRESSION_TRANSFORM",
    "PREPARED_METADATA_FILENAME",
    "PreparedInputs",
    "PreparedLine",
    "load_inputs",
    "prepare_line",
    "read_manifest",
    "read_prepared_line",
    "select_context_cells",
    "write_prepared_line",
]
