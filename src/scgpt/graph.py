"""Gene graph loading and construction utilities for scGPT retrievers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import numpy as np
import torch

from src.utils.metrics import normalize_condition


@dataclass(frozen=True)
class GeneGraphConfig:
    """Configuration for gene graph construction or loading."""

    enabled: bool = False
    source: str = "none"
    path: Path | None = None
    top_k: int = 50
    directed: bool = False
    include_controls: bool = True

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object] | None,
        fallback_path: Path | None = None,
    ) -> "GeneGraphConfig":
        """Build graph config from a model_config.graph mapping."""
        if config is None:
            return cls()
        source = str(config.get("source", "none")).lower()
        enabled = bool(config.get("enabled", source not in {"none", "disabled"}))
        path_value = config.get("path")
        path = Path(str(path_value)) if path_value else fallback_path
        return cls(
            enabled=enabled,
            source=source,
            path=path,
            top_k=int(config.get("top_k", 50)),
            directed=bool(config.get("directed", False)),
            include_controls=bool(config.get("include_controls", True)),
        )


@dataclass(frozen=True)
class GeneGraph:
    """Sparse gene graph aligned to model output gene order."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    diagnostics: dict[str, int | float | str | bool]


def resolve_graph_path(config: Mapping[str, object]) -> Path:
    """Resolve the graph path used by prepare/train/evaluate."""
    model_config = config.get("model_config", {})
    run_config = config.get("run_config", {})
    graph_config = (
        model_config.get("graph", {}) if isinstance(model_config, Mapping) else {}
    )
    if isinstance(graph_config, Mapping) and graph_config.get("path"):
        return Path(str(graph_config["path"]))
    study_name = "scgpt"
    if isinstance(run_config, Mapping) and run_config.get("study_name"):
        study_name = str(run_config["study_name"])
    top_k = 50
    if isinstance(graph_config, Mapping) and graph_config.get("top_k") is not None:
        top_k = int(graph_config["top_k"])
    return Path("data") / "scgpt_graphs" / f"{study_name}_coexpression_top{top_k}.csv"


def gene_names_from_adata(adata: ad.AnnData) -> list[str]:
    """Return model output gene names from AnnData var metadata."""
    if "gene_name" in adata.var.columns:
        return [str(value) for value in adata.var["gene_name"].tolist()]
    return [str(value) for value in adata.var_names.tolist()]


def graph_config_from_model_config(
    model_config: Mapping[str, object],
    fallback_path: Path | None = None,
) -> GeneGraphConfig:
    """Return a normalized gene graph config from model config."""
    graph_config = model_config.get("graph")
    if graph_config is None:
        return GeneGraphConfig()
    if not isinstance(graph_config, Mapping):
        raise ValueError("model_config.graph must be a mapping")
    return GeneGraphConfig.from_mapping(graph_config, fallback_path=fallback_path)


def load_gene_graph(
    config: GeneGraphConfig,
    gene_names: Sequence[str],
    device: torch.device | str | None = None,
) -> GeneGraph | None:
    """Load a sparse gene graph from an edge-list file."""
    if not config.enabled or config.source in {"none", "disabled"}:
        return None
    if config.source not in {"edge_list", "coexpression"}:
        raise ValueError(
            "model_config.graph.source must be 'edge_list', 'coexpression', or 'none'"
        )
    if config.path is None:
        raise ValueError("model_config.graph.path is required when graph is enabled")
    if not config.path.exists():
        raise FileNotFoundError(f"Gene graph file does not exist: {config.path}")

    name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    matched_edges = 0
    dropped_edges = 0
    with config.path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_edge_columns(reader.fieldnames)
        for row in reader:
            source_idx = _node_to_index(row["source"], name_to_idx, len(gene_names))
            target_idx = _node_to_index(row["target"], name_to_idx, len(gene_names))
            if source_idx is None or target_idx is None or source_idx == target_idx:
                dropped_edges += 1
                continue
            weight = float(row.get("weight") or 1.0)
            sources.append(source_idx)
            targets.append(target_idx)
            weights.append(weight)
            matched_edges += 1
            if not config.directed and config.source == "edge_list":
                sources.append(target_idx)
                targets.append(source_idx)
                weights.append(weight)

    edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return GeneGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        diagnostics={
            "source": config.source,
            "path": str(config.path),
            "directed": config.directed,
            "n_genes": len(gene_names),
            "matched_edges": matched_edges,
            "dropped_edges": dropped_edges,
            "n_edges": len(weights),
        },
    )


def build_coexpression_gene_graph(
    adata: ad.AnnData,
    train_conditions: Sequence[str],
    gene_names: Sequence[str],
    condition_key: str,
    control_key: str,
    top_k: int,
    directed: bool,
    include_controls: bool = True,
) -> GeneGraph:
    """Build a train-only top-k absolute-correlation coexpression graph."""
    if top_k < 1:
        raise ValueError("graph top_k must be at least 1")
    selected = _train_cell_mask(
        adata=adata,
        train_conditions=train_conditions,
        condition_key=condition_key,
        control_key=control_key,
        include_controls=include_controls,
    )
    expression = _expression_matrix(adata, selected)
    n_genes = len(gene_names)
    if expression.shape[1] != n_genes:
        raise ValueError(
            "gene_names length must match AnnData expression width: "
            f"{n_genes} != {expression.shape[1]}"
        )
    normalized_expression = _normalized_expression(expression)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    sample_count = min(top_k, max(0, n_genes - 1))
    for source_idx, target_indices, target_weights in _iter_topk_abs_correlations(
        normalized_expression,
        sample_count,
    ):
        for target_idx, weight in zip(target_indices, target_weights):
            if not np.isfinite(weight):
                continue
            sources.append(source_idx)
            targets.append(int(target_idx))
            weights.append(float(weight))
            if not directed:
                sources.append(int(target_idx))
                targets.append(source_idx)
                weights.append(float(weight))

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return GeneGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        diagnostics={
            "source": "coexpression",
            "directed": directed,
            "top_k": top_k,
            "n_cells": int(selected.sum()),
            "n_genes": n_genes,
            "n_edges": len(weights),
        },
    )


def save_gene_graph_edges(
    graph: GeneGraph,
    path: str | Path,
    gene_names: Sequence[str],
) -> None:
    """Save a sparse gene graph as a CSV edge list."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_index = graph.edge_index.detach().cpu()
    edge_weight = graph.edge_weight.detach().cpu()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "weight"])
        writer.writeheader()
        for edge_idx in range(edge_index.shape[1]):
            source_idx = int(edge_index[0, edge_idx])
            target_idx = int(edge_index[1, edge_idx])
            writer.writerow(
                {
                    "source": gene_names[source_idx],
                    "target": gene_names[target_idx],
                    "weight": f"{float(edge_weight[edge_idx]):.8g}",
                }
            )


def _validate_edge_columns(fieldnames: Sequence[str] | None) -> None:
    required = {"source", "target"}
    if fieldnames is None or not required.issubset(set(fieldnames)):
        raise ValueError("gene graph edge list must contain source and target columns")


def _node_to_index(
    value: str,
    name_to_idx: Mapping[str, int],
    n_genes: int,
) -> int | None:
    value = str(value)
    if value in name_to_idx:
        return name_to_idx[value]
    try:
        index = int(value)
    except ValueError:
        return None
    return index if 0 <= index < n_genes else None


def _train_cell_mask(
    adata: ad.AnnData,
    train_conditions: Sequence[str],
    condition_key: str,
    control_key: str,
    include_controls: bool,
) -> np.ndarray:
    train_condition_set = {
        normalize_condition(condition) for condition in train_conditions
    }
    conditions = [
        normalize_condition(str(condition))
        for condition in adata.obs[condition_key].tolist()
    ]
    mask = np.asarray([condition in train_condition_set for condition in conditions])
    if include_controls:
        mask = mask | (adata.obs[control_key].to_numpy() == 1)
    if not mask.any():
        raise ValueError("Cannot build coexpression graph without train/control cells")
    return mask


def _expression_matrix(adata: ad.AnnData, selected: np.ndarray) -> np.ndarray:
    layer = adata.layers["counts"] if "counts" in adata.layers else adata.X
    matrix = layer[selected]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    expression = np.asarray(matrix, dtype=np.float32)
    if expression.ndim != 2:
        raise ValueError("Expression matrix must be two-dimensional")
    return expression


def _normalized_expression(expression: np.ndarray) -> np.ndarray:
    centered = expression - expression.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    return centered / np.maximum(norms, 1.0e-8)


def _iter_topk_abs_correlations(
    normalized_expression: np.ndarray,
    top_k: int,
    block_size: int = 512,
):
    if top_k == 0:
        return
    n_genes = normalized_expression.shape[1]
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        correlations = np.abs(
            normalized_expression[:, start:stop].T @ normalized_expression
        ).astype(np.float32, copy=False)
        for offset, row in enumerate(correlations):
            source_idx = start + offset
            row[source_idx] = -np.inf
            target_indices = _topk_indices(row, top_k)
            yield source_idx, target_indices, row[target_indices]


def _topk_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= values.size:
        return np.argsort(-values)
    candidates = np.argpartition(-values, kth=top_k - 1)[:top_k]
    return candidates[np.argsort(-values[candidates])]
