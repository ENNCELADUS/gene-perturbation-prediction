"""Prepare Norman condition split artifacts for scGPT."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.scgpt.graph import (
    build_coexpression_gene_graph,
    gene_names_from_adata,
    graph_config_from_model_config,
    resolve_graph_path,
    save_gene_graph_edges,
)
from src.utils.data import (
    infer_gene_heldout_condition_split,
    get_condition_splits,
    load_adata,
    save_condition_split,
)


def run(config: dict) -> dict:
    """Generate a condition-level split artifact when config does not define one."""
    data_config = config["data_config"]
    split_path = _split_path(config)
    inline_split = data_config.get("condition_split", {})
    adata = None
    if isinstance(inline_split, Mapping) and any(inline_split.values()):
        split = get_condition_splits(config)
    else:
        adata = load_adata(data_config["h5ad_path"])
        split_config = data_config.get("split_config", {})
        if not isinstance(split_config, Mapping):
            raise ValueError("data_config.split_config must be a mapping")
        strategy = split_config.get("strategy", "gene_heldout")
        if strategy != "gene_heldout":
            raise ValueError("Only gene_heldout split strategy is supported")
        split = infer_gene_heldout_condition_split(
            adata=adata,
            condition_key=str(data_config.get("condition_key", "condition")),
            train_gene_fraction=float(split_config.get("train_gene_fraction", 0.7)),
            validation_gene_fraction=float(
                split_config.get("validation_gene_fraction", 0.1)
            ),
            test_gene_fraction=float(split_config.get("test_gene_fraction", 0.2)),
            min_cells_per_condition=int(split_config.get("min_cells_per_condition", 1)),
        )
    save_condition_split(split, split_path)
    conditions = split["conditions"] if "conditions" in split else split
    graph_result = _prepare_gene_graph(config, conditions, adata)
    raw_stats = split.get("stats", {})
    stats = raw_stats if isinstance(raw_stats, Mapping) else {}
    return {
        "split_path": str(split_path),
        "n_train": len(conditions["train"]),
        "n_validation": len(conditions["validation"]),
        "n_test": len(conditions["test"]),
        **graph_result,
        **stats,
    }


def _prepare_gene_graph(
    config: dict,
    conditions: Mapping[str, list[str]],
    adata,
) -> dict[str, object]:
    model_config = config.get("model_config", {})
    if not isinstance(model_config, Mapping):
        return {}
    graph_config = graph_config_from_model_config(
        model_config,
        fallback_path=resolve_graph_path(config),
    )
    if not graph_config.enabled or graph_config.source != "coexpression":
        return {}
    if adata is None:
        adata = load_adata(config["data_config"]["h5ad_path"])
    gene_names = gene_names_from_adata(adata)
    graph = build_coexpression_gene_graph(
        adata=adata,
        train_conditions=conditions["train"],
        gene_names=gene_names,
        condition_key=str(config["data_config"].get("condition_key", "condition")),
        control_key=str(config["data_config"].get("control_key", "control")),
        top_k=graph_config.top_k,
        directed=graph_config.directed,
        include_controls=graph_config.include_controls,
    )
    if graph_config.path is None:
        raise ValueError("coexpression graph requires a graph path")
    save_gene_graph_edges(graph, graph_config.path, gene_names)
    return {
        "graph_path": str(graph_config.path),
        "graph_source": "coexpression",
        "graph_n_edges": int(graph.edge_index.shape[1]),
        "graph_top_k": graph_config.top_k,
    }


def _split_path(config: Mapping[str, object]) -> Path:
    data_config = config.get("data_config", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data_config must be a mapping")
    path = data_config.get("condition_split_path")
    if path:
        return Path(str(path))
    run_config = config.get("run_config", {})
    study_name = "norman"
    if isinstance(run_config, Mapping) and run_config.get("study_name"):
        study_name = str(run_config["study_name"])
    h5ad_path = data_config.get("h5ad_path")
    if h5ad_path:
        return Path(str(h5ad_path)).with_name(f"{study_name}_condition_split.yaml")
    return Path("data") / study_name / f"{study_name}_condition_split.yaml"
