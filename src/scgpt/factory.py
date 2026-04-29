"""Factory helpers for scGPT gene-score models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch

from src.scgpt.graph import (
    graph_config_from_model_config,
    load_gene_graph,
    resolve_graph_path,
)
from src.scgpt.model import GeneScoreModel
from src.utils.distributed import log_primary_info


def build_gene_score_model(
    config: dict,
    n_genes: int,
    gene_ids: Sequence[int] | torch.Tensor,
    device: torch.device,
    gene_names: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> GeneScoreModel:
    """Build a configured scGPT gene-score model."""
    model_config = config["model_config"]
    pretrained_dir = Path(model_config.get("pretrained_dir", "model/scGPT"))
    graph = _load_model_graph(config, model_config, gene_names, device, logger)
    architecture_config = _architecture_config(model_config)
    return GeneScoreModel(
        n_genes=n_genes,
        checkpoint_path=pretrained_dir / "best_model.pt",
        vocab_path=pretrained_dir / "vocab.json",
        args_path=pretrained_dir / "args.json",
        score_gene_ids=gene_ids,
        freeze_encoder=bool(model_config.get("freeze_encoder", True)),
        freeze_layers_up_to=int(model_config.get("freeze_layers_up_to", 10)),
        score_mode=str(model_config.get("score_mode", "dot")),
        head_hidden_dim=int(model_config.get("head_hidden_dim", 512)),
        head_dropout=float(model_config.get("head_dropout", 0.2)),
        gene_graph_edge_index=graph.edge_index if graph is not None else None,
        gene_graph_edge_weight=graph.edge_weight if graph is not None else None,
        use_graph_encoder=bool(architecture_config.get("use_graph_encoder", False)),
        graph_message_layers=int(architecture_config.get("graph_message_layers", 2)),
        use_contrast_encoder=bool(
            architecture_config.get("use_contrast_encoder", True)
        ),
        use_slots=bool(architecture_config.get("use_slots", False)),
        n_target_slots=int(architecture_config.get("n_target_slots", 4)),
        slot_aggregation=str(architecture_config.get("slot_aggregation", "logsumexp")),
        use_cardinality_head=bool(
            architecture_config.get("use_cardinality_head", False)
        ),
        max_cardinality=int(architecture_config.get("max_cardinality", 4)),
        cardinality_loss_weight=float(
            architecture_config.get("cardinality_loss_weight", 0.1)
        ),
        use_cycle_loss=bool(architecture_config.get("use_cycle_loss", False)),
        cycle_loss_weight=float(architecture_config.get("cycle_loss_weight", 0.1)),
        alignment_heads=int(architecture_config.get("alignment_heads", 4)),
        use_fast_transformer=bool(model_config.get("use_fast_transformer", False)),
        fast_transformer_backend=str(
            model_config.get("fast_transformer_backend", "flash")
        ),
        device=device,
    )


def _architecture_config(model_config: dict) -> dict:
    architecture = model_config.get("architecture", {})
    if architecture is None:
        return {}
    if not isinstance(architecture, dict):
        raise ValueError("model_config.architecture must be a mapping")
    return architecture


def _load_model_graph(
    config: dict,
    model_config: dict,
    gene_names: list[str] | None,
    device: torch.device,
    logger: logging.Logger | None,
):
    graph_config = graph_config_from_model_config(
        model_config,
        fallback_path=resolve_graph_path(config),
    )
    if not graph_config.enabled:
        return None
    if gene_names is None:
        raise ValueError("gene_names are required when model_config.graph is enabled")
    graph = load_gene_graph(graph_config, gene_names=gene_names, device=device)
    if graph is not None and logger is not None:
        log_primary_info(
            logger,
            "Loaded scGPT gene graph: source=%s edges=%s matched=%s path=%s",
            graph.diagnostics.get("source"),
            graph.diagnostics.get("n_edges"),
            graph.diagnostics.get("matched_edges", graph.diagnostics.get("n_edges")),
            graph.diagnostics.get("path", ""),
        )
    return graph
