"""Evaluate scGPT gene-score model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.scgpt.data import GeneScoreDataset, collate_gene_score_batch
from src.scgpt.model import GeneScoreModel
from src.utils.data import get_condition_splits, load_adata
from src.utils.metrics import compute_gene_metrics, target_indices_for_conditions


def run(config: dict) -> dict:
    """Run gene-ranking evaluation for a finetuned scGPT scorer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_adata(config["data_config"]["h5ad_path"])
    split = get_condition_splits(config)
    pretrained_dir = Path(config["model_config"].get("pretrained_dir", "model/scGPT"))
    with (pretrained_dir / "vocab.json").open() as handle:
        vocab = json.load(handle)
    dataset = GeneScoreDataset(
        adata=adata,
        conditions=split["test"],
        vocab=vocab,
        n_bins=int(config["model_config"].get("preprocess_binning", 51)),
        condition_key=config["data_config"].get("condition_key", "condition"),
        control_key=config["data_config"].get("control_key", "control"),
        n_control_samples=int(config["data_config"].get("control_n_samples", 8)),
        seed=int(config["run_config"].get("seed", 42)),
    )
    model = _build_model(config, adata.n_vars, dataset.gene_ids, device)
    checkpoint_path = config["run_config"].get("load_checkpoint_path")
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training_config", {}).get("batch_size", 32)),
        shuffle=False,
        collate_fn=lambda batch: collate_gene_score_batch(batch, vocab, adata.n_vars),
    )
    scores = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["genes"].to(device),
                batch["values"].to(device),
                batch["padding_mask"].to(device),
                control_gene_ids=batch["control_genes"].to(device),
                control_values=batch["control_values"].to(device),
                control_padding_mask=batch["control_padding_mask"].to(device),
                control_counts=batch["control_counts"],
            )
            scores.extend(row.cpu().numpy() for row in logits)
            targets.extend(
                target_indices_for_conditions(
                    batch["conditions"], dataset.gene_name_to_idx
                )
            )
    top_k_values = config.get("evaluation_config", {}).get("top_k_values", [1, 5, 10])
    metrics = compute_gene_metrics(scores, targets, top_k_values)
    return {"metrics": metrics}


def _build_model(
    config: dict,
    n_genes: int,
    gene_ids,
    device: torch.device,
) -> GeneScoreModel:
    model_config = config["model_config"]
    pretrained_dir = Path(model_config.get("pretrained_dir", "model/scGPT"))
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
        device=device,
    )
