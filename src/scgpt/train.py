"""Train scGPT gene-score model with Hugging Face Accelerate."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.scgpt.data import GeneScoreDataset, collate_gene_score_batch
from src.scgpt.model import GeneScoreModel
from src.utils.data import get_condition_splits, load_adata
from src.utils.runtime import AccelerateRuntime


def run(config: dict) -> dict:
    """Run config-driven scGPT training."""
    runtime = AccelerateRuntime(config)
    adata = load_adata(config["data_config"]["h5ad_path"])
    split = get_condition_splits(config)
    pretrained_dir = Path(config["model_config"].get("pretrained_dir", "model/scGPT"))
    with (pretrained_dir / "vocab.json").open() as handle:
        vocab = json.load(handle)
    dataset = GeneScoreDataset(
        adata=adata,
        conditions=split["train"],
        vocab=vocab,
        n_bins=int(config["model_config"].get("preprocess_binning", 51)),
        condition_key=config["data_config"].get("condition_key", "condition"),
        control_key=config["data_config"].get("control_key", "control"),
        n_control_samples=int(config["data_config"].get("control_n_samples", 8)),
        seed=int(config["run_config"].get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training_config", {}).get("batch_size", 32)),
        shuffle=True,
        collate_fn=lambda batch: collate_gene_score_batch(batch, vocab, adata.n_vars),
        num_workers=int(config["data_config"].get("num_workers", 0)),
    )
    model = _build_model(config, adata.n_vars, dataset.gene_ids, runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training_config", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training_config", {}).get("weight_decay", 0.01)),
    )
    model, optimizer, loader = runtime.prepare(model, optimizer, loader)
    epochs = int(config.get("training_config", {}).get("epochs", 1))
    max_grad_norm = float(config.get("training_config", {}).get("max_grad_norm", 1.0))
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            logits = _forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["targets"].to(logits.device)
            )
            runtime.backward(loss)
            runtime.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    checkpoint_path = config["run_config"].get("save_checkpoint_path")
    if checkpoint_path:
        runtime.save_state_dict(model, checkpoint_path)
    return {"checkpoint_path": checkpoint_path, "n_train": len(dataset)}


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


def _forward(model, batch: dict) -> torch.Tensor:
    return model(
        batch["genes"],
        batch["values"],
        batch["padding_mask"],
        control_gene_ids=batch["control_genes"],
        control_values=batch["control_values"],
        control_padding_mask=batch["control_padding_mask"],
        control_counts=batch["control_counts"],
    )
