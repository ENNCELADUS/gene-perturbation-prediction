"""Train scGPT gene-score model with Hugging Face Accelerate."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.scgpt.data import GeneScoreDataset, collate_gene_score_batch
from src.scgpt.model import GeneScoreModel
from src.utils.data import get_condition_splits, load_adata
from src.utils.distributed import disable_tqdm, log_primary_info
from src.utils.runtime import AccelerateRuntime

LOGGER = logging.getLogger(__name__)


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
    progress_disabled = disable_tqdm(config)
    for epoch_idx in range(epochs):
        epoch_number = epoch_idx + 1
        log_primary_info(
            LOGGER,
            "scGPT train epoch %d/%d started",
            epoch_number,
            epochs,
        )
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        batch_iter = tqdm(
            loader,
            desc=f"scgpt train epoch {epoch_number}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=progress_disabled,
            leave=False,
        )
        for batch in batch_iter:
            optimizer.zero_grad()
            logits = _forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["targets"].to(logits.device)
            )
            loss_value = float(loss.detach().cpu())
            epoch_loss += loss_value
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss_value:.4f}")
            runtime.backward(loss)
            runtime.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        mean_loss = epoch_loss / n_batches if n_batches else 0.0
        log_primary_info(
            LOGGER,
            "scGPT train epoch %d/%d complete: batches=%d mean_loss=%.6f %s",
            epoch_number,
            epochs,
            n_batches,
            mean_loss,
            _gpu_memory_summary(runtime.device),
        )
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
        use_fast_transformer=bool(model_config.get("use_fast_transformer", False)),
        fast_transformer_backend=str(
            model_config.get("fast_transformer_backend", "flash")
        ),
        device=device,
    )


def _gpu_memory_summary(device: torch.device) -> str:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "gpu_memory=not_available"

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
    max_allocated_gb = torch.cuda.max_memory_allocated(device_index) / 1024**3
    return (
        "gpu_memory="
        f"allocated={allocated_gb:.3f}GB "
        f"reserved={reserved_gb:.3f}GB "
        f"max_allocated={max_allocated_gb:.3f}GB"
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
