#!/usr/bin/env python
"""
Train Linear Regression model for perturbation prediction on VCC dataset.

Usage:
    python src/linear.py --config src/configs/linear.yaml
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GEARS
from gears import PertData

# Import scGPT for vocab
scgpt_path = Path(__file__).parent.parent / "scGPT"
sys.path.insert(0, str(scgpt_path))

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

# Import utilities
from src.utils.data_split import (
    load_test_genes,
    get_perturbation_genes,
    create_train_val_split,
    filter_dataset_by_perts,
)
from src.utils.training import compute_validation_metrics
from src.model.linear import LinearPerturbationModel

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_genes: int,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
) -> float:
    """
    Train the linear model for one epoch.

    Args:
        model: Linear model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        n_genes: Number of genes
        device: Torch device
        epoch: Current epoch number
        config: Configuration dict
        logger: Logger instance

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    log_interval = config["logging"]["log_interval"]
    num_batches = len(train_loader)
    amp_enabled = config["training"]["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for batch_idx, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        # Extract data from batch - GEARS format
        x: torch.Tensor = batch_data.x
        ori_gene_values = x.squeeze(-1).view(batch_size, n_genes)
        target_gene_values = batch_data.y

        # Build pert_flags from pert_idx
        pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.float, device=device)
        pert_idx = batch_data.pert_idx
        for i in range(batch_size):
            idx = pert_idx[i] if isinstance(pert_idx, list) else pert_idx[i]
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            for p_idx in idx:
                if p_idx >= 0 and p_idx < n_genes:
                    pert_flags[i, p_idx] = 1

        # Forward pass
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(ori_gene_values, pert_flags)
            loss = criterion(output, target_gene_values)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                f"ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.4f}"
            )
            total_loss = 0
            start_time = time.time()

    return total_loss / max(1, num_batches % log_interval)


def evaluate_linear(
    model: nn.Module,
    loader: DataLoader,
    n_genes: int,
    device: torch.device,
) -> dict:
    """
    Evaluate linear model on a data loader.

    Returns:
        Dictionary with predictions and ground truth per perturbation
    """
    model.eval()

    pert_cat = []
    pred = []
    truth = []

    for batch_data in loader:
        batch_size = len(batch_data.y)
        batch_data.to(device)

        # Extract data from batch - GEARS format
        x: torch.Tensor = batch_data.x
        ori_gene_values = x.squeeze(-1).view(batch_size, n_genes)
        target_gene_values = batch_data.y

        # Build pert_flags from pert_idx
        pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.float, device=device)
        pert_idx = batch_data.pert_idx
        for i in range(batch_size):
            idx = pert_idx[i] if isinstance(pert_idx, list) else pert_idx[i]
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            for p_idx in idx:
                if p_idx >= 0 and p_idx < n_genes:
                    pert_flags[i, p_idx] = 1

        pert_cat.extend(batch_data.pert)

        with torch.no_grad():
            output = model(ori_gene_values, pert_flags)
            pred.extend(output.cpu())
            truth.extend(target_gene_values.cpu())

    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).detach().cpu().numpy().astype(np.float32),
        "truth": torch.stack(truth).detach().cpu().numpy().astype(np.float32),
    }


def save_linear_model(model: nn.Module, n_genes: int, save_dir: Path):
    """Save linear model checkpoint and config."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), save_dir / "best_model.pt")

    # Save model config
    with open(save_dir / "model_config.json", "w") as f:
        json.dump({"n_genes": n_genes}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train Linear model for perturbation")
    parser.add_argument("--config", default="src/configs/linear.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ========== Setup ==========
    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(
        config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    )

    save_dir = Path(config["paths"]["linear_model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}, Saving to {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: {config}")

    # ========== Load Data ==========
    logger.info("Loading data...")

    pert_data = PertData(config["paths"]["gears_data_dir"], default_pert_graph=False)
    pert_data.load(
        data_path=os.path.join(
            config["paths"]["gears_data_dir"], config["paths"]["dataset_name"]
        )
    )

    # ========== Load Vocabulary ==========
    vocab_file = Path(config["paths"]["model_dir"]) / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in config["data"]["special_tokens"]:
        if s not in vocab:
            vocab.append_token(s)

    # Map genes to vocabulary
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_names = np.array(genes)
    n_genes = len(genes)

    n_in_vocab = np.sum(pert_data.adata.var["id_in_vocab"] >= 0)
    logger.info(f"Dataset: {n_genes} genes, {n_in_vocab} in vocab")

    # ========== Create Train/Val Split ==========
    split_config = config["split"]
    test_genes = load_test_genes(split_config["test_genes_file"])
    pert_genes = get_perturbation_genes(pert_data)
    train_perts, val_perts = create_train_val_split(
        pert_genes, test_genes, split_config["train_ratio"], split_config["seed"]
    )

    logger.info(
        f"Test: {len(test_genes)}, Train: {len(train_perts)}, "
        f"Val: {len(val_perts)} perts"
    )

    # ========== Create Dataloaders ==========
    pert_data.prepare_split(split="no_test", seed=split_config["seed"])
    batch_size = config["optimizer"]["batch_size"]
    eval_batch_size = config["optimizer"]["eval_batch_size"]
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
    full_dataset = pert_data.dataloader["train_loader"].dataset

    train_indices = filter_dataset_by_perts(
        full_dataset, train_perts, include_ctrl=True
    )
    val_indices = filter_dataset_by_perts(full_dataset, val_perts, include_ctrl=False)
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    logger.info(f"Train cells: {len(train_indices)}, Val cells: {len(val_indices)}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=eval_batch_size, shuffle=False, num_workers=0
    )

    # ========== Initialize Model ==========
    model = LinearPerturbationModel(n_genes)
    model.to(device)
    logger.info(f"Linear model initialized with {n_genes} genes")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== Loss Function ==========
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "MSELoss")
    reduction = loss_cfg.get("reduction", "mean")

    if loss_type == "SmoothL1Loss":
        beta = loss_cfg.get("beta", 1.0)
        criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)
    elif loss_type == "MSELoss":
        criterion = nn.MSELoss(reduction=reduction)
    elif loss_type == "L1Loss":
        criterion = nn.L1Loss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    logger.info(f"Using loss: {loss_type}")

    # ========== Optimizer ==========
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 0),
    )

    # ========== Training Loop ==========
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    best_overall_score = 0.0
    best_model = None
    best_val_metrics = {}
    patience = 0

    logger.info(f"\n{'=' * 60}\nStarting training...\n{'=' * 60}")

    for epoch in range(1, config["optimizer"]["epochs"] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            n_genes,
            device,
            epoch,
            config,
            logger,
        )

        # Validation
        val_res = evaluate_linear(model, val_loader, n_genes, device)
        val_metrics = compute_validation_metrics(
            val_res, ctrl_adata, gene_names, config
        )

        logger.info(
            f"| epoch {epoch:3d} | time: {time.time() - epoch_start:5.2f}s | "
            f"loss: {train_loss:.4f} | PDS: {val_metrics['pds']:.4f} | "
            f"DES: {val_metrics['des']:.4f} | MAE: {val_metrics['mae']:.4f}"
        )
        logger.info(f"  overall_score: {val_metrics['overall_score']:.2f}")

        if val_metrics["overall_score"] > best_overall_score:
            best_overall_score = val_metrics["overall_score"]
            best_val_metrics = val_metrics.copy()
            best_model = copy.deepcopy(model)
            logger.info(f"  -> New best (overall_score={best_overall_score:.2f})")
            patience = 0
        else:
            patience += 1
            if patience >= config["optimizer"]["early_stop"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ========== Save ==========
    logger.info(
        f"\n{'=' * 60}\nTraining complete! Best: {best_val_metrics}\n{'=' * 60}"
    )
    save_linear_model(best_model, n_genes, save_dir)

    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(
            {
                "n_train": len(train_perts),
                "n_val": len(val_perts),
                "val_perts": val_perts,
                "best_metrics": best_val_metrics,
            },
            f,
            indent=2,
        )

    logger.info(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
