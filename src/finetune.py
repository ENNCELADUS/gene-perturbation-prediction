#!/usr/bin/env python
"""
Finetune scGPT for perturbation prediction on VCC dataset.

Usage (single GPU):
    python src/finetune.py --config src/configs/finetune.yaml

Usage (DDP with 4 GPUs):
    torchrun --nproc_per_node=4 src/finetune.py --config src/configs/finetune.yaml
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GEARS
from gears import PertData

# Import scGPT
scgpt_path = Path(__file__).parent.parent / "scGPT"
sys.path.insert(0, str(scgpt_path))

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

# Import utilities
from src.utils.ddp import setup_ddp, cleanup_ddp, is_main_process
from src.utils.data_split import (
    load_test_genes,
    get_perturbation_genes,
    create_train_val_split,
    filter_dataset_by_perts,
)
from src.utils.training import (
    load_pretrained_model,
    train_epoch,
    evaluate,
    compute_validation_metrics,
    save_model,
    freeze_encoder_layers,
)

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(save_dir: Path, rank: int = 0):
    """Setup scGPT logger with file handler (only on rank 0)."""
    logger = scg.logger
    if is_main_process(rank):
        scg.utils.add_file_handler(logger, save_dir / "run.log")
        logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return logger


def main():
    parser = argparse.ArgumentParser(description="Finetune scGPT for perturbation")
    parser.add_argument("--config", default="src/configs/finetune.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ========== Setup ==========
    rank, local_rank, world_size, is_distributed = setup_ddp()
    config = load_config(args.config)
    set_seed(args.seed)

    device = (
        torch.device(f"cuda:{local_rank}")
        if is_distributed
        else torch.device(
            config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
        )
    )

    save_dir = Path(config["paths"]["finetuned_model_dir"])
    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {device}, Saving to {save_dir}")

    if is_distributed:
        dist.barrier()

    logger = setup_logging(save_dir, rank)
    if is_main_process(rank):
        logger.info(f"Config: {config}")

    # ========== Load Data ==========
    if is_main_process(rank):
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
    vocab.set_default_index(vocab[config["model"]["pad_token"]])
    gene_ids = np.array(
        [
            vocab[g] if g in vocab else vocab[config["model"]["pad_token"]]
            for g in genes
        ],
        dtype=int,
    )
    n_genes = len(genes)

    if is_main_process(rank):
        logger.info(
            f"Dataset: {n_genes} genes, {np.sum(pert_data.adata.var['id_in_vocab'] >= 0)} in vocab"
        )

    # ========== Create Train/Val Split ==========
    split_config = config["split"]
    test_genes = load_test_genes(split_config["test_genes_file"])
    pert_genes = get_perturbation_genes(pert_data)
    train_perts, val_perts = create_train_val_split(
        pert_genes, test_genes, split_config["train_ratio"], split_config["seed"]
    )

    if is_main_process(rank):
        logger.info(
            f"Test: {len(test_genes)}, Train: {len(train_perts)}, Val: {len(val_perts)} perts"
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

    if is_main_process(rank):
        logger.info(f"Train cells: {len(train_indices)}, Val cells: {len(val_indices)}")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_subset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=max(1, batch_size // world_size),
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=0
        )

    val_loader = DataLoader(
        val_subset, batch_size=eval_batch_size, shuffle=False, num_workers=0
    )

    # ========== Load Model ==========
    model = load_pretrained_model(config, vocab, n_genes, device, logger)

    # ========== Freeze Encoder (if configured) ==========
    if config.get("training", {}).get("freeze_encoder", False):
        freeze_prefixes = config.get("training", {}).get(
            "freeze_prefixes", ["encoder", "value_encoder", "transformer_encoder"]
        )
        freeze_encoder_layers(
            model,
            freeze_prefixes=freeze_prefixes,
            logger=logger if is_main_process(rank) else None,
        )

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # ========== Optimizer/Scheduler ==========
    # Only optimize parameters that require gradients (respects freeze config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config["optimizer"]["schedule_interval"],
        gamma=config["optimizer"]["schedule_gamma"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["amp"])

    # ========== Training Loop ==========
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    best_overall_score = 0.0  # overall_score (0-100), higher is better
    best_model = None
    best_val_metrics = {}
    patience = 0

    if is_main_process(rank):
        logger.info(f"\n{'=' * 60}\nStarting training...\n{'=' * 60}")

    for epoch in range(1, config["optimizer"]["epochs"] + 1):
        epoch_start = time.time()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            gene_ids,
            config,
            epoch,
            logger if is_main_process(rank) else None,
        )

        if is_distributed:
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = loss_tensor.item()
            # All ranks wait here while rank 0 does validation
            # This prevents NCCL timeout during long validation
            dist.barrier()

        # Validation (rank 0 only)
        if is_main_process(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            val_res = evaluate(eval_model, val_loader, gene_ids, config, device)
            val_metrics = compute_validation_metrics(
                val_res, ctrl_adata, gene_names, config
            )

            logger.info(
                f"| epoch {epoch:3d} | time: {time.time() - epoch_start:5.2f}s | "
                f"loss: {train_loss:.4f} | PDS: {val_metrics['pds']:.4f} | "
                f"DES: {val_metrics['des']:.4f} | MAE: {val_metrics['mae']:.4f}"
            )
            logger.info(f"  overall_score: {val_metrics['overall_score']:.2f}")

            early_stop_flag = 0
            if val_metrics["overall_score"] > best_overall_score:
                best_overall_score = val_metrics["overall_score"]
                best_val_metrics = val_metrics.copy()
                best_model = copy.deepcopy(eval_model)
                save_model(eval_model, config, vocab, save_dir)
                logger.info(f"  -> New best (overall_score={best_overall_score:.2f})")
                patience = 0
            else:
                patience += 1
                if patience >= config["optimizer"]["early_stop"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    early_stop_flag = 1
        else:
            early_stop_flag = 0

        # Synchronize early stopping decision across all ranks
        if is_distributed:
            stop_tensor = torch.tensor([early_stop_flag], device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

        scheduler.step()

    # ========== Save ==========
    if is_main_process(rank):
        logger.info(
            f"\n{'=' * 60}\nTraining complete! Best: {best_val_metrics}\n{'=' * 60}"
        )
        save_model(best_model, config, vocab, save_dir)

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

    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
