#!/usr/bin/env python
"""
Finetune scGPT for perturbation prediction on VCC dataset.

Based on scGPT Tutorial_Perturbation.ipynb

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
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import GEARS
from gears import PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis

# Import scGPT
scgpt_path = Path(__file__).parent.parent / "scGPT"
sys.path.insert(0, str(scgpt_path))

import scgpt as scg
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained, map_raw_id_to_vocab_id, set_seed

import yaml

warnings.filterwarnings("ignore")


# ========== DDP Utility Functions ==========
def setup_ddp():
    """
    Initialize distributed training if running under torchrun.

    Returns:
        Tuple of (rank, local_rank, world_size, is_distributed)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp(is_distributed: bool):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


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


def load_pretrained_model(
    config: dict,
    vocab: GeneVocab,
    n_genes: int,
    device: torch.device,
    logger,
) -> TransformerGenerator:
    """
    Load pretrained scGPT model and initialize for finetuning.

    Args:
        config: Configuration dictionary
        vocab: Gene vocabulary
        n_genes: Number of genes in dataset
        device: Torch device
        logger: Logger instance

    Returns:
        Initialized TransformerGenerator model
    """
    model_dir = Path(config["paths"]["pretrained_model_dir"])
    model_file = model_dir / "best_model.pt"
    args_file = model_dir / "args.json"

    # Load pretrained model config
    with open(args_file, "r") as f:
        model_configs = json.load(f)

    logger.info(f"Loading pretrained model from {model_file}")
    logger.info(
        f"Model config: embsize={model_configs['embsize']}, "
        f"nlayers={model_configs['nlayers']}, nheads={model_configs['nheads']}"
    )

    # Get model hyperparameters from pretrained config
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    # Initialize model
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntoken=ntokens,
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,  # Not used for perturbation
        vocab=vocab,
        dropout=config["model"]["dropout"],
        pad_token=config["data"]["pad_token"],
        pad_value=config["data"]["pad_value"],
        pert_pad_id=config["data"]["pert_pad_id"],
        use_fast_transformer=config["model"]["use_fast_transformer"],
    )

    # Load pretrained weights using scGPT's utility
    # This handles Wqkv -> in_proj_ key conversion when flash-attn is unavailable
    load_param_prefixes = config.get("load_param_prefixes", None)
    pretrained_dict = torch.load(model_file, map_location=device)
    model = load_pretrained(
        model,
        pretrained_dict,
        strict=False,
        prefix=load_param_prefixes,
        verbose=True,
    )

    model.to(device)
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    gene_ids: np.ndarray,
    config: dict,
    epoch: int,
    logger,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    start_time = time.time()

    n_genes = len(gene_ids)
    device = next(model.parameters()).device
    amp_enabled = config["training"]["amp"]
    log_interval = config["logging"]["log_interval"]
    include_zero_gene = config["data"]["include_zero_gene"]
    max_seq_len = config["data"]["max_seq_len"]

    num_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        # Extract data from batch - GEARS format: x is (n_genes, 1) per cell
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 1)
        ori_gene_values = x.squeeze(-1).view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        # Build pert_flags from pert_idx (indices of perturbed genes)
        pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=device)
        pert_idx = batch_data.pert_idx  # (batch_size, num_perts) or list
        for i in range(batch_size):
            idx = pert_idx[i] if isinstance(pert_idx, list) else pert_idx[i]
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            for p_idx in idx:
                if p_idx >= 0 and p_idx < n_genes:  # -1 means no perturbation (ctrl)
                    pert_flags[i, p_idx] = 1

        # Prepare input
        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            # Sample if too many genes
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        # Forward pass
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=config["training"]["CLS"],
                CCE=config["training"]["CCE"],
                MVC=config["training"]["MVC"],
                ECS=config["training"]["ECS"],
            )
            output_values = output_dict["mlm_output"]

            # Compute loss on all positions
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = loss_mse = masked_mse_loss(
                output_values, target_values, masked_positions
            )

        # Backward pass
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["optimizer"]["grad_clip"],
            error_if_nonfinite=False if scaler.is_enabled() else True,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()

        # Logging (only if logger is provided, i.e., rank 0)
        if logger is not None and batch_idx % log_interval == 0 and batch_idx > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                f"ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.4f} | mse {cur_mse:5.4f}"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()

    return total_loss / max(1, num_batches % log_interval)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    gene_ids: np.ndarray,
    config: dict,
    device: torch.device,
) -> Dict:
    """
    Evaluate model on a data loader.

    Returns:
        Dictionary with predictions and ground truth
    """
    model.eval()

    n_genes = len(gene_ids)
    include_zero_gene = config["data"]["include_zero_gene"]

    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []

    for batch_data in loader:
        batch_data.to(device)
        pert_cat.extend(batch_data.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch_data,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch_data.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch_data.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # Compile results
    results = {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).detach().cpu().numpy().astype(np.float32),
        "truth": torch.stack(truth).detach().cpu().numpy().astype(np.float32),
        "pred_de": torch.stack(pred_de).detach().cpu().numpy().astype(np.float32),
        "truth_de": torch.stack(truth_de).detach().cpu().numpy().astype(np.float32),
    }

    return results


def compute_perturbation_metrics(results: Dict, ctrl_adata) -> Dict:
    """Compute perturbation prediction metrics."""
    from scipy.stats import pearsonr

    pred = results["pred"]
    truth = results["truth"]
    pred_de = results["pred_de"]
    truth_de = results["truth_de"]

    # Get control mean
    if hasattr(ctrl_adata.X, "toarray"):
        ctrl_mean = ctrl_adata.X.toarray().mean(axis=0)
    else:
        ctrl_mean = ctrl_adata.X.mean(axis=0)

    # Flatten for correlation
    pred_flat = pred.flatten()
    truth_flat = truth.flatten()
    pred_de_flat = pred_de.flatten()
    truth_de_flat = truth_de.flatten()

    # Compute metrics
    pearson_all = pearsonr(pred_flat, truth_flat)[0]
    pearson_de = pearsonr(pred_de_flat, truth_de_flat)[0]

    # Delta (change from control)
    pred_delta = pred - ctrl_mean
    truth_delta = truth - ctrl_mean
    pearson_delta = pearsonr(pred_delta.flatten(), truth_delta.flatten())[0]

    pred_de_delta = pred_de - ctrl_mean[results.get("de_idx", slice(None))]
    truth_de_delta = truth_de - ctrl_mean[results.get("de_idx", slice(None))]
    # Approximate DE delta correlation
    pearson_de_delta = pearsonr(pred_de.flatten(), truth_de.flatten())[0]

    return {
        "pearson": pearson_all,
        "pearson_de": pearson_de,
        "pearson_delta": pearson_delta,
        "pearson_de_delta": pearson_de_delta,
    }


def save_model(model: nn.Module, config: dict, vocab: GeneVocab, save_dir: Path):
    """Save finetuned model checkpoint."""
    # Unwrap DDP model if necessary
    model_to_save = model.module if isinstance(model, DDP) else model

    # Save model weights
    torch.save(model_to_save.state_dict(), save_dir / "best_model.pt")

    # Copy vocab
    vocab_src = Path(config["paths"]["pretrained_model_dir"]) / "vocab.json"
    vocab_dst = save_dir / "vocab.json"
    if vocab_src.exists():
        import shutil

        shutil.copy(vocab_src, vocab_dst)

    # Save model config (args.json)
    args_src = Path(config["paths"]["pretrained_model_dir"]) / "args.json"
    args_dst = save_dir / "args.json"
    if args_src.exists():
        import shutil

        shutil.copy(args_src, args_dst)


def main():
    parser = argparse.ArgumentParser(description="Finetune scGPT for perturbation")
    parser.add_argument(
        "--config",
        default="src/configs/finetune.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # ========== Setup DDP ==========
    rank, local_rank, world_size, is_distributed = setup_ddp()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(args.seed)

    # Setup device (use local_rank for DDP)
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(
            config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
        )

    if is_main_process(rank):
        print(f"Using device: {device}")
        if is_distributed:
            print(f"DDP enabled: world_size={world_size}")

    # Setup output directory (only rank 0 creates it)
    save_dir = Path(config["paths"]["output_dir"])
    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_dir}")

    # Synchronize before logging setup
    if is_distributed:
        dist.barrier()

    # Setup logging (only rank 0 logs to file)
    logger = setup_logging(save_dir, rank)
    if is_main_process(rank):
        logger.info(f"Config: {config}")
        logger.info(f"DDP: is_distributed={is_distributed}, world_size={world_size}")

    # ========== Load Data ==========
    if is_main_process(rank):
        logger.info("Loading data...")
    gears_data_dir = config["paths"]["gears_data_dir"]
    dataset_name = config["paths"]["dataset_name"]

    # Use default_pert_graph=False to include all perturbation genes from data
    # instead of filtering to essential genes list (avoids dropping 11 genes)
    pert_data = PertData(gears_data_dir, default_pert_graph=False)
    pert_data.load(data_path=os.path.join(gears_data_dir, dataset_name))

    # Prepare split (only rank 0 prepares, others wait)
    split_type = config["split"]["type"]
    split_seed = config["split"]["seed"]

    if is_distributed:
        if is_main_process(rank):
            pert_data.prepare_split(split=split_type, seed=split_seed)
        dist.barrier()
        if not is_main_process(rank):
            pert_data.prepare_split(split=split_type, seed=split_seed)
    else:
        pert_data.prepare_split(split=split_type, seed=split_seed)

    # Get dataloaders (we'll create our own train_loader with DistributedSampler)
    batch_size = config["optimizer"]["batch_size"]
    eval_batch_size = config["optimizer"]["eval_batch_size"]
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    # ========== Create Distributed Train Loader ==========
    if is_distributed:
        # Get the training dataset from PertData
        train_dataset = pert_data.dataloader["train_loader"].dataset
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        # Adjust batch size per GPU
        per_gpu_batch_size = batch_size // world_size
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=0,  # Avoid fork OOM on memory-constrained systems
            pin_memory=True,
        )
    else:
        train_loader = pert_data.dataloader["train_loader"]
        train_sampler = None

    # ========== Load Vocabulary ==========
    if is_main_process(rank):
        logger.info("Loading vocabulary...")
    vocab_file = Path(config["paths"]["pretrained_model_dir"]) / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)

    # Add special tokens
    special_tokens = config["data"]["special_tokens"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Map genes to vocabulary
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    if is_main_process(rank):
        logger.info(
            f"Matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}"
        )

    genes = pert_data.adata.var["gene_name"].tolist()
    vocab.set_default_index(vocab[config["data"]["pad_token"]])
    gene_ids = np.array(
        [
            vocab[gene] if gene in vocab else vocab[config["data"]["pad_token"]]
            for gene in genes
        ],
        dtype=int,
    )
    n_genes = len(genes)

    # ========== Load Model ==========
    if is_main_process(rank):
        logger.info("Loading pretrained model...")
    model = load_pretrained_model(config, vocab, n_genes, device, logger)
    if is_main_process(rank):
        logger.info(
            f"Model has {sum(p.numel() for p in model.parameters()):,} parameters"
        )

    # ========== Wrap Model with DDP ==========
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(rank):
            logger.info("Model wrapped with DistributedDataParallel")

    # ========== Setup Training ==========
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config["optimizer"]["schedule_interval"],
        gamma=config["optimizer"]["schedule_gamma"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"]["amp"])

    # ========== Training Loop ==========
    epochs = config["optimizer"]["epochs"]
    early_stop = config["optimizer"]["early_stop"]
    best_loss = float("inf")
    best_model = None
    patience = 0

    if is_main_process(rank):
        logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            gene_ids=gene_ids,
            config=config,
            epoch=epoch,
            logger=logger if is_main_process(rank) else None,
        )

        # Synchronize loss across processes
        if is_distributed:
            loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = loss_tensor.item()

        # Evaluate on validation (only on rank 0)
        if is_main_process(rank):
            if "val_loader" in pert_data.dataloader:
                val_loader = pert_data.dataloader["val_loader"]
                # Use unwrapped model for evaluation
                eval_model = model.module if isinstance(model, DDP) else model
                val_res = evaluate(eval_model, val_loader, gene_ids, config, device)
                ctrl_adata = pert_data.adata[
                    pert_data.adata.obs["condition"] == "ctrl"
                ]
                val_metrics = compute_perturbation_metrics(val_res, ctrl_adata)
                logger.info(f"Epoch {epoch} val_metrics: {val_metrics}")
                current_loss = -val_metrics["pearson"]  # Negative for "lower is better"
            else:
                current_loss = train_loss

            elapsed = time.time() - epoch_start
            logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s |")

            # Check for best model
            if current_loss < best_loss:
                best_loss = current_loss
                # Store unwrapped model
                model_to_copy = model.module if isinstance(model, DDP) else model
                best_model = copy.deepcopy(model_to_copy)
                logger.info(f"Best model at epoch {epoch}")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    # Broadcast early stop signal to all processes
                    if is_distributed:
                        early_stop_tensor = torch.tensor([1], device=device)
                        dist.broadcast(early_stop_tensor, src=0)
                    break
        else:
            current_loss = train_loss

        # Check for early stop signal from rank 0
        if is_distributed:
            early_stop_tensor = torch.tensor([0], device=device)
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item() == 1:
                break

        scheduler.step()

    # ========== Save Best Model (only rank 0) ==========
    if is_main_process(rank):
        logger.info("Saving best model...")
        save_model(best_model, config, vocab, save_dir)
        logger.info(f"Model saved to {save_dir}")

        # ========== Final Evaluation ==========
        if "test_loader" in pert_data.dataloader:
            logger.info("Evaluating on test set...")
            test_loader = pert_data.dataloader["test_loader"]
            test_res = evaluate(best_model, test_loader, gene_ids, config, device)
            ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
            test_metrics = compute_perturbation_metrics(test_res, ctrl_adata)
            logger.info(f"Test metrics: {test_metrics}")

            # Save test metrics
            with open(save_dir / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)

        logger.info("Training complete!")

    # ========== Cleanup DDP ==========
    cleanup_ddp(is_distributed)


if __name__ == "__main__":
    main()
