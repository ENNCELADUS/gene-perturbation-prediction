"""
Training utilities for scGPT perturbation prediction finetuning.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

# Import scGPT components (assumes scGPT is in path)
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained, map_raw_id_to_vocab_id

from src.utils.de_metrics import (
    compute_de_comparison_metrics,
    compute_mae_top2k,
    compute_pds,
    compute_pseudobulk_delta,
)


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
    model_dir = Path(config["paths"]["model_dir"])
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

        # Extract data from batch - GEARS format
        x: torch.Tensor = batch_data.x
        ori_gene_values = x.squeeze(-1).view(batch_size, n_genes)
        target_gene_values = batch_data.y

        # Build pert_flags from pert_idx
        pert_flags = torch.zeros(batch_size, n_genes, dtype=torch.long, device=device)
        pert_idx = batch_data.pert_idx
        for i in range(batch_size):
            idx = pert_idx[i] if isinstance(pert_idx, list) else pert_idx[i]
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            for p_idx in idx:
                if p_idx >= 0 and p_idx < n_genes:
                    pert_flags[i, p_idx] = 1

        # Prepare input
        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

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

            # SmoothL1Loss (Huber-style) per eval_metrics.md Section 4.1
            smooth_l1 = nn.SmoothL1Loss(beta=1.0, reduction="mean")
            loss = smooth_l1(output_values, target_values)

        # Backward pass
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["optimizer"]["grad_clip"],
            error_if_nonfinite=False if scaler.is_enabled() else True,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if logger is not None and batch_idx % log_interval == 0 and batch_idx > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                f"ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.4f}"
            )
            total_loss = 0
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
        Dictionary with predictions and ground truth per perturbation
    """
    model.eval()

    include_zero_gene = config["data"]["include_zero_gene"]

    pert_cat = []
    pred = []
    truth = []

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

    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).detach().cpu().numpy().astype(np.float32),
        "truth": torch.stack(truth).detach().cpu().numpy().astype(np.float32),
    }


def compute_validation_metrics(
    results: Dict,
    ctrl_adata,
    gene_names: np.ndarray,
    config: dict,
) -> Dict:
    """
    Compute validation metrics per eval_metrics.md Section 4.2.

    Raw metrics:
        - pds_mean_rank: Mean of perturbation ranks
        - mae_top2k: Mean MAE on top 2000 genes by |LFC|
        - des: Mean DES (Differential Expression Score)

    Normalized sub-scores (all higher is better, 0-1):
        - s_pds = (N - MeanRank) / (N - 1)
        - s_mae = 1 / (1 + mae_top2k / tau) where tau = median(mae_scores)
        - s_des = des

    Composite metric:
        - s_geo = (s_pds * s_mae * s_des)^(1/3)
        - l_geo = 1 - s_geo (lower is better, for early stopping)

    Args:
        results: Dict with pert_cat, pred, truth arrays
        ctrl_adata: Control cells AnnData
        gene_names: Gene names array
        config: Configuration dictionary

    Returns:
        Dict with raw metrics, normalized scores, and composite metrics
    """
    pred = results["pred"]
    truth = results["truth"]
    pert_cat = results["pert_cat"]

    # Get control mean
    if hasattr(ctrl_adata.X, "toarray"):
        ctrl_mean = ctrl_adata.X.toarray().mean(axis=0).flatten()
    else:
        ctrl_mean = np.asarray(ctrl_adata.X.mean(axis=0)).flatten()

    # Get control expression for DE analysis
    if hasattr(ctrl_adata.X, "toarray"):
        ctrl_expr = ctrl_adata.X.toarray()
    else:
        ctrl_expr = np.asarray(ctrl_adata.X)

    mae_top_k = config.get("metrics", {}).get("mae_top_k", 2000)

    # Group by perturbation
    unique_perts = [p for p in np.unique(pert_cat) if p != "ctrl"]

    pred_deltas = {}
    truth_deltas = {}
    target_gene_indices = {}
    des_scores = []
    mae_scores = []

    for pert in unique_perts:
        mask = pert_cat == pert
        pred_pert = pred[mask]
        truth_pert = truth[mask]

        if len(pred_pert) == 0:
            continue

        # Compute deltas for PDS
        pred_deltas[pert] = compute_pseudobulk_delta(pred_pert, ctrl_mean)
        truth_deltas[pert] = compute_pseudobulk_delta(truth_pert, ctrl_mean)

        # Get target gene index
        gene_name = pert.replace("+ctrl", "")
        try:
            target_idx = np.where(gene_names == gene_name)[0]
            target_gene_indices[pert] = target_idx[0] if len(target_idx) > 0 else -1
        except (IndexError, ValueError):
            target_gene_indices[pert] = -1

        # Compute DES
        try:
            de_metrics = compute_de_comparison_metrics(
                control_expr=ctrl_expr,
                pred_expr=pred_pert,
                truth_expr=truth_pert,
                gene_names=gene_names,
                fdr_threshold=0.05,
                threads=1,
            )
            des_scores.append(de_metrics["des"])
        except Exception:
            pass

        # Compute MAE_top2k
        mae = compute_mae_top2k(
            pred_expr=pred_pert,
            truth_expr=truth_pert,
            control_mean=ctrl_mean,
            top_k=mae_top_k,
        )
        mae_scores.append(mae)

    # Compute PDS (returns dict with mean_rank, npds, ranks, cosine_self)
    n_perts = len(pred_deltas)
    if pred_deltas:
        pds_result = compute_pds(pred_deltas, truth_deltas, target_gene_indices)
        mean_rank = pds_result["mean_rank"]
        npds = pds_result["npds"]
    else:
        mean_rank = np.nan
        npds = np.nan

    mean_des = np.nanmean(des_scores) if des_scores else np.nan
    mean_mae = np.nanmean(mae_scores) if mae_scores else np.nan

    # Compute normalized sub-scores per eval_metrics.md Section 4.2.2
    # s_pds = (N - MeanRank) / (N - 1), higher is better
    if n_perts > 1 and not np.isnan(mean_rank):
        s_pds = (n_perts - mean_rank) / (n_perts - 1)
    elif n_perts == 1 and not np.isnan(mean_rank):
        s_pds = 1.0 if mean_rank == 1 else 0.0
    else:
        s_pds = np.nan

    # s_mae = 1 / (1 + mae_top2k / tau), tau = median of per-perturbation MAE
    if mae_scores and not np.isnan(mean_mae):
        tau = float(np.median(mae_scores))
        if tau > 0:
            s_mae = 1.0 / (1.0 + mean_mae / tau)
        else:
            s_mae = 1.0  # Perfect case, tau=0 means all MAE=0
    else:
        s_mae = np.nan

    # s_des = des (already 0-1, higher is better)
    s_des = mean_des

    # Composite metric: s_geo = (s_pds * s_mae * s_des)^(1/3)
    if (
        not np.isnan(s_pds)
        and not np.isnan(s_mae)
        and not np.isnan(s_des)
        and s_pds > 0
        and s_mae > 0
        and s_des > 0
    ):
        s_geo = (s_pds * s_mae * s_des) ** (1 / 3)
    else:
        s_geo = np.nan

    # l_geo = 1 - s_geo (lower is better, for early stopping)
    l_geo = 1.0 - s_geo if not np.isnan(s_geo) else np.nan

    return {
        # Raw metrics
        "pds": float(npds),
        "pds_mean_rank": float(mean_rank),
        "des": float(mean_des),
        "mae_top2k": float(mean_mae),
        # Normalized sub-scores
        "s_pds": float(s_pds) if not np.isnan(s_pds) else np.nan,
        "s_mae": float(s_mae) if not np.isnan(s_mae) else np.nan,
        "s_des": float(s_des) if not np.isnan(s_des) else np.nan,
        # Composite metrics
        "s_geo": float(s_geo) if not np.isnan(s_geo) else np.nan,
        "l_geo": float(l_geo) if not np.isnan(l_geo) else np.nan,
    }


def save_model(model: nn.Module, config: dict, vocab: GeneVocab, save_dir: Path):
    """Save finetuned model checkpoint."""
    model_to_save = model.module if isinstance(model, DDP) else model

    torch.save(model_to_save.state_dict(), save_dir / "best_model.pt")

    # Copy vocab
    vocab_src = Path(config["paths"]["model_dir"]) / "vocab.json"
    vocab_dst = save_dir / "vocab.json"
    if vocab_src.exists():
        shutil.copy(vocab_src, vocab_dst)

    # Copy model config
    args_src = Path(config["paths"]["model_dir"]) / "args.json"
    args_dst = save_dir / "args.json"
    if args_src.exists():
        shutil.copy(args_src, args_dst)
