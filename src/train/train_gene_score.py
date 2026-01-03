"""
Training script for Route B1 gene-level scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from tqdm import tqdm
import json
import yaml
import os
import math

from ..data import load_perturb_data
from ..data.gene_score_collator import GeneScoreDataset, collate_gene_score_batch
from ..model.gene_score import GeneScoreModel


class BalancedConditionBatchSampler(Sampler[List[int]]):
    """Sample batches with a fixed number of conditions and cells per condition."""

    def __init__(
        self,
        condition_to_indices: Dict[str, List[int]],
        n_conditions: int,
        n_cells_per_condition: int,
        steps_per_epoch: int,
        seed: int = 42,
    ):
        self.condition_to_indices = condition_to_indices
        self.conditions = list(condition_to_indices.keys())
        self.n_conditions = n_conditions
        self.n_cells_per_condition = n_cells_per_condition
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            chosen_conditions = self.rng.choice(
                self.conditions,
                size=min(self.n_conditions, len(self.conditions)),
                replace=False,
            )
            batch = []
            for condition in chosen_conditions:
                indices = self.condition_to_indices[condition]
                replace = len(indices) < self.n_cells_per_condition
                sampled = self.rng.choice(
                    indices, size=self.n_cells_per_condition, replace=replace
                )
                batch.extend(sampled.tolist())
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch


class DistributedWeightedSampler(Sampler[int]):
    """Distributed sampler that draws samples proportionally to weights."""

    def __init__(
        self,
        weights: torch.Tensor | List[float],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        if num_replicas is None or rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("DistributedWeightedSampler requires initialized DDP.")
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if num_samples is None:
            num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.num_samples = num_samples
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        )
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Route B1 gene-score model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (for smoke test)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/gene_score", help="Output directory"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: GeneScoreModel, config: dict) -> torch.optim.Optimizer:
    weight_decay = config["training"].get("weight_decay", 0.01)
    head_lr = config["training"].get("head_learning_rate", 1e-4)
    backbone_lr = config["training"].get("backbone_learning_rate", 1e-5)

    param_groups = []
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    backbone_params = [p for p in model.backbone.model.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(f"Optimizer: head_lr={head_lr}, backbone_lr={backbone_lr}")
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps > 0 and total_steps > warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6
        )
    return scheduler


def compute_bce_pos_weight(
    gene_counts: np.ndarray,
    n_samples: int,
    cap: float,
    min_weight: float,
) -> np.ndarray:
    """Compute per-gene BCE pos_weight using global neg/pos ratio."""
    counts = gene_counts.astype(np.float64)
    denom = np.maximum(counts, 1.0)
    neg = max(n_samples, 1) - counts
    weights = neg / denom
    if cap is not None:
        weights = np.minimum(weights, cap)
    if min_weight is not None:
        weights = np.maximum(weights, min_weight)
    return weights


def sample_negatives(
    logits_row: torch.Tensor,
    neg_idx: torch.Tensor,
    num_negatives: int,
    hard_negative_ratio: float,
    hard_candidate_factor: float,
) -> torch.Tensor:
    """Sample negatives with a mix of hard and random choices."""
    if num_negatives <= 0 or neg_idx.numel() <= num_negatives:
        return neg_idx

    hard_k = int(num_negatives * hard_negative_ratio)
    hard_k = min(hard_k, neg_idx.numel())
    rand_k = max(num_negatives - hard_k, 0)

    hard_rel = None
    if hard_k > 0:
        candidate_size = int(num_negatives * hard_candidate_factor)
        candidate_size = max(candidate_size, hard_k)
        candidate_size = min(candidate_size, neg_idx.numel())
        if candidate_size < neg_idx.numel():
            perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[
                :candidate_size
            ]
            candidate_idx = neg_idx[perm]
        else:
            candidate_idx = neg_idx
        neg_scores = logits_row[candidate_idx]
        hard_rel = torch.topk(neg_scores, k=hard_k).indices
        hard_idx = candidate_idx[hard_rel]
    else:
        hard_idx = torch.empty(0, dtype=neg_idx.dtype, device=neg_idx.device)

    if rand_k > 0:
        if hard_rel is not None and neg_idx.numel() > hard_k:
            mask = torch.ones(neg_idx.numel(), dtype=torch.bool, device=neg_idx.device)
            mask[hard_rel] = False
            remaining = neg_idx[mask]
        else:
            remaining = neg_idx
        if remaining.numel() == 0:
            remaining = neg_idx
        if rand_k <= remaining.numel():
            perm = torch.randperm(remaining.numel(), device=neg_idx.device)[:rand_k]
            rand_idx = remaining[perm]
        else:
            rand_idx = remaining[
                torch.randint(0, remaining.numel(), (rand_k,), device=neg_idx.device)
            ]
    else:
        rand_idx = torch.empty(0, dtype=neg_idx.dtype, device=neg_idx.device)

    return torch.cat([hard_idx, rand_idx], dim=0)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: dict,
    ranking_pos_weights: Optional[torch.Tensor] = None,
    bce_pos_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute weighted ranking + BCE loss."""
    training_cfg = config.get("training", {})
    ranking_weight = training_cfg.get("ranking_loss_weight", 1.0)
    bce_weight = training_cfg.get("bce_loss_weight", 0.0)
    num_negatives = training_cfg.get("ranking_num_negatives", 128)
    margin = training_cfg.get("ranking_margin", 0.0)
    hard_negative_ratio = training_cfg.get("ranking_hard_negative_ratio", 0.5)
    hard_candidate_factor = training_cfg.get("ranking_hard_candidate_factor", 4.0)
    bce_pos_weight_enabled = training_cfg.get("bce_pos_weight_enabled", False)

    loss_items: dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=logits.device)
    ranking_losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []

    batch_size = logits.size(0)
    for i in range(batch_size):
        pos_idx = (targets[i] > 0).nonzero(as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        neg_idx = (targets[i] == 0).nonzero(as_tuple=False).squeeze(1)
        if neg_idx.numel() == 0:
            continue

        neg_sample = sample_negatives(
            logits[i],
            neg_idx,
            num_negatives,
            hard_negative_ratio,
            hard_candidate_factor,
        )

        if ranking_weight > 0:
            pos_scores = logits[i, pos_idx]
            neg_scores = logits[i, neg_sample]
            if margin > 0:
                pairwise = F.relu(
                    margin - (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
                )
            else:
                pairwise = F.softplus(neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1))
            if ranking_pos_weights is not None:
                pos_weights = ranking_pos_weights[pos_idx].to(pairwise.device)
                pairwise = pairwise * pos_weights.unsqueeze(1)
            ranking_losses.append(pairwise.mean())

        if bce_weight > 0:
            selected = torch.cat([pos_idx, neg_sample], dim=0)
            selected = torch.unique(selected)
            pos_weight = None
            if bce_pos_weight_enabled and bce_pos_weights is not None:
                pos_weight = bce_pos_weights[selected].to(logits.device)
            bce_losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[i, selected],
                    targets[i, selected],
                    pos_weight=pos_weight,
                )
            )

    if ranking_weight > 0:
        ranking_loss = (
            torch.stack(ranking_losses).mean()
            if ranking_losses
            else torch.tensor(0.0, device=logits.device)
        )
        total_loss = total_loss + ranking_weight * ranking_loss
        loss_items["ranking_loss"] = ranking_loss.item()

    if bce_weight > 0:
        bce_loss = (
            torch.stack(bce_losses).mean()
            if bce_losses
            else torch.tensor(0.0, device=logits.device)
        )
        total_loss = total_loss + bce_weight * bce_loss
        loss_items["bce_loss"] = bce_loss.item()

    loss_items["total_loss"] = total_loss.item()
    return total_loss, loss_items


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    config: dict,
    ranking_pos_weights: Optional[torch.Tensor] = None,
    bce_pos_weights: Optional[torch.Tensor] = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    sampler: DistributedSampler | None = None,
    max_steps: int | None = None,
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()

    total_loss = 0.0
    num_batches = 0

    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        genes = batch["genes"].to(device)
        values = batch["values"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        control_genes = batch["control_genes"].to(device)
        control_values = batch["control_values"].to(device)
        control_padding_mask = batch["control_padding_mask"].to(device)
        control_counts = batch["control_counts"]
        targets = batch["targets"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(
                genes,
                values,
                padding_mask,
                control_gene_ids=control_genes,
                control_values=control_values,
                control_padding_mask=control_padding_mask,
                control_counts=control_counts,
                control_chunk_size=config.get("training", {}).get(
                    "control_chunk_size", 0
                ),
                control_no_grad=config.get("training", {}).get("control_no_grad", True),
            )
            loss, loss_items = compute_loss(
                logits,
                targets,
                config,
                ranking_pos_weights=ranking_pos_weights,
                bce_pos_weights=bce_pos_weights,
            )

        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if "ranking_loss" in loss_items:
            postfix["rank"] = f"{loss_items['ranking_loss']:.4f}"
        if "bce_loss" in loss_items:
            postfix["bce"] = f"{loss_items['bce_loss']:.4f}"

        pbar.set_postfix(postfix)

    return {"train_loss": total_loss / num_batches if num_batches else 0.0}


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    config: dict,
    ranking_pos_weights: Optional[torch.Tensor] = None,
    bce_pos_weights: Optional[torch.Tensor] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            control_genes = batch["control_genes"].to(device)
            control_values = batch["control_values"].to(device)
            control_padding_mask = batch["control_padding_mask"].to(device)
            control_counts = batch["control_counts"]
            targets = batch["targets"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    genes,
                    values,
                    padding_mask,
                    control_gene_ids=control_genes,
                    control_values=control_values,
                    control_padding_mask=control_padding_mask,
                    control_counts=control_counts,
                    control_chunk_size=config.get("training", {}).get(
                        "control_chunk_size", 0
                    ),
                    control_no_grad=config.get("training", {}).get(
                        "control_no_grad", True
                    ),
                )
                loss, _ = compute_loss(
                    logits,
                    targets,
                    config,
                    ranking_pos_weights=ranking_pos_weights,
                    bce_pos_weights=bce_pos_weights,
                )

            total_loss += loss.item()
            num_batches += 1

    return {"val_loss": total_loss / num_batches if num_batches else 0.0}


def _setup_distributed() -> Dict[str, int | bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return {
            "enabled": True,
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
        }
    return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir == "results/gene_score":
        logging_cfg = config.get("logging", {})
        base_dir = logging_cfg.get("output_dir", "results/gene_score")
        args.output_dir = str(Path(base_dir))

    ddp = _setup_distributed()
    is_main = ddp["rank"] == 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = config.get("evaluation", {})
    checkpoint_path_cfg = eval_cfg.get("checkpoint_path")
    if checkpoint_path_cfg:
        checkpoint_path = Path(checkpoint_path_cfg)
    else:
        checkpoint_path = output_dir / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if is_main:
        print("=" * 60)
        print("Route B1 Gene-Score Training")
        print("=" * 60)
        print(f"Output: {output_dir}")
        print(f"Checkpoint: {checkpoint_path}")

    device = f"cuda:{ddp['local_rank']}" if torch.cuda.is_available() else "cpu"
    if is_main:
        print(f"Device: {device}")

    if is_main:
        print("\n[1/5] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    if is_main:
        print(f"  - Train conditions: {len(dataset.train_conditions)}")
        print(f"  - Val conditions: {len(dataset.val_conditions)}")

    if is_main:
        print("\n[2/5] Building datasets...")
    pretrained_dir = Path(config["model"].get("pretrained_dir", "model/scGPT"))
    model_args_path = pretrained_dir / "args.json"
    with open(model_args_path) as f:
        model_args = json.load(f)
    with open(pretrained_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_layers = int(model_args.get("nlayers", 12))
    training_cfg = config.get("training", {})
    unfreeze_last_n_layers = training_cfg.get("unfreeze_last_n_layers", 1)
    freeze_layers_up_to = max(n_layers - 1 - unfreeze_last_n_layers, -1)

    train_dataset = GeneScoreDataset(
        adata=dataset.adata,
        conditions=dataset.train_conditions,
        vocab=vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
        match_keys=config["data"].get("control_match_keys"),
        n_control_samples=config["data"].get("control_n_samples", 8),
        control_aggregate=training_cfg.get("control_aggregate", False),
        sample_weight_alpha=config.get("training", {}).get("sample_weight_alpha"),
        sample_weight_cap=config.get("training", {}).get("sample_weight_cap", 10.0),
        sample_weight_eps=config.get("training", {}).get("sample_weight_eps", 1.0),
    )

    val_dataset = GeneScoreDataset(
        adata=dataset.adata,
        conditions=dataset.val_conditions,
        vocab=train_dataset.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
        match_keys=config["data"].get("control_match_keys"),
        n_control_samples=config["data"].get("control_n_samples", 8),
        control_aggregate=training_cfg.get("control_aggregate", False),
        sample_weight_alpha=None,
    )

    n_genes = dataset.adata.n_vars

    ranking_pos_weights = None
    if training_cfg.get("ranking_pos_weight_alpha"):
        ranking_weights = train_dataset.get_gene_weights(
            alpha=training_cfg.get("ranking_pos_weight_alpha", 0.5),
            cap=training_cfg.get("ranking_pos_weight_cap", 10.0),
            eps=training_cfg.get("ranking_pos_weight_eps", 1.0),
        )
        ranking_pos_weights = torch.tensor(
            ranking_weights, dtype=torch.float32, device=device
        )

    bce_pos_weights = None
    if training_cfg.get("bce_pos_weight_enabled", False):
        bce_weights = compute_bce_pos_weight(
            train_dataset.gene_counts,
            n_samples=len(train_dataset),
            cap=training_cfg.get("bce_pos_weight_cap", 50.0),
            min_weight=training_cfg.get("bce_pos_weight_min", 1.0),
        )
        bce_pos_weights = torch.tensor(bce_weights, dtype=torch.float32, device=device)

    if is_main:
        print("\n[3/5] Loading model...")
    model = GeneScoreModel(
        n_genes=n_genes,
        checkpoint_path=Path(config["model"].get("pretrained_dir", "model/scGPT"))
        / "best_model.pt",
        vocab_path=Path(config["model"].get("pretrained_dir", "model/scGPT"))
        / "vocab.json",
        args_path=Path(config["model"].get("pretrained_dir", "model/scGPT"))
        / "args.json",
        freeze_encoder=config["model"].get("freeze_encoder", True),
        freeze_layers_up_to=freeze_layers_up_to,
        device=device,
        score_mode=config.get("head", {}).get("score_mode", "dot"),
        score_gene_ids=train_dataset.gene_ids,
        head_hidden_dim=config.get("head", {}).get("hidden_dim", 512),
        head_dropout=config.get("head", {}).get("dropout", 0.2),
    )

    if is_main:
        print("\n[4/5] Building dataloaders...")
    batch_size = config["training"].get("batch_size", 32)
    train_sampler = None
    val_sampler = None
    train_loader = None
    if ddp["enabled"]:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

    collate_vocab = train_dataset.vocab

    if config["training"].get("gene_weighted_sampling", False):
        if train_dataset.sample_weights is None:
            raise ValueError(
                "gene_weighted_sampling enabled but sample_weights are not computed. "
                "Set training.sample_weight_alpha > 0."
            )
        sampler_weights = torch.as_tensor(
            train_dataset.sample_weights, dtype=torch.double
        )
        if ddp["enabled"]:
            sampler = DistributedWeightedSampler(
                sampler_weights,
                num_samples=config["training"].get("weighted_samples_per_replica"),
                replacement=True,
                seed=config["training"].get("sampling_seed", 0),
            )
        else:
            sampler = WeightedRandomSampler(
                sampler_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=lambda batch: collate_gene_score_batch(
                batch, collate_vocab, n_genes
            ),
            num_workers=0,
        )
        train_sampler = sampler
    if train_loader is None and config["training"].get("balanced_sampling", False):
        if ddp["enabled"]:
            if is_main:
                print("Balanced sampling disabled under DDP; using DistributedSampler.")
        else:
            n_conditions = config["training"].get("balanced_sampling_n_conditions", 8)
            n_cells = config["training"].get("balanced_sampling_n_cells", 4)
            steps_per_epoch = max(1, len(train_dataset) // (n_conditions * n_cells))
            batch_sampler = BalancedConditionBatchSampler(
                train_dataset.condition_to_indices,
                n_conditions=n_conditions,
                n_cells_per_condition=n_cells,
                steps_per_epoch=steps_per_epoch,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=lambda batch: collate_gene_score_batch(
                    batch, collate_vocab, n_genes
                ),
                num_workers=0,
            )
    if train_loader is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: collate_gene_score_batch(
                batch, collate_vocab, n_genes
            ),
            num_workers=0,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda batch: collate_gene_score_batch(
            batch, collate_vocab, n_genes
        ),
        num_workers=0,
    )

    if is_main:
        print("\n[5/5] Training...")
    optimizer = build_optimizer(model, config)
    epochs = config["training"].get("epochs", 50)
    total_steps = len(train_loader) * epochs
    scheduler = build_scheduler(
        optimizer, total_steps, config["training"].get("warmup_ratio", 0.0)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    if ddp["enabled"]:
        model = DDP(
            model,
            device_ids=[ddp["local_rank"]] if torch.cuda.is_available() else None,
            output_device=ddp["local_rank"] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
        )

    best_val_loss = float("inf")
    early_stopping_patience = config["training"].get("early_stopping_patience")
    early_stopping_metric = config["training"].get("early_stopping_metric", "val_loss")
    if early_stopping_metric not in {"train_loss", "val_loss"}:
        if is_main:
            print(
                "Unknown early_stopping_metric="
                f"{early_stopping_metric}; defaulting to val_loss."
            )
        early_stopping_metric = "val_loss"
    early_stop_enabled = (
        early_stopping_patience is not None and early_stopping_patience > 0
    )
    best_metric = float("inf")
    patience = 0
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            ranking_pos_weights=ranking_pos_weights,
            bce_pos_weights=bce_pos_weights,
            scaler=scaler,
            sampler=train_sampler,
            max_steps=args.max_steps,
            epoch=epoch,
        )
        val_metrics = {"val_loss": float("inf")}
        if is_main:
            eval_model = model.module if isinstance(model, DDP) else model
            val_metrics = validate(
                eval_model,
                val_loader,
                device,
                config,
                ranking_pos_weights=ranking_pos_weights,
                bce_pos_weights=bce_pos_weights,
                use_amp=scaler.is_enabled(),
            )

        if is_main:
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

        if is_main and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_target = model.module if isinstance(model, DDP) else model
            save_target.save(checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

        early_stop = False
        if early_stop_enabled and is_main:
            metric_value = (
                train_metrics["train_loss"]
                if early_stopping_metric == "train_loss"
                else val_metrics["val_loss"]
            )
            if metric_value < best_metric:
                best_metric = metric_value
                patience = 0
            else:
                patience += 1
            if patience >= early_stopping_patience:
                early_stop = True
                print(
                    f"\nStopping early (patience={early_stopping_patience}, "
                    f"metric={early_stopping_metric})"
                )
        if ddp["enabled"]:
            stop_tensor = torch.tensor(
                int(early_stop), device=device if torch.cuda.is_available() else "cpu"
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            early_stop = bool(stop_tensor.item())
        if early_stop:
            break

        if args.max_steps:
            if is_main:
                print(f"\nStopping early (max_steps={args.max_steps})")
            break

    if is_main:
        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {checkpoint_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
