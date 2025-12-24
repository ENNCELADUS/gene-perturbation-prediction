"""
Training script for Route B1 gene-level scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import json
import yaml
import os

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


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    sampler: DistributedSampler | None = None,
    max_steps: int | None = None,
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    if sampler is not None:
        sampler.set_epoch(epoch)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        genes = batch["genes"].to(device)
        values = batch["values"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)

        logits = model(genes, values, padding_mask)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    return {"train_loss": total_loss / num_batches if num_batches else 0.0}


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            genes = batch["genes"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(genes, values, padding_mask)
            loss = loss_fn(logits, targets)

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

    ddp = _setup_distributed()
    is_main = ddp["rank"] == 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    if is_main:
        print("=" * 60)
        print("Route B1 Gene-Score Training")
        print("=" * 60)
        print(f"Output: {output_dir}")

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
        print(f"  - Train conditions: {len(dataset.train_conditions())}")
        print(f"  - Val conditions: {len(dataset.val_conditions())}")

    if is_main:
        print("\n[2/5] Building datasets...")
    pretrained_dir = Path(config["model"].get("pretrained_dir", "model/scGPT"))
    model_args_path = pretrained_dir / "args.json"
    with open(model_args_path) as f:
        model_args = json.load(f)
    with open(pretrained_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_layers = int(model_args.get("nlayers", 12))
    unfreeze_last_n_layers = config["training"].get("unfreeze_last_n_layers", 1)
    freeze_layers_up_to = max(n_layers - 1 - unfreeze_last_n_layers, -1)

    train_dataset = GeneScoreDataset(
        adata=dataset.train_adata(),
        conditions=dataset.train_conditions(),
        vocab=vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
    )

    val_dataset = GeneScoreDataset(
        adata=dataset.val_adata(),
        conditions=dataset.val_conditions(),
        vocab=train_dataset.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
    )

    n_genes = dataset.adata.n_vars

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
        head_hidden_dim=config.get("head", {}).get("hidden_dim", 512),
        head_dropout=config.get("head", {}).get("dropout", 0.2),
    )

    if is_main:
        print("\n[4/5] Building dataloaders...")
    batch_size = config["training"].get("batch_size", 32)
    train_sampler = None
    val_sampler = None
    if ddp["enabled"]:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

    if config["training"].get("balanced_sampling", False) and not ddp["enabled"]:
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
                batch, model.backbone.vocab, n_genes
            ),
            num_workers=0,
        )
    else:
        if ddp["enabled"] and config["training"].get("balanced_sampling", False):
            if is_main:
                print("Balanced sampling disabled under DDP; using DistributedSampler.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda batch: collate_gene_score_batch(
                batch, model.backbone.vocab, n_genes
            ),
            num_workers=0,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda batch: collate_gene_score_batch(
            batch, model.backbone.vocab, n_genes
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

    if ddp["enabled"]:
        model = DDP(
            model,
            device_ids=[ddp["local_rank"]] if torch.cuda.is_available() else None,
            output_device=ddp["local_rank"] if torch.cuda.is_available() else None,
        )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            sampler=train_sampler,
            max_steps=args.max_steps,
            epoch=epoch,
        )
        val_metrics = {"val_loss": float("inf")}
        if is_main:
            val_metrics = validate(model, val_loader, device)

        if is_main:
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

        if is_main and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_target = model.module if isinstance(model, DDP) else model
            save_target.save(output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

        if args.max_steps:
            if is_main:
                print(f"\nStopping early (max_steps={args.max_steps})")
            break

    if is_main:
        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {output_dir / 'best_model.pt'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
