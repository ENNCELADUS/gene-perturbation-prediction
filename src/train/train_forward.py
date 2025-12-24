"""
Training script for scGPT forward model.

Trains the model to predict perturbed expression from control + condition.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scanpy as sc

from ..data import load_perturb_data
from ..data.forward_collator import ForwardModelDataset, collate_forward_batch
from ..model.scgpt_forward import ScGPTForward


def parse_args():
    parser = argparse.ArgumentParser(description="Train scGPT forward model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (for smoke test)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/forward", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_forward_loss(
    model_outputs: Dict[str, torch.Tensor],
    target_values: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute forward model reconstruction loss.

    Args:
        model_outputs: Model outputs with mlm_output key
        target_values: Ground truth perturbed expression values
        padding_mask: Padding mask (True for padding positions)

    Returns:
        MSE loss on non-padding positions
    """
    pred = model_outputs["mlm_output"]  # (batch, seq_len)

    # Mask out padding positions
    valid_mask = ~padding_mask

    # MSE loss only on valid positions
    mse = torch.nn.functional.mse_loss(
        pred[valid_mask], target_values[valid_mask], reduction="mean"
    )

    return mse


def train_epoch(
    model: ScGPTForward,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    max_steps: int = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        # Move to device
        control_genes = batch["control_genes"].to(device)
        control_values = batch["control_values"].to(device)
        control_mask = batch["control_padding_mask"].to(device)

        perturbed_values = batch["perturbed_values"].to(device)
        perturbed_mask = batch["perturbed_padding_mask"].to(device)

        # Forward pass
        outputs = model(
            gene_ids=control_genes,
            values=control_values,
            padding_mask=control_mask,
        )

        # Compute loss
        loss = compute_forward_loss(outputs, perturbed_values, perturbed_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    return {
        "train_loss": total_loss / num_batches if num_batches > 0 else 0.0,
    }


def validate(
    model: ScGPTForward,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate the model."""
    model.model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            control_genes = batch["control_genes"].to(device)
            control_values = batch["control_values"].to(device)
            control_mask = batch["control_padding_mask"].to(device)

            perturbed_values = batch["perturbed_values"].to(device)
            perturbed_mask = batch["perturbed_padding_mask"].to(device)

            # Forward pass
            outputs = model(
                gene_ids=control_genes,
                values=control_values,
                padding_mask=control_mask,
            )

            # Compute loss
            loss = compute_forward_loss(outputs, perturbed_values, perturbed_mask)

            total_loss += loss.item()
            num_batches += 1

    return {
        "val_loss": total_loss / num_batches if num_batches > 0 else 0.0,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("scGPT Forward Model Training")
    print("=" * 60)
    print(f"Output: {output_dir}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\n[1/5] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    print(f"  - Total cells: {dataset.adata.n_obs}")
    print(f"  - Genes: {dataset.adata.n_vars}")
    print(f"  - Train conditions: {len(dataset.train_conditions())}")
    print(f"  - Val conditions: {len(dataset.val_conditions())}")

    # Load model
    print("\n[2/5] Loading model...")
    model = ScGPTForward(device=device)

    # Create datasets
    print("\n[3/5] Creating datasets...")
    train_dataset = ForwardModelDataset(
        adata=dataset.train_adata(),
        conditions=dataset.train_conditions(),
        vocab=model.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
    )

    val_dataset = ForwardModelDataset(
        adata=dataset.val_adata(),
        conditions=dataset.val_conditions(),
        vocab=model.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: collate_forward_batch(batch, model.vocab),
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_forward_batch(batch, model.vocab),
        num_workers=0,
    )

    # Setup optimizer and scheduler
    print("\n[4/5] Setting up optimizer...")
    optimizer = model.get_optimizer(lr_decoder=1e-4, lr_last_layer=1e-5)

    total_steps = len(train_loader) * 30  # Assume 30 epochs max
    warmup_steps = int(0.05 * total_steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )

    # Training loop
    print("\n[5/5] Training...")
    best_val_loss = float("inf")

    for epoch in range(30):  # Max 30 epochs
        print(f"\nEpoch {epoch + 1}/30")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_steps=args.max_steps,
            epoch=epoch,
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

        # Save checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save_finetuned(output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_finetuned(output_dir / f"checkpoint_epoch{epoch + 1}.pt")

        # Early stopping
        if args.max_steps:
            print(f"\nStopping early (max_steps={args.max_steps})")
            break

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
