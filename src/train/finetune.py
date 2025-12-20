"""
scGPT fine-tuning for reverse perturbation retrieval.

Implements three training modes:
1. Frozen: Inference-only (no training)
2. Head-only: Train retrieval head with frozen backbone
3. LoRA+Head: Train LoRA adapters + retrieval head

Usage:
    python -m src.train.finetune --config src/configs/scgpt_finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import yaml

from .losses import InfoNCELoss, ClassificationLoss
from .lora import (
    apply_lora_to_scgpt,
    get_lora_parameters,
    freeze_model,
    unfreeze_lora,
    count_trainable_parameters,
)


@dataclass
class TrainingConfig:
    """Configuration for scGPT fine-tuning."""

    # Mode: frozen | head_only | lora_head
    mode: str = "head_only"

    # Loss function: infonce | classification
    loss_fn: str = "infonce"

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"

    # LoRA config
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["out_proj", "linear1", "linear2"]
    )

    # Head config
    head_hidden_dim: int = 256
    head_output_dim: int = 128
    head_dropout: float = 0.2

    # InfoNCE config
    infonce_temperature: float = 0.07

    # Classification config
    label_smoothing: float = 0.1

    # Paths
    checkpoint_dir: str = "checkpoints/scgpt_finetune"
    scgpt_model_dir: str = "model/scGPT"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        config_dict = {}
        if "training" in data:
            config_dict.update(data["training"])
        if "lora" in data:
            for k, v in data["lora"].items():
                config_dict[f"lora_{k}"] = v
        if "head" in data:
            for k, v in data["head"].items():
                config_dict[f"head_{k}"] = v
        if "model" in data:
            # Support both pretrained_dir and checkpoint keys
            pretrained = data["model"].get("pretrained_dir") or data["model"].get(
                "checkpoint"
            )
            if pretrained:
                config_dict["scgpt_model_dir"] = pretrained

        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class RetrievalHead(nn.Module):
    """
    MLP projection head for retrieval embeddings.

    Projects scGPT CLS embeddings to a retrieval-optimized space.
    Architecture: [embsize -> hidden -> output_dim]
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
        normalize: bool = True,
    ):
        """
        Initialize retrieval head.

        Args:
            input_dim: Input embedding dimension (scGPT embsize)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
            normalize: Whether to L2-normalize output
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.normalize = normalize

        # Optional learnable temperature for similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.

        Args:
            x: Input embeddings [B, input_dim]

        Returns:
            Projected embeddings [B, output_dim]
        """
        x = self.mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


class CellEmbeddingDataset(Dataset):
    """
    Dataset for cell embeddings with condition labels.

    Stores pre-computed embeddings for efficient training.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        condition_labels: np.ndarray,
        condition_to_idx: Dict[str, int],
    ):
        """
        Initialize dataset.

        Args:
            embeddings: Cell embeddings [N, D]
            condition_labels: Condition names for each cell [N]
            condition_to_idx: Mapping from condition name to integer
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(
            [condition_to_idx[c] for c in condition_labels],
            dtype=torch.long,
        )
        self.condition_to_idx = condition_to_idx
        self.idx_to_condition = {v: k for k, v in condition_to_idx.items()}

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class FineTunableScGPTEncoder(nn.Module):
    """
    scGPT encoder with fine-tuning support.

    Wraps the scGPT TransformerModel and adds:
    - Retrieval head for embedding projection
    - LoRA adapters for parameter-efficient fine-tuning
    - Multiple training modes (frozen, head_only, lora_head)
    """

    def __init__(
        self,
        scgpt_model: nn.Module,
        config: TrainingConfig,
        num_conditions: Optional[int] = None,
    ):
        """
        Initialize fine-tunable encoder.

        Args:
            scgpt_model: Pretrained scGPT TransformerModel
            config: Training configuration
            num_conditions: Number of conditions (for classification loss)
        """
        super().__init__()

        self.scgpt_model = scgpt_model
        self.config = config

        # Get embedding dimension from model config
        self.embsize = scgpt_model.d_model if hasattr(scgpt_model, "d_model") else 512

        # Retrieval head
        self.retrieval_head = RetrievalHead(
            input_dim=self.embsize,
            hidden_dim=config.head_hidden_dim,
            output_dim=config.head_output_dim,
            dropout=config.head_dropout,
        )

        # Loss function
        if config.loss_fn == "infonce":
            self.loss_fn = InfoNCELoss(
                temperature=config.infonce_temperature,
                normalize=True,
            )
        elif config.loss_fn == "classification":
            if num_conditions is None:
                raise ValueError("num_conditions required for classification loss")
            self.loss_fn = ClassificationLoss(
                num_conditions=num_conditions,
                embedding_dim=config.head_output_dim,
                hidden_dim=config.head_hidden_dim,
                dropout=config.head_dropout,
                label_smoothing=config.label_smoothing,
            )
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

        # Apply training mode
        self._apply_training_mode()

    def _apply_training_mode(self):
        """Configure model for training mode."""
        mode = self.config.mode

        if mode == "frozen":
            # Freeze everything - inference only
            freeze_model(self.scgpt_model)
            freeze_model(self.retrieval_head)
            freeze_model(self.loss_fn)

        elif mode == "head_only":
            # Freeze backbone, train head + loss
            freeze_model(self.scgpt_model)
            # Head is trainable by default

        elif mode == "lora_head":
            # Freeze backbone, add LoRA, train LoRA + head + loss
            freeze_model(self.scgpt_model)
            apply_lora_to_scgpt(
                self.scgpt_model,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            unfreeze_lora(self.scgpt_model)

        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = count_trainable_parameters(self)
        print(f"[FineTunable] Mode: {mode}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / total_params:.4%}")

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through retrieval head.

        Note: This assumes embeddings are already extracted from scGPT.
        For end-to-end training with gene tokens, use encode_cells().

        Args:
            embeddings: Cell embeddings from scGPT [B, embsize]

        Returns:
            Projected embeddings [B, output_dim]
        """
        return self.retrieval_head(embeddings)

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            embeddings: Projected embeddings [B, output_dim]
            labels: Condition labels [B]

        Returns:
            Loss tensor
        """
        return self.loss_fn(embeddings, labels)


class ScGPTTrainer:
    """
    Trainer for scGPT fine-tuning.

    Handles:
    - Training loop with validation
    - Early stopping
    - Checkpoint saving
    - Logging
    """

    def __init__(
        self,
        model: FineTunableScGPTEncoder,
        config: TrainingConfig,
        device: str = "cuda",
        is_master: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: FineTunableScGPTEncoder
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.is_master = is_master

        # Optimizer only for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = None  # Set during training

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for embeddings, labels in dataloader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            projected = self.model(embeddings)
            loss = self.model.compute_loss(projected, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats.tolist()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for embeddings, labels in dataloader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            projected = self.model(embeddings)
            loss = self.model.compute_loss(projected, labels)

            total_loss += loss.item()
            num_batches += 1

        if dist.is_initialized():
            stats = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, num_batches = stats.tolist()

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )

        if self.is_master:
            print(f"\n{'=' * 60}")
            print(f"Starting training: {self.config.mode} mode")
            print(f"Loss function: {self.config.loss_fn}")
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            print(f"{'=' * 60}\n")

        for epoch in range(self.config.epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)

            # Log
            if self.is_master:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    if self.is_master:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        # Save final
        self._save_checkpoint("final")

        if self.is_master:
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not self.is_master:
            return

        checkpoint_path = self.checkpoint_dir / f"{name}_{self.config.mode}.pt"
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        # Save only trainable parts
        state = {
            "config": asdict(self.config),
            "retrieval_head": base_model.retrieval_head.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }

        # Save loss function state if it has parameters
        if hasattr(base_model.loss_fn, "state_dict"):
            state["loss_fn"] = base_model.loss_fn.state_dict()

        # Save LoRA weights if applicable
        if self.config.mode == "lora_head":
            lora_state = {}
            for name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear

                if isinstance(module, LoRALinear):
                    lora_state[name] = {
                        "lora_A": module.lora_A.data,
                        "lora_B": module.lora_B.data,
                    }
            state["lora"] = lora_state

        torch.save(state, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.retrieval_head.load_state_dict(state["retrieval_head"])

        if "loss_fn" in state and hasattr(base_model.loss_fn, "load_state_dict"):
            base_model.loss_fn.load_state_dict(state["loss_fn"])

        if "lora" in state:
            for name, module in base_model.scgpt_model.named_modules():
                from .lora import LoRALinear

                if isinstance(module, LoRALinear) and name in state["lora"]:
                    module.lora_A.data = state["lora"][name]["lora_A"]
                    module.lora_B.data = state["lora"][name]["lora_B"]

        self.history = state.get("history", {})
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint: {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="scGPT Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/scgpt_finetune.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frozen", "head_only", "lora_head"],
        default=None,
        help="Training mode (overrides config)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["infonce", "classification"],
        default=None,
        help="Loss function (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a quick test without full training",
    )
    return parser.parse_args()


def _setup_ddp() -> Tuple[bool, str, int]:
    """Initialize DDP if launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, f"cuda:{local_rank}", int(os.environ.get("RANK", "0"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return False, device, 0


def main():
    """Main entry point for fine-tuning."""
    args = parse_args()

    # Load config
    if Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Override from CLI
    if args.mode:
        config.mode = args.mode
    if args.loss:
        config.loss_fn = args.loss

    ddp_enabled, device, rank = _setup_ddp()
    is_master = rank == 0

    if is_master:
        print(f"\n{'=' * 60}")
        print("scGPT Fine-tuning for Reverse Perturbation Retrieval")
        print(f"{'=' * 60}")
        print(f"Mode: {config.mode}")
        print(f"Loss: {config.loss_fn}")
        print(f"{'=' * 60}\n")

    if args.dry_run:
        if is_master:
            print("[Dry run] Configuration loaded successfully.")
            print(f"  LoRA rank: {config.lora_rank}")
            print(f"  Head hidden dim: {config.head_hidden_dim}")
            print(f"  Learning rate: {config.learning_rate}")
        return

    # Import heavy dependencies only when needed
    import sys
    from pathlib import Path as PathLib

    # Add scGPT to path
    scgpt_path = PathLib(__file__).parent.parent.parent / "scGPT"
    if str(scgpt_path) not in sys.path:
        sys.path.insert(0, str(scgpt_path))

    # Load scGPT model
    from src.model import ScGPTEncoder

    if is_master:
        print("Loading scGPT model...")
    encoder = ScGPTEncoder(model_dir=config.scgpt_model_dir)
    encoder._load_model()
    scgpt_model = encoder._model

    # Load dataset
    from src.data import load_perturb_data

    if is_master:
        print("Loading dataset...")
    dataset = load_perturb_data(h5ad_path="data/norman/perturb_processed.h5ad")

    # Get condition mapping
    conditions = dataset.all_conditions
    condition_to_idx = {c: i for i, c in enumerate(conditions)}
    num_conditions = len(conditions)
    if is_master:
        print(f"  Conditions: {num_conditions}")

    # Create model
    model = FineTunableScGPTEncoder(
        scgpt_model=scgpt_model,
        config=config,
        num_conditions=num_conditions,
    )

    # Extract embeddings (pre-compute for efficiency)
    if is_master:
        print("Extracting embeddings...")
    train_adata = dataset.get_ref_adata_for_conditions(conditions)
    embeddings = encoder.encode_adata(train_adata)
    labels = train_adata.obs[dataset.condition_col].values

    # Create datasets
    train_dataset = CellEmbeddingDataset(
        embeddings=embeddings,
        condition_labels=labels,
        condition_to_idx=condition_to_idx,
    )

    # Simple 80/20 split for train/val
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val]
    )

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = torch.utils.data.DistributedSampler(train_subset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_subset, shuffle=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    # Train
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device), device_ids=[int(device.split(":")[-1])]
        )
    trainer = ScGPTTrainer(model, config, device=device, is_master=is_master)

    history = trainer.train(train_loader, val_loader)

    # Save history
    if is_master:
        history_path = Path(config.checkpoint_dir) / f"history_{config.mode}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved: {history_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
