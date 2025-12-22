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
import math
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

    # Data paths
    data_h5ad_path: str = "data/norman/perturb_processed.h5ad"

    # Cell-level split config
    split_min_cells_per_condition: int = 50
    split_query_fraction: float = 0.2
    split_min_query_cells: int = 10
    split_seed: int = 42
    split_output_path: Optional[str] = None

    # Condition-level split config (optional)
    track: Optional[str] = None
    condition_split_train_ratio: float = 0.7
    condition_split_val_ratio: float = 0.1
    condition_split_test_ratio: float = 0.2
    condition_split_seed: int = 42
    condition_split_output_path: Optional[str] = None
    condition_split_n_holdout_genes: int = 5

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
    mask_perturbed: bool = True

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
    condition_order: Optional[List[str]] = None

    # Balanced sampling (for contrastive training)
    balanced_sampling: bool = False
    balanced_sampling_n_conditions: int = 8
    balanced_sampling_n_cells: int = 4
    balanced_sampling_seed: int = 42

    # Paths
    checkpoint_dir: str = "model/scgpt_finetune"
    scgpt_model_dir: str = "model/scGPT"
    scgpt_raw_layer_key: Optional[str] = None
    scgpt_preprocess: bool = False
    scgpt_preprocess_normalize_total: float | bool = 1e4
    scgpt_preprocess_log1p: bool = True
    scgpt_preprocess_binning: Optional[int] = None
    scgpt_preprocess_result_binned_key: str = "X_binned"
    scgpt_preprocess_result_normed_key: str = "X_normed"
    scgpt_preprocess_result_log1p_key: str = "X_log1p"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        config_dict = {}
        if "data" in data:
            h5ad_path = data["data"].get("h5ad_path")
            if h5ad_path:
                config_dict["data_h5ad_path"] = h5ad_path
        if "split" in data:
            split_cfg = data["split"]
            config_dict["split_min_cells_per_condition"] = split_cfg.get(
                "min_cells_per_condition",
                config_dict.get("split_min_cells_per_condition"),
            )
            config_dict["split_query_fraction"] = split_cfg.get(
                "query_fraction", config_dict.get("split_query_fraction")
            )
            config_dict["split_min_query_cells"] = split_cfg.get(
                "min_query_cells", config_dict.get("split_min_query_cells")
            )
            config_dict["split_seed"] = split_cfg.get(
                "seed", config_dict.get("split_seed")
            )
            config_dict["split_output_path"] = split_cfg.get("output_path")
        if "track" in data:
            config_dict["track"] = data.get("track")
        if "condition_split" in data:
            cond_cfg = data["condition_split"]
            config_dict["condition_split_train_ratio"] = cond_cfg.get(
                "train_ratio", config_dict.get("condition_split_train_ratio")
            )
            config_dict["condition_split_val_ratio"] = cond_cfg.get(
                "val_ratio", config_dict.get("condition_split_val_ratio")
            )
            config_dict["condition_split_test_ratio"] = cond_cfg.get(
                "test_ratio", config_dict.get("condition_split_test_ratio")
            )
            config_dict["condition_split_seed"] = cond_cfg.get(
                "seed", config_dict.get("condition_split_seed")
            )
            config_dict["condition_split_output_path"] = cond_cfg.get("output_path")
            if "n_holdout_genes" in cond_cfg:
                config_dict["condition_split_n_holdout_genes"] = cond_cfg.get(
                    "n_holdout_genes"
                )
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
            config_dict["scgpt_raw_layer_key"] = data["model"].get("raw_layer_key")
            config_dict["scgpt_preprocess"] = data["model"].get(
                "preprocess", config_dict.get("scgpt_preprocess")
            )
            config_dict["scgpt_preprocess_normalize_total"] = data["model"].get(
                "preprocess_normalize_total",
                config_dict.get("scgpt_preprocess_normalize_total"),
            )
            config_dict["scgpt_preprocess_log1p"] = data["model"].get(
                "preprocess_log1p", config_dict.get("scgpt_preprocess_log1p")
            )
            config_dict["scgpt_preprocess_binning"] = data["model"].get(
                "preprocess_binning", config_dict.get("scgpt_preprocess_binning")
            )
            config_dict["scgpt_preprocess_result_binned_key"] = data["model"].get(
                "preprocess_result_binned_key",
                config_dict.get("scgpt_preprocess_result_binned_key"),
            )
            config_dict["scgpt_preprocess_result_normed_key"] = data["model"].get(
                "preprocess_result_normed_key",
                config_dict.get("scgpt_preprocess_result_normed_key"),
            )
            config_dict["scgpt_preprocess_result_log1p_key"] = data["model"].get(
                "preprocess_result_log1p_key",
                config_dict.get("scgpt_preprocess_result_log1p_key"),
            )

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


class CellTokenDataset(Dataset):
    """Dataset that returns tokenized genes/expressions with labels."""

    def __init__(
        self,
        adata,
        gene_ids: np.ndarray,
        condition_labels: np.ndarray,
        condition_to_idx: Dict[str, int],
        cls_token_id: int,
        pad_value: float,
        input_layer_key: Optional[str] = None,
    ):
        if input_layer_key and input_layer_key != "X":
            if input_layer_key not in adata.layers:
                raise ValueError(
                    f"Layer '{input_layer_key}' not found in AnnData.layers"
                )
            matrix = adata.layers[input_layer_key]
        else:
            matrix = adata.X
        self.count_matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        self.gene_ids = gene_ids
        self.labels = torch.tensor(
            [condition_to_idx[c] for c in condition_labels],
            dtype=torch.long,
        )
        self.cls_token_id = cls_token_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.count_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx].astype(np.float32)
        genes = self.gene_ids[nonzero_idx].astype(np.int64)
        genes = np.insert(genes, 0, self.cls_token_id)
        values = np.insert(values, 0, self.pad_value).astype(np.float32)
        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "label": self.labels[idx],
        }


def _extract_labels(dataset) -> np.ndarray:
    """Extract labels from datasets or subsets for balanced sampling."""
    if isinstance(dataset, torch.utils.data.Subset):
        base_labels = _extract_labels(dataset.dataset)
        return np.asarray(base_labels)[dataset.indices]
    if hasattr(dataset, "labels"):
        labels = dataset.labels
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        return np.asarray(labels)
    raise ValueError("Dataset does not expose labels for balanced sampling.")


class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Sample batches with multiple cells per condition."""

    def __init__(
        self,
        labels: np.ndarray,
        n_conditions_per_batch: int,
        n_cells_per_condition: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.labels = np.asarray(labels)
        self.n_conditions_per_batch = n_conditions_per_batch
        self.n_cells_per_condition = n_cells_per_condition
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        if self.n_conditions_per_batch <= 0 or self.n_cells_per_condition <= 0:
            raise ValueError("Balanced sampler requires positive batch settings.")

        self.label_to_indices: Dict[int, np.ndarray] = {}
        for label in np.unique(self.labels):
            label = int(label)
            self.label_to_indices[label] = np.where(self.labels == label)[0]
        self.unique_labels = list(self.label_to_indices.keys())

        self.batch_size = self.n_conditions_per_batch * self.n_cells_per_condition
        if self.drop_last:
            self.num_batches = max(1, len(self.labels) // self.batch_size)
        else:
            self.num_batches = max(1, math.ceil(len(self.labels) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        for _ in range(self.num_batches):
            if len(self.unique_labels) >= self.n_conditions_per_batch:
                batch_labels = rng.choice(
                    self.unique_labels,
                    size=self.n_conditions_per_batch,
                    replace=False,
                )
            else:
                batch_labels = rng.choice(
                    self.unique_labels,
                    size=self.n_conditions_per_batch,
                    replace=True,
                )

            batch_indices: List[int] = []
            for label in batch_labels:
                indices = self.label_to_indices[int(label)]
                replace = len(indices) < self.n_cells_per_condition
                sampled = rng.choice(
                    indices, size=self.n_cells_per_condition, replace=replace
                )
                batch_indices.extend(sampled.tolist())
            yield batch_indices


def _prepare_scgpt_adata(adata, vocab, gene_col: str):
    """Add vocab ids and filter to genes present in scGPT vocab."""
    adata = adata.copy()
    gene_names = (
        adata.var[gene_col].values if gene_col in adata.var else adata.var.index.values
    )
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in gene_names]
    valid_genes = adata.var["id_in_vocab"] >= 0
    n_valid = int(valid_genes.sum())
    n_total = len(valid_genes)
    if n_valid < n_total:
        print(f"  [scGPT] Using {n_valid}/{n_total} genes in vocabulary")
    adata = adata[:, valid_genes].copy()
    gene_ids = np.array(adata.var["id_in_vocab"])
    return adata, gene_ids


def _make_scgpt_collate_fn(collator):
    """Attach labels to scGPT data collator output."""

    def collate(examples):
        labels = torch.stack([ex["label"] for ex in examples])
        base_examples = [
            {"genes": ex["genes"], "expressions": ex["expressions"]} for ex in examples
        ]
        batch = collator(base_examples)
        batch["labels"] = labels
        return batch

    return collate


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

    def encode_tokens(
        self,
        input_gene_ids: torch.Tensor,
        expressions: torch.Tensor,
        pad_token_id: int,
        batch_labels: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode gene/expression tokens into CLS embeddings."""
        src_key_padding_mask = input_gene_ids.eq(pad_token_id)
        layer_output = self.scgpt_model._encode(
            input_gene_ids,
            expressions,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
        )
        embeddings = layer_output[:, 0, :]
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

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
        end_to_end: bool = False,
        pad_token_id: Optional[int] = None,
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
        self.end_to_end = end_to_end
        self.pad_token_id = pad_token_id

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
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        for batch in dataloader:
            if self.end_to_end:
                input_gene_ids = batch["gene"].to(self.device)
                expressions = batch["expr"].to(self.device)
                labels = batch["labels"].to(self.device)
                cls_embeddings = base_model.encode_tokens(
                    input_gene_ids,
                    expressions,
                    pad_token_id=self.pad_token_id,
                )
                projected = base_model.retrieval_head(cls_embeddings)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                projected = self.model(embeddings)

            self.optimizer.zero_grad()
            loss = base_model.compute_loss(projected, labels)

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
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        for batch in dataloader:
            if self.end_to_end:
                input_gene_ids = batch["gene"].to(self.device)
                expressions = batch["expr"].to(self.device)
                labels = batch["labels"].to(self.device)
                cls_embeddings = base_model.encode_tokens(
                    input_gene_ids,
                    expressions,
                    pad_token_id=self.pad_token_id,
                )
                projected = base_model.retrieval_head(cls_embeddings)
                loss = base_model.compute_loss(projected, labels)
            else:
                embeddings, labels = batch
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                projected = self.model(embeddings)
                loss = base_model.compute_loss(projected, labels)

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
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)

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
    encoder = ScGPTEncoder(
        model_dir=config.scgpt_model_dir,
        raw_layer_key=config.scgpt_raw_layer_key,
        preprocess=config.scgpt_preprocess,
        preprocess_normalize_total=config.scgpt_preprocess_normalize_total,
        preprocess_log1p=config.scgpt_preprocess_log1p,
        preprocess_binning=config.scgpt_preprocess_binning,
        preprocess_result_binned_key=config.scgpt_preprocess_result_binned_key,
        preprocess_result_normed_key=config.scgpt_preprocess_result_normed_key,
        preprocess_result_log1p_key=config.scgpt_preprocess_result_log1p_key,
    )
    encoder._load_model()
    scgpt_model = encoder._model

    # Load dataset
    from src.data import load_perturb_data, mask_perturbed_genes
    from src.data import ConditionSplitter, ConditionSplit

    if is_master:
        print("Loading dataset...")
    dataset = load_perturb_data(
        h5ad_path=config.data_h5ad_path,
        split_path=config.split_output_path,
        min_cells_per_condition=config.split_min_cells_per_condition,
        query_fraction=config.split_query_fraction,
        min_query_cells=config.split_min_query_cells,
        seed=config.split_seed,
    )
    if (
        config.split_output_path
        and dataset.split is not None
        and not Path(config.split_output_path).exists()
    ):
        dataset.split.save(config.split_output_path)

    if config.track and config.track != "in_dist":
        cond_split = None
        if (
            config.condition_split_output_path
            and Path(config.condition_split_output_path).exists()
        ):
            cond_split = ConditionSplit.load(config.condition_split_output_path)
        else:
            splitter = ConditionSplitter(
                train_ratio=config.condition_split_train_ratio,
                val_ratio=config.condition_split_val_ratio,
                test_ratio=config.condition_split_test_ratio,
                seed=config.condition_split_seed,
            )
            if config.track == "unseen_gene":
                cond_split = splitter.split_unseen_gene(
                    dataset.all_conditions,
                    n_holdout_genes=config.condition_split_n_holdout_genes,
                )
            else:
                cond_split = splitter.split(dataset.all_conditions, track=config.track)
            if config.condition_split_output_path:
                cond_split.save(config.condition_split_output_path)
        dataset.apply_condition_split(cond_split)
    elif config.track == "in_dist" and is_master:
        print("  [Info] in_dist track uses cell-level split; skipping condition split.")

    # Get condition mapping
    conditions = dataset.all_conditions
    condition_to_idx = {c: i for i, c in enumerate(conditions)}
    num_conditions = len(conditions)
    config.condition_order = list(conditions)
    if is_master:
        print(f"  Conditions: {num_conditions}")

    # Create model
    model = FineTunableScGPTEncoder(
        scgpt_model=scgpt_model,
        config=config,
        num_conditions=num_conditions,
    )

    condition_sets = dataset.get_condition_sets()
    train_conditions = condition_sets.get("train") or conditions
    val_conditions = condition_sets.get("val") or []

    if config.mode == "lora_head":
        if is_master:
            print("Preparing token datasets for LoRA fine-tuning...")
        train_adata = dataset.get_ref_adata_for_conditions(train_conditions)
        if config.mask_perturbed:
            train_adata = mask_perturbed_genes(
                train_adata,
                condition_col=dataset.condition_col,
                layer=config.scgpt_raw_layer_key,
            )
        train_adata, train_gene_ids = encoder.prepare_adata(train_adata)
        train_labels = train_adata.obs[dataset.condition_col].values
        train_dataset = CellTokenDataset(
            adata=train_adata,
            gene_ids=train_gene_ids,
            condition_labels=train_labels,
            condition_to_idx=condition_to_idx,
            cls_token_id=encoder._vocab["<cls>"],
            pad_value=encoder._model_configs["pad_value"],
            input_layer_key=encoder.resolve_input_layer_key(train_adata),
        )

        if val_conditions:
            val_adata = dataset.get_ref_adata_for_conditions(val_conditions)
            if config.mask_perturbed:
                val_adata = mask_perturbed_genes(
                    val_adata,
                    condition_col=dataset.condition_col,
                    layer=config.scgpt_raw_layer_key,
                )
            val_adata, val_gene_ids = encoder.prepare_adata(val_adata)
            val_labels = val_adata.obs[dataset.condition_col].values
            val_dataset = CellTokenDataset(
                adata=val_adata,
                gene_ids=val_gene_ids,
                condition_labels=val_labels,
                condition_to_idx=condition_to_idx,
                cls_token_id=encoder._vocab["<cls>"],
                pad_value=encoder._model_configs["pad_value"],
                input_layer_key=encoder.resolve_input_layer_key(val_adata),
            )
            train_subset = train_dataset
            val_subset = val_dataset
        else:
            n_train = int(0.8 * len(train_dataset))
            n_val = len(train_dataset) - n_train
            generator = torch.Generator().manual_seed(config.split_seed)
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [n_train, n_val], generator=generator
            )

        from scgpt.data_collator import DataCollator

        pad_token_id = encoder._vocab[encoder._model_configs["pad_token"]]
        pre_binned = config.scgpt_preprocess and not (
            config.scgpt_preprocess_binning is False
            or config.scgpt_preprocess_binning == 0
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=pad_token_id,
            pad_value=encoder._model_configs["pad_value"],
            do_mlm=False,
            do_binning=not pre_binned,
            max_length=encoder.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        collate_fn = _make_scgpt_collate_fn(collator)
    else:
        # Extract embeddings (pre-compute for efficiency)
        if is_master:
            print("Extracting embeddings...")
        train_adata = dataset.get_ref_adata_for_conditions(train_conditions)
        if config.mask_perturbed:
            train_adata = mask_perturbed_genes(
                train_adata,
                condition_col=dataset.condition_col,
                layer=config.scgpt_raw_layer_key,
            )
        train_embeddings = encoder.encode_adata(train_adata)
        train_labels = train_adata.obs[dataset.condition_col].values

        train_dataset = CellEmbeddingDataset(
            embeddings=train_embeddings,
            condition_labels=train_labels,
            condition_to_idx=condition_to_idx,
        )

        if val_conditions:
            val_adata = dataset.get_ref_adata_for_conditions(val_conditions)
            if config.mask_perturbed:
                val_adata = mask_perturbed_genes(
                    val_adata,
                    condition_col=dataset.condition_col,
                    layer=config.scgpt_raw_layer_key,
                )
            val_embeddings = encoder.encode_adata(val_adata)
            val_labels = val_adata.obs[dataset.condition_col].values
            val_dataset = CellEmbeddingDataset(
                embeddings=val_embeddings,
                condition_labels=val_labels,
                condition_to_idx=condition_to_idx,
            )
            train_subset = train_dataset
            val_subset = val_dataset
        else:
            n_train = int(0.8 * len(train_dataset))
            n_val = len(train_dataset) - n_train
            generator = torch.Generator().manual_seed(config.split_seed)
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [n_train, n_val], generator=generator
            )
        collate_fn = None
        pad_token_id = None

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = torch.utils.data.DistributedSampler(train_subset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_subset, shuffle=False)

    use_balanced_sampling = (
        config.balanced_sampling and config.loss_fn == "infonce" and not ddp_enabled
    )
    if config.balanced_sampling and not use_balanced_sampling and is_master:
        print("Balanced sampling disabled (requires infonce loss and non-DDP mode).")
    if use_balanced_sampling:
        labels = _extract_labels(train_subset)
        batch_sampler = BalancedBatchSampler(
            labels=labels,
            n_conditions_per_batch=config.balanced_sampling_n_conditions,
            n_cells_per_condition=config.balanced_sampling_n_cells,
            seed=config.balanced_sampling_seed,
            drop_last=True,
        )
        if is_master:
            print(
                "Using balanced sampling: "
                f"{config.balanced_sampling_n_conditions} conditions x "
                f"{config.balanced_sampling_n_cells} cells"
            )
        train_loader = DataLoader(
            train_subset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            drop_last=True,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    # Train
    if ddp_enabled:
        allow_unused = config.loss_fn == "infonce" or config.mode == "lora_head"
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[int(device.split(":")[-1])],
            find_unused_parameters=allow_unused,
        )
    trainer = ScGPTTrainer(
        model,
        config,
        device=device,
        is_master=is_master,
        end_to_end=config.mode == "lora_head",
        pad_token_id=pad_token_id,
    )

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
