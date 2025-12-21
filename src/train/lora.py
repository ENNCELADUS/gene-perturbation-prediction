"""
LoRA (Low-Rank Adaptation) for scGPT fine-tuning.

Implements parameter-efficient fine-tuning by adding low-rank
decomposition matrices to transformer attention and FFN layers.

Reference: LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps an existing linear layer and adds low-rank decomposition:
    output = W @ x + (B @ A) @ x * scaling

    Where:
    - W is the frozen original weight
    - A is [rank, in_features]
    - B is [out_features, rank]
    - scaling = alpha / rank
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        """
        Initialize LoRA wrapper.

        Args:
            original_linear: The linear layer to adapt
            rank: LoRA rank (low-rank dimension)
            alpha: LoRA alpha for scaling
            dropout: Dropout rate for LoRA layers
        """
        super().__init__()

        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # LoRA matrices (match original layer device/dtype for multi-GPU safety)
        weight = self.original_linear.weight
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype)
        )

        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [*, in_features]

        Returns:
            Output tensor [*, out_features]
        """
        # Original forward
        result = self.original_linear(x)

        # LoRA forward: x @ A^T @ B^T * scaling
        lora_x = self.lora_dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)

        return result + lora_out * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        """Expose wrapped weight for compatibility with callers expecting nn.Linear."""
        return self.original_linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Expose wrapped bias for compatibility with callers expecting nn.Linear."""
        return self.original_linear.bias

    @property
    def in_features(self) -> int:
        """Expose wrapped in_features for compatibility with nn.Linear."""
        return self.original_linear.in_features

    @property
    def out_features(self) -> int:
        """Expose wrapped out_features for compatibility with nn.Linear."""
        return self.original_linear.out_features

    def merge_weights(self):
        """
        Merge LoRA weights into original linear layer.

        This is useful for inference to avoid LoRA overhead.
        """
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.original_linear.weight.add_(delta_W)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def apply_lora_to_scgpt(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA adapters to scGPT transformer encoder.

    Targets the following layers in each TransformerEncoderLayer:
    - self_attn.out_proj: Attention output projection
    - linear1, linear2: FFN layers

    Args:
        model: scGPT TransformerModel
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout
        target_modules: List of module names to apply LoRA to.
                       Default: ["out_proj", "linear1", "linear2"]

    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = ["out_proj", "linear1", "linear2"]

    lora_layers = []

    # Access transformer encoder layers
    if hasattr(model, "transformer_encoder"):
        encoder = model.transformer_encoder

        for layer_idx, layer in enumerate(encoder.layers):
            # Apply LoRA to attention output projection
            if "out_proj" in target_modules and hasattr(layer.self_attn, "out_proj"):
                original = layer.self_attn.out_proj
                lora_layer = LoRALinear(original, rank, alpha, dropout)
                layer.self_attn.out_proj = lora_layer
                lora_layers.append(f"layer{layer_idx}.self_attn.out_proj")

            # Apply LoRA to FFN linear1
            if "linear1" in target_modules and hasattr(layer, "linear1"):
                original = layer.linear1
                lora_layer = LoRALinear(original, rank, alpha, dropout)
                layer.linear1 = lora_layer
                lora_layers.append(f"layer{layer_idx}.linear1")

            # Apply LoRA to FFN linear2
            if "linear2" in target_modules and hasattr(layer, "linear2"):
                original = layer.linear2
                lora_layer = LoRALinear(original, rank, alpha, dropout)
                layer.linear2 = lora_layer
                lora_layers.append(f"layer{layer_idx}.linear2")

    print(f"[LoRA] Applied to {len(lora_layers)} layers:")
    for name in lora_layers[:6]:  # Print first 6
        print(f"  - {name}")
    if len(lora_layers) > 6:
        print(f"  ... and {len(lora_layers) - 6} more")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters for optimization.

    Args:
        model: Model with LoRA adapters

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_trainable_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model: nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_lora(model: nn.Module):
    """Unfreeze only LoRA parameters."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
