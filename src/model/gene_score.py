"""
Gene-level scoring model for Route B1.

Wraps a frozen scGPT backbone and a trainable gene-scoring head.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from .scgpt_forward import ScGPTForward


class GeneScoreModel(nn.Module):
    """scGPT backbone + gene-scoring head."""

    def __init__(
        self,
        n_genes: int,
        checkpoint_path: str = "model/scGPT/best_model.pt",
        vocab_path: str = "model/scGPT/vocab.json",
        args_path: str = "model/scGPT/args.json",
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = ScGPTForward(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            args_path=args_path,
            freeze_encoder=freeze_encoder,
            freeze_layers_up_to=freeze_layers_up_to,
            device=device,
        )

        self._freeze_unused_heads()

        self.head = nn.Sequential(
            nn.Linear(self.backbone.args["embsize"], head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, n_genes),
        )

        self.to(device)

    def _freeze_unused_heads(self) -> None:
        """Freeze decoder heads that are not used for gene scoring."""
        if hasattr(self.backbone.model, "cls_decoder"):
            for param in self.backbone.model.cls_decoder.parameters():
                param.requires_grad = False
        if hasattr(self.backbone.model, "decoder"):
            for param in self.backbone.model.decoder.parameters():
                param.requires_grad = False
        if hasattr(self.backbone.model, "mvc_decoder"):
            for param in self.backbone.model.mvc_decoder.parameters():
                param.requires_grad = False
        if hasattr(self.backbone.model, "flag_encoder"):
            for param in self.backbone.model.flag_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone.model(
            src=gene_ids,
            values=values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        cell_emb = outputs["cell_emb"]
        return self.head(cell_emb)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=True)
