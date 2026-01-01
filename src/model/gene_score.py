"""
Gene-level scoring model for Route B1.

Wraps a frozen scGPT backbone and a trainable gene-scoring head that
matches cell embeddings to gene embeddings.
"""

from __future__ import annotations

from typing import Optional, Sequence
from contextlib import nullcontext
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
        score_mode: str = "dot",
        score_gene_ids: Optional[Sequence[int] | torch.Tensor] = None,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.score_mode = score_mode

        self.backbone = ScGPTForward(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            args_path=args_path,
            freeze_encoder=freeze_encoder,
            freeze_layers_up_to=freeze_layers_up_to,
            device=device,
        )

        self._freeze_unused_heads()

        emb_dim = self.backbone.args["embsize"]
        if self.score_mode == "dot":
            self.head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
        elif self.score_mode == "mlp":
            self.head = nn.Sequential(
                nn.Linear(emb_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            raise ValueError(
                f"score_mode must be 'dot' or 'mlp', got {self.score_mode}"
            )

        self.register_buffer(
            "score_gene_ids", torch.empty(0, dtype=torch.long), persistent=False
        )

        self.to(device)
        if score_gene_ids is not None:
            self.set_score_gene_ids(score_gene_ids)

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

    def set_score_gene_ids(self, gene_ids: Sequence[int] | torch.Tensor) -> None:
        """Set the gene token IDs used for scoring (order defines output)."""
        gene_ids_tensor = torch.as_tensor(gene_ids, dtype=torch.long)
        if gene_ids_tensor.ndim != 1:
            raise ValueError("score_gene_ids must be a 1D tensor or sequence.")
        if gene_ids_tensor.numel() != self.n_genes:
            raise ValueError(
                f"score_gene_ids length {gene_ids_tensor.numel()} does not match "
                f"n_genes={self.n_genes}."
            )
        device = next(self.parameters()).device
        self.score_gene_ids = gene_ids_tensor.to(device=device)

    def _resolve_score_gene_ids(
        self, score_gene_ids: Optional[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        if score_gene_ids is None:
            if self.score_gene_ids.numel() == 0:
                raise ValueError(
                    "score_gene_ids not set. Call set_score_gene_ids(...) or pass "
                    "score_gene_ids to forward."
                )
            return self.score_gene_ids.to(device=device)
        return score_gene_ids.to(device=device)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_gene_ids: Optional[torch.Tensor] = None,
        control_values: Optional[torch.Tensor] = None,
        control_padding_mask: Optional[torch.Tensor] = None,
        control_counts: Optional[int] = None,
        control_chunk_size: Optional[int] = None,
        control_no_grad: bool = True,
        score_gene_ids: Optional[torch.Tensor] = None,
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
        if control_gene_ids is not None:
            if control_values is None or control_padding_mask is None:
                raise ValueError(
                    "control_values and control_padding_mask are required when "
                    "control_gene_ids is provided."
                )
            if control_chunk_size is None or control_chunk_size <= 0:
                control_chunk_size = control_gene_ids.size(0)
            control_embs = []
            ctx = torch.no_grad() if control_no_grad else nullcontext()
            with ctx:
                for start in range(0, control_gene_ids.size(0), control_chunk_size):
                    end = start + control_chunk_size
                    control_outputs = self.backbone.model(
                        src=control_gene_ids[start:end],
                        values=control_values[start:end],
                        src_key_padding_mask=control_padding_mask[start:end],
                        batch_labels=None,
                        CLS=False,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                    )
                    control_embs.append(control_outputs["cell_emb"])
            control_emb = torch.cat(control_embs, dim=0)
            batch_size = cell_emb.size(0)
            if control_counts is None:
                if control_emb.size(0) % batch_size != 0:
                    raise ValueError(
                        "control_counts is required when control batch size is "
                        "not a multiple of the perturbed batch size."
                    )
                controls_per_sample = control_emb.size(0) // batch_size
            else:
                controls_per_sample = int(control_counts)
            control_emb = control_emb.view(batch_size, controls_per_sample, -1)
            control_mean = control_emb.mean(dim=1)
            cell_emb = cell_emb - control_mean

        score_gene_ids = self._resolve_score_gene_ids(
            score_gene_ids, device=cell_emb.device
        )
        gene_emb = self.backbone.model.encoder(score_gene_ids)

        if self.score_mode == "dot":
            cell_proj = self.head(cell_emb)
            return torch.matmul(cell_proj, gene_emb.transpose(0, 1))

        cell_rep = cell_emb.unsqueeze(1).expand(-1, gene_emb.size(0), -1)
        gene_rep = gene_emb.unsqueeze(0).expand(cell_emb.size(0), -1, -1)
        pair = torch.cat([cell_rep, gene_rep], dim=-1)
        return self.head(pair).squeeze(-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=True)
