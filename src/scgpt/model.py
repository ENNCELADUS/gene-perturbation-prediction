"""scGPT backbone and gene-scoring head."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from src.utils.distributed import (
    log_primary_info,
    suppress_torchtext_deprecation_warning,
)

LOGGER = logging.getLogger(__name__)


class ScGPTBackbone(nn.Module):
    """Load a pretrained scGPT transformer."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        args_path: str | Path,
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.args_path = Path(args_path)
        self.freeze_encoder = freeze_encoder
        self.freeze_layers_up_to = freeze_layers_up_to
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        self.device = device
        with self.vocab_path.open() as vocab_file:
            self.vocab = json.load(vocab_file)
        with self.args_path.open() as args_file:
            self.args = json.load(args_file)
        self.model = self._build_model()
        self._load_checkpoint()
        if self.freeze_encoder:
            self._apply_freeze_strategy()
        self.model.to(self.device)

    def _build_model(self):
        scgpt_path = Path(__file__).parents[2] / "scGPT"
        if str(scgpt_path) not in sys.path:
            sys.path.insert(0, str(scgpt_path))
        suppress_torchtext_deprecation_warning()
        from scgpt.model.model import TransformerModel

        return TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.args["embsize"],
            nhead=self.args["nheads"],
            d_hid=self.args["d_hid"],
            nlayers=self.args["nlayers"],
            nlayers_cls=self.args.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self.vocab,
            dropout=self.args["dropout"],
            pad_token=self.args["pad_token"],
            pad_value=self.args["pad_value"],
            do_mvc=self.args.get("MVC", True),
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=self.args["input_emb_style"],
            n_input_bins=self.args.get("n_bins", 51),
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            explicit_zero_prob=False,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend=self.fast_transformer_backend,
            pre_norm=False,
        )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_state = self.model.state_dict()
        compatible = {
            key: value
            for key, value in checkpoint.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        model_state.update(compatible)
        self.model.load_state_dict(model_state, strict=True)
        log_primary_info(LOGGER, "Loaded %d scGPT parameters", len(compatible))

    def _apply_freeze_strategy(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for layer_idx in range(self.freeze_layers_up_to + 1, self.args["nlayers"]):
            for param in self.model.transformer_encoder.layers[layer_idx].parameters():
                param.requires_grad = True


class GeneScoreModel(nn.Module):
    """scGPT cell embedding plus trainable gene-scoring head."""

    def __init__(
        self,
        n_genes: int,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        args_path: str | Path,
        score_gene_ids: Sequence[int] | torch.Tensor | None = None,
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,
        score_mode: str = "dot",
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.score_mode = score_mode
        self.backbone = ScGPTBackbone(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            args_path=args_path,
            freeze_encoder=freeze_encoder,
            freeze_layers_up_to=freeze_layers_up_to,
            use_fast_transformer=use_fast_transformer,
            fast_transformer_backend=fast_transformer_backend,
            device=device,
        )
        emb_dim = int(self.backbone.args["embsize"])
        if score_mode == "dot":
            self.head = nn.Sequential(
                nn.Linear(emb_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, emb_dim),
            )
        elif score_mode == "mlp":
            self.head = nn.Sequential(
                nn.Linear(emb_dim * 2, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            raise ValueError("score_mode must be 'dot' or 'mlp'")
        self.register_buffer(
            "score_gene_ids", torch.empty(0, dtype=torch.long), persistent=False
        )
        if score_gene_ids is not None:
            self.set_score_gene_ids(score_gene_ids)

    def set_score_gene_ids(self, gene_ids: Sequence[int] | torch.Tensor) -> None:
        """Set the token ids used for the output score order."""
        gene_ids_tensor = torch.as_tensor(gene_ids, dtype=torch.long)
        if gene_ids_tensor.ndim != 1 or gene_ids_tensor.numel() != self.n_genes:
            raise ValueError("score_gene_ids must be 1D with length n_genes")
        self.score_gene_ids = gene_ids_tensor.to(next(self.parameters()).device)

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        control_gene_ids: torch.Tensor | None = None,
        control_values: torch.Tensor | None = None,
        control_padding_mask: torch.Tensor | None = None,
        control_counts: int | None = None,
        control_chunk_size: int | None = None,
        control_no_grad: bool = True,
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
            cell_emb = cell_emb - self._control_mean(
                control_gene_ids,
                control_values,
                control_padding_mask,
                control_counts,
                control_chunk_size,
                control_no_grad,
                batch_size=cell_emb.size(0),
            )
        gene_emb = self.backbone.model.encoder(self.score_gene_ids.to(cell_emb.device))
        if self.score_mode == "dot":
            return self.head(cell_emb) @ gene_emb.transpose(0, 1)
        cell_rep = cell_emb.unsqueeze(1).expand(-1, gene_emb.size(0), -1)
        gene_rep = gene_emb.unsqueeze(0).expand(cell_emb.size(0), -1, -1)
        return self.head(torch.cat([cell_rep, gene_rep], dim=-1)).squeeze(-1)

    def _control_mean(
        self,
        control_gene_ids: torch.Tensor,
        control_values: torch.Tensor | None,
        control_padding_mask: torch.Tensor | None,
        control_counts: int | None,
        control_chunk_size: int | None,
        control_no_grad: bool,
        batch_size: int,
    ) -> torch.Tensor:
        if control_values is None or control_padding_mask is None:
            raise ValueError("control_values and control_padding_mask are required")
        chunk_size = control_chunk_size or control_gene_ids.size(0)
        control_embs = []
        context = torch.no_grad() if control_no_grad else nullcontext()
        with context:
            for start in range(0, control_gene_ids.size(0), chunk_size):
                output = self.backbone.model(
                    src=control_gene_ids[start : start + chunk_size],
                    values=control_values[start : start + chunk_size],
                    src_key_padding_mask=control_padding_mask[
                        start : start + chunk_size
                    ],
                    batch_labels=None,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                )
                control_embs.append(output["cell_emb"])
        control_emb = torch.cat(control_embs, dim=0)
        controls_per_sample = control_counts or control_emb.size(0) // batch_size
        return control_emb.view(batch_size, controls_per_sample, -1).mean(dim=1)
