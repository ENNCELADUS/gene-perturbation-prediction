"""Shared scGPT backbone loader for inverse perturbation models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

from scgpt.model.model import TransformerModel  # noqa: E402

logger = logging.getLogger(__name__)


class ScGPTBackbone(nn.Module):
    """Load a pretrained scGPT transformer and apply the shared freeze policy."""

    def __init__(
        self,
        checkpoint_path: str = "model/scGPT/best_model.pt",
        vocab_path: str = "model/scGPT/vocab.json",
        args_path: str = "model/scGPT/args.json",
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.args_path = Path(args_path)
        self.freeze_encoder = freeze_encoder
        self.freeze_layers_up_to = freeze_layers_up_to
        self.device = device

        with open(self.vocab_path) as vocab_file:
            self.vocab = json.load(vocab_file)
        with open(self.args_path) as args_file:
            self.args = json.load(args_file)

        self.model = self._build_model()
        self._load_checkpoint()

        if self.freeze_encoder:
            self._apply_freeze_strategy()

        self.model.to(self.device)

    def _build_model(self) -> TransformerModel:
        """Build the scGPT TransformerModel using pretrained checkpoint args."""
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
            use_fast_transformer=True,
            fast_transformer_backend="flash",
            pre_norm=False,
        )

    def _load_checkpoint(self) -> None:
        """Load compatible pretrained weights into the scGPT model."""
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_state = self.model.state_dict()
        compatible = {}
        mismatched = []
        unexpected = []

        for key, value in checkpoint.items():
            if key not in model_state:
                unexpected.append(key)
                continue
            if model_state[key].shape != value.shape:
                mismatched.append(key)
                continue
            compatible[key] = value

        model_state.update(compatible)
        self.model.load_state_dict(model_state, strict=True)

        missing = [key for key in model_state if key not in compatible]
        missing = [
            key for key in missing if not key.startswith(("cls_decoder.",))
        ]
        unexpected = [
            key for key in unexpected if not key.startswith(("flag_encoder.",))
        ]

        logger.info(
            "Loaded scGPT checkpoint from %s with %d matched parameters.",
            self.checkpoint_path,
            len(compatible),
        )
        if missing:
            logger.warning(
                "Missing checkpoint keys (%d): %s",
                len(missing),
                missing[:5],
            )
        if unexpected:
            logger.warning(
                "Unexpected checkpoint keys (%d): %s", len(unexpected), unexpected[:5]
            )
        if mismatched:
            logger.warning(
                "Mismatched checkpoint shapes (%d): %s", len(mismatched), mismatched[:5]
            )

    def _apply_freeze_strategy(self) -> None:
        """Freeze most of scGPT while leaving the final transformer blocks trainable."""
        for param in self.model.parameters():
            param.requires_grad = False

        for layer_idx in range(self.freeze_layers_up_to + 1, self.args["nlayers"]):
            layer = self.model.transformer_encoder.layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.model.encoder, "enc_norm"):
            for param in self.model.encoder.enc_norm.parameters():
                param.requires_grad = True

        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        logger.info(
            "Applied scGPT freeze policy: %d trainable / %d total parameters.",
            trainable_params,
            total_params,
        )

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        mvc: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the scGPT transformer and return its output dictionary."""
        return self.model(
            src=gene_ids,
            values=values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=mvc,
            ECS=False,
            do_sample=False,
        )

    def get_gene_id(self, gene_name: str) -> Optional[int]:
        """Return a token id for a gene name when present in the scGPT vocab."""
        return self.vocab.get(gene_name)
