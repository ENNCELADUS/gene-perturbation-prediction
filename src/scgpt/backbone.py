"""scGPT pretrained transformer backbone wrapper."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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
