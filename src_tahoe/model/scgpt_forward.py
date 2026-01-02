"""
scGPT forward model wrapper for perturbation prediction.

Loads pretrained scGPT checkpoint and applies freeze strategy for finetuning.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import json

# Import scGPT model bypassing tokenizer to avoid torchtext ABI incompatibility
import sys
from pathlib import Path

# Add scGPT to sys.path
scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

# Now import model directly - this will load scgpt.model.model but not scgpt.__init__
from scgpt.model.model import TransformerModel


class ScGPTForward(nn.Module):
    """
    scGPT forward model wrapper for perturbation prediction.

    Freeze strategy:
    - Frozen: encoder, flag_encoder, value_encoder, transformer layers 0-10
    - Trainable: decoder, mvc_decoder, transformer layer 11, enc_norm
    """

    def __init__(
        self,
        checkpoint_path: str = "model/scGPT/best_model.pt",
        vocab_path: str = "model/scGPT/vocab.json",
        args_path: str = "model/scGPT/args.json",
        freeze_encoder: bool = True,
        freeze_layers_up_to: int = 10,  # Freeze layers 0-10, train layer 11
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.args_path = Path(args_path)
        self.freeze_encoder = freeze_encoder
        self.freeze_layers_up_to = freeze_layers_up_to
        self.device = device

        # Load vocab and args
        with open(self.vocab_path) as f:
            self.vocab = json.load(f)
        with open(self.args_path) as f:
            self.args = json.load(f)

        # Build model from checkpoint config
        self.model = self._build_model()

        # Load checkpoint weights
        self._load_checkpoint()

        # Apply freeze strategy
        if self.freeze_encoder:
            self._apply_freeze_strategy()

        # Move to device
        self.model.to(self.device)

    def _build_model(self) -> TransformerModel:
        """Build scGPT TransformerModel from checkpoint args."""
        model = TransformerModel(
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
            do_dab=False,  # Not used in forward modeling
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=self.args["input_emb_style"],
            n_input_bins=self.args.get("n_bins", 51),
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            explicit_zero_prob=False,
            use_fast_transformer=True,  # Checkpoint uses FlashAttention format
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        return model

    def _load_checkpoint(self):
        """Load pretrained weights from checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
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

        missing = [key for key in model_state.keys() if key not in compatible]
        allowed_missing_prefixes = ("cls_decoder.",)
        allowed_unexpected_prefixes = ("flag_encoder.",)

        missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        unexpected = [
            key for key in unexpected if not key.startswith(allowed_unexpected_prefixes)
        ]

        print(f"  ✓ Loaded checkpoint ({len(compatible)} matched params)")
        if missing:
            print(f"  ⚠ Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        if mismatched:
            print(f"  ⚠ Mismatched shapes ({len(mismatched)}): {mismatched[:5]}...")

    def _apply_freeze_strategy(self):
        """
        Apply freeze strategy for finetuning:
        - Freeze encoder, flag_encoder, value_encoder
        - Freeze transformer layers 0 to freeze_layers_up_to
        - Keep decoder, mvc_decoder trainable
        - Keep last transformer layer(s) trainable
        - Keep enc_norm trainable
        """
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze decoder and mvc_decoder
        if hasattr(self.model, "decoder"):
            for param in self.model.decoder.parameters():
                param.requires_grad = True

        if hasattr(self.model, "mvc_decoder"):
            for param in self.model.mvc_decoder.parameters():
                param.requires_grad = True

        # Unfreeze last transformer layer(s)
        for layer_idx in range(self.freeze_layers_up_to + 1, self.args["nlayers"]):
            layer = self.model.transformer_encoder.layers[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze encoder norm
        if hasattr(self.model.encoder, "enc_norm"):
            for param in self.model.encoder.enc_norm.parameters():
                param.requires_grad = True

        # Print freeze summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        print(f"Freeze strategy applied:")
        print(f"  Total params: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(
            f"  Trainable:    {trainable_params:,} ({trainable_params / 1e6:.2f}M, {100 * trainable_params / total_params:.1f}%)"
        )
        print(
            f"  Frozen:       {frozen_params:,} ({frozen_params / 1e6:.2f}M, {100 * frozen_params / total_params:.1f}%)"
        )

    def forward(
        self,
        gene_ids: torch.Tensor,  # (batch, seq_len) - gene token IDs
        values: torch.Tensor,  # (batch, seq_len) - binned expression values
        padding_mask: torch.Tensor,  # (batch, seq_len) - True for padding
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through scGPT model.

        Args:
            gene_ids: Gene token IDs, shape (batch, seq_len)
            values: Binned expression values, shape (batch, seq_len)
            padding_mask: Padding mask (True for padding), shape (batch, seq_len)

        Returns:
            Dictionary with:
            - mlm_output: Predicted expression, shape (batch, seq_len)
            - cell_emb: Cell embedding, shape (batch, embsize)
            - mvc_output: MVC prediction (if do_mvc=True), shape (batch, seq_len)
        """
        outputs = self.model(
            src=gene_ids,
            values=values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=self.args.get("MVC", True),
            ECS=False,
            do_sample=False,
        )
        return outputs

    def get_optimizer(
        self,
        lr_decoder: float = 1e-4,
        lr_last_layer: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with parameter groups for different learning rates.

        Args:
            lr_decoder: Learning rate for decoder/mvc_decoder
            lr_last_layer: Learning rate for last transformer layer + enc_norm
            weight_decay: Weight decay for AdamW

        Returns:
            AdamW optimizer with parameter groups
        """
        param_groups = []

        # Decoder params (higher LR)
        decoder_params = []
        if hasattr(self.model, "decoder"):
            decoder_params.extend(
                [p for p in self.model.decoder.parameters() if p.requires_grad]
            )
        if hasattr(self.model, "mvc_decoder"):
            decoder_params.extend(
                [p for p in self.model.mvc_decoder.parameters() if p.requires_grad]
            )

        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": lr_decoder})

        # Last transformer layer + enc_norm (lower LR)
        last_layer_params = []
        for layer_idx in range(self.freeze_layers_up_to + 1, self.args["nlayers"]):
            layer = self.model.transformer_encoder.layers[layer_idx]
            last_layer_params.extend([p for p in layer.parameters() if p.requires_grad])

        if hasattr(self.model.encoder, "enc_norm"):
            last_layer_params.extend(
                [p for p in self.model.encoder.enc_norm.parameters() if p.requires_grad]
            )

        if last_layer_params:
            param_groups.append({"params": last_layer_params, "lr": lr_last_layer})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        print(f"Optimizer created:")
        print(f"  Decoder group: {len(decoder_params)} params, LR={lr_decoder}")
        print(
            f"  Last layer group: {len(last_layer_params)} params, LR={lr_last_layer}"
        )

        return optimizer

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_gene_id(self, gene_name: str) -> Optional[int]:
        """Get gene ID from gene name."""
        return self.vocab.get(gene_name, None)

    def save_finetuned(self, save_path: str):
        """Save finetuned model weights."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved finetuned model to {save_path}")


def main():
    """Test model loading and freeze strategy."""
    print("=" * 60)
    print("ScGPT Forward Model - Loading Test")
    print("=" * 60)

    model = ScGPTForward()

    print("\nVocab size:", model.get_vocab_size())
    print("\nSample gene IDs:")
    for gene in ["SAMD11", "PLEKHN1", "LINC00115", "TNFRSF18"]:
        gene_id = model.get_gene_id(gene)
        print(f"  {gene}: {gene_id}")

    print("\nModel summary:")
    print(f"  Device: {model.device}")
    print(f"  Input style: {model.args['input_emb_style']}")
    print(f"  Num bins: {model.args.get('n_bins', 51)}")

    print("\n" + "=" * 60)
    print("✓ Model loaded successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
