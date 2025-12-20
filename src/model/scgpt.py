"""
scGPT-based expression encoder.

Uses pretrained scGPT model (frozen) to extract cell embeddings
via CLS token pooling from the transformer's last layer.
"""

from typing import Optional

import numpy as np

from .base import BaseEncoder


class ScGPTEncoder(BaseEncoder):
    """
    scGPT-based expression encoder.

    Uses pretrained scGPT model (frozen) to extract cell embeddings
    via CLS token pooling from the transformer's last layer.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        gene_col: str = "gene_name",
        max_length: int = 1200,
        batch_size: int = 64,
        device: str = "cuda",
        use_fast_transformer: bool = True,
        **unused_kwargs,
    ):
        """
        Initialize scGPT encoder.

        Args:
            model_dir: Path to pretrained model directory containing
                       best_model.pt, vocab.json, and args.json
            gene_col: Column name in adata.var with gene names
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            device: Device to use ('cuda' or 'cpu')
            use_fast_transformer: Whether to use flash attention
        """
        resolved_model_dir = model_dir or checkpoint
        self.model_dir = resolved_model_dir or "model/scGPT"
        self.gene_col = gene_col
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.use_fast_transformer = use_fast_transformer

        # Lazy-loaded components
        self._model = None
        self._vocab = None
        self._model_configs = None
        self._loaded = False

    def _load_model(self):
        """Load scGPT model, vocab, and config."""
        if self._loaded:
            return

        import json
        import torch
        from pathlib import Path

        # Import scGPT components
        import sys

        scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
        if str(scgpt_path) not in sys.path:
            sys.path.insert(0, str(scgpt_path))

        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        model_dir = Path(self.model_dir)
        vocab_file = model_dir / "vocab.json"
        config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"

        # Load vocabulary
        self._vocab = GeneVocab.from_file(vocab_file)
        special_tokens = ["<pad>", "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self._vocab:
                self._vocab.append_token(s)
        self._vocab.set_default_index(self._vocab["<pad>"])

        # Load model config
        with open(config_file, "r") as f:
            self._model_configs = json.load(f)

        # Set required config keys
        self._model_configs.setdefault("pad_token", "<pad>")
        self._model_configs.setdefault("pad_value", -2)

        # Determine device
        if self.device == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)

        # Build model
        self._model = TransformerModel(
            ntoken=len(self._vocab),
            d_model=self._model_configs["embsize"],
            nhead=self._model_configs["nheads"],
            d_hid=self._model_configs["d_hid"],
            nlayers=self._model_configs["nlayers"],
            nlayers_cls=self._model_configs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self._vocab,
            dropout=self._model_configs.get("dropout", 0.2),
            pad_token=self._model_configs["pad_token"],
            pad_value=self._model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )

        # Load pretrained weights
        load_pretrained(
            self._model, torch.load(model_file, map_location=device), verbose=False
        )
        self._model.to(device)
        self._model.eval()

        self._loaded = True

    def fit(self, X: np.ndarray) -> "ScGPTEncoder":
        """Fit is no-op for pretrained model."""
        # Load model on first fit call
        self._load_model()
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode expression using scGPT.

        Note: For proper scGPT encoding, use encode_adata() with an AnnData
        object that includes gene name mapping.
        """
        raise NotImplementedError(
            "ScGPTEncoder.encode() requires AnnData with gene names. "
            "Use encode_adata() instead."
        )

    def encode_adata(self, adata) -> np.ndarray:
        """
        Encode AnnData object to cell embeddings.

        Args:
            adata: AnnData object with gene names in var

        Returns:
            Cell embeddings (n_cells, embsize), normalized
        """
        import sys
        from pathlib import Path

        # Ensure model is loaded
        self._load_model()

        # Import scGPT cell embedding function
        scgpt_path = Path(__file__).parent.parent.parent / "scGPT"
        if str(scgpt_path) not in sys.path:
            sys.path.insert(0, str(scgpt_path))

        from scgpt.tasks.cell_emb import get_batch_cell_embeddings

        # Prepare gene IDs
        adata = adata.copy()
        gene_names = (
            adata.var[self.gene_col].values
            if self.gene_col in adata.var
            else adata.var.index.values
        )
        adata.var["id_in_vocab"] = [
            self._vocab[g] if g in self._vocab else -1 for g in gene_names
        ]

        # Filter to genes in vocabulary
        valid_genes = adata.var["id_in_vocab"] >= 0
        n_valid = valid_genes.sum()
        n_total = len(valid_genes)
        if n_valid < n_total:
            print(f"  [scGPT] Using {n_valid}/{n_total} genes in vocabulary")
        adata = adata[:, valid_genes]

        gene_ids = np.array(adata.var["id_in_vocab"])

        # Get cell embeddings
        cell_embeddings = get_batch_cell_embeddings(
            adata,
            cell_embedding_mode="cls",
            model=self._model,
            vocab=self._vocab,
            max_length=self.max_length,
            batch_size=self.batch_size,
            model_configs=self._model_configs,
            gene_ids=gene_ids,
            use_batch_labels=False,
        )

        return cell_embeddings

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        self._load_model()
        return self._model_configs["embsize"]

    def get_model(self):
        """
        Get underlying TransformerModel for fine-tuning.

        Returns:
            scGPT TransformerModel
        """
        self._load_model()
        return self._model

    def get_vocab(self):
        """
        Get vocabulary for token encoding.

        Returns:
            GeneVocab instance
        """
        self._load_model()
        return self._vocab

    def get_model_configs(self) -> dict:
        """
        Get model configuration dictionary.

        Returns:
            Model config dict with embsize, nheads, etc.
        """
        self._load_model()
        return self._model_configs
