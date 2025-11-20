import sys
import json
import torch
import logging
from pathlib import Path

# Add scGPT to path (reusing logic from wrapper.py)
current_dir = Path(__file__).parent.parent.parent
scgpt_path = current_dir / "scGPT"

if not scgpt_path.exists():
    scgpt_path = Path.cwd() / "scGPT"

if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

try:
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
except ImportError:
    logging.warning("Could not import scgpt directly. Checking path...")
    if not scgpt_path.exists():
        raise ImportError(f"scGPT directory not found at {scgpt_path}")
    else:
        raise


class BaselineWrapper:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None
        self._load_vocab()

    def _load_vocab(self):
        # We need the vocab to ensure data loader compatibility
        model_dir = Path(self.config["paths"]["model_dir"])
        vocab_file = model_dir / "vocab.json"

        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        self.vocab = GeneVocab.from_file(vocab_file)

        # Ensure special tokens (matching ScGPTWrapper logic)
        special_tokens = [self.config["model"]["pad_token"], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.logger.info(f"Loaded vocab from {vocab_file}")

    def predict(self, batch_data, gene_ids, include_zero_gene="batch-wise", amp=True):
        """
        Baseline prediction: Identity mapping of the input control expression.
        batch_data.x has shape (batch, 2, n_genes)
        index 0 is expression, index 1 is perturbation flags
        """
        # Extract expression from batch_data
        # batch_data is expected to be the BatchData object from loader.py
        # which has .x attribute

        input_expression = batch_data.x[:, 0, :]

        # Return as is (Identity / Control Mean Baseline)
        # shape: (batch, n_genes)
        return input_expression
