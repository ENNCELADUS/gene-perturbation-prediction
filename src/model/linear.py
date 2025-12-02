"""
Linear Regression Model for Perturbation Prediction.

A simple linear model that learns to predict perturbed expression
from control expression and perturbation flags.
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add scGPT to path (reusing logic from baseline.py)
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


class LinearPerturbationModel(nn.Module):
    """
    Simple linear regression model for perturbation prediction.

    Input: concatenated [control_expression, pert_flags] of shape (batch, 2*n_genes)
    Output: predicted perturbed expression of shape (batch, n_genes)
    """

    def __init__(self, n_genes: int):
        super().__init__()
        self.n_genes = n_genes
        # Linear layer: input is [control_expr, pert_flags], output is perturbed_expr
        self.linear = nn.Linear(2 * n_genes, n_genes)

    def forward(
        self, control_expr: torch.Tensor, pert_flags: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            control_expr: Control expression values (batch, n_genes)
            pert_flags: Perturbation flags (batch, n_genes), 1 for perturbed gene

        Returns:
            Predicted perturbed expression (batch, n_genes)
        """
        # Concatenate control expression and perturbation flags
        x = torch.cat([control_expr, pert_flags.float()], dim=1)  # (batch, 2*n_genes)
        return self.linear(x)


class LinearWrapper:
    """
    Linear Regression Model Wrapper for Perturbation Prediction.

    This model learns a linear mapping from control expression + perturbation flags
    to perturbed expression. Compatible with the evaluation pipeline in main.py.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False
        self._load_vocab()

    def _load_vocab(self):
        """Load vocabulary from scGPT model directory."""
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

    def load_model(self, model_dir: Path = None):
        """
        Load trained model from checkpoint.

        Args:
            model_dir: Directory containing model checkpoint (default: from config)
        """
        if model_dir is None:
            model_dir = Path(self.config["paths"]["linear_model_dir"])

        model_file = model_dir / "best_model.pt"
        config_file = model_dir / "model_config.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model config
        with open(config_file, "r") as f:
            model_config = json.load(f)

        n_genes = model_config["n_genes"]
        self.model = LinearPerturbationModel(n_genes)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self._is_fitted = True

        self.logger.info(f"Loaded linear model from {model_file} (n_genes={n_genes})")

    def predict(self, batch_data, gene_ids, include_zero_gene="batch-wise", amp=True):
        """
        Predict perturbed expression from batch data.

        Args:
            batch_data: BatchData object with .x attribute of shape (batch, 2, n_genes)
                        x[:, 0] is control expression, x[:, 1] is pert_flags
            gene_ids: Gene IDs (unused, kept for API compatibility)
            include_zero_gene: Unused, kept for API compatibility
            amp: Whether to use automatic mixed precision

        Returns:
            Tensor of shape (batch, n_genes) with predicted perturbed expression
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Linear model has not been loaded. "
                "Call load_model() with the model directory first."
            )

        # Extract control expression and perturbation flags from batch_data
        # batch_data.x has shape (batch, 2, n_genes)
        x = batch_data.x
        if hasattr(x, "to"):
            x = x.to(self.device)

        control_expr = x[:, 0]  # (batch, n_genes)
        pert_flags = x[:, 1]  # (batch, n_genes)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp and self.device.type == "cuda"):
                predictions = self.model(control_expr, pert_flags)

        return predictions
