import sys
import json
import torch
import logging
import numpy as np
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
    """
    Average Baseline Model.

    This baseline pre-computes the mean expression across all perturbed samples
    in the training dataset. At prediction time, it returns this pre-computed
    mean for all samples, ignoring the actual input.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None
        self.mean_expression = None  # Pre-computed mean from training data
        self._is_fitted = False
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

    def fit(self, train_loader):
        """
        Pre-compute mean expression from the training dataset.

        Args:
            train_loader: DataLoader yielding (batch_data, target) tuples.
                          target has shape (batch, n_genes) representing
                          perturbed expression values.
        """
        self.logger.info(
            "Fitting baseline: computing mean expression from training data..."
        )

        all_targets = []
        for batch_data, target in train_loader:
            # target contains the perturbed expression values
            if isinstance(target, torch.Tensor):
                all_targets.append(target.cpu().numpy())
            else:
                all_targets.append(np.array(target))

        # Concatenate all targets and compute mean
        all_targets = np.concatenate(all_targets, axis=0)
        self.mean_expression = np.mean(all_targets, axis=0)

        self._is_fitted = True
        self.logger.info(
            f"Baseline fitted: mean expression computed from {all_targets.shape[0]} samples, "
            f"{self.mean_expression.shape[0]} genes"
        )

    def fit_from_adata(self, adata, condition_key="condition", control_key="ctrl"):
        """
        Pre-compute mean expression from an AnnData object (perturbed samples only).

        Args:
            adata: AnnData object with expression in .X
            condition_key: Column in .obs indicating perturbation condition
            control_key: Value in condition_key that indicates control samples
        """
        self.logger.info(
            "Fitting baseline from AnnData: computing mean perturbed expression..."
        )

        # Select only perturbed (non-control) samples
        perturbed_mask = adata.obs[condition_key] != control_key
        perturbed_adata = adata[perturbed_mask]

        # Get expression matrix
        if hasattr(perturbed_adata.X, "toarray"):
            expr_matrix = perturbed_adata.X.toarray()
        else:
            expr_matrix = np.array(perturbed_adata.X)

        # Compute mean across all perturbed samples
        self.mean_expression = np.mean(expr_matrix, axis=0)

        self._is_fitted = True
        self.logger.info(
            f"Baseline fitted: mean expression computed from {expr_matrix.shape[0]} perturbed samples, "
            f"{self.mean_expression.shape[0]} genes"
        )

    def predict(self, batch_data, gene_ids, include_zero_gene="batch-wise", amp=True):
        """
        Baseline prediction: Return pre-computed mean expression from training data.

        Args:
            batch_data: BatchData object with .x attribute of shape (batch, 2, n_genes)
            gene_ids: Gene IDs for the prediction
            include_zero_gene: Unused, kept for API compatibility
            amp: Unused, kept for API compatibility

        Returns:
            Tensor of shape (batch, n_genes) with pre-computed mean expression
            broadcast to batch size.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Baseline model has not been fitted. "
                "Call fit() or fit_from_adata() with training data first."
            )

        batch_size = batch_data.x.shape[0]
        n_genes = batch_data.x.shape[2]

        # Broadcast pre-computed mean to batch size
        # mean_expression shape: (n_genes,) -> (batch, n_genes)
        if isinstance(self.mean_expression, np.ndarray):
            mean_tensor = torch.from_numpy(self.mean_expression).float()
        else:
            mean_tensor = self.mean_expression.float()

        # Handle case where stored mean has different gene count than input
        if mean_tensor.shape[0] != n_genes:
            self.logger.warning(
                f"Gene count mismatch: mean has {mean_tensor.shape[0]} genes, "
                f"input has {n_genes} genes. Using input gene count."
            )
            # Truncate or pad as needed
            if mean_tensor.shape[0] > n_genes:
                mean_tensor = mean_tensor[:n_genes]
            else:
                padding = torch.zeros(n_genes - mean_tensor.shape[0])
                mean_tensor = torch.cat([mean_tensor, padding])

        # Broadcast to batch size
        predictions = mean_tensor.unsqueeze(0).expand(batch_size, -1)

        # Move to same device as input
        if batch_data.x.is_cuda:
            predictions = predictions.cuda()

        return predictions
