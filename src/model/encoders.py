"""
Expression encoders for perturbation prediction.

Supports:
- PCA: Simple baseline encoder
- scGPT: Foundation model encoder (placeholder for Stage 2)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA


class BaseEncoder(ABC):
    """Abstract base class for expression encoders."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseEncoder":
        """Fit encoder on training data."""
        pass

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode expression matrix to embeddings."""
        pass


class PCAEncoder(BaseEncoder):
    """PCA-based expression encoder (baseline)."""

    def __init__(self, n_components: int = 50):
        """
        Initialize PCA encoder.

        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        """
        Fit PCA on expression data.

        Args:
            X: Expression matrix (n_samples, n_genes)

        Returns:
            self
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.pca.fit(X)
        self._fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode expression to PCA space.

        Args:
            X: Expression matrix (n_samples, n_genes)

        Returns:
            Embeddings (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before encode()")
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.pca.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio per component."""
        return self.pca.explained_variance_ratio_


class ScGPTEncoder(BaseEncoder):
    """
    scGPT-based expression encoder.

    Placeholder for Stage 2 implementation.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        freeze: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
    ):
        """
        Initialize scGPT encoder.

        Args:
            checkpoint: Path to pretrained weights
            freeze: Whether to freeze encoder weights
            use_lora: Whether to use LoRA for fine-tuning
            lora_rank: LoRA rank
        """
        self.checkpoint = checkpoint
        self.freeze = freeze
        self.use_lora = use_lora
        self.lora_rank = lora_rank

        # TODO: Load scGPT model in Stage 2
        self.model = None

    def fit(self, X: np.ndarray) -> "ScGPTEncoder":
        """Fit is no-op for pretrained model."""
        # TODO: Implement optional fine-tuning
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode expression using scGPT."""
        raise NotImplementedError("scGPT encoder will be implemented in Stage 2")


def get_encoder(
    encoder_type: str,
    **kwargs,
) -> BaseEncoder:
    """
    Factory function to get encoder by type.

    Args:
        encoder_type: 'pca' or 'scgpt'
        **kwargs: Encoder-specific arguments

    Returns:
        Encoder instance
    """
    encoders = {
        "pca": PCAEncoder,
        "scgpt": ScGPTEncoder,
    }

    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder: {encoder_type}. Available: {list(encoders.keys())}")

    return encoders[encoder_type](**kwargs)
