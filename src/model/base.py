"""
Base encoder class for perturbation prediction.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


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
