"""Model components for reverse perturbation prediction."""

from .base import BaseEncoder
from .pca import PCAEncoder
from .scgpt import ScGPTEncoder
from .logreg import LogRegClassifier


def get_encoder(encoder_type: str, **kwargs) -> BaseEncoder:
    """Factory function to get encoder by type."""
    encoders = {
        "pca": PCAEncoder,
        "scgpt": ScGPTEncoder,
        "logreg": LogRegClassifier,
    }
    if encoder_type not in encoders:
        raise ValueError(
            f"Unknown encoder: {encoder_type}. Available: {list(encoders.keys())}"
        )
    return encoders[encoder_type](**kwargs)


__all__ = [
    "BaseEncoder",
    "PCAEncoder",
    "ScGPTEncoder",
    "LogRegClassifier",
    "get_encoder",
]
