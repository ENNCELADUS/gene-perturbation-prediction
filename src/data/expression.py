"""STATE's expression space: whole-library normalize_total, then log1p.

This reproduces arc-state's ``preprocess_train``.
"""

import numpy as np
from scipy import sparse


def library_sizes(matrix) -> np.ndarray:
    """UMI total per cell over every column of the source matrix."""
    return np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()


def log_normalize(matrix, library_size: np.ndarray, target_sum: float) -> np.ndarray:
    """``log1p(x * target_sum / library_size)``; cells with no UMI stay zero."""
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    scale = np.zeros_like(library_size, dtype=np.float64)
    nonzero = library_size > 0
    scale[nonzero] = target_sum / library_size[nonzero]
    return np.log1p(dense * scale[:, None]).astype(np.float32)


def median_library_size(*sizes: np.ndarray) -> float:
    """Median library size over the pooled cells of every given source."""
    return float(np.median(np.concatenate(sizes)))
