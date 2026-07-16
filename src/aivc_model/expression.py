"""Deterministic finite-value handling for normalized expression matrices."""

from pathlib import Path

import numpy as np


def compute_finite_feature_means(
    matrix: object,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Compute per-feature means from finite control entries only."""
    sums = np.zeros(len(column_indices), dtype=np.float64)
    counts = np.zeros(len(column_indices), dtype=np.int64)
    for start in range(0, len(row_indices), chunk_size):
        rows = row_indices[start : start + chunk_size]
        chunk = matrix[rows, :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        values = np.asarray(chunk, dtype=np.float32)[:, column_indices]
        finite = np.isfinite(values)
        sums += np.where(finite, values, 0.0).sum(axis=0, dtype=np.float64)
        counts += finite.sum(axis=0, dtype=np.int64)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise ValueError(
            f"control cells have no finite values for features {missing[:10].tolist()}"
        )
    return (sums / counts).astype(np.float32)


def replace_nonfinite(values: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """Return a float32 copy with NaN/Inf replaced column-wise."""
    array = np.asarray(values, dtype=np.float32).copy()
    if array.ndim != 2 or fill_values.shape != (array.shape[1],):
        raise ValueError("expression and fill-value shapes are inconsistent")
    rows, columns = np.nonzero(~np.isfinite(array))
    array[rows, columns] = fill_values[columns]
    if not np.isfinite(array).all():
        raise ValueError("expression sanitization did not produce finite values")
    return array


def assert_finite_npy(path: Path, *, chunk_size: int = 4096) -> None:
    """Validate a numeric NPY without materializing the full array."""
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        if not np.isfinite(np.asarray(matrix[start:stop])).all():
            raise ValueError(
                f"{path.name} contains nonfinite values in rows {start}:{stop}"
            )
