"""Build per-gene gwps response bags and a shared K562 control template.

Reads a gwps h5ad file backed (not fully in memory); uses the ``X_hvg`` obsm
embedding if present, otherwise falls back to ``X``. Subsamples control cells
to ``config.control_template_size`` and each perturbation gene to
``config.cells_per_bag``, both with a seeded RNG. Symbol keys are upper-cased.

Reuses the chunked, backed-h5ad reading approach of
``aivc_model.prepare.load_gene_bags`` without loading the full 1.99M-cell
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from sl_dl_model.config import SLDLConfig


@dataclass(frozen=True)
class GwpsBags:
    """Per-gene response bags and a shared K562 control template.

    Attributes:
        control_template: Float32 array of shape ``(T, D)`` — subsampled
            control cells in STATE HVG input space.
        bags_by_symbol: Dict mapping upper-case gene symbol to float32 array
            of shape ``(n_cells, D)`` — subsampled response cells for that
            perturbation gene.
        input_dim: Feature dimension ``D`` shared by all arrays.
    """

    control_template: np.ndarray
    bags_by_symbol: dict[str, np.ndarray]
    input_dim: int


def _embed_matrix(adata: ad.AnnData, embed_key: str | None) -> np.ndarray:
    """Return the embedding matrix from obsm or fall back to X.

    Args:
        adata: AnnData object to extract from.
        embed_key: Key in ``adata.obsm``; if None or absent, use ``adata.X``.

    Returns:
        Float32 array of shape ``(n_obs, D)``.
    """
    if embed_key and embed_key in adata.obsm:
        return np.asarray(adata.obsm[embed_key], dtype=np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def build_gwps_bags(config: SLDLConfig, rng_seed: int = 17) -> GwpsBags:
    """Read gwps h5ad and build subsampled per-gene bags + control template.

    Reads the h5ad from ``config.gwps_h5ad`` (backed=False for compatibility
    with synthetic test AnnData; large files should be pre-cached via
    ``save_bags_npz``). Uses the ``X_hvg`` obsm embedding when available.
    Control cells are identified by the ``non-targeting`` label in
    ``obs["gene"]``.

    Args:
        config: SLDLConfig with ``gwps_h5ad``, ``control_template_size``, and
            ``cells_per_bag`` fields.
        rng_seed: Integer seed for reproducible subsampling.

    Returns:
        GwpsBags with control template, per-gene bags (upper-case keys), and
        input dimension.

    Raises:
        ValueError: If ``obs["gene"]`` column is absent.
    """
    rng = np.random.default_rng(rng_seed)
    adata = ad.read_h5ad(config.gwps_h5ad)

    if "gene" not in adata.obs.columns:
        raise ValueError("h5ad obs must have a 'gene' column")

    matrix = _embed_matrix(adata, "X_hvg")
    genes = adata.obs["gene"].astype(str).to_numpy()
    control_label = "non-targeting"

    # --- control template ---
    control_rows = np.where(genes == control_label)[0]
    if len(control_rows) > config.control_template_size:
        control_rows = rng.choice(
            control_rows, size=config.control_template_size, replace=False
        )
    control_template = matrix[np.sort(control_rows)]

    # --- per-gene bags (upper-cased, excluding control) ---
    bags: dict[str, np.ndarray] = {}
    for symbol in np.unique(genes):
        if symbol == control_label:
            continue
        rows = np.where(genes == symbol)[0]
        if len(rows) == 0:
            continue
        if len(rows) > config.cells_per_bag:
            rows = rng.choice(rows, size=config.cells_per_bag, replace=False)
        bags[str(symbol).upper()] = matrix[np.sort(rows)]

    return GwpsBags(
        control_template=control_template,
        bags_by_symbol=bags,
        input_dim=int(matrix.shape[1]),
    )


def save_bags_npz(bags: GwpsBags, path: Path) -> None:
    """Cache bags to a flat NPZ using ragged offsets.

    Format:
        control_template: float32 array (T, D).
        symbols: object array of upper-case gene symbols (n_symbols,).
        flat: float32 array (total_cells, D) — all bags concatenated.
        offsets: int64 array (n_symbols + 1,) — start/end indices into flat.
        input_dim: scalar int64.

    Args:
        bags: GwpsBags to serialize.
        path: Destination path; parent directories are created as needed.
    """
    symbols = sorted(bags.bags_by_symbol)
    arrays = [bags.bags_by_symbol[s] for s in symbols]
    offsets = np.cumsum([0] + [a.shape[0] for a in arrays])
    flat = (
        np.vstack(arrays) if arrays else np.zeros((0, bags.input_dim), dtype=np.float32)
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        control_template=bags.control_template,
        symbols=np.array(symbols, dtype=object),
        flat=flat.astype(np.float32),
        offsets=offsets.astype(np.int64),
        input_dim=np.int64(bags.input_dim),
    )


def load_bags_npz(path: Path) -> GwpsBags:
    """Load bags cached by :func:`save_bags_npz`.

    Args:
        path: Path to the NPZ written by ``save_bags_npz``.

    Returns:
        GwpsBags reconstructed from the flat ragged representation.
    """
    with np.load(path, allow_pickle=True) as payload:
        control = np.asarray(payload["control_template"], dtype=np.float32)
        symbols = np.asarray(payload["symbols"], dtype=object)
        flat = np.asarray(payload["flat"], dtype=np.float32)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        input_dim = int(payload["input_dim"])

    bags = {
        str(symbols[i]): flat[offsets[i] : offsets[i + 1]] for i in range(len(symbols))
    }
    return GwpsBags(
        control_template=control,
        bags_by_symbol=bags,
        input_dim=input_dim,
    )
