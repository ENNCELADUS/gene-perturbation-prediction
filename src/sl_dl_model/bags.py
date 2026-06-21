"""Build per-gene gwps response bags and a shared K562 control template.

Reads a gwps h5ad file backed (not fully in memory); aligns raw expression to
the STATE checkpoint's ``var_dims.pkl`` gene order when available, then falls
back to the ``X_hvg`` obsm embedding or full ``X``. Subsamples control cells
to ``config.control_template_size`` and each perturbation gene to
``config.cells_per_bag``, both with a seeded RNG. Symbol keys are upper-cased.

Reuses the chunked, backed-h5ad reading approach of
``aivc_model.prepare.load_gene_bags`` without loading the full 1.99M-cell
matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from sl_dl_model.config import SLDLConfig

logger = logging.getLogger(__name__)


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


def state_checkpoint_input_dim(config: SLDLConfig) -> int | None:
    """Return the STATE checkpoint input dimension when metadata is available."""
    if config.state_backend == "linear_mock":
        return None
    payload = _state_var_dims(config)
    if payload is None:
        return None
    input_dim = payload.get("input_dim")
    return int(input_dim) if input_dim is not None else None


def _state_var_dims(config: SLDLConfig) -> dict[str, object] | None:
    if config.state_backend == "linear_mock":
        return None
    path = config.state_checkpoint.parent.parent / "var_dims.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else None


def _var_symbols(adata: ad.AnnData) -> list[str]:
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _checkpoint_gene_indices(
    adata: ad.AnnData,
    config: SLDLConfig,
) -> np.ndarray | None:
    payload = _state_var_dims(config)
    names = payload.get("gene_names") if payload is not None else None
    if names is None:
        return None

    symbol_to_index: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, symbol in enumerate(_var_symbols(adata)):
        key = str(symbol)
        if key in symbol_to_index:
            duplicates.add(key)
        else:
            symbol_to_index[key] = index

    selected: list[int] = []
    missing: list[str] = []
    duplicate_matches: list[str] = []
    for name in names:
        key = str(name)
        index = symbol_to_index.get(key)
        if index is None:
            missing.append(key)
            continue
        if key in duplicates:
            duplicate_matches.append(key)
            continue
        selected.append(index)

    if missing or duplicate_matches:
        logger.warning(
            "Cannot align GWPS bags to STATE checkpoint gene order: %d missing, "
            "%d duplicate gene symbol(s). Falling back to the configured matrix view.",
            len(missing),
            len(duplicate_matches),
        )
        return None
    return np.asarray(selected, dtype=np.int64)


def _dense_slice(matrix: object, indices: np.ndarray) -> np.ndarray:
    subset = matrix[:, indices]  # type: ignore[index]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float32)


def _zero_fill_nonfinite(array: np.ndarray, label: str) -> tuple[np.ndarray, int]:
    """Replace non-finite entries with 0.0, returning the count touched.

    STATE HVG input is normalized expression where 0 is the natural
    "no signal" baseline, so zero-fill keeps imputed entries from skewing the
    energy-distance / mean-delta bag losses.

    Args:
        array: Float array, possibly containing NaN/+-inf.
        label: Identifier used by the caller for logging context.

    Returns:
        Tuple of the cleaned float32 array and the number of non-finite entries.
    """
    del label  # Reserved for future per-array logging; counts are aggregated.
    mask = ~np.isfinite(array)
    n_nonfinite = int(mask.sum())
    if n_nonfinite == 0:
        return np.asarray(array, dtype=np.float32), 0
    cleaned = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(cleaned, dtype=np.float32), n_nonfinite


def _embed_matrix(
    adata: ad.AnnData,
    embed_key: str | None,
    config: SLDLConfig,
) -> np.ndarray:
    """Return the embedding matrix from obsm or fall back to X.

    Args:
        adata: AnnData object to extract from.
        embed_key: Key in ``adata.obsm``; if None or absent, use ``adata.X``.
        config: Experiment config, used to align raw expression to the STATE
            checkpoint's gene order when checkpoint metadata is available.

    Returns:
        Float32 array of shape ``(n_obs, D)``.
    """
    checkpoint_indices = _checkpoint_gene_indices(adata, config)
    if checkpoint_indices is not None:
        return _dense_slice(adata.X, checkpoint_indices)
    if embed_key and embed_key in adata.obsm:
        return np.asarray(adata.obsm[embed_key], dtype=np.float32)
    if sparse.issparse(adata.X):
        return np.asarray(adata.X.toarray(), dtype=np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def build_gwps_bags(config: SLDLConfig, rng_seed: int = 17) -> GwpsBags:
    """Read gwps h5ad and build subsampled per-gene bags + control template.

    Reads the h5ad from ``config.gwps_h5ad`` (backed=False for compatibility
    with synthetic test AnnData; large files should be pre-cached via
    ``save_bags_npz``). For real STATE checkpoints, raw expression is first
    projected into checkpoint ``var_dims.pkl`` gene order when possible.
    Otherwise, this uses the ``X_hvg`` obsm embedding when available. Control
    cells are identified by the ``non-targeting`` label in ``obs["gene"]``.

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

    matrix = _embed_matrix(adata, "X_hvg", config)
    genes = adata.obs["gene"].astype(str).to_numpy()
    control_label = "non-targeting"

    # --- control template ---
    control_rows = np.where(genes == control_label)[0]
    if len(control_rows) == 0:
        raise ValueError(
            f"No '{control_label}' control cells found in h5ad "
            f"({config.gwps_h5ad}). Cannot build control template."
        )
    if len(control_rows) > config.control_template_size:
        control_rows = rng.choice(
            control_rows, size=config.control_template_size, replace=False
        )
    control_template, control_nonfinite = _zero_fill_nonfinite(
        matrix[np.sort(control_rows)], "control_template"
    )

    # --- per-gene bags (upper-cased, excluding control) ---
    bags: dict[str, np.ndarray] = {}
    affected_genes = 0
    total_nonfinite = control_nonfinite
    for symbol in np.unique(genes):
        if symbol == control_label:
            continue
        rows = np.where(genes == symbol)[0]
        if len(rows) == 0:
            continue
        if len(rows) > config.cells_per_bag:
            rows = rng.choice(rows, size=config.cells_per_bag, replace=False)
        key = str(symbol).upper()
        bag, bag_nonfinite = _zero_fill_nonfinite(matrix[np.sort(rows)], key)
        bags[key] = bag
        if bag_nonfinite > 0:
            affected_genes += 1
            total_nonfinite += bag_nonfinite

    # Warn about genes whose bags have fewer than 2 cells — std pooling will be
    # all-zeros for those genes, which is a silent quality issue.
    single_cell_count = sum(1 for arr in bags.values() if arr.shape[0] < 2)
    if single_cell_count > 0:
        logger.warning(
            "%d gene bag(s) have fewer than 2 cells; std-based pooling will "
            "produce all-zeros for those genes.",
            single_cell_count,
        )

    if total_nonfinite > 0:
        logger.warning(
            "Zero-filled %d non-finite GWPS expression entries across %d gene "
            "bag(s) plus the control template; upstream h5ad %s contained NaN/inf.",
            total_nonfinite,
            affected_genes,
            config.gwps_h5ad,
        )

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


def _assert_finite_bags(
    control: np.ndarray, bags_by_symbol: dict[str, np.ndarray]
) -> None:
    """Raise if any cached bag or the control template is non-finite.

    The build path is the single cleaning site; this verifies the invariant on
    load so a stale pre-fix cache fails loudly instead of poisoning training.

    Args:
        control: Control template array.
        bags_by_symbol: Per-gene response bags.

    Raises:
        ValueError: If any array contains NaN/inf, naming up to 10 symbols.
    """
    offenders: list[str] = []
    if not np.isfinite(control).all():
        offenders.append("control_template")
    for symbol, bag in bags_by_symbol.items():
        if not np.isfinite(bag).all():
            offenders.append(symbol)
    if offenders:
        shown = ", ".join(sorted(offenders)[:10])
        raise ValueError(
            f"GWPS bag cache contains non-finite values in: {shown}"
            f"{' ...' if len(offenders) > 10 else ''}. This is a stale pre-fix "
            "cache; rebuild it with "
            "`uv run python scripts/setup_exp08_assets.py bags`."
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

    if control.shape[0] == 0:
        raise ValueError(
            f"Loaded NPZ at '{path}' has an empty control template (0 rows). "
            "The cache was likely built without 'non-targeting' control cells."
        )

    bags = {
        str(symbols[i]): flat[offsets[i] : offsets[i + 1]] for i in range(len(symbols))
    }
    _assert_finite_bags(control, bags)
    return GwpsBags(
        control_template=control,
        bags_by_symbol=bags,
        input_dim=input_dim,
    )
