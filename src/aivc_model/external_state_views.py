"""External-test-set alignment to a (possibly two-view) reference `GeneBags`.

The AIVC response encoder is always supervised in gene/HVG space. When a
reference `GeneBags` carries a distinct ST-input space (`reference.target_bags
is not None`, see `state_views.StateViews`), an external test source's cells
must be aligned to *both* the reference input space and its target space
independently — one shared alignment is no longer correct once the two spaces
differ in width or feature identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aivc_model.prepare import ExternalSourceConfig, GeneBags


def gene_symbol_matched_matrix(
    adata: ad.AnnData,
    source: ExternalSourceConfig,
    reference_names: np.ndarray,
    reference_fill_values: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """Fill-value-imputed matrix aligning `adata` to `reference_names` by gene symbol.

    Returns:
        A tuple of (matrix, matched_count, reference_feature_count,
        source_feature_count).
    """
    from aivc_model.prepare import _dense_slice, _var_symbols

    reference_name_list = reference_names.astype(str).tolist()
    source_names = _var_symbols(adata, source.var_gene_symbol_col)
    source_to_index = {name: index for index, name in enumerate(source_names)}
    fill_values = np.asarray(reference_fill_values, dtype=np.float32)
    matrix = np.tile(fill_values[None, :], (adata.n_obs, 1)).astype(np.float32)
    matched_reference_indices = []
    matched_source_indices = []
    for ref_index, name in enumerate(reference_name_list):
        source_index = source_to_index.get(str(name))
        if source_index is None:
            continue
        matched_reference_indices.append(ref_index)
        matched_source_indices.append(source_index)
    if matched_reference_indices:
        matrix[:, np.asarray(matched_reference_indices, dtype=np.int64)] = _dense_slice(
            adata.X,
            np.asarray(matched_source_indices, dtype=np.int64),
        )
    return (
        matrix,
        len(matched_reference_indices),
        len(reference_name_list),
        len(source_names),
    )


def external_target_view(
    adata: ad.AnnData,
    source: ExternalSourceConfig,
    reference: GeneBags,
) -> np.ndarray:
    """Align an external source to the reference's distinct target space."""
    target_names = reference.effective_target_feature_names
    target_fill_values = reference.effective_target_fill_values
    if target_names is None:
        raise ValueError(
            "Cannot align external target expression without reference target "
            "feature names"
        )
    if target_fill_values is None:
        raise ValueError(
            "Cannot align external target expression without reference target "
            "feature fills"
        )
    target_matrix, matched, _total, _source_total = gene_symbol_matched_matrix(
        adata, source, target_names, target_fill_values
    )
    if matched == 0:
        raise ValueError(
            f"External source {source.name!r} matched 0 reference target features"
        )
    if target_matrix.shape[1] != reference.effective_target_dim:
        msg = (
            f"External target alignment produced dim {target_matrix.shape[1]}, "
            f"expected {reference.effective_target_dim}"
        )
        raise ValueError(msg)
    return target_matrix


def external_target_context(
    reference: GeneBags,
) -> tuple[np.ndarray | None, np.ndarray] | None:
    """Return (target feature names, target fill values) for the external loader.

    Returns None when `reference` carries a single shared space — the common,
    default case — so the caller can skip target-space work entirely and let
    `GeneBags.effective_target_*` fall back to the input-space fields.
    """
    if reference.target_bags is None:
        return None
    fill_values = np.asarray(reference.effective_target_fill_values, dtype=np.float32)
    if (
        fill_values.shape != (reference.effective_target_dim,)
        or not np.isfinite(fill_values).all()
    ):
        raise ValueError("External test reference target feature fills must be finite")
    return reference.effective_target_feature_names, fill_values


def merge_external_target_bags(
    row_metadata: pd.DataFrame,
    target_bags: list[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Merge external target-space bags in the same per-gene order as
    `prepare._merge_external_gene_rows`, so input and target bags stay
    index-paired.
    """
    rows = row_metadata.reset_index(drop=True).copy()
    rows["_source_row"] = np.arange(len(rows), dtype=np.int64)
    merged: list[np.ndarray] = []
    for _, group in rows.groupby("perturbation_gene", sort=True):
        source_indices = group["_source_row"].to_numpy(dtype=np.int64)
        merged.append(
            np.vstack([target_bags[index] for index in source_indices]).astype(
                np.float32
            )
        )
    return tuple(merged)
