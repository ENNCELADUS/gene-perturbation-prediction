"""ST-input vs. response-encoder-target feature-space container.

Arc's StateTransitionPerturbationModel (ST) and the AIVC response encoder read
two conceptually independent feature spaces from the same AnnData: ST's input
(today 2000-d HVG expression; for the Tx1 arm, a 2560-d frozen basal-cell
embedding) and the response encoder's supervision target (always gene/HVG
expression). `StateViews` is the container that keeps the two explicit instead
of letting one matrix silently serve both roles.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class StateViews:
    """Resolved ST-input and response-encoder-target feature matrices.

    ``input_matrix`` feeds the frozen state-adapter (ST); ``target_matrix``
    feeds the response encoder that supervises the observed response. When the
    two roles share one space (today's default, ``state.input_view ==
    "checkpoint_hvg"``), ``target_matrix is input_matrix`` — the identical
    object, not merely an equal copy, so legacy numerics cannot drift.
    """

    input_matrix: np.ndarray
    input_feature_names: np.ndarray | None
    target_matrix: np.ndarray
    target_feature_names: np.ndarray | None


def obsm_input_view(
    adata: ad.AnnData,
    embed_key: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ST-input view from a raw obsm embedding (e.g. frozen Tx1).

    Args:
        adata: Source AnnData.
        embed_key: The ``adata.obsm`` key holding the embedding, from
            ``config.data.state_embed_key``.

    Returns:
        The float32 embedding matrix and synthetic per-column feature names.

    Raises:
        ValueError: If ``embed_key`` is falsy, or absent from ``adata.obsm``.
    """
    if not embed_key:
        raise ValueError(
            "state.input_view='obsm' requires data.state_embed_key to name an obsm key"
        )
    if embed_key not in adata.obsm:
        raise ValueError(f"AnnData is missing obsm[{embed_key!r}]")
    matrix = np.asarray(adata.obsm[embed_key], dtype=np.float32)
    names = np.asarray(
        [f"{embed_key}_{index}" for index in range(matrix.shape[1])],
        dtype=object,
    )
    return matrix, names
