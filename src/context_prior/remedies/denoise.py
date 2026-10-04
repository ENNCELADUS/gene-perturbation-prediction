"""Low-rank denoising: bridged pseudo-bulk queries are replaced by their
reconstruction from the leading bulk components; the prior is fitted on bulk as
in the reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine
from src.data.context_pca import fit_context_pca


def _denoise(frame: pd.DataFrame, pca) -> pd.DataFrame:
    """Orthogonal projection of the standardised rows on the components (their
    rows are orthonormal), back in the gene space."""
    z = (frame.to_numpy(dtype=np.float64) - pca.mean) / pca.scale
    rebuilt = (z @ pca.components.T) @ pca.components
    return pd.DataFrame(
        rebuilt * pca.scale + pca.mean, index=frame.index, columns=frame.columns
    )


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """Affine-bridged inputs with ``val`` and ``test`` projected onto the ``rank``
    leading components of the standardised training bulk, and each out-of-fold
    row onto components fitted without its fold's bulk.

    ``expression``, the ``oracle`` query and ``oof_bulk`` are the reference's.
    """
    if set(setting) != {"rank"}:
        raise ValueError(
            f"the denoising bridge takes exactly 'rank', got {dict(setting)}"
        )
    rank = int(setting["rank"])
    inputs = affine.build(base, {})
    pca = fit_context_pca(base.bulk.to_numpy(dtype=np.float64), rank)
    queries = dict(inputs.queries)
    queries["val"] = _denoise(queries["val"], pca)
    queries["test"] = _denoise(queries["test"], pca)
    paired = list(base.paired)
    parts = []
    for fold in sorted({base.folds[m] for m in paired}):
        held = [m for m in paired if base.folds[m] == fold]
        local = fit_context_pca(
            base.bulk.drop(index=held).to_numpy(dtype=np.float64), rank
        )
        parts.append(_denoise(inputs.oof_paired.loc[held], local))
    return replace(inputs, queries=queries, oof_paired=pd.concat(parts).loc[paired])
