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


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """Affine-bridged inputs with ``val``, ``test`` and ``oof_paired`` projected
    onto the ``rank`` leading components of the standardised training bulk.

    ``expression`` and the ``oracle`` query are the reference's. The components
    have orthonormal rows, so ``(z @ C.T) @ C`` is the orthogonal projection of
    the standardised row ``z``; the result stays in the gene space.
    """
    if set(setting) != {"rank"}:
        raise ValueError(
            f"the denoising bridge takes exactly 'rank', got {dict(setting)}"
        )
    inputs = affine.build(base, {})
    pca = fit_context_pca(base.bulk.to_numpy(dtype=np.float64), int(setting["rank"]))

    def denoise(frame: pd.DataFrame) -> pd.DataFrame:
        z = (frame.to_numpy(dtype=np.float64) - pca.mean) / pca.scale
        rebuilt = (z @ pca.components.T) @ pca.components
        return pd.DataFrame(
            rebuilt * pca.scale + pca.mean, index=frame.index, columns=frame.columns
        )

    queries = dict(inputs.queries)
    queries["val"] = denoise(queries["val"])
    queries["test"] = denoise(queries["test"])
    return replace(inputs, queries=queries, oof_paired=denoise(inputs.oof_paired))
