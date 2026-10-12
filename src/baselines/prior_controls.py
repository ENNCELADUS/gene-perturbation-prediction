"""Controls built on the linear context prior.

``context_prior`` is the prior export alone. ``context_prior+context_pca_ridge[tx1]``
adds the Tx1 context-PCA ridge of the control ladder (mean and variance of the
line's Tx1 cell embeddings, standardised on the training lines, 8 principal
components, a ridge per gene at alpha 1) fitted on what the prior leaves on the
labelled training lines: the linear special case of the head on the same target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.prepared import PreparedInputs

PRIOR = "context_prior"
PRIOR_PLUS_TX1_RIDGE = "context_prior+context_pca_ridge[tx1]"
COLUMNS = [
    "slice",
    "model_id",
    "gene_symbol",
    "method",
    "gene_effect",
    "residual",
    "residual_prediction",
]


def _tx1_view(inputs: PreparedInputs, lines: list[str]) -> np.ndarray:
    """The ladder's Tx1 view: mean and variance of the line's Tx1 cell embeddings."""
    return np.stack(
        [
            np.concatenate(
                (
                    inputs.lines[m].controls_tx1.mean(0),
                    inputs.lines[m].controls_tx1.var(0),
                )
            )
            for m in lines
        ]
    ).astype(np.float64)


def _per_gene_ridge(
    x_train: np.ndarray, targets: pd.DataFrame, x_eval: np.ndarray, alpha: float
) -> np.ndarray:
    """A ridge per gene column on the rows where that gene has a label; genes with
    the same labelled rows share one multi-output fit (identical per-gene fits)."""
    values = targets.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    out = np.empty((len(x_eval), values.shape[1]))
    patterns: dict[bytes, list[int]] = {}
    for column in range(values.shape[1]):
        patterns.setdefault(observed[:, column].tobytes(), []).append(column)
    for key, columns in patterns.items():
        rows = np.frombuffer(key, dtype=bool)
        model = Ridge(alpha=alpha).fit(x_train[rows], values[rows][:, columns])
        out[:, columns] = model.predict(x_eval).reshape(len(x_eval), len(columns))
    return out


def prior_control_rows(
    inputs: PreparedInputs,
    split: str,
    *,
    pca_components: int = 8,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Both controls' prediction rows for ``split``'s labelled lines."""
    if inputs.prior is None:
        raise ValueError("prior controls need a prior export, and no prior is set")
    train = list(inputs.split.supervised_train)
    lines = list(getattr(inputs.split, split))
    genes = list(inputs.genes)
    scale = inputs.residual_scale.loc[genes].to_numpy()
    offsets = inputs.prior.values.loc[[*train, *lines], genes] * scale
    residual = inputs.labels.pivot(
        index="model_id", columns="gene_symbol", values="residual"
    ).reindex(index=train, columns=genes)
    view = _tx1_view(inputs, [*train, *lines])
    spread = view[: len(train)].max(axis=0) - view[: len(train)].min(axis=0)
    view = view[:, spread > 1e-12]  # the ladder's constant-feature rule
    scaler = StandardScaler().fit(view[: len(train)])
    scaled = scaler.transform(view)
    components = min(pca_components, len(train) - 1, scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(scaled[: len(train)])
    scores = pca.transform(scaled)
    ridge = _per_gene_ridge(
        scores[: len(train)],
        residual - offsets.loc[train],
        scores[len(train) :],
        ridge_alpha,
    )
    stacked = offsets.loc[lines] + ridge
    rows = inputs.labels.loc[inputs.labels.model_id.isin(lines)]

    def frame(method: str, matrix: pd.DataFrame) -> pd.DataFrame:
        values = matrix.to_numpy()[
            matrix.index.get_indexer(rows.model_id),
            matrix.columns.get_indexer(rows.gene_symbol),
        ]
        return rows.assign(slice=split, method=method, residual_prediction=values)[
            COLUMNS
        ]

    return pd.concat(
        [frame(PRIOR, offsets.loc[lines]), frame(PRIOR_PLUS_TX1_RIDGE, stacked)],
        ignore_index=True,
    )
