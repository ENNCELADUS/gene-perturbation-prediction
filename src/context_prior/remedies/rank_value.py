"""Cilantro-SL's input reductions on the prior's input: Geneformer's rank-value
encoding and its top-k truncation, applied to bulk and pseudo-bulk alike before the
affine bridge.

A line's values go back to linear scale (``expm1`` of the quantile-normalised log
values), are divided by each gene's median over the training-side bulk rows where
it is nonzero (Geneformer's normalisation by corpus medians), and are ranked within
the line, highest first. ``encoding: rank_value`` gives a gene ``1 - rank / G``, or
``max(0, 1 - rank / k)`` with ``top_k``; ``encoding: expression`` (needs ``top_k``)
keeps the normalised expression of the same top ``k`` genes and zeroes the rest, so
the two encodings differ only in what the kept genes carry.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.context_prior.bridge import BridgeBase, BridgeInputs, fit_bridge
from src.context_prior.remedies import affine

ENCODINGS = ("rank_value", "expression")


def nonzero_medians(bulk: pd.DataFrame) -> pd.Series:
    """Each gene's median linear value over the rows where it is nonzero; a gene
    nonzero in no row gets infinity, so it always ranks last."""
    linear = np.expm1(bulk.to_numpy(dtype=np.float64))
    linear[linear <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN: handled below
        medians = np.nanmedian(linear, axis=0)
    return pd.Series(np.where(np.isfinite(medians), medians, np.inf), bulk.columns)


def encode(
    frame: pd.DataFrame, medians: pd.Series, encoding: str, top_k: int | None
) -> pd.DataFrame:
    """One line per row, encoded as the module describes."""
    values = frame.to_numpy(dtype=np.float64)
    relative = np.expm1(values) / medians.loc[frame.columns].to_numpy()
    rank = rankdata(-relative, axis=1, method="average") - 1.0  # 0 is the top
    genes = values.shape[1]
    if encoding == "rank_value":
        encoded = (
            1.0 - rank / genes if top_k is None else np.maximum(0.0, 1.0 - rank / top_k)
        )
    elif encoding == "expression":
        encoded = np.where(rank < top_k, values, 0.0)
    else:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    return pd.DataFrame(encoded, index=frame.index, columns=frame.columns)


def _setting(setting: Mapping[str, Any]) -> tuple[str, int | None]:
    if set(setting) != {"encoding", "top_k"}:
        raise ValueError(
            f"the rank-value bridge takes 'encoding' and 'top_k', got {dict(setting)}"
        )
    encoding, top_k = setting["encoding"], setting["top_k"]
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    if top_k is not None and not (isinstance(top_k, int) and top_k > 0):
        raise ValueError(f"top_k must be a positive integer or null, got {top_k!r}")
    if encoding == "expression" and top_k is None:
        raise ValueError("expression encoding without top_k is the affine reference")
    return encoding, top_k


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """The affine bridge on encoded sources. Medians come from the training-side
    bulk; each out-of-fold row's from the bulk outside its fold."""
    encoding, top_k = _setting(setting)

    def encoded(medians: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
        return encode(frame, medians, encoding, top_k)

    medians = nonzero_medians(base.bulk)
    inputs = affine.build(
        replace(
            base,
            bulk=encoded(medians, base.bulk),
            oracle=encoded(medians, base.oracle),
            pseudobulk=encoded(medians, base.pseudobulk),
        ),
        {},
    )
    paired = list(base.paired)
    bridged, bulk = [], []
    for fold in sorted({base.folds[m] for m in paired}):
        held = [m for m in paired if base.folds[m] == fold]
        fit = [m for m in paired if base.folds[m] != fold]
        local = nonzero_medians(base.bulk.drop(index=held))
        bridge = fit_bridge(
            encoded(local, base.pseudobulk.loc[fit]), encoded(local, base.bulk.loc[fit])
        )
        bridged.append(bridge.apply(encoded(local, base.pseudobulk.loc[held])))
        bulk.append(encoded(local, base.bulk.loc[held]))
    return replace(
        inputs,
        oof_paired=pd.concat(bridged).loc[paired],
        oof_bulk=pd.concat(bulk).loc[paired],
    )
