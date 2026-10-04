"""Contrastive-PCA remedy: project the directions the two sources do not share out
of both, then bridge as the affine reference does."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pandas as pd

from src.context_prior.alignment import fit_alignment
from src.context_prior.bridge import fit_bridge
from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine

#: The setting's keys: directions of pseudo-bulk-excess and bulk-excess variance.
SETTING_KEYS = ("pseudo_components", "bulk_components")


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """The alignment fitted on the paired lines, applied to every bulk, oracle and
    pseudo-bulk row; then the affine bridge on the aligned sources. Out-of-fold
    rows refit the alignment as well as the bridge without each fold and are
    compared with the held lines' bulk in that fold's aligned space."""
    if set(setting) != set(SETTING_KEYS):
        raise ValueError(
            f"the contrastive remedy takes exactly {list(SETTING_KEYS)}, "
            f"got {dict(setting)}"
        )
    counts = {key: setting[key] for key in SETTING_KEYS}
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer, got {value!r}")
    paired = list(base.paired)
    alignment = fit_alignment(
        base.pseudobulk.loc[paired], base.bulk.loc[paired], **counts
    )
    aligned = replace(
        base,
        bulk=alignment.apply(base.bulk, source="bulk"),
        oracle=alignment.apply(base.oracle, source="bulk"),
        pseudobulk=alignment.apply(base.pseudobulk, source="pseudo"),
    )
    inputs = affine.build(aligned, {})
    rows, targets = [], []
    for fold in sorted({base.folds[m] for m in paired}):
        fit = [m for m in paired if base.folds[m] != fold]
        held = [m for m in paired if base.folds[m] == fold]
        local = fit_alignment(base.pseudobulk.loc[fit], base.bulk.loc[fit], **counts)
        pseudo = local.apply(base.pseudobulk.loc[paired], source="pseudo")
        bulk = local.apply(base.bulk.loc[paired], source="bulk")
        bridge = fit_bridge(pseudo.loc[fit], bulk.loc[fit])
        rows.append(bridge.apply(pseudo.loc[held]))
        targets.append(bulk.loc[held])
    return replace(
        inputs,
        oof_paired=pd.concat(rows).loc[paired],
        oof_bulk=pd.concat(targets).loc[paired],
    )
