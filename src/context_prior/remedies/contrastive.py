"""Contrastive-PCA remedy: project the directions the two sources do not share out
of both, then bridge as the affine reference does."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.alignment import fit_alignment
from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine

#: The setting's keys: directions of pseudo-bulk-excess and bulk-excess variance.
SETTING_KEYS = ("pseudo_components", "bulk_components")


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """The alignment fitted on the paired lines, applied to every bulk, oracle and
    pseudo-bulk row; then the affine bridge on the aligned sources, so queries and
    out-of-fold rows are in the space of the aligned bulk."""
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
    return affine.build(aligned, {})
