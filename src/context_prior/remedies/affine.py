"""The reference bridge: one affine map per gene, pseudo-bulk to bulk."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.context_prior.bridge import fit_bridge
from src.context_prior.bridging import BridgeBase, BridgeInputs, oof_bridged


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """Bulk as the prior's rows; validation and test pseudo-bulk through the
    bridge fitted on every paired line."""
    if setting:
        raise ValueError(f"the affine bridge takes no settings, got {dict(setting)}")
    paired = list(base.paired)
    bridge = fit_bridge(base.pseudobulk.loc[paired], base.bulk.loc[paired])
    return BridgeInputs(
        expression=base.bulk,
        queries={
            "val": bridge.apply(base.pseudobulk.loc[list(base.val)]),
            "test": bridge.apply(base.pseudobulk.loc[list(base.test)]),
            "oracle": base.oracle,
        },
        oof_paired=oof_bridged(base.pseudobulk, base.bulk, paired, paired, base.folds),
    )
