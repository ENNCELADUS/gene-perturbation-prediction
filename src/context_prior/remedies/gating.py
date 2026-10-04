"""Reliability gating: gene-level blocks read only genes whose out-of-fold bridge
quality reaches the threshold; others count as undefined."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.bridge import bridge_quality
from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.remedies import affine


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """The affine remedy's inputs, with the gene space limited to genes whose
    out-of-fold bridged pseudo-bulk correlates with bulk across the paired lines
    at or above ``setting["threshold"]`` (an undefined correlation never does)."""
    if set(setting) != {"threshold"}:
        raise ValueError(f"gating takes exactly a threshold, got {dict(setting)}")
    inputs = affine.build(base, {})
    quality = bridge_quality(inputs.oof_paired, base.bulk.loc[list(base.paired)])
    threshold = float(setting["threshold"])
    space = tuple(g for g in base.bulk.columns if quality[g] >= threshold)
    return replace(inputs, gene_space=space)
