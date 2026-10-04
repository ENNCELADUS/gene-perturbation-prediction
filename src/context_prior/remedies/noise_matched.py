"""Noise-matched fitting: gene-level blocks are fitted on the single-cell training
lines' out-of-fold bridged pseudo-bulk, the inputs they meet at query time, so
their weights learn the bridge's noise."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.context_prior.bridging import BridgeBase, BridgeInputs, oof_bridged
from src.context_prior.remedies import affine


def build(base: BridgeBase, setting: Mapping[str, Any]) -> BridgeInputs:
    """The affine remedy's inputs, with every single-cell training line's out-of-
    fold bridged pseudo-bulk (with or without bulk RNA) as the gene-level rows."""
    if setting:
        raise ValueError(
            f"noise-matched fitting takes no settings, got {dict(setting)}"
        )
    inputs = affine.build(base, {})
    rows = oof_bridged(
        base.pseudobulk, base.bulk, base.paired, base.single_cell_train, base.folds
    )
    return replace(inputs, gene_rows=rows)
