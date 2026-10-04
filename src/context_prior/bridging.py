"""Shared inputs of the bridge remedies: the quantile-normalised sources, out-of-
fold bridging, and diagnostics of how well bridged pseudo-bulk tracks bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.context_prior.bridge import bridge_quality, fit_bridge

#: Per-gene bridge correlations counted at or above each threshold.
THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class BridgeBase:
    """Quantile-normalised sources every remedy starts from, keyed by ModelID.

    Attributes:
        bulk: Training-side bulk rows.
        oracle: Validation lines' bulk rows (off-contract; scored, never fitted).
        pseudobulk: Pseudo-bulk of the 226 lines.
        paired: Labelled single-cell training lines with bulk RNA: the bridge's
            fit lines.
        single_cell_train: Every labelled single-cell training line, with or
            without bulk RNA.
        val: Validation lines.
        test: Test lines.
        folds: Patient-grouped fold of each line of ``single_cell_train``.
    """

    bulk: pd.DataFrame
    oracle: pd.DataFrame
    pseudobulk: pd.DataFrame
    paired: tuple[str, ...]
    single_cell_train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    folds: Mapping[str, int]


@dataclass(frozen=True)
class BridgeInputs:
    """What a remedy hands the prior.

    Attributes:
        expression: Rows the prior is fitted on (training-side bulk, possibly
            transformed).
        queries: ``val`` and ``test`` (bridged pseudo-bulk) and ``oracle``
            (validation bulk), in the space of ``expression``.
        oof_paired: Out-of-fold bridged rows of the paired lines, for diagnostics.
        gene_space: Expression genes the gene-level blocks may read; None for all.
        gene_rows: Rows the gene-level blocks are fitted on instead of
            ``expression``; None for ``expression``'s fit rows.
    """

    expression: pd.DataFrame
    queries: Mapping[str, pd.DataFrame]
    oof_paired: pd.DataFrame
    gene_space: tuple[str, ...] | None = None
    gene_rows: pd.DataFrame | None = None


def oof_bridged(
    pseudobulk: pd.DataFrame,
    bulk: pd.DataFrame,
    paired: Sequence[str],
    lines: Sequence[str],
    folds: Mapping[str, int],
) -> pd.DataFrame:
    """Bridged pseudo-bulk of ``lines``, each from a bridge fitted on the paired
    lines outside its fold; rows in ``lines`` order."""
    parts = []
    for fold in sorted({folds[m] for m in lines}):
        fit = [m for m in paired if folds[m] != fold]
        held = [m for m in lines if folds[m] == fold]
        bridge = fit_bridge(pseudobulk.loc[fit], bulk.loc[fit])
        parts.append(bridge.apply(pseudobulk.loc[held]))
    return pd.concat(parts).loc[list(lines)]


def bridge_diagnostics(
    oof_paired: pd.DataFrame,
    bulk_paired: pd.DataFrame,
    selective: Sequence[str],
    paralogs: pd.DataFrame,
) -> dict:
    """Per-gene Pearson across lines between out-of-fold bridged pseudo-bulk and
    bulk: quartiles over the defined genes, and counts at or above each threshold
    for every gene, the selective genes and the selective genes' paralogs present
    in the space (an undefined correlation counts below every threshold)."""
    quality = bridge_quality(
        oof_paired, bulk_paired.loc[oof_paired.index, oof_paired.columns]
    )
    present = set(quality.index)
    selective_present = [g for g in selective if g in present]
    partner = paralogs.loc[paralogs["gene"].isin(set(selective)), "paralog"]
    paralogs_present = sorted(set(partner) & present)

    def counts(genes: Sequence[str]) -> dict:
        values = quality.loc[list(genes)]
        return {
            "total": len(genes),
            **{str(t): int((values >= t).sum()) for t in THRESHOLDS},
        }

    defined = quality.dropna()
    return {
        "median": float(defined.median()),
        "q25": float(defined.quantile(0.25)),
        "q75": float(defined.quantile(0.75)),
        "all": counts(list(quality.index)),
        "selective": counts(selective_present),
        "selective_paralogs": counts(paralogs_present),
    }
