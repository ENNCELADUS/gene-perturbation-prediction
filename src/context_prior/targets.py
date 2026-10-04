"""The metric's fixed definitions, fitted on the labelled training lines only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.data.geneeffect import (
    fit_residual_scale,
    fit_selective_genes,
    fit_variable_gene_membership,
)
from src.data.residual_target import fit_gene_means
from src.data.splits import FixedSplit, assert_fit_eligible


@dataclass(frozen=True)
class Definitions:
    """Panel genes with a training mean, their means, gene sets and residual SD."""

    genes: tuple[str, ...]
    gene_means: pd.Series
    selective: tuple[str, ...]
    variable: tuple[str, ...]
    residual_scale: pd.Series


def fit_definitions(
    gene_effect: pd.DataFrame,
    split: FixedSplit,
    genes: Sequence[str],
    settings: Mapping[str, float],
) -> Definitions:
    """Fit as ``load_inputs`` does, on ``split.supervised_train`` only.

    ``settings`` holds the joint config's ``features`` keys. Genes without a
    training mean (fewer than three labels) leave the panel.
    """
    train = split.supervised_train
    for model_id in train:
        assert_fit_eligible(model_id, split)
    candidates = [g for g in genes if g in gene_effect.columns]
    labels = (
        gene_effect.loc[list(train), candidates]
        .rename_axis(index="model_id", columns="gene_symbol")
        .reset_index()
        .melt(id_vars="model_id", var_name="gene_symbol", value_name="gene_effect")
    )
    means = fit_gene_means(labels, train)
    kept = tuple(g for g in candidates if g in means.index)
    labels = labels.loc[labels["gene_symbol"].isin(kept)].copy()
    labels["residual"] = labels["gene_effect"] - labels["gene_symbol"].map(means)
    variable = fit_variable_gene_membership(
        labels,
        train,
        kept,
        min_observations=int(settings["variable_gene_min_observations"]),
        percentile=float(settings["variable_gene_percentile"]),
    )
    selective = fit_selective_genes(
        labels,
        train,
        kept,
        min_lines=int(settings["selective_min_lines"]),
        max_fraction=float(settings["selective_max_fraction"]),
    )
    scale = fit_residual_scale(
        labels,
        train,
        kept,
        floor_percentile=float(settings["residual_sd_floor_percentile"]),
    )
    return Definitions(
        genes=kept,
        gene_means=means.loc[list(kept)],
        selective=tuple(g for g in kept if g in selective),
        variable=tuple(g for g in kept if g in variable),
        residual_scale=scale.loc[list(kept)],
    )


def residual_frame(
    gene_effect: pd.DataFrame, lines: Sequence[str], definitions: Definitions
) -> pd.DataFrame:
    """Lines x genes ``y - mu_hat``; NaN where a line has no label."""
    frame = gene_effect.reindex(index=list(lines), columns=list(definitions.genes))
    return frame.sub(definitions.gene_means, axis=1)
