"""The fitted linear context prior every prior command shares: the run's inputs
(``RunBase``), the bridged inputs a setting builds, the prior's stages and fits, and
the reference row's predictions."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.context_prior.bridging import BridgeBase, BridgeInputs
from src.context_prior.folds import patient_folds
from src.context_prior.prior import PriorInputs, PriorSpec, Stage, fit_prior, total
from src.context_prior.reference import Reference, load_reference
from src.context_prior.space import (
    fill_unmeasured,
    measured_genes,
    quantile_normalize,
    quantile_reference,
)
from src.context_prior.targets import Definitions, fit_definitions, residual_frame
from src.context_prior.views import fit_expression_components
from src.data.depmap import read_depmap_matrix, read_models
from src.data.extra_lines import load_bulk_lines
from src.data.splits import FixedSplit, load_geneeffect_split
from src.eval.accuracy import score
from src.experiments.joint.config import load_config

SCORED = ("val", "test")


# ----------------------------------------------------------------------------
# Run data
# ----------------------------------------------------------------------------


def scaled_residual(
    gene_effect: pd.DataFrame, lines: Sequence[str], definitions: Definitions
) -> pd.DataFrame:
    """Lines x genes ``(y - mu_hat) / sigma``; NaN where a label is missing.

    Raises when a line has no GeneEffect row at all: it would otherwise read as an
    all-missing line and count as zero residual in every fit.
    """
    absent = [m for m in lines if m not in gene_effect.index]
    if absent:
        raise ValueError(f"no GeneEffect row for {len(absent)} lines: {absent[:10]}")
    frame = residual_frame(gene_effect, lines, definitions)
    return frame.div(definitions.residual_scale, axis=1)


@dataclass(frozen=True)
class RunBase:
    """Everything the experiments share, keyed by ModelID.

    Attributes:
        config: The prior config.
        split: The split: single-cell training, validation and test lines.
        gene_effect: GeneEffect of the split's lines and the kept labelled
            bulk-only lines, over the definitions' genes.
        definitions: Gene order, means, selective and variable genes and residual
            SD, fitted on the labelled single-cell training lines.
        labelled: Labelled training-side lines with bulk RNA: the prior's fit lines.
        models: Every line's ``patient_id`` and ``lineage``.
        reference: Pinned reference tables.
        bridge: The quantile-normalised sources and folds the remedies start from.
        filled_lines: Training lines whose pseudo-bulk took the training mean for
            some gene of the space.
        profile: The training side's mean sorted bulk profile, which bulk and
            pseudo-bulk are quantile-normalised to.
        pseudobulk_fill: Each space gene's pseudo-bulk mean over the training
            lines measuring it: what an unmeasured value takes.
    """

    config: Mapping[str, Any]
    split: FixedSplit
    gene_effect: pd.DataFrame
    definitions: Definitions
    labelled: tuple[str, ...]
    models: pd.DataFrame
    reference: Reference
    bridge: BridgeBase
    filled_lines: int
    profile: np.ndarray
    pseudobulk_fill: pd.Series

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        """Residual of ``lines`` in residual-SD units; reads their labels."""
        return scaled_residual(self.gene_effect, lines, self.definitions)

    def normalize_pseudobulk(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        """Any pseudo-bulk rows (prepared transform, any genes) in the prior's
        space, as ``bridge.pseudobulk`` holds the prepared rows: the space's genes,
        an unmeasured value given the training mean, quantile-normalised to the
        training side's profile."""
        space = list(self.pseudobulk_fill.index)
        return quantile_normalize(
            pseudobulk.loc[:, space].fillna(self.pseudobulk_fill), self.profile
        )


def load_base(config: Mapping[str, Any]) -> RunBase:
    """Read and normalise everything once.

    Test lines' bulk RNA and the excluded lines are dropped as soon as the bulk
    matrix is read; validation lines' bulk RNA feeds the oracle query alone. The
    space is the bulk genes every validation and test line measures in
    pseudo-bulk; a training line whose source lacks one takes the training mean.
    Bulk and pseudo-bulk are quantile-normalised to the training side's profile.
    The single-cell training lines are the split's labelled training lines. A
    line of a dropped lineage leaves the training side, single-cell or bulk-only:
    the prior neither fits nor encodes it and the bridge does not pair it, though
    a single-cell training line still takes a fold and an export row.
    """
    from src.data.prepared import read_manifest
    from src.data.pseudobulk import read_pseudobulk
    from src.experiments.prepare import prepare_pseudobulk

    joint = load_config(Path(config["joint_config"]))
    paths = config["paths"]
    split = load_geneeffect_split(Path(joint["paths"]["split"]))
    extra = load_bulk_lines(Path(joint["paths"]["split"]), split)
    models = read_models(Path(paths["model"]))
    dropped = set(config["training_side"]["exclude_lineages"])

    def kept(ids: Sequence[str]) -> list[str]:
        return [m for m in ids if models.at[m, "lineage"] not in dropped]

    bulk = read_depmap_matrix(Path(paths["bulk_expression"]))
    bulk = bulk.drop(index=bulk.index.intersection([*split.test, *extra.excluded]))
    side = [
        m
        for m in kept([*split.train, *extra.labelled, *extra.unlabelled])
        if m in bulk.index
    ]
    oracle_lines = [m for m in split.val if m in bulk.index]
    on_side = set(side)
    labelled = tuple(
        m for m in (*split.supervised_train, *extra.labelled) if m in on_side
    )
    paired = tuple(m for m in split.supervised_train if m in on_side)

    gene_effect = read_depmap_matrix(Path(joint["paths"]["gene_effect"]))
    needed = {*split.all_model_ids, *kept(extra.labelled)}
    gene_effect = gene_effect.loc[gene_effect.index.isin(needed)]

    prepared = Path(joint["prepared_root"])
    panel = list(read_manifest(prepared)["common_gene_panel"])
    prepare_pseudobulk(joint, list(bulk.columns))
    pseudo = read_pseudobulk(prepared).loc[list(split.all_model_ids)]
    space = measured_genes(pseudo, list(bulk.columns), [*split.val, *split.test])
    pseudo = pseudo.loc[:, space]
    fill = pseudo.loc[list(split.train)].mean(axis=0, skipna=True)
    pseudo, filled = fill_unmeasured(pseudo, split.train)
    definitions = fit_definitions(gene_effect, split, panel, joint["features"])
    gene_effect = gene_effect.loc[:, list(definitions.genes)]

    # Training-side rows first, then the oracle's validation rows; nothing else.
    bulk = bulk.loc[[*side, *oracle_lines], space]
    profile = quantile_reference(bulk.iloc[: len(side)])
    normalized = quantile_normalize(bulk, profile)
    del bulk
    single_cell_train = tuple(split.supervised_train)
    bridge = BridgeBase(
        bulk=normalized.loc[side],
        oracle=normalized.loc[oracle_lines],
        pseudobulk=quantile_normalize(pseudo, profile),
        paired=paired,
        single_cell_train=single_cell_train,
        val=tuple(split.val),
        test=tuple(split.test),
        folds=patient_folds(
            list(single_cell_train),
            models["patient_id"].to_dict(),
            n_folds=int(config["prior"]["folds"]),
            seed=int(config["seed"]),
        ),
    )
    return RunBase(
        config=config,
        split=split,
        gene_effect=gene_effect,
        definitions=definitions,
        labelled=labelled,
        models=models,
        reference=load_reference(
            Path(paths["reference"]),
            blocked={*split.val, *split.test, *extra.excluded},
        ),
        bridge=bridge,
        filled_lines=int((filled > 0).sum()),
        profile=profile,
        pseudobulk_fill=fill,
    )


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------


def prior_inputs(base: RunBase, inputs: BridgeInputs) -> PriorInputs:
    """The prior's inputs from a remedy's rows; labels of every labelled
    training-side line and every single-cell training line (for gene rows)."""
    lines = list(dict.fromkeys([*base.labelled, *base.bridge.single_cell_train]))
    return PriorInputs(
        expression=inputs.expression,
        residual=base.truth(lines),
        components=fit_expression_components(
            inputs.expression, int(base.config["prior"]["components"])
        ),
        reference=base.reference,
        lineage=base.models.loc[list(inputs.expression.index), "lineage"],
        patients=base.models["patient_id"].to_dict(),
        gene_space=inputs.gene_space,
        gene_rows=inputs.gene_rows,
    )


def build(base: RunBase, kind: str, setting: Mapping[str, Any]) -> BridgeInputs:
    module = importlib.import_module(f"src.context_prior.remedies.{kind}")
    return module.build(base.bridge, setting)


class Fitter:
    """Fits the prior of a list of stages on the labelled training lines and
    scores its total prediction of every query."""

    def __init__(self, base: RunBase, inputs: BridgeInputs) -> None:
        self.base = base
        self.inputs = inputs
        self.prior = prior_inputs(base, inputs)
        self.truth = {
            name: base.truth(list(query.index))
            for name, query in inputs.queries.items()
        }

    def __call__(
        self, stages: Sequence[Stage]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        fitted = fit_prior(
            PriorSpec(tuple(stages)),
            self.prior,
            fit_lines=list(self.base.labelled),
            encoder_lines=list(self.prior.expression.index),
        )
        predictions = {
            name: total(fitted.predict(query))
            for name, query in self.inputs.queries.items()
        }
        scores = {
            name: score(frame, self.truth[name], self.base.definitions)
            for name, frame in predictions.items()
        }
        return predictions, scores

    def components(self) -> dict[float, tuple[dict, dict]]:
        """Every components penalty of the config, keyed by penalty."""
        return {
            float(p): self([Stage("expression_components", float(p))])
            for p in self.base.config["penalties"]["components"]
        }


def stages(
    config: Mapping[str, Any],
    block_set: str,
    components_penalty: float,
    gene_penalty: float | None,
) -> list[Stage]:
    """The prior's stages for one row: a block set at one penalty pair."""
    blocks = list(config["block_sets"][block_set])
    selected = int(config["prior"]["selected_genes"])
    return [Stage("expression_components", float(components_penalty))] + [
        Stage(b, float(gene_penalty), selected if b == "data_selected" else 0)
        for b in blocks[1:]
    ]


def reference_predictions(base: RunBase) -> dict[str, pd.DataFrame]:
    """The pinned reference row (its experiment's first setting): its ``val`` and
    ``test`` predictions."""
    reference = base.config["reference"]
    experiment = base.config["experiments"][reference["experiment"]]
    fit = Fitter(base, build(base, experiment["kind"], experiment["settings"][0]))
    predictions, _ = fit(
        stages(
            base.config,
            reference["block_set"],
            reference["components_penalty"],
            reference["gene_penalty"],
        )
    )
    return {name: predictions[name] for name in SCORED}
