"""What the noise diagnosis reads: the prior run it diagnoses (its bound config,
export and inputs, ``Diagnosis``), the state programs' members, and the reference row
refitted as each section needs it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.context_prior.bridge import Bridge, fit_bridge
from src.context_prior.prior import (
    FittedPrior,
    PriorInputs,
    PriorSpec,
    fit_prior,
    total,
)
from src.data.prior_offsets import checked_prior, read_prior_offsets
from src.experiments.joint.config import load_config
from src.eval.accuracy import gain, score
from src.experiments.prior.base import (
    RunBase,
    build,
    load_base,
    prior_inputs,
    stages,
)


#: A reproduced export row may differ from the written one by float rounding only.
OUT_DIR = "noise"


SECTIONS = (
    "cell_sampling",
    "learning_curve",
    "deposit_confound",
    "deposit_swap",
    "cell_states",
)
REPRODUCTION_TOLERANCE = 1e-6

#: Lines whose source lists less of the scoring set than this are named in
#: ``cell_states`` (their unlisted genes score as unexpressed).
LISTED_FLOOR = 0.9


@dataclass
class Diagnosis:
    """Everything the sections share.

    Attributes:
        config: The noise config.
        base: The prior run's data, as its own runner loads it.
        export: The prior run's export (lines x genes, residual-SD units): out of
            fold on the training lines, the full fit on validation and test.
        joint: The joint config the prior run reads (prepared root, split).
    """

    config: Mapping[str, Any]
    base: RunBase
    export: pd.DataFrame
    joint: Mapping[str, Any]

    @property
    def train(self) -> list[str]:
        """The labelled single-cell training lines (the export's training rows)."""
        return list(self.base.bridge.single_cell_train)

    @property
    def queries(self) -> dict[str, list[str]]:
        return {"val": list(self.base.bridge.val), "test": list(self.base.bridge.test)}

    @property
    def prepared(self) -> Path:
        return Path(self.joint["prepared_root"])

    def spec(self) -> PriorSpec:
        reference = self.base.config["reference"]
        return PriorSpec(
            tuple(
                stages(
                    self.base.config,
                    reference["block_set"],
                    reference["components_penalty"],
                    reference["gene_penalty"],
                )
            )
        )

    def reference_inputs(self) -> PriorInputs:
        """The reference row's prior inputs; its bridge must be the affine one,
        whose bridged queries this module rebuilds."""
        reference = self.base.config["reference"]
        experiment = self.base.config["experiments"][reference["experiment"]]
        if experiment["kind"] != "affine":
            raise ValueError(
                f"the diagnosis rebuilds the affine bridge, not {experiment['kind']!r}"
            )
        return prior_inputs(self.base, build(self.base, "affine", {}))

    def raw_pseudobulk(self) -> pd.DataFrame:
        from src.data.pseudobulk import read_pseudobulk

        return read_pseudobulk(self.prepared)

    def pseudobulk_genes(self) -> list[str]:
        from src.data.pseudobulk import PSEUDOBULK_DIR

        manifest = self.prepared / PSEUDOBULK_DIR / "manifest.json"
        return list(json.loads(manifest.read_text())["genes"])

    def subsamples(self) -> pd.DataFrame:
        """Cell-draw pseudo-bulk, prepared on first use (reads raw cells)."""
        from src.experiments.prepare import prepare_noise_inputs

        path = prepare_noise_inputs(
            self.joint,
            self.pseudobulk_genes(),
            cells=self.config["cell_sampling"]["cells"],
            programs=program_members(self.base, self.config),
        )
        return pd.read_parquet(path.parent / "subsamples.parquet")

    def reading(
        self, frame: pd.DataFrame, lines: Sequence[str], *, measured_only: bool = False
    ) -> pd.DataFrame:
        """``lines``' rows of a raw pseudo-bulk frame, in the prior's space. With
        ``measured_only`` a value the source does not measure stays missing
        instead of taking the training mean: agreement between two readings of a
        line must not count the same filled value twice."""
        rows = frame.loc[list(lines)]
        normalized = self.base.normalize_pseudobulk(rows)
        if measured_only:
            normalized = normalized.where(rows.loc[:, normalized.columns].notna())
        return normalized


def load_diagnosis(config: Mapping[str, Any]) -> Diagnosis:
    prior_run = Path(config["prior_run"])
    base_config = json.loads((prior_run / "run_config.json").read_text())["config"]
    base = load_base(base_config)
    every = [
        *base.bridge.single_cell_train,
        *base.bridge.val,
        *base.bridge.test,
    ]
    export = checked_prior(
        read_prior_offsets(prior_run / "export"),
        genes=base.definitions.genes,
        residual_scale=base.definitions.residual_scale,
        lines=every,
    )
    joint = load_config(Path(base_config["joint_config"]))
    return Diagnosis(config, base, export.values.loc[every], joint)


def program_members(base: RunBase, config: Mapping[str, Any]) -> dict[str, list[str]]:
    """The config's programs and their member genes, from the pinned tables."""
    root = Path(base.config["paths"]["reference"])
    cycle = pd.read_csv(root / "cell_cycle.tsv", sep="\t")
    meta = pd.read_csv(root / "meta_programs.tsv", sep="\t")
    tables = {
        **{phase: list(group["gene"]) for phase, group in cycle.groupby("phase")},
        **{name: list(group["gene"]) for name, group in meta.groupby("program")},
    }
    unknown = [p for p in config["cell_states"]["programs"] if p not in tables]
    if unknown:
        raise ValueError(f"programs absent from the pinned tables: {unknown}")
    return {p: tables[p] for p in config["cell_states"]["programs"]}


def reproduces(name: str, rebuilt: pd.DataFrame, written: pd.DataFrame) -> None:
    """Fail closed when a rebuilt export row differs from the written one: the
    diagnosis would otherwise score another prior than the run's."""
    difference = float(np.nanmax(np.abs(rebuilt.to_numpy() - written.to_numpy())))
    if not difference <= REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"{name}: rebuilt predictions differ from the export by {difference:.3g}"
        )


@dataclass
class ReferenceFit:
    """The reference row fitted on the whole training side, and its bridge."""

    fitted: FittedPrior
    bridge: Bridge

    def predict(self, reading: pd.DataFrame) -> pd.DataFrame:
        """Predictions (residual-SD units) of pseudo-bulk rows in the space."""
        return total(self.fitted.predict(self.bridge.apply(reading)))


def reference_fit(diagnosis: Diagnosis) -> ReferenceFit:
    base = diagnosis.base
    prior = diagnosis.reference_inputs()
    fitted = fit_prior(
        diagnosis.spec(),
        prior,
        fit_lines=list(base.labelled),
        encoder_lines=list(prior.expression.index),
    )
    paired = list(base.bridge.paired)
    bridge = fit_bridge(
        base.bridge.pseudobulk.loc[paired], base.bridge.bulk.loc[paired]
    )
    fit = ReferenceFit(fitted, bridge)
    for split, lines in diagnosis.queries.items():
        reproduces(
            split,
            fit.predict(base.bridge.pseudobulk.loc[lines]),
            diagnosis.export.loc[lines],
        )
    return fit


def scores_and_gain(
    diagnosis: Diagnosis,
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    lines: Sequence[str],
) -> dict:
    """Metrics of ``predictions`` on ``lines`` and the gain over ``reference``."""
    base = diagnosis.base
    truth = base.truth(list(lines))
    return {
        **score(predictions.loc[list(lines)], truth, base.definitions),
        "gain": gain(
            predictions.loc[list(lines)],
            reference.loc[list(lines)],
            truth,
            base.definitions,
            int(diagnosis.config["bootstrap_repeats"]),
        ),
    }
