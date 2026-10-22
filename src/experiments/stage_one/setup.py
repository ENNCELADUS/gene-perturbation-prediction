"""A ladder config and everything it names: the prior, joint and noise configs at once,
and the built ladder, the split and the labels on first use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from src.context_prior.targets import Definitions
from src.data import ladder_reader as reader
from src.data.extra_lines import ExtraLines
from src.data.splits import FixedSplit


@dataclass(frozen=True)
class Setup:
    config: Mapping[str, Any]
    prior: Mapping[str, Any]
    joint: Mapping[str, Any]
    noise: Mapping[str, Any]

    @property
    def root(self) -> Path:
        return Path(self.config["ladder_root"])

    @property
    def prepared(self) -> Path:
        return Path(self.joint["prepared_root"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def label_source(self) -> Path:
        """The GeneEffect matrix the labels are read from."""
        return Path(self.joint["paths"]["gene_effect"])

    @cached_property
    def ladder(self) -> reader.Ladder:
        return reader.read_ladder(self.root)

    @cached_property
    def split(self) -> FixedSplit:
        from src.data.splits import load_geneeffect_split

        return load_geneeffect_split(Path(self.joint["paths"]["split"]))

    @cached_property
    def bulk(self) -> ExtraLines:
        from src.data.extra_lines import load_bulk_lines

        return load_bulk_lines(Path(self.joint["paths"]["split"]), self.split)

    @property
    def bulk_labelled(self) -> tuple[str, ...]:
        return tuple(self.bulk.labelled)

    @property
    def excluded(self) -> tuple[str, ...]:
        """Lines outside the split sharing a patient with a held-out line."""
        return tuple(self.bulk.excluded)

    @cached_property
    def gene_effect(self) -> pd.DataFrame:
        from src.data.depmap import read_depmap_matrix

        return read_depmap_matrix(self.label_source)

    @property
    def definitions(self) -> Definitions:
        """The protocol's definitions (377-line mean, residual SD, gene sets), as
        ``genes.parquet`` records them."""
        genes = self.ladder.genes
        axis = [str(g) for g in genes.index]
        return Definitions(
            genes=tuple(axis),
            gene_means=genes["mu_g"].astype(float),
            selective=tuple(g for g, s in zip(axis, genes["selective"]) if s),
            variable=tuple(g for g, v in zip(axis, genes["variable"]) if v),
            residual_scale=genes["sigma_g"].astype(float),
        )

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        return reader.scoring_truth(self.ladder, self.gene_effect, lines)

    def targets(self, fold: int | None) -> reader.Targets:
        return reader.training_targets(
            self.ladder,
            self.gene_effect,
            self.split,
            self.bulk_labelled,
            exclude_fold=fold,
        )


def load_setup(config: Mapping[str, Any] | str | Path) -> Setup:
    """A ladder config (a path, or one already loaded) with the configs it names."""
    from src.experiments.joint.config import load_config
    from src.experiments.prior.config import load_noise_config, load_prior_config
    from src.experiments.stage_one.config import (
        load_ladder_config,
        validate_ladder_config,
    )

    if isinstance(config, Mapping):
        validate_ladder_config(config)
    else:
        config = load_ladder_config(Path(config))
    prior = load_prior_config(Path(config["prior_config"]))
    return Setup(
        config=config,
        prior=prior,
        joint=load_config(Path(prior["joint_config"])),
        noise=load_noise_config(Path(config["noise_config"])),
    )
