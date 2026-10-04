"""Train-only metric definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.targets import fit_definitions, residual_frame
from src.data.splits import FixedSplit

SETTINGS = {
    "variable_gene_min_observations": 3,
    "variable_gene_percentile": 50,
    "selective_min_lines": 1,
    "selective_max_fraction": 0.9,
    "residual_sd_floor_percentile": 10,
}


def test_definitions_use_labelled_training_lines_only():
    rng = np.random.default_rng(0)
    lines = [f"T{i}" for i in range(6)] + ["V1", "V2"]
    effect = pd.DataFrame(
        rng.normal(-0.3, 0.4, (8, 4)), index=lines, columns=["A", "B", "C", "D"]
    )
    effect.loc["V1"] = 100.0  # a validation value must not move any fit
    effect.loc["T0":"T4", "D"] = np.nan  # D has one training label: dropped
    split = FixedSplit(train=tuple(lines[:6]), val=("V1", "V2"), test=())
    definitions = fit_definitions(effect, split, ["A", "B", "C", "D"], SETTINGS)
    assert definitions.genes == ("A", "B", "C")
    assert np.isclose(definitions.gene_means["A"], effect.loc["T0":"T5", "A"].mean())
    residual = residual_frame(effect, ["V2"], definitions)
    assert np.isclose(
        residual.loc["V2", "B"], effect.loc["V2", "B"] - definitions.gene_means["B"]
    )
