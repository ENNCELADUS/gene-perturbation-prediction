"""Train-only metric definitions and the fast selective Spearman."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.targets import fit_definitions, residual_frame
from src.data.splits import FixedSplit
from src.eval.geneeffect import aggregate_geneeffect
from src.eval.metrics import macro_gene_spearman

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


def test_fast_selector_matches_aggregate_geneeffect():
    rng = np.random.default_rng(1)
    genes, lines = [f"G{i}" for i in range(5)], [f"L{i}" for i in range(7)]
    truth = rng.normal(size=(5, 7))
    prediction = rng.normal(size=(5, 7))
    truth[0, :3] = np.nan
    prediction[1] = 0.5  # constant: undefined
    prediction[2, :4] = 1.0  # ties
    frame = pd.DataFrame(
        {
            "model_id": np.tile(lines, 5),
            "gene_symbol": np.repeat(genes, 7),
            "residual": truth.ravel(),
            "residual_prediction": prediction.ravel(),
        }
    ).dropna(subset=["residual"])
    frame["gene_effect"] = frame["residual"]
    frame["geneeffect_prediction"] = frame["residual_prediction"]
    metrics, _, _ = aggregate_geneeffect(
        frame, model_ids=lines, genes=genes, variable_genes=genes, selective_genes=genes
    )
    assert np.isclose(
        macro_gene_spearman(truth, prediction), metrics["selective_spearman"]
    )
