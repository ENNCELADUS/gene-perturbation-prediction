"""Expression components, pathway scores and out-of-sample predicted genotype."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.context_prior.reference import Reference
from src.context_prior.views import (
    expression_components,
    fit_expression_components,
    fit_genotype,
    pathway_scores,
)


def reference(lines, drivers, msi):
    return Reference(
        paralogs=pd.DataFrame(columns=["gene", "paralog", "identity"]),
        complexes=pd.DataFrame(columns=["complex_id", "gene"]),
        hallmark=pd.DataFrame({"gene_set": ["S", "S", "T"], "gene": ["A", "B", "ZZ"]}),
        progeny=pd.DataFrame(
            {"pathway": ["P", "P"], "gene": ["A", "C"], "weight": [1.0, -1.0]}
        ),
        drivers=pd.DataFrame(drivers, index=pd.Index(lines, name="model_id")),
        msi=pd.Series(msi, index=pd.Index(lines, name="model_id"), name="msi_score"),
    )


def test_components_are_eigen_scaled_and_named():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(20, 6)), columns=list("ABCDEF"))
    pca = fit_expression_components(frame, 3)
    scores = expression_components(pca, frame)
    assert list(scores.columns) == ["pc1", "pc2", "pc3"]
    assert np.isclose(scores["pc1"].std(ddof=0), 1.0)


def test_pathway_scores_skip_absent_genes():
    frame = pd.DataFrame([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], columns=["A", "B", "C"])
    ref = reference(["L"], {"KRAS": [0]}, [1.0])
    scores = pathway_scores(frame, ref.hallmark, ref.progeny)
    assert list(scores.columns) == ["hallmark:S", "progeny:P"]  # T has no present gene
    assert np.isclose(scores.iloc[0]["hallmark:S"], (3 / 3 + 2 / 3) / 2 - 0.5)
    assert np.isclose(scores.iloc[0]["progeny:P"], 3.0 - 1.0)


def test_genotype_features_are_out_of_sample_and_handle_a_constant_driver():
    rng = np.random.default_rng(0)
    lines = [f"L{i}" for i in range(60)]
    components = pd.DataFrame(rng.normal(size=(60, 4)), index=lines)
    mutated = (components[0] > 0).astype(int).tolist()
    ref = reference(lines, {"KRAS": mutated, "BRAF": [0] * 60}, components[1].tolist())
    lineage = pd.Series(["Lung"] * 30 + ["Bowel"] * 30, index=lines)
    folds = {m: i % 5 for i, m in enumerate(lines)}
    encoders, own = fit_genotype(components, ref, lineage, folds)
    assert list(own.index) == lines
    assert (own["driver:BRAF"] == 0.0).all()
    assert own.loc[components[0] > 1, "driver:KRAS"].mean() > 0.7
    query = encoders.predict(components.iloc[:3])
    assert list(query.columns) == list(own.columns)


def test_float_inexact_constant_column_does_not_move_components():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("ABCDE"))
    frame["E"] = 0.1  # mean of 60 copies of 0.1 is not exactly 0.1
    pca = fit_expression_components(frame, 3)
    query = frame.iloc[:2].copy()
    moved = query.assign(E=0.5)
    assert np.allclose(
        expression_components(pca, query), expression_components(pca, moved), atol=1e-6
    )
