"""The stagewise prior on synthetic lines: recovery, missing labels, query keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.folds import patient_folds
from src.context_prior.prior import (
    INNER_FOLD_SEED,
    INNER_FOLDS,
    PriorInputs,
    PriorSpec,
    Stage,
    _context_view,
    crossfit,
    fit_prior,
    total,
)
from src.context_prior.reference import Reference
from src.context_prior.views import (
    expression_components,
    fit_expression_components,
    fit_genotype,
)

SPACE = [f"E{i}" for i in range(12)]
GENES = ["E0", "E1", "Q"]  # Q has no expression column


def synthetic(seed=0, lines=80):
    rng = np.random.default_rng(seed)
    ids = [f"L{i}" for i in range(lines)]
    expression = pd.DataFrame(rng.normal(size=(lines, 12)), index=ids, columns=SPACE)
    residual = pd.DataFrame(
        {
            "E0": expression["E3"] + 0.1 * rng.normal(size=lines),
            "E1": -expression["E1"] + 0.1 * rng.normal(size=lines),
            "Q": expression["E5"] - expression["E6"],
        },
        index=ids,
    )
    residual.iloc[0, 0] = np.nan  # a missing training label
    reference = Reference(
        paralogs=pd.DataFrame({"gene": ["E0"], "paralog": ["E2"], "identity": [50.0]}),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["E1", "E4"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["E7"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["E8"], "weight": [1.0]}),
        drivers=pd.DataFrame(
            {"KRAS": (expression["E9"] > 0).astype(int)},
            index=pd.Index(ids, name="model_id"),
        ),
        msi=pd.Series(
            expression["E10"].to_numpy(), index=pd.Index(ids, name="model_id")
        ),
    )
    inputs = PriorInputs(
        expression=expression,
        residual=residual,
        components=fit_expression_components(expression, 6),
        reference=reference,
        lineage=pd.Series(["Lung"] * lines, index=ids),
        patients={m: m for m in ids},
    )
    return inputs, ids


ALL_BLOCKS = PriorSpec(
    (
        Stage("expression_components", 0.01),
        Stage("pathway_scores", 1.0),
        Stage("predicted_genotype", 1.0),
        Stage("own_expression", 1.0),
        Stage("partners", 1.0),
        Stage("data_selected", 0.01, selected=2),
    )
)


def test_every_block_fits_and_recovers_signal():
    inputs, ids = synthetic()
    fitted = fit_prior(ALL_BLOCKS, inputs, fit_lines=ids[:60], encoder_lines=ids)
    # Score on many held-out lines of the same model: on 20 lines the sample
    # correlation of a prior whose true correlation is 0.85 spreads from 0.7 to 0.9.
    held_out, _ = synthetic(seed=1, lines=1000)
    stages = fitted.predict(held_out.expression)
    assert list(stages) == [s.block for s in ALL_BLOCKS.stages]
    prediction = total(stages)
    for gene in GENES:
        assert prediction[gene].corr(held_out.residual[gene]) > 0.8
    assert np.isfinite(prediction.to_numpy()).all()


def test_fit_lines_must_be_encoder_lines():
    inputs, ids = synthetic()
    with pytest.raises(ValueError, match="encoder"):
        fit_prior(ALL_BLOCKS, inputs, fit_lines=ids, encoder_lines=ids[:40])


def test_a_missing_label_counts_as_zero_residual():
    inputs, ids = synthetic()
    filled = PriorInputs(**{**vars(inputs), "residual": inputs.residual.fillna(0.0)})
    query = inputs.expression.loc[ids[60:]]

    def predict(prior_inputs):
        fitted = fit_prior(
            ALL_BLOCKS, prior_inputs, fit_lines=ids[:60], encoder_lines=ids
        )
        return total(fitted.predict(query))

    assert np.allclose(predict(inputs), predict(filled))


def test_a_feature_constant_on_the_fit_lines_does_not_move_a_query():
    inputs, ids = synthetic()
    expression = inputs.expression.copy()
    expression.loc[ids[:60], "E1"] = 0.1  # E1's own expression, inexact in binary
    inputs = PriorInputs(**{**vars(inputs), "expression": expression})
    spec = PriorSpec(
        (
            Stage("own_expression", 1.0),
            Stage("partners", 1.0),
            Stage("data_selected", 0.01, selected=2),
        )
    )
    fitted = fit_prior(spec, inputs, fit_lines=ids[:60], encoder_lines=ids[:60])
    at_fit = expression.loc[ids[60:]].assign(E1=0.1)
    moved = expression.loc[ids[60:]].assign(E1=5.0)
    assert np.allclose(
        total(fitted.predict(at_fit)), total(fitted.predict(moved)), atol=1e-9
    )


def test_reduced_rank_applies_to_context_stages_only():
    inputs, ids = synthetic()
    spec = PriorSpec(
        (
            Stage("expression_components", 0.01),
            Stage("data_selected", 0.01, selected=2),
        ),
        rank=1,
    )
    fitted = fit_prior(spec, inputs, fit_lines=ids[:60], encoder_lines=ids)
    stages = fitted.predict(inputs.expression.loc[ids[60:]])

    def singular(frame):
        values = frame.to_numpy()
        return np.linalg.svd(values - values.mean(axis=0), compute_uv=False)

    context = singular(stages["expression_components"])
    selected = singular(stages["data_selected"])
    assert context[1] < 1e-10 * context[0]
    assert selected[1] > 1e-3 * selected[0]


def test_genotype_trains_on_out_of_sample_features_and_queries_on_full_encoders():
    inputs, ids = synthetic()
    fit_rows, encoder_rows = inputs.expression.loc[ids[:60]], inputs.expression
    features, train = _context_view(
        "predicted_genotype", inputs, fit_rows, encoder_rows
    )
    folds = patient_folds(
        ids, inputs.patients, n_folds=INNER_FOLDS, seed=INNER_FOLD_SEED
    )
    components = expression_components(inputs.components, encoder_rows)
    encoders, own = fit_genotype(components, inputs.reference, inputs.lineage, folds)
    assert np.allclose(train, own.loc[ids[:60]].to_numpy())
    assert not np.allclose(train, features(fit_rows))
    query = inputs.expression.loc[ids[60:]]
    full = encoders.predict(expression_components(inputs.components, query))
    assert np.allclose(features(query), full.to_numpy())


def test_crossfit_predicts_each_fold_from_the_others_keyed_by_query():
    inputs, ids = synthetic()
    folds = {m: i % 4 for i, m in enumerate(ids)}
    queries = {
        k: inputs.expression.loc[[m for m in ids if folds[m] == k]] + 0.5
        for k in range(4)
    }
    spec = PriorSpec((Stage("expression_components", 0.1),), rank=2)
    predicted = crossfit(
        spec, inputs, folds=folds, queries=queries, labelled=ids, encoder_lines=ids
    )
    frame = predicted["expression_components"]
    assert sorted(frame.index) == sorted(ids)
    fold_zero = [m for m in ids if folds[m] == 0]
    outside = [m for m in ids if folds[m] != 0]
    fitted = fit_prior(spec, inputs, fit_lines=outside, encoder_lines=outside)
    direct = fitted.predict(queries[0])["expression_components"]
    assert np.allclose(frame.loc[fold_zero], direct)
    # Keyed by the query row, never the same line's bulk row.
    bulk = fitted.predict(inputs.expression.loc[fold_zero])["expression_components"]
    assert not np.allclose(frame.loc[fold_zero], bulk)


def test_crossfit_refuses_a_query_line_of_another_fold():
    inputs, ids = synthetic()
    folds = {m: i % 4 for i, m in enumerate(ids)}
    queries = {0: inputs.expression.loc[[ids[0], ids[1]]]}  # L1 is in fold 1
    spec = PriorSpec((Stage("expression_components", 0.1),))
    with pytest.raises(ValueError, match="fold"):
        crossfit(
            spec, inputs, folds=folds, queries=queries, labelled=ids, encoder_lines=ids
        )
