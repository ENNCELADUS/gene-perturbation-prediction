"""The noise-matched remedy: the affine inputs plus out-of-fold bridged gene rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.context_prior.bridging import BridgeBase, BridgeInputs, oof_bridged
from src.context_prior.prior import PriorInputs, PriorSpec, Stage, fit_prior, total
from src.context_prior.reference import Reference
from src.context_prior.remedies import affine, noise_matched
from src.context_prior.views import fit_expression_components
from tests.test_context_prior_bridging import GENES, base


def test_inputs_are_the_affine_remedy_s_plus_gene_rows():
    b = base()
    reference = affine.build(b, {})
    inputs = noise_matched.build(b, {})
    # Components-only predictions read expression and queries alone: identical.
    pd.testing.assert_frame_equal(inputs.expression, reference.expression)
    assert set(inputs.queries) == set(reference.queries)
    for name, frame in reference.queries.items():
        pd.testing.assert_frame_equal(inputs.queries[name], frame)
    pd.testing.assert_frame_equal(inputs.oof_paired, reference.oof_paired)
    assert inputs.gene_space is None
    assert reference.gene_rows is None and inputs.gene_rows is not None
    with pytest.raises(ValueError, match="no settings"):
        noise_matched.build(b, {"rank": 4})


def test_gene_rows_are_every_single_cell_training_line_bridged_out_of_fold():
    b = base()
    rows = noise_matched.build(b, {}).gene_rows
    expected = oof_bridged(b.pseudobulk, b.bulk, b.paired, b.single_cell_train, b.folds)
    pd.testing.assert_frame_equal(rows, expected)
    without_bulk = [m for m in b.single_cell_train if m not in b.bulk.index]
    assert without_bulk == ["S18", "S19"]
    assert list(rows.index) == list(b.single_cell_train)


def noisy_copy(seed=0, lines=200, unpaired=20, noise=1.0):
    """Pseudo-bulk is a noisy copy of bulk (noise SD ``noise``, signal SD 1); each
    gene's residual is its own bulk expression. The last ``unpaired`` single-cell
    lines have labels but no bulk row."""
    rng = np.random.default_rng(seed)
    sc = [f"S{i}" for i in range(lines)]
    val = [f"V{i}" for i in range(6)]
    test = [f"T{i}" for i in range(6)]
    signal = pd.DataFrame(
        rng.normal(size=(lines + 12, len(GENES))),
        index=[*sc, *val, *test],
        columns=GENES,
    )
    paired = sc[: lines - unpaired]
    bridge_base = BridgeBase(
        bulk=signal.loc[paired],
        oracle=signal.loc[val],
        pseudobulk=signal + noise * rng.normal(size=signal.shape),
        paired=tuple(paired),
        single_cell_train=tuple(sc),
        val=tuple(val),
        test=tuple(test),
        folds={m: i % 5 for i, m in enumerate(sc)},
    )
    residual = signal.loc[sc] + 0.1 * rng.normal(size=(lines, len(GENES)))
    return bridge_base, residual


def as_prior_inputs(inputs: BridgeInputs, residual: pd.DataFrame) -> PriorInputs:
    ids = list(inputs.expression.index)
    no_pairs = Reference(
        paralogs=pd.DataFrame(columns=["gene", "paralog", "identity"]),
        complexes=pd.DataFrame(columns=["complex_id", "gene"]),
        hallmark=pd.DataFrame(columns=["gene_set", "gene"]),
        progeny=pd.DataFrame(columns=["pathway", "gene", "weight"]),
        drivers=pd.DataFrame(index=pd.Index(ids, name="model_id")),
        msi=pd.Series(0.0, index=pd.Index(ids, name="model_id")),
    )
    return PriorInputs(
        expression=inputs.expression,
        residual=residual,
        components=fit_expression_components(inputs.expression, 4),
        reference=no_pairs,
        lineage=pd.Series("Lung", index=ids),
        patients={m: m for m in ids},
        gene_space=inputs.gene_space,
        gene_rows=inputs.gene_rows,
    )


def test_own_expression_weight_is_learnt_on_the_bridged_rows():
    # Own expression alone. With expression components fitted first, the gene
    # rows' target is the residual minus the components' prediction from the same
    # noisy rows; that prediction's noise correlates with the noisy own feature,
    # which then learns to cancel it, so the pooled weight need not shrink.
    b, residual = noisy_copy()
    own = PriorSpec((Stage("own_expression", np.inf),))  # pooled weight only
    labelled = list(b.paired)

    def fit(inputs: BridgeInputs) -> tuple[float, np.ndarray]:
        fitted = fit_prior(
            own,
            as_prior_inputs(inputs, residual),
            fit_lines=labelled,
            encoder_lines=labelled,
        )
        models = {block: model for block, _, model in fitted.stages}
        query = total(fitted.predict(inputs.queries["val"])).to_numpy()
        return float(models["own_expression"].pooled[0]), query

    clean, clean_query = fit(affine.build(b, {}))
    matched, matched_query = fit(noise_matched.build(b, {}))
    # The weight is per SD of the feature on its fit rows. Bridged pseudo-bulk
    # correlates about 0.7 with bulk at equal noise and signal SD; the bulk-fitted
    # weight is about 1.
    assert 0.9 < clean < 1.1
    assert 0 < matched < 0.8 * clean
    # The affine bridge regresses bulk on pseudo-bulk, so its rows are already
    # shrunk toward the mean by the bridge's reliability: the smaller weight
    # meets rows of smaller SD, and the queries' predictions barely move.
    moved = np.std(matched_query - clean_query) / np.std(clean_query)
    assert moved < 0.15
