"""Reliability gating: the affine remedy's inputs with a gene space of the genes the
bridge carries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.context_prior.bridge import bridge_quality
from src.context_prior.remedies import affine, gating
from tests.test_context_prior_bridging import GENES, base


def gated_base():
    """The bridging test's sources with G0-G3 pseudo-bulk replaced by noise (the
    bridge cannot carry them) and G4 constant in paired bulk (undefined quality)."""
    b = base()
    rng = np.random.default_rng(5)
    pseudo = b.pseudobulk.copy()
    pseudo[GENES[:4]] = rng.normal(size=(len(pseudo), 4))
    bulk = b.bulk.copy()
    bulk.loc[list(b.paired), "G4"] = 1.0
    return replace(b, pseudobulk=pseudo, bulk=bulk)


def quality(b):
    """Out-of-fold bridge quality on the paired lines, computed independently."""
    paired = list(b.paired)
    return bridge_quality(affine.build(b, {}).oof_paired, b.bulk.loc[paired])


def test_inputs_are_the_affine_remedys_except_the_gene_space():
    b = gated_base()
    reference = affine.build(b, {})
    gated = gating.build(b, {"threshold": 0.5})
    assert gated.expression.equals(reference.expression)
    assert set(gated.queries) == set(reference.queries)
    for name, frame in reference.queries.items():
        assert gated.queries[name].equals(frame)
    assert gated.oof_paired.equals(reference.oof_paired)
    assert gated.gene_rows is None
    assert reference.gene_space is None and gated.gene_space is not None


def test_gene_space_holds_the_genes_bridged_at_or_above_the_threshold():
    b = gated_base()
    q = quality(b)
    assert np.isnan(q["G4"])
    space = gating.build(b, {"threshold": 0.5}).gene_space
    assert space == tuple(g for g in GENES if q[g] >= 0.5)
    assert space == tuple(GENES[5:])  # the noise genes and the undefined one are out
    # A gene whose quality equals the threshold is in; an undefined one never is.
    assert "G7" in gating.build(b, {"threshold": float(q["G7"])}).gene_space
    every = gating.build(b, {"threshold": -1.0}).gene_space
    assert every == tuple(g for g in GENES if g != "G4")


def test_a_threshold_above_every_quality_leaves_an_empty_gene_space():
    assert gating.build(gated_base(), {"threshold": 1.5}).gene_space == ()


def test_the_setting_is_exactly_a_threshold():
    for setting in ({}, {"threshold": 0.5, "rank": 4}):
        with pytest.raises(ValueError, match="threshold"):
            gating.build(base(), setting)
