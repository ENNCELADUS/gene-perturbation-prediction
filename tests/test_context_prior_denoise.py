"""The low-rank denoising bridge remedy."""

from __future__ import annotations

import numpy as np
import pytest

from src.context_prior.remedies import affine, denoise
from src.data.context_pca import fit_context_pca
from tests.test_context_prior_bridging import GENES, base


def project(frame, pca):
    z = (frame.to_numpy() - pca.mean) / pca.scale
    return (z @ pca.components.T) @ pca.components * pca.scale + pca.mean


def test_expression_and_oracle_are_the_references():
    b = base()
    reference = affine.build(b, {})
    inputs = denoise.build(b, {"rank": 4})
    assert inputs.expression is b.bulk
    assert inputs.queries["oracle"].equals(reference.queries["oracle"])
    assert set(inputs.queries) == {"val", "test", "oracle"}
    assert list(inputs.oof_paired.index) == list(reference.oof_paired.index)
    assert list(inputs.oof_paired.columns) == GENES


def test_denoised_rows_are_rank_r_reconstructions():
    b = base()
    reference = affine.build(b, {})
    inputs = denoise.build(b, {"rank": 4})
    pca = fit_context_pca(b.bulk.to_numpy(), 4)
    for name in ("val", "test"):
        got = inputs.queries[name]
        assert not np.allclose(got, reference.queries[name])
        assert np.allclose(project(got, pca), got.to_numpy())  # idempotent
    # Each out-of-fold row lies on components fitted without its fold's bulk.
    for fold in sorted({b.folds[m] for m in b.paired}):
        held = [m for m in b.paired if b.folds[m] == fold]
        local = fit_context_pca(b.bulk.drop(index=held).to_numpy(), 4)
        rows = inputs.oof_paired.loc[held]
        assert np.allclose(project(rows, local), rows.to_numpy())


def test_full_rank_equals_the_affine_queries():
    b = base()
    reference = affine.build(b, {})
    inputs = denoise.build(b, {"rank": len(GENES)})
    for name in ("val", "test"):
        assert np.allclose(inputs.queries[name], reference.queries[name])
    assert np.allclose(inputs.oof_paired, reference.oof_paired)


@pytest.mark.parametrize("setting", [{}, {"rank": 4, "extra": 1}, {"k": 4}])
def test_setting_keys_are_exactly_rank(setting):
    with pytest.raises(ValueError, match="rank"):
        denoise.build(base(), setting)
