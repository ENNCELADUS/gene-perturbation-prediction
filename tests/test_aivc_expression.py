from pathlib import Path

import numpy as np
import pytest

from aivc_model.expression import (
    assert_finite_npy,
    compute_finite_feature_means,
    replace_nonfinite,
)


def test_control_means_ignore_nonfinite_and_preserve_negative_values() -> None:
    matrix = np.asarray(
        [[-2.0, np.nan, 1.0], [2.0, 4.0, np.inf], [8.0, 6.0, 3.0]],
        dtype=np.float32,
    )
    means = compute_finite_feature_means(
        matrix,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        chunk_size=1,
    )
    np.testing.assert_allclose(means, [0.0, 4.0, 1.0])


def test_replace_nonfinite_uses_feature_means_only() -> None:
    values = np.asarray([[-1.0, np.nan], [np.inf, 5.0]], dtype=np.float32)
    result = replace_nonfinite(values, np.asarray([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(result, [[-1.0, 3.0], [2.0, 5.0]])


def test_assert_finite_npy_reports_the_first_bad_chunk(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.asarray([[1.0], [np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match=r"bad.npy.*rows 1:2"):
        assert_finite_npy(path, chunk_size=1)
