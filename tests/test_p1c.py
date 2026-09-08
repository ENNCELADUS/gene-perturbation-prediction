import numpy as np
import pandas as pd
import pytest


def test_bias_decomposition_splits_shared_from_condition_specific():
    from src.eval.p1c_tier0 import bias_decomposition

    observed = np.zeros((4, 3))
    predicted = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0], [1.0, -2.0, 0]])
    result = bias_decomposition(predicted, observed)
    # mean_g ||e||^2 = (1 + 1 + 5 + 5)/4 = 3 ; ||ē||^2 = 1 ; residual = 2
    assert result["total_mse"] == pytest.approx(3.0 / 3)  # per-coordinate mean
    assert result["shared_bias"] == pytest.approx(1.0 / 3)
    assert result["condition_specific"] == pytest.approx(2.0 / 3)
    assert result["shared_bias_fraction"] == pytest.approx(1 / 3)
    assert result["n"] == 4


def test_decompose_effects_reads_npz_by_method_and_anchor(tmp_path):
    from src.eval.p1c_tier0 import decompose_effects

    keys = np.array([("model", "A", "G"), ("model", "A", "H"), ("no_change", "A", "G")])
    predicted = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    observed = np.zeros((3, 2))
    np.savez(
        tmp_path / "effects.npz", keys=keys, predicted=predicted, observed=observed
    )
    frame = decompose_effects(tmp_path / "effects.npz", method="model")
    assert frame.model_id.tolist() == ["A"] and frame.n.tolist() == [2]
    assert frame.shared_bias_fraction.iloc[0] == pytest.approx(1.0)


def test_matched_coverage_restricts_references_to_perturbation_mean_support():
    from src.eval.p1c_tier0 import matched_coverage

    rows = []
    for gene, covered in (("G", True), ("H", False)):
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="no_change",
                response_loss=1.0,
            )
        )
        rows.append(
            dict(
                role="val",
                model_id="A",
                panel="all",
                gene=gene,
                method="global_mean",
                response_loss=2.0,
            )
        )
        if covered:
            rows.append(
                dict(
                    role="val",
                    model_id="A",
                    panel="all",
                    gene=gene,
                    method="perturbation_mean",
                    response_loss=3.0,
                )
            )
    frame = matched_coverage(pd.DataFrame(rows))
    row = frame[(frame.model_id == "A") & (frame.reference == "no_change")].iloc[0]
    assert row.covered == 1 and row.total == 2
    assert row.perturbation_mean_loss == 3.0 and row.reference_loss_on_covered == 1.0


def test_cross_context_pearson_is_undefined_for_constant_effect_references():
    from src.eval.p1b import cross_context

    bundle = {
        "keys": [("a", "G"), ("b", "G"), ("j", "G")],
        "splits": {"train": [], "val": [0, 1], "external": [2]},
        "anchors": ["a", "b"],
        "external": "j",
    }
    # Identical predicted effects in both anchors -> zero difference up to float noise.
    noise = np.array([1e-7, -1e-7, 0.0])
    effects = {
        ("global_mean", "a", "G"): (
            np.array([1.0, 2.0, 3.0]) + noise,
            np.array([1.0, 0.0, 0.0]),
        ),
        ("global_mean", "b", "G"): (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    }
    frame = pd.DataFrame({"method": ["global_mean"]})
    result = cross_context(
        frame,
        effects,
        {**bundle, "splits": {"train": [], "val": [0, 1], "external": [2]}},
    )
    assert len(result) == 1 and np.isnan(result.effect_difference_pearson.iloc[0])
