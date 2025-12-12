import numpy as np
import pytest

from src.utils.de_metrics import (
    compute_de_comparison_metrics,
    compute_des,
    compute_mae_top2k,
    compute_overall_score,
    compute_pds,
)


def test_compute_pds_deterministic():
    truth_deltas = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
        "C": np.array([0.0, 0.0, 1.0]),
    }
    pred_deltas = {k: v.copy() for k, v in truth_deltas.items()}

    results = compute_pds(pred_deltas, truth_deltas)
    assert results["mean_rank"] == 1.0
    assert results["npds"] == pytest.approx(1.0 / 3.0)
    assert all(rank == 1 for rank in results["ranks"].values())
    assert all(sim == pytest.approx(1.0) for sim in results["cosine_self"].values())


def test_compute_des_pred_smaller_or_equal():
    truth = {0, 1, 2, 3}
    pred = {1, 3}
    pred_log2fc = np.array([0.1, 0.2, 0.3, 0.4])
    des, n_intersect = compute_des(truth, pred, pred_log2fc)
    assert n_intersect == 2
    assert des == pytest.approx(0.5)


def test_compute_des_pred_larger_truncates_by_abs_log2fc():
    truth = {0, 1, 2}
    pred = {0, 1, 2, 3, 4}
    pred_log2fc = np.array([1.0, 0.9, 0.8, 100.0, 90.0])
    des, n_intersect = compute_des(truth, pred, pred_log2fc)
    assert n_intersect == 1
    assert des == pytest.approx(1.0 / 3.0)


def test_compute_de_comparison_metrics_pdex_perfect_prediction():
    rng = np.random.default_rng(0)
    n_ctrl = 40
    n_pert = 40
    n_genes = 20
    gene_names = np.array([f"g{i}" for i in range(n_genes)], dtype=object)

    control_expr = rng.normal(loc=1.0, scale=0.05, size=(n_ctrl, n_genes)).astype(
        np.float32
    )
    truth_expr = control_expr[:n_pert].copy()
    truth_expr[:, :5] += 2.0  # strong shift to ensure DE detection

    pred_expr = truth_expr.copy()

    results = compute_de_comparison_metrics(
        control_expr=control_expr,
        pred_expr=pred_expr,
        truth_expr=truth_expr,
        gene_names=gene_names,
        fdr_threshold=0.05,
        threads=1,
    )

    assert results["n_de_truth"] > 0
    assert results["des"] == pytest.approx(1.0)
    assert results["n_de_pred"] == results["n_de_truth"]
    assert results["n_intersect"] == results["n_de_truth"]


def test_compute_mae_top2k_constant_error():
    rng = np.random.default_rng(0)
    n_cells = 10
    n_genes = 50
    control_mean = rng.normal(loc=1.0, scale=0.1, size=(n_genes,)).astype(np.float32)
    truth_expr = rng.normal(loc=1.0, scale=0.1, size=(n_cells, n_genes)).astype(
        np.float32
    )
    error_val = 0.5
    pred_expr = truth_expr + error_val

    mae = compute_mae_top2k(
        pred_expr=pred_expr,
        truth_expr=truth_expr,
        control_mean=control_mean,
        top_k=20,
    )
    assert np.isclose(mae, error_val, atol=1e-6)


def test_compute_overall_score():
    """Test the aggregation logic."""
    from src.utils.de_metrics import BASELINE_DES, BASELINE_MAE_TOP2000, BASELINE_PDS

    # Case 1: Better-than-baseline scores should yield positive scaled values.
    input_pds = 0.1  # Lower normalized rank is better
    input_mae = 0.05  # Lower than baseline 0.1258
    input_des = 0.8  # Higher than baseline 0.0442

    scores = compute_overall_score(input_pds, input_mae, input_des)

    assert scores["pds_scaled"] > 0
    assert scores["mae_scaled"] > 0
    assert scores["des_scaled"] > 0

    # Case 2: Worse than baseline (should clip to 0)
    scores_bad = compute_overall_score(
        pds=BASELINE_PDS + 0.1, mae=BASELINE_MAE_TOP2000 + 0.1, des=BASELINE_DES - 0.01
    )
    assert scores_bad["pds_scaled"] == 0.0
    assert scores_bad["mae_scaled"] == 0.0
    assert scores_bad["des_scaled"] == 0.0
    assert scores_bad["overall_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
