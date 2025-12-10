import pytest
import scanpy as sc
import numpy as np
import pandas as pd
from src.utils.de_metrics import (
    compute_pds,
    compute_de_comparison_metrics,
    compute_mae_top2k,
    compute_pseudobulk_delta,
    compute_des,
)

TEST_DATA_PATH = "tests/data/metrics_test_data.h5ad"


@pytest.fixture(scope="module")
def adata_fixture():
    """Load the sampled test data."""
    try:
        adata = sc.read_h5ad(TEST_DATA_PATH)
    except FileNotFoundError:
        pytest.skip(
            f"Test data not found at {TEST_DATA_PATH}. Run scripts/sample_test_data.py first."
        )

    # Identify NTC (mocked in sampling script as the first one or named explicitly)
    # The sampling script uses 'target_gene'.
    # We need to know which one is NTC.
    # Based on the script output, 'AKT2' was used as mock NTC.
    # But ideally we dynamically find it or the test assumes known structure.
    # For robust testing with the mock data, let's just pick one as control randomly
    # if we can't match a standard name, but we know 'AKT2' is used as mock.

    # We'll use the first one in the list as NTC effectively.
    return adata


def get_pseudobulk(adata, pert_col="target_gene"):
    """Helper to compute pseudobulk for testing."""
    # Simple mean per group
    pseudobulks = {}
    for group in adata.obs[pert_col].unique():
        subset = adata[adata.obs[pert_col] == group]
        # Use X (log1p normalized)
        if hasattr(subset.X, "toarray"):
            expr = subset.X.toarray()
        else:
            expr = subset.X
        pseudobulks[group] = expr.mean(axis=0)
    return pseudobulks


def test_compute_pds(adata_fixture):
    """Test Perturbation Discrimination Score (PDS)."""
    adata = adata_fixture
    pert_col = "target_gene"

    # Identify control (mock)
    conditions = list(adata.obs[pert_col].unique())
    # Assume first is NTC for this test setup
    ntc_name = conditions[0]
    perturbations = conditions[1:]

    if len(perturbations) < 1:
        pytest.skip("Not enough perturbations for PDS test.")

    # Compute deltas
    pseudobulks = get_pseudobulk(adata, pert_col)
    ntc_expr = pseudobulks[ntc_name]

    truth_deltas = {}
    pred_deltas = {}

    # For PDS, we need predicted and truth.
    # To test the metric, we can mock predictions.
    # 1. Perfect prediction: pred == truth -> Should have best rank (1) and high cosine
    # 2. Noisy prediction: pred = truth + noise

    for p in perturbations:
        truth = pseudobulks[p] - ntc_expr
        truth_deltas[p] = truth

        # Create a "good" prediction (closely dependent on truth)
        # Add small noise
        noise = np.random.normal(0, 0.01, size=truth.shape)
        pred_deltas[p] = truth + noise

    # Add a "bad" prediction for one to see if rank drops?
    # But PDS computes rank of *true* perturbation for a *given* prediction.
    # If prediction is perfect effectively, it should be closest to its own truth.

    # Compute PDS
    results = compute_pds(pred_deltas, truth_deltas)

    assert "mean_rank" in results
    assert "npds" in results
    assert "ranks" in results

    # With reliable predictions (truth + small noise), ranks should be 1 (perfect discrimination)
    # assuming the perturbations are distinct enough.
    # Note: If perturbations are very similar, rank might not be 1.
    print(f"PDS Ranks: {results['ranks']}")

    # Check range
    assert 1 <= results["mean_rank"] <= len(perturbations)
    assert 0 <= results["npds"] <= 1.0

    # Check self cosine similarity - should be high for good predictions
    for p, sim in results["cosine_self"].items():
        assert sim > 0.9  # given we added only small noise


def test_compute_des(adata_fixture):
    """Test Differential Expression Score (DES)."""
    adata = adata_fixture
    pert_col = "target_gene"
    conditions = list(adata.obs[pert_col].unique())
    ntc_name = conditions[0]
    pert_name = conditions[1]

    # Get expression logic
    def get_expr(name):
        subset = adata[adata.obs[pert_col] == name]
        if hasattr(subset.X, "toarray"):
            return subset.X.toarray()
        return subset.X

    control_expr = get_expr(ntc_name)
    truth_expr = get_expr(pert_name)  # Treating real data as truth

    # Create mock prediction:
    # 1. Perfect prediction: pred = truth
    # 2. Or shifted prediction
    pred_expr = truth_expr.copy()

    # Gene names
    gene_names = np.array(adata.var_names)

    # Test DES function
    # Note: this uses Wilcoxon test which might be slow or sensitive to sample size.
    # We sampled 50 cells, should be enough for some DE.

    results = compute_de_comparison_metrics(
        control_expr=control_expr,
        pred_expr=pred_expr,
        truth_expr=truth_expr,
        gene_names=gene_names,
    )

    assert "des" in results
    des = results["des"]

    # Since pred == truth, DES should be 1.0 (or close to it, if DE set is empty or something)
    # If no DE genes are found, our current implementation returns 0.0, 0 intersect.

    n_de_truth = results["n_de_truth"]
    if n_de_truth > 0:
        assert des == 1.0
        assert results["n_intersect"] == n_de_truth
    else:
        # If no DE genes found (possible with small sample/weak perturbation), DES is 0.
        assert des == 0.0


def test_compute_mae_top2k(adata_fixture):
    """Test MAE top 2k metric."""
    adata = adata_fixture
    pert_col = "target_gene"
    conditions = list(adata.obs[pert_col].unique())
    ntc_name = conditions[0]
    pert_name = conditions[1]

    def get_expr(name):
        subset = adata[adata.obs[pert_col] == name]
        if hasattr(subset.X, "toarray"):
            return subset.X.toarray()
        return subset.X

    control_expr = get_expr(ntc_name)
    truth_expr = get_expr(pert_name)

    # Mock prediction: truth + constant error
    error_val = 0.5
    pred_expr = truth_expr + error_val

    # Control mean needed for ranking genes
    control_mean = control_expr.mean(axis=0)

    mae = compute_mae_top2k(
        pred_expr=pred_expr,
        truth_expr=truth_expr,
        control_mean=control_mean,
        top_k=100,  # use smaller k for test
    )

    # MAE should be exactly error_val (since we added constant error to all genes,
    # meaningful of subset selection doesn't change the error value for those selected)
    assert np.isclose(mae, error_val, atol=1e-5)


def test_compute_overall_score():
    """Test the aggregation logic."""
    from src.utils.de_metrics import (
        compute_overall_score,
        BASELINE_PDS,
        BASELINE_DES,
        BASELINE_MAE_TOP2000,
    )

    # Case 1: Perfect scores compared to baseline
    # PDS (rank based): normalized score. Lower rank is better.
    # In `compute_pds`: nPDS = MeanRank / N.
    # But `compute_overall_score` expects input `pds` as "1 - nPDS" ?
    # Let's check `compute_overall_score` docstring in source:
    # "pds: PDS score (1 - npds, higher is better, range 0-1)"

    # So if MeanRank = 1, N=2 -> nPDS = 0.5. Input pds = 0.5.
    # If MeanRank = 1, N=100 -> nPDS = 0.01. Input pds = 0.99.

    # Let's assume a good model:
    input_pds = 0.9
    input_mae = 0.05  # Lower than baseline 0.1258
    input_des = 0.8  # Higher than baseline 0.0442

    scores = compute_overall_score(input_pds, input_mae, input_des)

    assert scores["pds_scaled"] > 0
    assert scores["mae_scaled"] > 0
    assert scores["des_scaled"] > 0

    # Case 2: Worse than baseline (should clip to 0)
    scores_bad = compute_overall_score(
        pds=BASELINE_PDS - 0.1, mae=BASELINE_MAE_TOP2000 + 0.1, des=BASELINE_DES - 0.01
    )
    assert scores_bad["pds_scaled"] == 0.0
    assert scores_bad["mae_scaled"] == 0.0
    assert scores_bad["des_scaled"] == 0.0
    assert scores_bad["overall_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
