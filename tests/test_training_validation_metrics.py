import sys
import types
from pathlib import Path

import pytest
import anndata as ad
import numpy as np

# Skip when heavy dependencies are missing (e.g., CI/lightweight envs)
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

# Ensure project root is importable and stub pdex (optional dependency not needed here)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "pdex",
    types.SimpleNamespace(parallel_differential_expression=None),
)

from src.utils.training import compute_validation_metrics


def test_compute_validation_metrics_fast_des_fallback():
    gene_names = np.array(["g1", "g2"])

    # Simple control dataset (two control cells, two genes)
    ctrl_adata = ad.AnnData(
        X=np.array([[0.0, 0.0], [0.0, 0.0]]),
    )

    # Two perturbed cells for a single perturbation
    results = {
        "pert_cat": np.array(["g1", "g1"]),
        "pred": np.array([[2.0, 1.0], [2.0, 1.0]]),
        "truth": np.array([[2.0, 1.0], [2.0, 1.0]]),
    }

    config = {
        "metrics": {
            "compute_de_metrics": False,  # force fast DES fallback
            "mae_top_k": 2,
            "des_top_k": 2,
        }
    }

    metrics = compute_validation_metrics(results, ctrl_adata, gene_names, config)

    assert not np.isnan(metrics["des"])
    assert np.isclose(metrics["des"], 1.0)
