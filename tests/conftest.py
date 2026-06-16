"""Shared fixtures for sl_benchmark_baseline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_synthetic_benchmark(path: Path) -> Path:
    """Write a tiny CV1-shaped benchmark CSV with 2 folds and both classes.

    Each fold has 8 train rows (4 pos, 4 neg) and 6 test rows (3 pos, 3 neg).
    Gene effects are chosen so positives skew more negative (more essential).
    """
    rows = []
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(12)]
    pair_counter = 0
    for fold_id in (0, 1):
        for role, n_each in (("train", 4), ("test", 3)):
            for label in (1, 0):
                for _ in range(n_each):
                    a = genes[rng.integers(0, len(genes))]
                    b = genes[rng.integers(0, len(genes))]
                    while b == a:
                        b = genes[rng.integers(0, len(genes))]
                    # positives more essential (more negative gene effect)
                    base = -1.0 if label == 1 else 0.2
                    ea = base + rng.normal(0, 0.1)
                    eb = base + rng.normal(0, 0.1)
                    rows.append(
                        {
                            "pair_id": f"P{pair_counter}",
                            "fold_id": fold_id,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": a,
                            "gene_b_symbol": b,
                            "gene_a_k562_gene_effect": ea,
                            "gene_b_k562_gene_effect": eb,
                        }
                    )
                    pair_counter += 1
    frame = pd.DataFrame(rows)
    csv_path = path / "synthetic_sl.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_benchmark_csv(tmp_path: Path) -> Path:
    """Provide a path to a freshly written synthetic benchmark CSV."""
    return _write_synthetic_benchmark(tmp_path)
