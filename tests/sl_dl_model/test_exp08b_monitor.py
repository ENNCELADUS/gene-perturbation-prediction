from __future__ import annotations

import numpy as np

from sl_dl_model.exp08b_generator import (
    bag_energy_metric,
    compute_monitor_rows,
    nearest_neighbor_copy_predictions,
    pooled_vector_metrics,
)


def test_pooled_vector_metrics_report_direction_and_magnitude() -> None:
    pred = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    real = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    metrics = pooled_vector_metrics(pred, real)

    assert metrics["pooled_cosine"] > 0.999
    assert metrics["pooled_mse"] == 0.0
    assert metrics["pooled_l2"] == 0.0


def test_bag_energy_metric_is_zero_for_identical_bags() -> None:
    bag = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert bag_energy_metric(bag, bag) < 1e-6


def test_nearest_neighbor_copy_uses_train_covered_only() -> None:
    esm = {
        "TRAIN_A": np.array([1.0, 0.0], dtype=np.float32),
        "TRAIN_B": np.array([0.0, 1.0], dtype=np.float32),
        "VAL": np.array([0.9, 0.1], dtype=np.float32),
        "OUTSIDE": np.array([0.91, 0.09], dtype=np.float32),
    }
    bags = {
        "TRAIN_A": np.full((2, 2), 1.0, dtype=np.float32),
        "TRAIN_B": np.full((2, 2), 2.0, dtype=np.float32),
        "OUTSIDE": np.full((2, 2), 9.0, dtype=np.float32),
    }

    copied = nearest_neighbor_copy_predictions(
        val_symbols={"VAL"},
        train_covered_symbols={"TRAIN_A", "TRAIN_B"},
        esm_vectors=esm,
        real_bags=bags,
    )

    np.testing.assert_allclose(copied["VAL"], bags["TRAIN_A"])


def test_compute_monitor_rows_has_generator_and_nn_rows() -> None:
    pred_bags = {"VAL": np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}
    real_bags = {"VAL": np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}
    nn_bags = {"VAL": np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)}

    rows = compute_monitor_rows(
        epoch=2,
        split_type="CV2",
        fold_id=0,
        pred_bags=pred_bags,
        real_bags=real_bags,
        nn_copy_bags=nn_bags,
    )

    assert {row["predictor"] for row in rows} == {"generator", "esm2_nn_copy"}
    assert all(row["split_type"] == "CV2" for row in rows)
    assert all(row["fold_id"] == 0 for row in rows)
    assert all(row["epoch"] == 2 for row in rows)
    gen = [row for row in rows if row["predictor"] == "generator"][0]
    assert gen["pooled_cosine"] > 0.999
    assert gen["pooled_mse"] == 0.0
