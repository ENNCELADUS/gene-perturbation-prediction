from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sl_profile_baseline.data import load_feng_fold, load_profile_universe
from sl_profile_baseline.config import ProfileBaselineConfig
from sl_profile_baseline.evaluate import run_cv
from sl_profile_baseline.evaluate import _ranking_metrics_low_memory
from sl_benchmark_baseline.metrics import official_ranking_metrics
from sl_profile_baseline.features import (
    PROFILE_FEATURE_NAMES,
    build_profile_pair_features,
    build_raw_pair_features,
)


def _write_split(path: Path) -> None:
    positive = np.empty((4, 1), dtype=object)
    negative = np.empty((4, 1), dtype=object)
    for samples in (positive, negative):
        samples[0, 0] = np.zeros((3, 3), dtype=float)
        samples[1, 0] = np.zeros((3, 3), dtype=float)
    positive[2, 0] = np.array([[0, 1], [1, 2]], dtype=int)
    positive[3, 0] = np.array([[0, 2]], dtype=int)
    negative[2, 0] = np.array([[0, 2], [2, 1]], dtype=int)
    negative[3, 0] = np.array([[1, 2]], dtype=int)
    np.save(path, np.array([positive, negative], dtype=object))


def test_load_feng_fold_preserves_official_train_test_pairs(tmp_path: Path) -> None:
    split_path = tmp_path / "CV1_1.npy"
    _write_split(split_path)

    fold = load_feng_fold(split_path, fold_id=0)

    assert fold.train_pairs.tolist() == [[0, 1], [1, 2], [0, 2], [2, 1]]
    assert fold.train_labels.tolist() == [1, 1, 0, 0]
    assert fold.test_pairs.tolist() == [[0, 2], [1, 2]]
    assert fold.test_labels.tolist() == [1, 0]


def test_load_profile_universe_maps_symbols_and_tracks_coverage(
    tmp_path: Path,
) -> None:
    entities = pd.DataFrame(
        {
            "unified_id": [0, 1, 2],
            "entity_type_name": ["Gene", "Gene", "Gene"],
            "entity_name": ["A", "B", "C"],
        }
    )
    entities_path = tmp_path / "fin_entities.csv"
    entities.to_csv(entities_path, index=False)
    gene_effect = pd.DataFrame(
        {
            "A (1)": [-1.0, -0.8, np.nan],
            "B (2)": [0.0, -0.2, -0.1],
        },
        index=["CL1", "CL2", "CL3"],
    )
    gene_effect_path = tmp_path / "CRISPRGeneEffect.csv"
    gene_effect.to_csv(gene_effect_path)

    universe = load_profile_universe(
        entities_path,
        gene_effect_path,
        candidate_gene_count=3,
        min_finite_lines=2,
        dependency_threshold=-0.5,
    )

    assert universe.symbols == ("A", "B", "C")
    assert universe.profiles.shape == (3, 3)
    assert universe.coverage.tolist() == [True, True, False]
    assert universe.finite_counts.tolist() == [2, 3, 0]
    assert np.isfinite(universe.profiles).all()
    assert universe.profiles[2].tolist() == [0.0, 0.0, 0.0]


def test_profile_features_are_swap_invariant_and_finite() -> None:
    profiles = np.array(
        [
            [-1.0, -0.5, 0.0, 0.2],
            [0.0, -0.2, -0.8, 0.1],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    coverage = np.array([True, True, False])
    pairs = np.array([[0, 1], [2, 0]], dtype=int)

    forward = build_profile_pair_features(
        profiles, coverage, pairs, dependency_threshold=-0.5
    )
    reverse = build_profile_pair_features(
        profiles, coverage, pairs[:, ::-1], dependency_threshold=-0.5
    )

    assert forward.shape == (2, len(PROFILE_FEATURE_NAMES))
    np.testing.assert_allclose(forward, reverse, atol=1e-6)
    assert np.isfinite(forward).all()
    assert forward[0, PROFILE_FEATURE_NAMES.index("coverage_both")] == 1.0
    assert forward[1, PROFILE_FEATURE_NAMES.index("coverage_both")] == 0.0


def test_raw_features_are_swap_invariant() -> None:
    profiles = np.array(
        [[-1.0, 0.0, 0.5], [0.2, -0.4, 0.5]], dtype=np.float32
    )
    pairs = np.array([[0, 1]], dtype=int)

    forward = build_raw_pair_features(profiles, pairs)
    reverse = build_raw_pair_features(profiles, pairs[:, ::-1])

    np.testing.assert_allclose(forward, reverse)
    np.testing.assert_allclose(
        forward,
        np.array([[-0.8, -0.4, 1.0, 1.2, 0.4, 0.0]], dtype=np.float32),
    )


def test_run_cv_writes_reproducible_artifacts(tmp_path: Path) -> None:
    sl_root = tmp_path / "sl"
    split_dir = sl_root / "data" / "data_split"
    split_dir.mkdir(parents=True)
    _write_split(split_dir / "CV1_1.npy")
    pd.DataFrame(
        {
            "unified_id": [0, 1, 2],
            "entity_type_name": ["Gene", "Gene", "Gene"],
            "entity_name": ["A", "B", "C"],
        }
    ).to_csv(sl_root / "data" / "fin_entities.csv", index=False)
    gene_effect_path = tmp_path / "CRISPRGeneEffect.csv"
    pd.DataFrame(
        {
            "A (1)": [-1.0, -0.8, 0.0, 0.1],
            "B (2)": [-0.9, -0.7, 0.1, 0.0],
            "C (3)": [0.0, -0.1, -0.8, -0.9],
        },
        index=["CL1", "CL2", "CL3", "CL4"],
    ).to_csv(gene_effect_path)
    output_dir = tmp_path / "out"
    config = ProfileBaselineConfig(
        sl_root=sl_root,
        gene_effect_csv=gene_effect_path,
        output_dir=output_dir,
        split_types=("CV1",),
        folds=(0,),
        ranking_k=(1, 2),
        models=("pearson_abs", "summary_logreg"),
        ranking_models=("pearson_abs", "summary_logreg"),
        min_finite_lines=2,
        score_matrix_chunk_rows=2,
    )

    summary = run_cv(config)

    assert not summary.empty
    assert set(summary["model"]) == {"pearson_abs", "summary_logreg"}
    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "slice_counts.csv").exists()
    assert (output_dir / "gene_mapping.csv").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["entities_csv"].endswith("fin_entities.csv")
    assert len(manifest["entities_sha256"]) == 64
    assert "DepMap GeneEffect profiles remain available" in manifest[
        "cv2_cv3_auxiliary_exposure"
    ]


def test_low_memory_ranking_matches_official_metrics() -> None:
    matrix = np.array(
        [
            [0.0, 0.9, 0.3, 0.2],
            [0.9, 0.0, 0.8, 0.1],
            [0.3, 0.8, 0.0, 0.7],
            [0.2, 0.1, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    positives = np.array([[0, 2], [1, 3]], dtype=int)
    seen = np.array([[0, 1]], dtype=int)

    expected = official_ranking_metrics(matrix, positives, seen, (1, 2))
    observed = _ranking_metrics_low_memory(matrix.copy(), positives, seen, (1, 2))

    assert observed == expected
