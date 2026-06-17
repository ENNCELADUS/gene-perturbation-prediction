from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_run_cv_writes_outputs_and_aggregates(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "run"
    config = SLBaselineConfig(
        input_csv=synthetic_benchmark_csv,
        output_dir=output_dir,
        folds=(0, 1),
        ranking_k=(2, 5),
    )
    summary = run_cv(config)

    assert (output_dir / "fold_metrics.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "manifest.json").exists()

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["model"].unique()) == {"A", "B", "C"}
    assert set(fold_metrics["split_type"].unique()) == {"CV1"}
    assert set(fold_metrics["fold_id"].unique()) == {0, 1}
    assert {"split_type", "model", "fold_id", "metric", "value"}.issubset(
        fold_metrics.columns
    )
    metric_names = set(fold_metrics["metric"].unique())
    assert "f1@0.5" not in metric_names
    assert {"auroc", "aupr", "f1"}.issubset(metric_names)
    assert {"ndcg@2", "recall@2", "precision@2"}.issubset(metric_names)
    assert {"map@2", "map@5"}.issubset(metric_names)

    assert {"split_type", "model", "metric", "mean", "std"}.issubset(summary.columns)
    assert len(summary) == len(fold_metrics["metric"].unique()) * 3

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "input_csv_sha256" in manifest
    assert "leakage_notes" in manifest
    assert "ranking_semantics" in manifest
    assert "official_metric_source" in manifest
    assert "candidate_gene_count" in manifest
    assert manifest["split_types"] == ["CV1"]
    assert manifest["seed"] == config.seed


def test_run_cv_preserves_all_cv_split_boundaries(
    synthetic_all_cv_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "all_cv_run"
    config = SLBaselineConfig(
        input_csv=synthetic_all_cv_benchmark_csv,
        output_dir=output_dir,
        split_types=("CV1", "CV2"),
        folds=(0,),
        ranking_k=(2, 4),
    )
    summary = run_cv(config)

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["split_type"].unique()) == {"CV1", "CV2"}
    assert (fold_metrics.groupby(["split_type", "fold_id"]).size() > 0).all()
    assert {"CV1", "CV2"} == set(summary["split_type"].unique())
    assert not summary.duplicated(["split_type", "model", "metric"]).any()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["split_types"] == ["CV1", "CV2"]


def test_run_cv_rejects_missing_requested_split_type(
    synthetic_all_cv_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    config = SLBaselineConfig(
        input_csv=synthetic_all_cv_benchmark_csv,
        output_dir=tmp_path / "missing_split_run",
        split_types=("CV3",),
        folds=(0,),
    )

    try:
        run_cv(config)
    except ValueError as error:
        assert "split_types" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing split_type")


def test_build_gene_universe_populates_embeddings_and_coverage(
    synthetic_benchmark_csv: Path, synthetic_bags_npz: Path
) -> None:
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.embeddings import load_gene_embeddings
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    frame = load_benchmark(synthetic_benchmark_csv)
    table = load_gene_embeddings(synthetic_bags_npz)  # covers G0,G1,G2
    universe = _build_gene_universe(frame, embedding_table=table,
                                    fallback_strategy="zero")
    assert universe.embeddings is not None
    assert universe.embeddings.shape == (len(universe.symbols), table.dim)
    assert universe.coverage_mask.shape == (len(universe.symbols),)
    # G0/G1/G2 are in the synthetic benchmark gene pool G0..G11
    covered = {s for s, m in zip(universe.symbols, universe.coverage_mask) if m == 1.0}
    assert covered.issubset(set(universe.symbols))
    assert {"G0", "G1", "G2"}.issubset(covered)


def test_build_gene_universe_without_embeddings_is_none(
    synthetic_benchmark_csv: Path,
) -> None:
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.evaluate import _build_gene_universe

    frame = load_benchmark(synthetic_benchmark_csv)
    universe = _build_gene_universe(frame)
    assert universe.embeddings is None
    assert universe.coverage_mask is None


def test_augmented_score_matrix_is_square_with_zero_diagonal(
    synthetic_benchmark_csv: Path, synthetic_bags_npz: Path
) -> None:
    import numpy as np
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.data import load_benchmark
    from sl_benchmark_baseline.embeddings import load_gene_embeddings
    from sl_benchmark_baseline.evaluate import (
        _build_augmented_score_matrix,
        _build_gene_universe,
    )
    from sl_benchmark_baseline.features import (
        Standardizer,
        build_augmented_pair_features,
    )
    from sl_benchmark_baseline.models import LogRegTranscriptModel

    frame = load_benchmark(synthetic_benchmark_csv)
    table = load_gene_embeddings(synthetic_bags_npz)
    universe = _build_gene_universe(frame, embedding_table=table,
                                    fallback_strategy="zero")
    # build a tiny train feature set to fit the standardizer and a model
    ea = frame["gene_a_k562_gene_effect"].to_numpy()
    eb = frame["gene_b_k562_gene_effect"].to_numpy()
    dim = table.dim
    emb_a = np.zeros((len(frame), dim))
    emb_b = np.zeros((len(frame), dim))
    flag = np.ones(len(frame))
    raw = build_augmented_pair_features(ea, eb, emb_a, emb_b, flag, flag, True)
    standardizer = Standardizer.fit(raw)
    model = LogRegTranscriptModel(SLBaselineConfig())
    from sl_benchmark_baseline.models import FoldData
    model.fit(FoldData(df=frame, features=standardizer.transform(raw),
                       labels=frame["sl_label"].to_numpy(dtype=int)))
    matrix = _build_augmented_score_matrix(model, universe, standardizer, True)
    n = len(universe.symbols)
    assert matrix.shape == (n, n)
    assert np.allclose(np.diag(matrix), 0.0)


def test_augmented_run_cv_emits_transcript_models_and_covered_slice(
    synthetic_augmented_benchmark_csv: Path,
    synthetic_augmented_bags_npz: Path,
    tmp_path: Path,
) -> None:
    import json
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "aug_run"
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv,
        output_dir=output_dir,
        bags_npz=synthetic_augmented_bags_npz,
        folds=(0, 1),
        ranking_k=(2, 5),
    )
    summary = run_cv(config)

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    # baseline A/B + transcript variants; degree probe C excluded in augmented mode
    assert set(fold_metrics["model"].unique()) == {
        "A", "B", "A_transcript", "B_transcript"
    }
    assert "slice" in fold_metrics.columns
    slices = set(fold_metrics["slice"].unique())
    assert "full_universe" in slices
    assert "covered_pairs" in slices
    # covered_pairs slice only emitted for transcript models
    covered = fold_metrics[fold_metrics["slice"] == "covered_pairs"]
    assert set(covered["model"].unique()).issubset({"A_transcript", "B_transcript"})
    assert {"split_type", "model", "slice", "metric"}.issubset(summary.columns)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["augmented"] is True
    assert manifest["embedding_method"] == config.embedding_method


def test_nonaugmented_run_cv_unchanged_models(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "base_run"
    config = SLBaselineConfig(
        input_csv=synthetic_benchmark_csv, output_dir=output_dir,
        folds=(0, 1), ranking_k=(2, 5),
    )
    run_cv(config)
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    assert set(fold_metrics["model"].unique()) == {"A", "B", "C"}
    assert set(fold_metrics["slice"].unique()) == {"full_universe"}


def test_augmented_manifest_records_coverage_fields(
    synthetic_augmented_benchmark_csv: Path,
    synthetic_augmented_bags_npz: Path,
    tmp_path: Path,
) -> None:
    import json
    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "aug_manifest_run"
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv, output_dir=output_dir,
        bags_npz=synthetic_augmented_bags_npz, folds=(0,), ranking_k=(2, 5),
        fallback_strategy="global_mean", include_coverage_flag=False,
    )
    run_cv(config)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["augmented"] is True
    assert manifest["bags_npz"].endswith("synthetic_augmented_bags.npz")
    assert manifest["fallback_strategy"] == "global_mean"
    assert manifest["include_coverage_flag"] is False
    assert "gwps_coverage_gene_count" in manifest
    assert manifest["models"] == ["A", "B", "A_transcript", "B_transcript"]


def test_augmented_manifest_records_pair_coverage_fraction(
    synthetic_augmented_benchmark_csv: Path,
    synthetic_augmented_bags_npz: Path,
    tmp_path: Path,
) -> None:
    import json

    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "aug_pair_cov_run"
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv,
        output_dir=output_dir,
        bags_npz=synthetic_augmented_bags_npz,
        folds=(0,),
        ranking_k=(2, 5),
    )
    run_cv(config)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    # The augmented fixture has both both-covered pairs (C*,C*) and mixed pairs
    # (C*,U*), so the pair-level fraction is strictly between 0 and 1.
    assert manifest["gwps_coverage_pair_fraction"] is not None
    assert 0.0 < manifest["gwps_coverage_pair_fraction"] < 1.0


def test_nonaugmented_manifest_pair_fraction_is_none(
    synthetic_benchmark_csv: Path, tmp_path: Path
) -> None:
    import json

    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    output_dir = tmp_path / "base_pair_cov_run"
    config = SLBaselineConfig(
        input_csv=synthetic_benchmark_csv, output_dir=output_dir,
        folds=(0,), ranking_k=(2, 5),
    )
    run_cv(config)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["gwps_coverage_pair_fraction"] is None


def test_augmented_run_warns_on_zero_coverage_bags(
    synthetic_augmented_benchmark_csv: Path, tmp_path: Path, caplog
) -> None:
    """A bags NPZ disjoint from the benchmark universe must warn, not silently
    produce all-fallback transcript features."""
    import logging

    import numpy as np

    from sl_benchmark_baseline.config import SLBaselineConfig
    from sl_benchmark_baseline.evaluate import run_cv

    # Bags whose gene symbols do not intersect the benchmark (C*/U* genes).
    disjoint_npz = tmp_path / "disjoint_bags.npz"
    np.savez_compressed(
        disjoint_npz,
        cell_delta_pcs=np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        bag_offsets=np.array([0, 2], dtype=np.int64),
        perturbation_gene=np.asarray(["ZZZ_NOTREAL"], dtype=object),
    )
    config = SLBaselineConfig(
        input_csv=synthetic_augmented_benchmark_csv,
        output_dir=tmp_path / "zero_cov_run",
        bags_npz=disjoint_npz,
        folds=(0,),
        ranking_k=(2, 5),
    )
    with caplog.at_level(logging.WARNING):
        run_cv(config)
    assert any("Low gwps gene coverage" in rec.message for rec in caplog.records)
