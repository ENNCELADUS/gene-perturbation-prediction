from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dependency_baseline.config import (
    BaselineConfig,
    CvConfig,
    DataConfig,
    ExternalEvaluationConfig,
    FeatureConfig,
    SelectionConfig,
    ViabilityAxisArtifactConfig,
    ViabilityAxisConfig,
)
from dependency_baseline.artifacts import organize_artifacts
from dependency_baseline.evaluation import (
    fit_final,
    regression_metrics,
    run_cv,
    summarize_results,
)
from dependency_baseline.features import build_features
from dependency_baseline.models import (
    NuisanceResidualizer,
    ResidualizedPCAWithScores,
    ViabilityAxisResidualizer,
    build_model_specs,
    compatible_model_feature_shape,
    fit_estimator,
)
from dependency_baseline.program_scores import build_program_scores
from dependency_baseline.viability_axis import (
    file_sha256,
    parse_coefficient_csv,
)


def test_build_features_and_run_quick_cv(tmp_path: Path) -> None:
    h5ad_path, overlap_path = _write_synthetic_replogle_inputs(tmp_path)
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        ),
        features=FeatureConfig(chunk_size=4, top_abs_delta_sizes=(2, 3)),
        cv=CvConfig(
            n_splits=2,
            n_repeats=1,
            random_state=7,
            stratify_bins=3,
            pca_components=(2,),
            model_set="quick",
        ),
    )

    feature_paths = build_features(config)
    assert feature_paths.features_npz.exists()
    assert feature_paths.metadata_path.exists()
    assert feature_paths.qa_report_md.exists()
    assert feature_paths.features_npz.parent.name == "features"
    assert feature_paths.metadata_path.name == "feature_metadata.parquet"

    metadata = pd.read_parquet(feature_paths.metadata_path)
    assert len(metadata) == 6
    assert metadata["depmap_gene_effect"].notna().all()
    assert metadata["observed_n_cells"].min() == 3

    feature_data = np.load(feature_paths.features_npz, allow_pickle=True)
    assert feature_data["delta"].shape == (6, 5)
    assert feature_data["response_burden"].shape[0] == 6
    assert feature_data["program_scores"].shape[0] == 6

    external_features_npz = tmp_path / "synthetic_external_features.npz"
    np.savez_compressed(
        external_features_npz,
        delta=feature_data["delta"][:4],
        response_burden=feature_data["response_burden"][:4],
        program_scores=feature_data["program_scores"][:4],
        program_score_columns=feature_data["program_score_columns"],
        y=feature_data["y"][:4],
        n_cells=feature_data["n_cells"][:4],
        target_gene_index=feature_data["target_gene_index"][:4],
        perturbation_gene=feature_data["perturbation_gene"][:4],
    )
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
            external_evaluations=(
                ExternalEvaluationConfig(
                    name="synthetic_holdout",
                    features_npz=external_features_npz,
                ),
            ),
        ),
        features=config.features,
        cv=config.cv,
    )

    cv_paths = run_cv(config)
    assert cv_paths.run_dir.exists()
    assert cv_paths.manifest_json.exists()
    assert cv_paths.config_json.exists()
    assert cv_paths.log_file.exists()
    assert cv_paths.splits_path.exists()
    assert cv_paths.model_manifest_path.exists()
    assert cv_paths.topk_candidates_path.exists()
    assert cv_paths.summary_csv == cv_paths.run_dir / "results" / "summary_metrics.csv"
    assert cv_paths.summary_csv.exists()
    assert cv_paths.fold_metrics_path == (
        cv_paths.run_dir / "artifacts" / "fold_metrics.parquet"
    )
    assert cv_paths.predictions_path == (
        cv_paths.run_dir / "artifacts" / "predictions.parquet"
    )
    assert cv_paths.fold_metrics_path.exists()
    assert cv_paths.predictions_path.exists()
    assert not (cv_paths.run_dir / "fold_metrics.csv").exists()
    assert not (cv_paths.run_dir / "predictions.csv").exists()
    assert not (cv_paths.run_dir / "summary_metrics.csv").exists()
    summary = pd.read_csv(cv_paths.summary_csv)
    fold_metrics = pd.read_parquet(cv_paths.fold_metrics_path)
    predictions = pd.read_parquet(cv_paths.predictions_path)
    splits = pd.read_parquet(cv_paths.splits_path)
    model_manifest = pd.read_parquet(cv_paths.model_manifest_path)
    topk_candidates = pd.read_parquet(cv_paths.topk_candidates_path)
    assert {
        "internal_cv_all",
        "internal_cv_target_index_valid",
        "external:synthetic_holdout",
    }.issubset(set(summary["evaluation_scope"]))
    assert {"delta_all", "delta_mask_target", "n_cells_only"}.issubset(
        set(summary["feature_set"])
    )
    assert {"mean_label", "ridge"}.issubset(set(summary["model"]))
    assert {"unweighted", "sqrt_n_cells"}.issubset(set(summary["weighting"]))
    target_valid_summary = summary.loc[
        summary["evaluation_scope"] == "internal_cv_target_index_valid"
    ]
    assert set(target_valid_summary["feature_set"]) == {
        "delta_all",
        "delta_mask_target",
    }
    target_valid_predictions = predictions.loc[
        predictions["evaluation_scope"] == "internal_cv_target_index_valid"
    ]
    assert (target_valid_predictions["target_gene_index"] >= 0).all()
    assert {"fit_seconds", "spearman_defined", "pearson_defined"}.issubset(
        set(fold_metrics.columns)
    )
    assert fold_metrics["fit_seconds"].ge(0).all()
    mean_label_rows = fold_metrics["model"] == "mean_label"
    assert not fold_metrics.loc[mean_label_rows, "spearman_defined"].any()
    assert not fold_metrics.loc[mean_label_rows, "pearson_defined"].any()
    assert not predictions.empty
    assert not splits.empty
    checkpoint_exists = model_manifest["checkpoint_path"].map(
        lambda value: Path(value).exists()
    )
    assert checkpoint_exists.all()
    assert {"rank", "predicted_dependency_score", "top_k"}.issubset(
        set(topk_candidates.columns)
    )
    test_splits = splits.loc[splits["split"] == "test"]
    merged = predictions.loc[
        predictions["evaluation_scope"].str.startswith("internal_cv")
    ].merge(
        test_splits,
        on=["evaluation_scope", "fold", "perturbation_gene"],
        how="left",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()

    single_paths = run_cv(
        config,
        run_id="single_job",
        selection=SelectionConfig(
            scopes=("internal_cv_all",),
            features=("delta_all",),
            models=("ridge",),
            folds=(0,),
            weightings=("unweighted",),
        ),
    )
    single_metrics = pd.read_parquet(single_paths.fold_metrics_path)
    assert len(single_metrics) == 2
    internal_single = single_metrics.loc[
        single_metrics["evaluation_scope"] == "internal_cv_all"
    ]
    assert len(internal_single) == 1
    assert internal_single.iloc[0]["model"] == "ridge"
    assert internal_single.iloc[0]["feature_set"] == "delta_all"
    assert internal_single.iloc[0]["fold"] == 0

    resumed_paths = run_cv(
        config,
        run_id="single_job",
        resume=True,
        selection=SelectionConfig(
            scopes=("internal_cv_all",),
            features=("delta_all",),
            models=("ridge",),
            folds=(0,),
            weightings=("unweighted",),
        ),
    )
    resumed_metrics = pd.read_parquet(resumed_paths.fold_metrics_path)
    assert len(resumed_metrics) == len(single_metrics)

    pca_forest_config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        ),
        features=config.features,
        cv=config.cv,
        models={
            "mean_label": {"enabled": False},
            "ridge": {"enabled": False},
            "elastic_net": {"enabled": False},
            "pca_ridge": {"enabled": False},
            "random_forest": {"enabled": False},
            "pca_random_forest": {
                "enabled": True,
                "components": [2],
                "n_estimators": 5,
                "min_samples_leaf": 1,
                "n_jobs": 1,
            },
            "xgboost": {"enabled": False},
        },
    )
    pca_forest_specs = build_model_specs(pca_forest_config)
    assert [spec.name for spec in pca_forest_specs] == ["pca2_random_forest"]
    assert compatible_model_feature_shape(
        "pca2_random_forest",
        "delta_all",
        (3, 5),
    )
    assert not compatible_model_feature_shape(
        "pca2_random_forest",
        "response_burden",
        (3, 6),
    )
    pca_forest_paths = run_cv(
        pca_forest_config,
        feature_paths.features_npz,
        run_id="pca_forest_job",
        selection=SelectionConfig(
            scopes=("internal_cv_all",),
            features=("delta_all",),
            models=("pca2_random_forest",),
            folds=(0,),
            weightings=("unweighted",),
        ),
    )
    pca_forest_metrics = pd.read_parquet(pca_forest_paths.fold_metrics_path)
    assert len(pca_forest_metrics) == 1
    assert pca_forest_metrics.iloc[0]["model"] == "pca2_random_forest"
    assert pca_forest_metrics.iloc[0]["feature_set"] == "delta_all"

    final_paths = fit_final(
        config,
        run_id="final_job",
        selection=SelectionConfig(
            features=("delta_all",),
            models=("ridge",),
            weightings=("unweighted",),
        ),
    )
    assert final_paths.log_file.exists()
    final_manifest = pd.read_parquet(final_paths.final_model_manifest_path)
    final_rankings = pd.read_parquet(final_paths.final_rankings_path)
    assert len(final_manifest) == 1
    assert Path(final_manifest.iloc[0]["checkpoint_path"]).exists()
    assert {"rank", "predicted_dependency_score"}.issubset(set(final_rankings.columns))

    summary_path, ranking_summary_path = summarize_results(cv_paths.run_dir)
    assert summary_path == cv_paths.summary_csv
    assert ranking_summary_path == (
        cv_paths.run_dir / "artifacts" / "ranking_summary.parquet"
    )
    assert summary_path.exists()
    assert ranking_summary_path.exists()


def test_regression_metrics_skip_correlation_for_constant_predictions() -> None:
    metrics = regression_metrics(
        np.asarray([-2.0, -1.0, 0.0]),
        np.asarray([0.5, 0.5, 0.5]),
    )

    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])
    assert metrics["spearman_defined"] is False
    assert metrics["pearson_defined"] is False


def test_nar_coefficient_parsing_and_feature_scoring(tmp_path: Path) -> None:
    coefficient_path = tmp_path / "nar_local.csv"
    coefficient_path.write_text(
        ",pr_gene_symbol,coefficient\n1,GENE1,2.0\n2,HOUSE,-1.0\nINTERCEPT,0.5,0.5\n",
        encoding="utf-8",
    )
    coefficients, intercept = parse_coefficient_csv(coefficient_path)
    assert intercept == 0.5
    assert coefficients.to_dict() == {"GENE1": 2.0, "HOUSE": -1.0}

    h5ad_path, overlap_path = _write_synthetic_replogle_inputs(tmp_path)
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        ),
        features=FeatureConfig(chunk_size=4, top_abs_delta_sizes=(2, 3)),
        cv=CvConfig(n_splits=2, n_repeats=1, pca_components=(2,)),
        viability_axis=ViabilityAxisConfig(
            enabled=True,
            artifacts=(
                ViabilityAxisArtifactConfig(
                    name="local",
                    url=str(coefficient_path),
                    sha256=file_sha256(coefficient_path),
                ),
            ),
        ),
    )

    feature_paths = build_features(config)
    feature_data = np.load(feature_paths.features_npz, allow_pickle=True)
    assert feature_data["nar_viability_scores"].shape == (6, 1)
    assert feature_data["nar_viability_score_columns"].tolist() == ["nar_local_score"]
    summary = json.loads(feature_paths.summary_json.read_text(encoding="utf-8"))
    model_summary = summary["viability_axis_models"][0]
    assert model_summary["n_matched_expression_genes"] == 2
    assert model_summary["n_missing_expression_genes"] == 0


def test_viability_axis_residualizer_removes_score_linear_signal() -> None:
    score = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    delta = np.column_stack(
        [
            10.0 + 2.0 * score[:, 0],
            -1.0 - 3.0 * score[:, 0],
            np.asarray([1.0, 1.0, 1.0, 1.0]),
        ]
    ).astype(np.float32)
    x = np.hstack([delta, score])

    residualizer = ViabilityAxisResidualizer(n_score_columns=1).fit(x)
    residual = residualizer.transform(x)

    assert np.allclose(residual[:, :2], 0.0, atol=1e-5)
    assert np.allclose(residual[:, 2], 0.0, atol=1e-5)


def test_program_scores_align_symbols_and_report_missing_genes() -> None:
    delta = np.asarray(
        [
            [1.0, 2.0, -1.0],
            [3.0, 4.0, -2.0],
        ],
        dtype=np.float32,
    )
    result = build_program_scores(
        delta=delta,
        gene_symbols=["E2F1", "MCM2", "HOUSE"],
        program_sets=("cell_cycle_e2f",),
    )

    assert result.score_columns == ("program_cell_cycle_e2f_mean_delta",)
    assert np.allclose(result.scores.iloc[:, 0].to_numpy(), [1.5, 3.5])
    assert result.qa_rows[0]["n_matched_expression_genes"] == 2
    assert result.qa_rows[0]["n_missing_expression_genes"] > 0


def test_residualized_pca_with_scores_appends_nuisance_scores() -> None:
    score = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    delta = np.column_stack(
        [
            10.0 + 2.0 * score[:, 0],
            np.asarray([0.0, 1.0, 0.0, 1.0]),
        ]
    ).astype(np.float32)
    x = np.hstack([delta, score])

    transformed = ResidualizedPCAWithScores(
        n_score_columns=1,
        n_components=1,
        random_state=7,
    ).fit_transform(x)

    assert transformed.shape == (4, 2)
    assert np.allclose(transformed[:, -1], score[:, 0])
    assert np.allclose(
        NuisanceResidualizer(n_score_columns=1).fit(x).transform(x)[:, 0],
        0.0,
        atol=1e-5,
    )


def test_signal_decomposition_model_specs_fit_synthetic_data(tmp_path: Path) -> None:
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=tmp_path / "missing.h5ad",
            overlap_csv=tmp_path / "missing.csv",
            output_dir=tmp_path / "outputs",
        ),
        features=FeatureConfig(),
        cv=CvConfig(n_splits=2, n_repeats=1, pca_components=(2,)),
        models={
            "mean_label": {"enabled": False},
            "ridge": {"enabled": False},
            "elastic_net": {"enabled": False},
            "pca_ridge": {"enabled": False},
            "random_forest": {"enabled": False},
            "pca_random_forest": {"enabled": False},
            "xgboost": {"enabled": False},
            "nar_viability_axis": {"enabled": False},
            "signal_decomposition": {
                "enabled": True,
                "n_score_columns": 2,
                "pca_components": 2,
                "n_estimators": 5,
                "min_samples_leaf": 1,
                "n_jobs": 1,
            },
        },
    )
    specs = build_model_specs(config)
    spec_by_name = {spec.name: spec for spec in specs}
    assert {
        "nuisance_score_ridge",
        "nuisance_resid_pca2_ridge",
        "nuisance_resid_pca2_random_forest",
        "nuisance_resid_pca2_plus_scores_ridge",
        "nuisance_resid_pca2_plus_scores_random_forest",
        "nuisance_resid_lasso",
        "program_score_ridge",
        "program_score_elastic_net",
        "program_score_random_forest",
    }.issubset(spec_by_name)

    rng = np.random.default_rng(7)
    nuisance_x = rng.normal(size=(8, 5)).astype(np.float32)
    residual_x = rng.normal(size=(8, 7)).astype(np.float32)
    program_x = rng.normal(size=(8, 4)).astype(np.float32)
    y = rng.normal(size=8)
    weights = np.ones(8)
    fit_inputs = {
        "nuisance_score_ridge": nuisance_x,
        "nuisance_resid_pca2_ridge": residual_x,
        "nuisance_resid_pca2_random_forest": residual_x,
        "nuisance_resid_pca2_plus_scores_ridge": residual_x,
        "nuisance_resid_pca2_plus_scores_random_forest": residual_x,
        "nuisance_resid_lasso": residual_x,
        "program_score_ridge": program_x,
        "program_score_elastic_net": program_x,
        "program_score_random_forest": program_x,
    }
    for model_name, x in fit_inputs.items():
        fitted, fit_seconds = fit_estimator(
            spec_by_name[model_name],
            x,
            y,
            weights,
            "unweighted",
        )
        assert fit_seconds >= 0
        assert fitted.predict(x).shape == (8,)


def test_organize_artifacts_migrates_legacy_layout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame({"feature_row": [0], "perturbation_gene": ["GENE1"]}).to_csv(
        results_dir / "replogle_k562_feature_metadata.csv",
        index=False,
    )
    np.savez_compressed(results_dir / "replogle_k562_delta_features.npz", x=[1])
    (results_dir / "replogle_k562_feature_qa.md").write_text("qa", encoding="utf-8")
    (results_dir / "replogle_k562_feature_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    run_dir = results_dir / "runs" / "legacy_run"
    rankings_dir = run_dir / "rankings"
    rankings_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "job_key": ["job"],
            "evaluation_scope": ["internal_cv_all"],
            "fold": [0],
            "feature_set": ["delta_all"],
            "model": ["ridge"],
            "weighting": ["unweighted"],
            "spearman": [0.1],
        }
    ).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "evaluation_scope": ["internal_cv_all"],
            "feature_set": ["delta_all"],
            "model": ["ridge"],
            "weighting": ["unweighted"],
            "n_folds": [1],
            "spearman_mean": [0.1],
        }
    ).to_csv(run_dir / "summary_metrics.csv", index=False)
    pd.DataFrame({"job_key": ["job"], "perturbation_gene": ["GENE1"]}).to_csv(
        run_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame({"rank": [1], "perturbation_gene": ["GENE1"]}).to_csv(
        rankings_dir / "internal_cv_all__delta_all__ridge__unweighted.csv",
        index=False,
    )
    (run_dir / "completed_jobs.jsonl").write_text(
        '{"job_key": "job"}\n',
        encoding="utf-8",
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "legacy_run.log").write_text("log", encoding="utf-8")

    organize_artifacts(results_dir, logs_dir)

    assert (results_dir / "features" / "replogle_k562_delta_features.npz").exists()
    assert (results_dir / "features" / "feature_metadata.parquet").exists()
    assert not (results_dir / "replogle_k562_feature_metadata.csv").exists()
    assert (run_dir / "results" / "summary_metrics.csv").exists()
    assert (run_dir / "artifacts" / "fold_metrics.parquet").exists()
    assert (run_dir / "artifacts" / "predictions.parquet").exists()
    assert (run_dir / "artifacts" / "completed_jobs.jsonl").exists()
    assert (
        run_dir
        / "artifacts"
        / "rankings"
        / "internal_cv_all__delta_all__ridge__unweighted.parquet"
    ).exists()
    assert (run_dir / "logs" / "run.log").read_text(encoding="utf-8") == "log"
    assert not (run_dir / "fold_metrics.csv").exists()
    assert not (run_dir / "predictions.csv").exists()
    legacy_ranking_csv = (
        rankings_dir / "internal_cv_all__delta_all__ridge__unweighted.csv"
    )
    assert not legacy_ranking_csv.exists()


def _write_synthetic_replogle_inputs(tmp_path: Path) -> tuple[Path, Path]:
    genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"]
    expression_symbols = ["GENE1", "GENE2", "GENE3", "GENE4", "HOUSE"]
    labels = ["non-targeting"] * 4
    rows = [np.ones(5, dtype=np.float32) for _ in range(4)]
    effects = {
        "GENE1": -2.0,
        "GENE2": -1.5,
        "GENE3": -1.0,
        "GENE4": -0.5,
        "GENE5": 0.0,
        "GENE6": 0.2,
    }
    for index, gene in enumerate(genes):
        for _ in range(3):
            labels.append(gene)
            signal = abs(effects[gene])
            row = np.ones(5, dtype=np.float32)
            row[index % 4] += signal
            row[4] += signal / 2
            rows.append(row)

    obs = pd.DataFrame({"gene": labels, "cell_line": "k562"})
    var = pd.DataFrame(
        {"gene_name": expression_symbols}, index=[f"ENSG{i}" for i in range(5)]
    )
    adata = ad.AnnData(X=np.vstack(rows), obs=obs, var=var)
    h5ad_path = tmp_path / "synthetic_replogle.h5ad"
    adata.write_h5ad(h5ad_path)

    overlap = pd.DataFrame(
        {
            "perturbation_gene": ["non-targeting", *genes, "MATCHED_NAN", "UNMATCHED"],
            "depmap_gene_column": [
                np.nan,
                *(f"{gene} ({i})" for i, gene in enumerate(genes)),
                "MATCHED_NAN (999)",
                np.nan,
            ],
            "has_depmap_label": [False, *([True] * len(genes)), True, False],
            "depmap_gene_effect": [
                np.nan,
                *(effects[gene] for gene in genes),
                np.nan,
                np.nan,
            ],
            "n_cells_or_pseudobulk": [4, *([3] * len(genes)), 1, 1],
            "is_control_candidate": [True, *([False] * len(genes)), False, False],
            "modality": "CRISPRi",
            "source_dataset": "synthetic",
        }
    )
    overlap_path = tmp_path / "overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    return h5ad_path, overlap_path
