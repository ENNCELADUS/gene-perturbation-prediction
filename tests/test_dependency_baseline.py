from __future__ import annotations

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
)
from dependency_baseline.evaluation import fit_final, regression_metrics, run_cv
from dependency_baseline.features import build_features
from dependency_baseline.models import (
    build_model_specs,
    compatible_model_feature_shape,
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
    assert feature_paths.metadata_csv.exists()
    assert feature_paths.qa_report_md.exists()

    metadata = pd.read_csv(feature_paths.metadata_csv)
    assert len(metadata) == 6
    assert metadata["depmap_gene_effect"].notna().all()
    assert metadata["observed_n_cells"].min() == 3

    feature_data = np.load(feature_paths.features_npz, allow_pickle=True)
    assert feature_data["delta"].shape == (6, 5)
    assert feature_data["response_burden"].shape[0] == 6

    external_features_npz = tmp_path / "synthetic_external_features.npz"
    np.savez_compressed(
        external_features_npz,
        delta=feature_data["delta"][:4],
        response_burden=feature_data["response_burden"][:4],
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
    assert cv_paths.splits_csv.exists()
    assert cv_paths.model_manifest_csv.exists()
    assert cv_paths.topk_candidates_csv.exists()
    assert (cv_paths.run_dir / "fold_metrics.parquet").exists()
    assert (cv_paths.run_dir / "predictions.parquet").exists()
    summary = pd.read_csv(cv_paths.summary_csv)
    fold_metrics = pd.read_csv(cv_paths.fold_metrics_csv)
    predictions = pd.read_csv(cv_paths.predictions_csv)
    splits = pd.read_csv(cv_paths.splits_csv)
    model_manifest = pd.read_csv(cv_paths.model_manifest_csv)
    topk_candidates = pd.read_csv(cv_paths.topk_candidates_csv)
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
    single_metrics = pd.read_csv(single_paths.fold_metrics_csv)
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
    resumed_metrics = pd.read_csv(resumed_paths.fold_metrics_csv)
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
    pca_forest_metrics = pd.read_csv(pca_forest_paths.fold_metrics_csv)
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
    final_manifest = pd.read_csv(final_paths.final_model_manifest_csv)
    final_rankings = pd.read_csv(final_paths.final_rankings_csv)
    assert len(final_manifest) == 1
    assert Path(final_manifest.iloc[0]["checkpoint_path"]).exists()
    assert {"rank", "predicted_dependency_score"}.issubset(set(final_rankings.columns))


def test_regression_metrics_skip_correlation_for_constant_predictions() -> None:
    metrics = regression_metrics(
        np.asarray([-2.0, -1.0, 0.0]),
        np.asarray([0.5, 0.5, 0.5]),
    )

    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])
    assert metrics["spearman_defined"] is False
    assert metrics["pearson_defined"] is False


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
