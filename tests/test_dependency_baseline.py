from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from vcc_dependency_baseline.config import (
    BaselineConfig,
    CvConfig,
    DataConfig,
    FeatureConfig,
)
from vcc_dependency_baseline.evaluation import regression_metrics, run_cv
from vcc_dependency_baseline.features import build_features


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

    cv_paths = run_cv(config)
    summary = pd.read_csv(cv_paths.summary_csv)
    predictions = pd.read_csv(cv_paths.predictions_csv)
    assert {"delta_all", "delta_mask_target", "n_cells_only"}.issubset(
        set(summary["feature_set"])
    )
    assert {"mean_label", "ridge"}.issubset(set(summary["model"]))
    assert {"unweighted", "sqrt_n_cells"}.issubset(set(summary["weighting"]))
    assert not predictions.empty


def test_regression_metrics_skip_correlation_for_constant_predictions() -> None:
    metrics = regression_metrics(
        np.asarray([-2.0, -1.0, 0.0]),
        np.asarray([0.5, 0.5, 0.5]),
    )

    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["pearson"])


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
