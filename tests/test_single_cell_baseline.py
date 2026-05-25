from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dependency_baseline.cell_bags import build_cell_bags
from dependency_baseline.config import (
    BaselineConfig,
    CvConfig,
    DataConfig,
    SingleCellConfig,
)
from dependency_baseline.single_cell import DeepSetsRegressor, run_single_cell_cv


def test_build_cell_bags_creates_aligned_single_cell_artifacts(
    tmp_path: Path,
) -> None:
    h5ad_path, overlap_path = _write_synthetic_replogle_inputs(tmp_path)
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        ),
        cv=CvConfig(n_splits=2, n_repeats=1, random_state=7, model_set="quick"),
    )

    paths = build_cell_bags(config)

    assert paths.bags_npz.exists()
    assert paths.metadata_path.exists()
    assert paths.summary_json.exists()
    assert paths.bags_npz.parent.name == "single_cell_bags"

    payload = np.load(paths.bags_npz, allow_pickle=True)
    metadata = pd.read_parquet(paths.metadata_path)
    assert metadata["perturbation_gene"].tolist() == [
        "GENE1",
        "GENE2",
        "GENE3",
        "GENE4",
        "GENE5",
        "GENE6",
    ]
    assert metadata["observed_n_cells"].tolist() == [3, 3, 3, 3, 3, 3]
    assert payload["cell_delta_pcs"].shape[0] == 18
    assert payload["cell_delta_pcs"].shape[1] <= 5
    assert payload["bag_offsets"].tolist() == [0, 3, 6, 9, 12, 15, 18]
    np.testing.assert_allclose(payload["y"], metadata["depmap_gene_effect"].to_numpy())
    assert (
        payload["perturbation_gene"].astype(str).tolist()
        == metadata["perturbation_gene"].tolist()
    )


def test_deepsets_regressor_ignores_padding_in_bag_predictions() -> None:
    train_bags = (
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32),
    )
    y = np.asarray([-1.0, 1.0, -2.0, 2.0], dtype=np.float32)
    model = DeepSetsRegressor(
        input_dim=2,
        hidden_units=(8,),
        bag_hidden_units=(8,),
        max_cells_per_bag=4,
        max_epochs=20,
        patience=5,
        batch_size=2,
        random_state=3,
        device="cpu",
    )
    model.fit(train_bags, y)

    original = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    padded = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    pred_original = model.predict((original,))
    pred_padded = model.predict((padded,), observed_counts=np.asarray([2]))

    np.testing.assert_allclose(pred_original, pred_padded, atol=1e-6)


def test_run_single_cell_cv_writes_comparable_artifacts(tmp_path: Path) -> None:
    h5ad_path, overlap_path = _write_synthetic_replogle_inputs(tmp_path)
    config = BaselineConfig(
        data=DataConfig(
            h5ad_path=h5ad_path,
            overlap_csv=overlap_path,
            output_dir=tmp_path / "outputs",
        ),
        cv=CvConfig(
            n_splits=2,
            n_repeats=1,
            random_state=7,
            stratify_bins=3,
            model_set="quick",
        ),
        single_cell=SingleCellConfig(
            n_hvg=5,
            n_pcs=3,
            max_cells_per_bag=4,
            hidden_units=(8,),
            bag_hidden_units=(8,),
            max_epochs=5,
            patience=2,
            batch_size=2,
            device="cpu",
        ),
    )
    bag_paths = build_cell_bags(config)

    cv_paths = run_single_cell_cv(config, bag_paths.bags_npz, run_id="single_cell")

    assert cv_paths.run_dir.exists()
    assert cv_paths.fold_metrics_path.exists()
    assert cv_paths.predictions_path.exists()
    assert cv_paths.summary_csv.exists()
    assert cv_paths.model_manifest_path.exists()
    assert cv_paths.splits_path.exists()
    fold_metrics = pd.read_parquet(cv_paths.fold_metrics_path)
    predictions = pd.read_parquet(cv_paths.predictions_path)
    model_manifest = pd.read_parquet(cv_paths.model_manifest_path)
    assert set(fold_metrics["feature_set"]) == {"single_cell_pc_delta"}
    assert set(fold_metrics["model"]) == {"deepsets_pca3_meanpool"}
    assert {"unweighted", "sqrt_n_cells"}.issubset(set(fold_metrics["weighting"]))
    assert predictions["perturbation_gene"].nunique() == 6
    assert (
        model_manifest["checkpoint_path"].map(lambda value: Path(value).exists()).all()
    )


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
        {"gene_name": expression_symbols},
        index=[f"ENSG{i}" for i in range(5)],
    )
    adata = ad.AnnData(X=np.vstack(rows), obs=obs, var=var)
    h5ad_path = tmp_path / "synthetic_replogle.h5ad"
    adata.write_h5ad(h5ad_path)

    overlap = pd.DataFrame(
        {
            "perturbation_gene": ["non-targeting", *genes, "MATCHED_NAN"],
            "depmap_gene_column": [
                np.nan,
                *(f"{gene} ({i})" for i, gene in enumerate(genes)),
                "MATCHED_NAN (999)",
            ],
            "has_depmap_label": [False, *([True] * len(genes)), True],
            "depmap_gene_effect": [
                np.nan,
                *(effects[gene] for gene in genes),
                np.nan,
            ],
            "n_cells_or_pseudobulk": [4, *([3] * len(genes)), 1],
            "is_control_candidate": [True, *([False] * len(genes)), False],
            "modality": "CRISPRi",
            "source_dataset": "synthetic",
        }
    )
    overlap_path = tmp_path / "overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    return h5ad_path, overlap_path
