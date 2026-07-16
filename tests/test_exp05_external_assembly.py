"""Tests for fixed-split external dataset assembly."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from aivc_model.prepare import GeneBags, merge_gene_bag_pool
from scripts.assemble_exp05_fixed_datasets import (
    build_dixit_overlap,
    build_pool_labels,
    combine_external_overlap,
    labels_from_predictions,
)
from scripts.build_exp05_xatlas_overlap import build_xatlas_overlap


def _write_h5ad(
    path: Path,
    obs: pd.DataFrame,
) -> None:
    ad.AnnData(
        X=sparse.csr_matrix(np.ones((len(obs), 2), dtype=np.float32)),
        obs=obs,
        var=pd.DataFrame(index=["A", "B"]),
    ).write_h5ad(path)


def test_labels_from_predictions_uses_internal_scope(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["A", "B", "A"],
            "y_true": [-1.0, 0.2, 99.0],
            "evaluation_scope": [
                "internal_outer_test",
                "internal_outer_test",
                "external_test",
            ],
        }
    ).to_csv(path, index=False)
    result = labels_from_predictions(path)
    assert result.to_dict("records") == [
        {"perturbation_gene": "A", "depmap_gene_effect": -1.0},
        {"perturbation_gene": "B", "depmap_gene_effect": 0.2},
    ]


def test_dixit_overlap_keeps_exact_single_gene_conditions(tmp_path: Path) -> None:
    path = tmp_path / "dixit.h5ad"
    _write_h5ad(
        path,
        pd.DataFrame(
            {
                "perturbation_name": [
                    "A",
                    "A",
                    "B+C",
                    "control",
                    "INTERGENIC1",
                ]
            }
        ),
    )
    result = build_dixit_overlap(path, {"A": -0.5, "B": 0.1}, min_cells=2)
    assert result[["source_perturbation_label", "perturbation_gene"]].to_dict(
        "records"
    ) == [{"source_perturbation_label": "A", "perturbation_gene": "A"}]


def test_combined_overlap_deduplicates_source_conditions() -> None:
    row = {
        "source_dataset": "adamson_pilot",
        "source_perturbation_label": "A_guide",
        "perturbation_gene": "A",
        "perturbation_label_type": "single_gene",
        "has_depmap_label": True,
        "depmap_gene_effect": -0.5,
    }
    result = combine_external_overlap(pd.DataFrame([row, row]), pd.DataFrame([row]))
    assert len(result) == 1


def test_pool_labels_add_only_replogle_unseen_genes() -> None:
    replogle = pd.DataFrame(
        {
            "perturbation_gene": ["A", "B"],
            "depmap_gene_effect": [-1.0, 0.2],
        }
    )
    supplement = pd.DataFrame(
        {
            "perturbation_gene": ["B", "C", "C"],
            "depmap_gene_effect": [0.2, -0.4, -0.4],
        }
    )
    result = build_pool_labels(replogle, supplement)
    assert result["perturbation_gene"].tolist() == ["A", "B", "C"]


def _bags(genes: list[str], values: list[float], source: str) -> GeneBags:
    matrices = tuple(
        np.full((2, 2), index + 1, dtype=np.float32)
        for index in range(len(genes))
    )
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.asarray(values, dtype=np.float32),
        input_bags=matrices,
        latent_bags=matrices,
        control_input=np.zeros((2, 2), dtype=np.float32),
        control_latent=np.zeros((2, 2), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=np.asarray(["G1", "G2"], dtype=object),
        metadata=pd.DataFrame(
            {"perturbation_gene": genes, "source_dataset": source}
        ),
        input_dim=2,
        latent_dim=2,
    )


def test_gene_bag_pool_combines_shared_gene_and_adds_unseen_gene() -> None:
    result = merge_gene_bag_pool(
        _bags(["A", "B"], [-1.0, 0.2], "replogle"),
        _bags(["B", "C"], [0.2, -0.4], "adamson"),
        "depmap_gene_effect",
    )
    assert result.genes.tolist() == ["A", "B", "C"]
    assert [len(bag) for bag in result.input_bags] == [2, 4, 2]
    assert result.metadata.loc[1, "source_dataset"] == "adamson;replogle"


def test_xatlas_overlap_uses_passed_guide_pairs(tmp_path: Path) -> None:
    path = tmp_path / "xatlas.h5ad"
    _write_h5ad(
        path,
        pd.DataFrame(
            {
                "gene_target": ["A", "A", "B", "Non-Targeting"],
                "pass_guide_filter": [1, 1, 0, 1],
            }
        ),
    )
    result = build_xatlas_overlap(
        path,
        {"A": -1.0, "B": 0.5},
        target_col="gene_target",
        pass_filter_col="pass_guide_filter",
        min_cells=2,
    )
    assert result[["perturbation_gene", "depmap_gene_effect"]].to_dict("records") == [
        {"perturbation_gene": "A", "depmap_gene_effect": -1.0}
    ]
