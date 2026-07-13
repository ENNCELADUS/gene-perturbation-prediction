"""Tests for the exp05 GWPS/DepMap label-universe builder."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.build_exp05_gwps_labels import build_gwps_label_table


def test_build_gwps_label_table_keeps_numeric_intersection(tmp_path: Path) -> None:
    adata = ad.AnnData(np.ones((5, 2), dtype=np.float32))
    adata.obs["gene"] = ["non-targeting", "TP53", "TP53", "KRAS", "NO_LABEL"]
    h5ad = tmp_path / "gwps.h5ad"
    adata.write_h5ad(h5ad)
    effects = pd.DataFrame(
        [[-0.7, 0.2, np.nan]],
        index=["ACH-000551"],
        columns=["TP53 (7157)", "KRAS (3845)", "NO_LABEL (1)"],
    )
    csv = tmp_path / "CRISPRGeneEffect.csv"
    effects.to_csv(csv)

    result = build_gwps_label_table(h5ad, csv, "ACH-000551")

    assert result.to_dict("records") == [
        {
            "perturbation_gene": "KRAS",
            "depmap_model_id": "ACH-000551",
            "depmap_entrez_id": "3845",
            "depmap_gene_effect": 0.2,
            "has_depmap_label": True,
        },
        {
            "perturbation_gene": "TP53",
            "depmap_model_id": "ACH-000551",
            "depmap_entrez_id": "7157",
            "depmap_gene_effect": -0.7,
            "has_depmap_label": True,
        },
    ]


def test_build_gwps_label_table_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ACH-000551"):
        build_gwps_label_table(
            tmp_path / "gwps.h5ad",
            tmp_path / "CRISPRGeneEffect.csv",
            "ACH-000551",
        )
