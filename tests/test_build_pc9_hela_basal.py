from __future__ import annotations

import gzip
import io

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse

import scripts.build_pc9_hela_basal as builder


def test_repair_hela_excel_symbols_is_position_aware() -> None:
    symbols = pd.Index(
        ["MAPT-IT1", "1-Mar", "2-Mar", "1-Mar", "10-Mar", "2-Mar", "1-Sep", "1-Dec"]
    )

    assert builder._repair_hela_symbols(symbols) == [
        "MAPT-IT1",
        "MARC1",
        "MARC2",
        "MARCH1",
        "MARCH10",
        "MARCH2",
        "SEPTIN1",
        "DEC1",
    ]


def test_minimum_qc_uses_cells_by_genes_orientation() -> None:
    matrix = sparse.csr_matrix(
        np.vstack([np.full(501, 2, dtype=np.int64), np.zeros(501, dtype=np.int64)])
    )
    var = pd.DataFrame(
        {"gene_symbol": ["MT-ND1", *[f"G{i}" for i in range(500)]]},
        index=[f"ENSG{i:011d}" for i in range(1, 502)],
    )
    adata = ad.AnnData(X=matrix, obs=pd.DataFrame(index=["pass", "fail"]), var=var)

    assert builder._minimum_qc_mask(adata).tolist() == [True, False]


def test_build_pc9_transposes_gene_by_cell_matrix(monkeypatch, tmp_path) -> None:
    source = tmp_path / "pc9"
    source.mkdir()
    genes = pd.DataFrame(
        {
            0: [f"ENSG{i:011d}" for i in range(1, 502)],
            1: ["MT-ND1", *[f"G{i}" for i in range(500)]],
        }
    )
    genes.to_csv(
        source / "GSM4932159_sample1_genes.tsv.gz",
        sep="\t",
        header=False,
        index=False,
    )
    pd.Series(["pass", "fail"]).to_csv(
        source / "GSM4932159_sample1_barcodes.tsv.gz", header=False, index=False
    )
    matrix = sparse.coo_matrix(
        np.column_stack(
            [np.full(501, 2, dtype=np.int64), np.zeros(501, dtype=np.int64)]
        )
    )
    payload = io.BytesIO()
    scipy_io.mmwrite(payload, matrix)
    with gzip.open(source / "GSM4932159_sample1_matrix.mtx.gz", "wb") as handle:
        handle.write(payload.getvalue())

    captured: dict[str, ad.AnnData] = {}

    def fake_write(adata: ad.AnnData, output) -> str:
        captured["adata"] = adata.copy()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"formal")
        return "0" * 64

    monkeypatch.setattr(builder, "_write_verified", fake_write)

    result = builder.build_pc9(tmp_path)

    assert captured["adata"].shape == (1, 501)
    assert captured["adata"].obs_names.tolist() == ["pass"]
    assert result["qc_filter"]["cells_removed"] == 1
