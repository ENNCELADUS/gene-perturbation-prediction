from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.verify_tx1_obsm_width import (
    _TorchBertPadding,
    tiny_dmso_adata,
    validate_load_result,
)


def test_tiny_dmso_adata_keeps_raw_counts_and_required_axes(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    pd.DataFrame(
        {
            "genes": [np.array([1, 3, 4]), np.array([3, 5])],
            "expressions": [np.array([-2.0, 2.0, 1.0]), np.array([3.0, 0.0])],
            "cell_line_id": ["CVCL_A", "CVCL_B"],
        }
    ).to_parquet(shard_dir / "part-00000.parquet")
    metadata = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C"],
            "ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
            "token_id": [3, 4, 5],
        }
    )
    metadata_path = tmp_path / "genes.parquet"
    metadata.to_parquet(metadata_path)
    adata = tiny_dmso_adata(shard_dir, metadata_path, cells=2)
    assert adata.shape == (2, 2)
    assert adata.var["ensembl_id"].tolist() == ["ENSG1", "ENSG2"]
    assert adata.obs["cell_type"].tolist() == ["CVCL_A", "CVCL_B"]
    assert np.all(adata.X.data >= 0)


def test_torch_padding_fallback_matches_unpadding_contract() -> None:
    hidden = torch.arange(12).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    values, indices, cumulative, maximum = _TorchBertPadding.unpad_input(hidden, mask)
    assert indices.tolist() == [0, 1, 3]
    assert cumulative.tolist() == [0, 2, 3]
    assert maximum == 2
    assert values.tolist() == [[0, 1], [2, 3], [6, 7]]


def test_checkpoint_load_rejects_missing_backbone_keys() -> None:
    class LoadResult:
        missing_keys = ["model.transformer.block.0.weight"]
        unexpected_keys: list[str] = []

    with np.testing.assert_raises(ValueError):
        validate_load_result(LoadResult())


def test_checkpoint_load_allows_removed_chemical_head() -> None:
    class LoadResult:
        missing_keys: list[str] = []
        unexpected_keys = [
            "model.chem_encoder.weight",
            "model.chemical_encoder.weight",
            "model.mlm_head.weight",
        ]

    report = validate_load_result(LoadResult())
    assert report["complete_backbone_load"] is True
