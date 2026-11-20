from __future__ import annotations


import numpy as np
import torch

from src.model.tx1 import TorchBertPadding, validate_load_result


def test_torch_padding_fallback_matches_unpadding_contract() -> None:
    hidden = torch.arange(12).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    values, indices, cumulative, maximum = TorchBertPadding.unpad_input(hidden, mask)
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


def test_gene_sums_add_each_gene_over_the_cells_that_hold_it():
    from src.model.tx1 import gene_sums

    ids = torch.tensor([[7, 3, 0], [3, 9, 0], [5, 0, 0]])
    emb = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    sums, counts = gene_sums(ids, emb, torch.tensor([3, 9, 4]))
    assert counts.tolist() == [2, 1, 0]
    assert sums.tolist() == [[8.0, 10.0], [8.0, 9.0], [0.0, 0.0]]
    sums, counts = gene_sums(ids, emb, torch.tensor([3, -1, 9, 3]))
    assert counts.tolist() == [2, 0, 1, 2]
    assert sums.tolist() == [[8.0, 10.0], [0.0, 0.0], [8.0, 9.0], [8.0, 10.0]]


def test_gene_sums_refuse_a_token_twice_in_one_cell():
    from src.model.tx1 import gene_sums

    ids = torch.tensor([[3, 3, 0], [3, 9, 0]])
    with np.testing.assert_raises(ValueError):
        gene_sums(ids, torch.zeros(2, 3, 2), torch.tensor([3, 9]))
