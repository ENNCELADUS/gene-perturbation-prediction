from __future__ import annotations


import numpy as np
import torch

from src.model.tx1 import _TorchBertPadding, validate_load_result


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
