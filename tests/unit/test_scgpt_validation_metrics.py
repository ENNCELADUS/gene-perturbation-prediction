from __future__ import annotations

import torch

from src.scgpt.losses import BinaryCrossEntropyGeneScoreLoss
from src.scgpt.train import _evaluate_validation


class _FakeRuntime:
    device = torch.device("cpu")


class _FixedLogitModel(torch.nn.Module):
    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del gene_ids, padding_mask, kwargs
        return torch.tensor(
            [[0.9, 0.8, 0.7, 0.1]],
            dtype=values.dtype,
            device=values.device,
        )


def test_validation_metrics_rank_only_candidate_perturbation_genes() -> None:
    batch = {
        "genes": torch.tensor([[1, 2, 3, 4]]),
        "values": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "padding_mask": torch.tensor([[False, False, False, False]]),
        "control_genes": torch.tensor([[1, 2, 3, 4]]),
        "control_values": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "control_padding_mask": torch.tensor([[False, False, False, False]]),
        "control_counts": 1,
        "targets": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }

    _, metrics = _evaluate_validation(
        model=_FixedLogitModel(),
        loader=[batch],
        loss_fn=BinaryCrossEntropyGeneScoreLoss(),
        runtime=_FakeRuntime(),
        top_k_values=[1],
        candidate_indices=[3],
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["mrr"] == 1.0
