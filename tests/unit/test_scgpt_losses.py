from __future__ import annotations

import torch
import torch.nn.functional as F

from src.scgpt.losses import GeneScoreLossConfig, build_gene_score_loss


def test_sampled_pairwise_loss_uses_top_false_positive_hard_negatives() -> None:
    logits = torch.tensor([[0.2, 2.0, 1.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="sampled_pairwise",
            hard_negatives=2,
            random_negatives=0,
            temperature=1.0,
            margin=0.0,
        )
    )

    loss = loss_fn(logits, targets)

    expected = torch.stack(
        [
            F.softplus(logits[0, 1] - logits[0, 0]),
            F.softplus(logits[0, 2] - logits[0, 0]),
        ]
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] > 0


def test_sampled_pairwise_loss_skips_rows_without_positive_targets() -> None:
    logits = torch.tensor(
        [[0.2, 1.0, -0.5], [0.3, 0.1, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="sampled_pairwise",
            hard_negatives=1,
            random_negatives=0,
        )
    )

    loss = loss_fn(logits, targets)

    expected = F.softplus(logits[1, 0] - logits[1, 1])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad[0], torch.zeros_like(logits.grad[0]))


def test_bce_loss_config_matches_torch_bce_with_logits() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.7, 1.2]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss_fn = build_gene_score_loss(GeneScoreLossConfig(name="bce"))

    loss = loss_fn(logits, targets)

    assert torch.allclose(loss, F.binary_cross_entropy_with_logits(logits, targets))


def test_pairwise_loss_supports_semi_hard_negative_mining() -> None:
    logits = torch.tensor([[1.0, 5.0, 1.1, 0.9, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="sampled_pairwise",
            hard_negative_source="semi_hard",
            hard_negatives=2,
            random_negatives=0,
        )
    )

    loss = loss_fn(logits, targets)

    expected = torch.stack(
        [
            F.softplus(logits[0, 2] - logits[0, 0]),
            F.softplus(logits[0, 3] - logits[0, 0]),
        ]
    ).mean()
    assert torch.allclose(loss, expected)


def test_sampled_softmax_loss_uses_positive_and_sampled_negative_scores() -> None:
    logits = torch.tensor([[0.2, 2.0, 1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="sampled_softmax",
            hard_negatives=2,
            random_negatives=0,
            temperature=1.0,
        )
    )

    loss = loss_fn(logits, targets)

    expected = torch.logsumexp(logits[0, [0, 1, 2]], dim=0) - torch.logsumexp(
        logits[0, [0]], dim=0
    )
    assert torch.allclose(loss, expected)


def test_asymmetric_loss_matches_bce_when_focusing_is_disabled() -> None:
    logits = torch.tensor([[0.2, -0.4], [0.7, 1.2]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="asymmetric",
            asymmetric_gamma_positive=0.0,
            asymmetric_gamma_negative=0.0,
            asymmetric_clip=0.0,
        )
    )

    loss = loss_fn(logits, targets)

    assert torch.allclose(loss, F.binary_cross_entropy_with_logits(logits, targets))


def test_sampled_lambdarank_downweights_low_rank_swaps() -> None:
    logits = torch.tensor([[2.0, 3.0, 1.0, -5.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_fn = build_gene_score_loss(
        GeneScoreLossConfig(
            name="sampled_lambdarank",
            hard_negatives=2,
            random_negatives=0,
        )
    )

    loss = loss_fn(logits, targets)

    high_rank_swap = F.softplus(logits[0, 1] - logits[0, 0]) * (
        (1.0 / torch.log2(torch.tensor(2.0))) - (1.0 / torch.log2(torch.tensor(3.0)))
    )
    low_rank_swap = F.softplus(logits[0, 2] - logits[0, 0]) * (
        (1.0 / torch.log2(torch.tensor(3.0))) - (1.0 / torch.log2(torch.tensor(4.0)))
    )
    expected = torch.stack([high_rank_swap, low_rank_swap]).mean()
    assert torch.allclose(loss, expected)
