"""GeneEffect objectives, gene-blocked sampler, optimizer groups and schedule."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from src.model.losses import blocked_pearson_loss, geneeffect_loss
from src.training.sampling import GeneBlockSampler
from src.training.trainer import make_optimizer, make_scheduler


def loss(prediction, target, scale, objective, gene_index, selective):
    return geneeffect_loss(
        torch.tensor(prediction),
        torch.tensor(target),
        torch.tensor(scale),
        objective=objective,
        gene_index=torch.tensor(gene_index),
        selective=torch.tensor(selective),
    )


def test_huber_is_delta_one_on_residual_units():
    prediction, target = [0.0, 3.0, -0.5], [0.5, 0.0, -0.5]
    expected = (0.5 * 0.5**2 + (3.0 - 0.5) + 0.0) / 3
    value = loss(prediction, target, [9.0, 9.0, 9.0], "huber", [0, 1, 2], [1, 1, 1])
    assert value.item() == pytest.approx(expected)


def test_standardized_mse_divides_each_row_by_its_scale():
    value = loss(
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.5, 2.0, 1.0],
        "standardized_mse",
        [0, 1, 2],
        [True, False, True],
    )
    assert value.item() == pytest.approx((2.0**2 + 0.5**2 + 0.0) / 3)


def test_pearson_term_skips_non_selective_and_small_genes():
    # Gene 0: selective, 3 rows, perfectly correlated -> 1 - r = 0.
    # Gene 1: selective, 3 rows, perfectly anti-correlated -> 1 - r = 2.
    # Gene 2: not selective (anti-correlated, would add 2). Gene 3: selective, 2 rows.
    prediction = [1.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 1.0]
    target = [0.1, 0.2, 0.4, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.9]
    genes = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    selective = [True] * 6 + [False] * 3 + [True] * 2
    scale = [0.1] * 3 + [0.5] * 3 + [1.0] * 3 + [2.0] * 2
    standardized = loss(prediction, target, scale, "standardized_mse", genes, selective)
    blocks = loss(prediction, target, scale, "pearson_blocks", genes, selective)
    assert blocks.item() == pytest.approx(standardized.item() + 1.0, rel=1e-5)


def test_pearson_term_is_zero_without_an_eligible_gene():
    prediction = torch.tensor([1.0, 2.0, 0.5, 0.7], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 0.2, 0.1])
    term = blocked_pearson_loss(
        prediction,
        target,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([True, True, False, False]),
    )
    assert term.item() == 0.0
    term.backward()
    assert torch.equal(prediction.grad, torch.zeros(4))


def test_constant_prediction_or_target_gives_finite_loss_and_gradient():
    # Gene 0 has a constant prediction (r = 0, term 1); gene 1 a constant target,
    # which is not eligible.
    prediction = torch.tensor([0.2, 0.2, 0.2, 1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([0.1, 0.5, 0.3, 0.4, 0.4, 0.4])
    value = geneeffect_loss(
        prediction,
        target,
        torch.ones(6),
        objective="pearson_blocks",
        gene_index=torch.tensor([4, 4, 4, 7, 7, 7]),
        selective=torch.ones(6, dtype=torch.bool),
    )
    mse = (prediction.detach() - target).square().mean()
    assert value.item() == pytest.approx(mse.item() + 1.0, rel=1e-5)
    value.backward()
    assert torch.isfinite(prediction.grad).all()


def test_objectives_compute_in_fp32():
    prediction = torch.tensor([0.5, -0.25, 1.0, 0.0], dtype=torch.bfloat16)
    target = torch.tensor([0.0, 0.5, 0.75, -0.5])
    for objective in ("huber", "standardized_mse", "pearson_blocks"):
        value = geneeffect_loss(
            prediction,
            target,
            torch.full((4,), 0.3),
            objective=objective,
            gene_index=torch.tensor([0, 0, 0, 0]),
            selective=torch.ones(4, dtype=torch.bool),
        )
        assert value.dtype == torch.float32 and torch.isfinite(value)
    with pytest.raises(ValueError, match="unknown GeneEffect objective"):
        geneeffect_loss(
            prediction,
            target,
            torch.ones(4),
            objective="mse",
            gene_index=torch.zeros(4, dtype=torch.long),
            selective=torch.ones(4, dtype=torch.bool),
        )


def test_gene_blocks_cover_each_gene_at_most_once_with_equal_rank_counts():
    sizes = [3, 0, 2, 4, 1, 0, 3, 3, 2, 5, 1, 2, 4, 3]  # 12 genes with rows
    offsets = np.cumsum([0, *sizes])
    rows_by_gene = [np.arange(offsets[g], offsets[g + 1]) for g in range(len(sizes))]
    row_gene = np.repeat(np.arange(len(sizes)), sizes)
    world, per_block = 3, 2  # 6 blocks -> 2 per rank, none dropped
    for epoch in (0, 1):
        batches = [
            list(
                GeneBlockSampler(
                    rows_by_gene,
                    genes_per_block=per_block,
                    epoch=epoch,
                    rank=rank,
                    world=world,
                )
            )
            for rank in range(world)
        ]
        assert [len(b) for b in batches] == [2, 2, 2]
        genes = []
        for rank_batches in batches:
            for batch in rank_batches:
                block = sorted(set(row_gene[batch]))
                assert len(block) == per_block
                # Every row of each block gene, nothing else.
                assert sorted(batch) == sorted(
                    np.concatenate([rows_by_gene[g] for g in block]).tolist()
                )
                genes += block
        assert len(genes) == len(set(genes)) == 12
        assert all(sizes[g] for g in genes)

    # 12 genes in blocks of 5 over 2 ranks: 2 blocks, one per rank, 2 genes unused.
    for rank in range(2):
        sampler = GeneBlockSampler(
            rows_by_gene, genes_per_block=5, epoch=3, rank=rank, world=2
        )
        assert len(sampler) == len(list(sampler)) == 1
    first = list(
        GeneBlockSampler(rows_by_gene, genes_per_block=2, epoch=0, rank=0, world=1)
    )
    second = list(
        GeneBlockSampler(rows_by_gene, genes_per_block=2, epoch=1, rank=0, world=1)
    )
    assert first != second


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perturbations = nn.Linear(2, 3)
        self.state = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))


class FakeModel(nn.Module):
    """The attributes ``make_optimizer`` reads: backbone parts, head, ``uses_state``."""

    def __init__(self, uses_state: bool) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        self.head = nn.Linear(3, 1)
        self.uses_state = uses_state

    def forward(self, x):
        return self.head(self.backbone.state(self.backbone.perturbations(x)))


def train_config(state_mode="frozen", **overrides):
    return {
        "train": {
            "state_learning_rate": 1e-5,
            "adapter_learning_rate": 1e-4,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.01,
            "state_mode": state_mode,
            "warmup_epochs": 1,
            "max_epochs": 3,
            **overrides,
        }
    }


@pytest.mark.parametrize(
    ("uses_state", "state_mode", "groups"),
    [
        (True, "frozen", {"adapter": 1e-4, "head": 1e-3}),
        (True, "trainable", {"state": 1e-5, "adapter": 1e-4, "head": 1e-3}),
        (False, "trainable", {"head": 1e-3}),
    ],
)
def test_optimizer_groups_hold_only_trained_parameters(uses_state, state_mode, groups):
    torch.manual_seed(0)
    model = FakeModel(uses_state)
    optimizer = make_optimizer(model, train_config(state_mode))
    assert {g["name"]: g["lr"] for g in optimizer.param_groups} == groups
    held = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert held == {id(p) for p in model.parameters() if p.requires_grad}


def test_frozen_state_is_unchanged_by_an_update_while_the_adapter_moves():
    torch.manual_seed(0)
    model = FakeModel(uses_state=True)
    optimizer = make_optimizer(model, train_config("frozen"))
    before = copy.deepcopy(model.state_dict())
    model(torch.randn(8, 2)).square().mean().backward()
    optimizer.step()
    after = model.state_dict()
    for name, value in before.items():
        if name.startswith("backbone.state."):
            torch.testing.assert_close(after[name], value, rtol=0, atol=0)
        else:
            assert not torch.equal(after[name], value), name


def test_schedule_warms_up_over_one_epoch_then_decays_to_zero():
    weight = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [weight], "lr": 1e-3}])
    scheduler = make_scheduler(optimizer, train_config(max_epochs=3), 4)
    rates = []
    for _ in range(12):  # 3 epochs of 4 updates
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:4] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3])
    assert rates[4] == pytest.approx(1e-3)  # cosine starts at the full rate
    assert rates[-1] == pytest.approx(1e-3 * 0.5 * (1 + math.cos(math.pi * 7 / 8)))
    assert all(a > b for a, b in zip(rates[4:], rates[5:]))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_schedule_state_restores_mid_run():
    def build():
        weight = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([{"params": [weight], "lr": 1e-3}])
        return optimizer, make_scheduler(optimizer, train_config(max_epochs=3), 4)

    optimizer, scheduler = build()
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    saved = (optimizer.state_dict(), scheduler.state_dict())
    restored_optimizer, restored_scheduler = build()
    restored_optimizer.load_state_dict(saved[0])
    restored_scheduler.load_state_dict(saved[1])
    for opt, sched in (
        (optimizer, scheduler),
        (restored_optimizer, restored_scheduler),
    ):
        opt.step()
        sched.step()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch == 6
