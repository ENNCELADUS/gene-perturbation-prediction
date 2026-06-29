from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_generator import (
    EmaBagScale,
    FixedWarmupBagScale,
    build_bag_scale,
    select_generator_bag_sets,
)
from sl_dl_model.pert_vocab import load_pert_vocab


def test_generator_validation_split_comes_only_from_train_covered() -> None:
    train_symbols = {"A", "B", "C", "D", "E", "TEST_ONLY"}
    covered_symbols = {"A", "B", "C", "D", "E", "OUTSIDE_TRAIN"}

    train_bag, val_bag = select_generator_bag_sets(
        train_symbols=train_symbols,
        covered_symbols=covered_symbols,
        val_fraction=0.4,
        seed=17,
    )

    assert train_bag | val_bag == {"A", "B", "C", "D", "E"}
    assert train_bag.isdisjoint(val_bag)
    assert "TEST_ONLY" not in train_bag | val_bag
    assert "OUTSIDE_TRAIN" not in train_bag | val_bag
    assert len(val_bag) == 1


def test_generator_validation_split_is_deterministic() -> None:
    kwargs = {
        "train_symbols": {"A", "B", "C", "D", "E", "F"},
        "covered_symbols": {"A", "B", "C", "D", "E", "F"},
        "val_fraction": 0.2,
        "seed": 23,
    }

    first = select_generator_bag_sets(**kwargs)
    second = select_generator_bag_sets(**kwargs)

    assert first == second


def test_fixed_warmup_bag_scale_uses_median_and_clamp() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    for value in (torch.tensor(10.0), torch.tensor(2.0), torch.tensor(6.0)):
        scale.observe(value)

    chosen = scale.finalize()

    assert chosen == 6.0
    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_fixed_warmup_bag_scale_clamps_small_values() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    scale.observe(torch.tensor(0.0))
    scale.observe(torch.tensor(1e-8))

    chosen = scale.finalize()

    assert chosen == 1e-3
    assert torch.isclose(scale.normalize(torch.tensor(1e-3)), torch.tensor(1.0))


def test_fixed_warmup_bag_scale_requires_observations() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)

    try:
        scale.finalize()
    except ValueError as exc:
        assert "no bag losses observed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ema_bag_scale_updates_and_normalizes() -> None:
    scale = EmaBagScale(min_scale=1e-3, decay=0.5)

    scale.observe(torch.tensor(10.0))
    assert scale.value == 10.0
    scale.observe(torch.tensor(2.0))

    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_build_bag_scale_selects_fixed_or_ema() -> None:
    fixed = build_bag_scale(Exp08bConfig(bag_scale_mode="fixed_warmup"))
    ema = build_bag_scale(Exp08bConfig(bag_scale_mode="ema", bag_scale_ema_decay=0.9))

    assert isinstance(fixed, FixedWarmupBagScale)
    assert isinstance(ema, EmaBagScale)


def test_build_bag_scale_rejects_unknown_mode() -> None:
    try:
        build_bag_scale(Exp08bConfig(bag_scale_mode="mystery"))
    except ValueError as exc:
        assert "unknown bag_scale_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_pert_vocab_reads_checkpoint_sibling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    torch.save(
        {"A": np.eye(3, dtype=np.float32)[0]},
        checkpoint.parent.parent / "pert_onehot_map.pt",
    )

    loaded = load_pert_vocab(checkpoint)

    assert loaded is not None
    assert set(loaded) == {"A"}
    np.testing.assert_allclose(loaded["A"], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_load_pert_vocab_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    """Missing pert_onehot_map.pt must return None, NOT {}.

    The exp08 `_ensure_pert_vocab` raise-on-missing contract (train.py) keys off
    a `None` return to distinguish "absent" from "present but empty". Returning
    `{}` here would silently disable the distill anchor — the OOD-token fix the
    spec mandates at full weight (§3.2) — and would also break the existing
    `tests/sl_dl_model/test_train.py::test_distill_required_but_missing_vocab_raises`.
    """
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert load_pert_vocab(checkpoint) is None
