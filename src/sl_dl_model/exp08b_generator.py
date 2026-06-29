"""Exp08b Step-1 generator: leakage-safe split and bag-loss scale helpers.

This module is used by Task 2 (split + scale) and will be extended by Task 3
(Step1GeneratorTrainer).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from sl_dl_model.exp08b_config import Exp08bConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage-safe generator validation split
# ---------------------------------------------------------------------------


def select_generator_bag_sets(
    *,
    train_symbols: set[str],
    covered_symbols: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split bag-covered train symbols into train-bag and val-bag sets.

    Only symbols that appear in **both** ``train_symbols`` and
    ``covered_symbols`` are eligible.  This prevents:

    - test-split leakage (symbols only in the test split of the SL fold
      cannot appear in the generator val set)
    - bag-coverage leakage (symbols without a real perturbation bag in the
      GWPS h5ad cannot receive bag supervision)

    The split is deterministic for a given ``seed``.

    Args:
        train_symbols: Upper-case gene symbols in this fold's train SL pairs.
        covered_symbols: Upper-case symbols that have a real bag in GwpsBags.
        val_fraction: Fraction of eligible symbols to allocate to validation.
            Rounded to the nearest integer; minimum 0.
        seed: Random seed for reproducibility.

    Returns:
        ``(train_bag, val_bag)`` — disjoint sets that together equal
        ``train_symbols & covered_symbols``.
    """
    eligible = sorted(
        {s.upper() for s in train_symbols} & {s.upper() for s in covered_symbols}
    )
    if not eligible:
        return set(), set()
    # Reserve at least 1 for train_bag: compute n_val from (n-1) eligible slots.
    # This gives floor((n-1) * val_fraction) which is always < n, guaranteeing
    # at least 1 symbol remains in the train-bag set.
    n_val = max(0, math.floor((len(eligible) - 1) * val_fraction))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))
    val_indices = set(idx[:n_val].tolist())
    train_bag: set[str] = set()
    val_bag: set[str] = set()
    for i, sym in enumerate(eligible):
        if i in val_indices:
            val_bag.add(sym)
        else:
            train_bag.add(sym)
    return train_bag, val_bag


# ---------------------------------------------------------------------------
# Bag-loss scale normalizers
# ---------------------------------------------------------------------------


class FixedWarmupBagScale:
    """Median bag-loss scale chosen from detached warmup observations."""

    def __init__(self, *, min_scale: float) -> None:
        self.min_scale = float(min_scale)
        self._observed: list[float] = []
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether a fixed scale has been selected."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Record one detached bag-loss value."""
        self._observed.append(float(loss.detach().cpu()))

    def finalize(self) -> float:
        """Select median observed scale, clamped to ``min_scale``."""
        if not self._observed:
            raise ValueError("no bag losses observed during warmup")
        finite = [x for x in self._observed if np.isfinite(x)]
        if not finite:
            raise ValueError("no finite bag losses observed during warmup")
        median = float(np.median(np.asarray(finite, dtype=float)))
        self.value = max(median, self.min_scale)
        logger.debug("FixedWarmupBagScale finalized to %.4g", self.value)
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the selected fixed scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been finalized")
        return loss / float(self.value)


class EmaBagScale:
    """EMA-normalized bag-loss scale for the normalization ablation."""

    def __init__(self, *, min_scale: float, decay: float) -> None:
        self.min_scale = float(min_scale)
        self.decay = float(decay)
        self.value: float | None = None

    @property
    def ready(self) -> bool:
        """Return whether at least one finite scale has been observed."""
        return self.value is not None

    def observe(self, loss: torch.Tensor) -> None:
        """Update the EMA scale from one detached bag-loss value."""
        current = max(float(loss.detach().cpu()), self.min_scale)
        if not np.isfinite(current):
            return
        if self.value is None:
            self.value = current
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * current
        self.value = max(float(self.value), self.min_scale)

    def finalize(self) -> float:
        """Return the current EMA scale."""
        if self.value is None:
            raise ValueError("no finite bag losses observed for EMA scale")
        return self.value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize a bag loss by the current EMA scale."""
        if self.value is None:
            raise RuntimeError("bag scale has not been initialized")
        return loss / float(self.value)


def build_bag_scale(config: Exp08bConfig) -> FixedWarmupBagScale | EmaBagScale:
    """Build the configured bag-loss normalizer."""
    if config.bag_scale_mode == "fixed_warmup":
        return FixedWarmupBagScale(min_scale=config.bag_scale_min)
    if config.bag_scale_mode == "ema":
        return EmaBagScale(
            min_scale=config.bag_scale_min,
            decay=config.bag_scale_ema_decay,
        )

    raise ValueError(f"unknown bag_scale_mode: {config.bag_scale_mode!r}")
