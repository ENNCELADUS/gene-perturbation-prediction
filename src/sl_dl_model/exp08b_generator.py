"""Exp08b Step-1 generator: leakage-safe split and bag-loss scale helpers.

This module is used by Task 2 (split + scale) and will be extended by Task 3
(Step1GeneratorTrainer).
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

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
    eligible = sorted(train_symbols & covered_symbols)
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


class BagScale(Protocol):
    """Protocol for bag-loss normalizers."""

    @property
    def value(self) -> float:
        """Current scale value."""
        ...

    def observe(self, loss: torch.Tensor) -> None:
        """Record a raw bag loss for scale estimation."""
        ...

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Divide *loss* by the current scale."""
        ...


class FixedWarmupBagScale:
    """Accumulate bag losses during warmup, then fix the median as the scale.

    The scale is fixed to the median of all observed values, clamped to
    ``min_scale`` from below.  Call :meth:`finalize` once at the end of the
    warmup phase; after that :meth:`normalize` is valid.

    Args:
        min_scale: Lower bound for the scale to avoid division by near-zero.
    """

    def __init__(self, min_scale: float = 1e-3) -> None:
        self._min_scale = min_scale
        self._observations: list[float] = []
        self._value: float | None = None

    @property
    def value(self) -> float:
        """Fixed scale value (only valid after :meth:`finalize`)."""
        if self._value is None:
            raise RuntimeError("bag scale has not been finalized")
        return self._value

    def observe(self, loss: torch.Tensor) -> None:
        """Record a raw bag loss value.

        Args:
            loss: Scalar tensor with the raw (un-normalized) bag loss.
        """
        self._observations.append(float(loss.detach()))

    def finalize(self) -> float:
        """Fix the scale to the median of observed values.

        Clamps to :attr:`min_scale` from below.

        Returns:
            The finalized scale value.

        Raises:
            ValueError: If no observations have been recorded.
        """
        if not self._observations:
            raise ValueError("no bag losses observed; cannot finalize scale")
        median = float(np.median(self._observations))
        self._value = max(self._min_scale, median)
        logger.debug("FixedWarmupBagScale finalized to %.4g", self._value)
        return self._value

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Divide *loss* by the fixed scale.

        Args:
            loss: Raw bag loss tensor.

        Returns:
            Normalized loss tensor.

        Raises:
            RuntimeError: If :meth:`finalize` has not been called.
        """
        if self._value is None:
            raise RuntimeError("bag scale has not been initialized")
        return loss / float(self._value)


class EmaBagScale:
    """Exponential moving average bag-loss normalizer.

    Updates the scale online with each :meth:`observe` call.  The first
    observation seeds the EMA exactly (no warmup cold-start issue).

    Args:
        min_scale: Lower bound for the scale to avoid division by near-zero.
        decay: EMA smoothing factor in ``[0, 1)``. Higher = slower adaptation.
    """

    def __init__(self, min_scale: float = 1e-3, decay: float = 0.95) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self._min_scale = min_scale
        self._decay = decay
        self._value: float | None = None

    @property
    def value(self) -> float:
        """Current EMA scale value.

        Raises:
            RuntimeError: If no observation has been received yet.
        """
        if self._value is None:
            raise RuntimeError("bag scale has not been initialized")
        return self._value

    def observe(self, loss: torch.Tensor) -> None:
        """Update the EMA with a new raw bag loss.

        Args:
            loss: Scalar tensor with the raw (un-normalized) bag loss.
        """
        x = float(loss.detach())
        if self._value is None:
            self._value = max(self._min_scale, x)
        else:
            self._value = self._decay * self._value + (1.0 - self._decay) * x
            self._value = max(self._min_scale, self._value)

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Divide *loss* by the current EMA scale.

        Args:
            loss: Raw bag loss tensor.

        Returns:
            Normalized loss tensor.

        Raises:
            RuntimeError: If :meth:`observe` has not been called.
        """
        if self._value is None:
            raise RuntimeError("bag scale has not been initialized")
        return loss / float(self._value)


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
