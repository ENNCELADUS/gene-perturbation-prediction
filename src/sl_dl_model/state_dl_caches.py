"""Label-free cache loading for STATE-backed SL-DL producers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sl_dl_model.config import SLDLConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateDlCaches:
    """Shared, fold-independent caches for STATE-backed producers."""

    esm: object
    bags: object
    input_dim: int
    output_dim: int


def load_state_dl_caches(config: SLDLConfig) -> StateDlCaches:
    """Load ESM2 and GWPS bag caches once for fold-local producers.

    Args:
        config: Run configuration with ``esm2_npz`` and optional ``bags_npz``.

    Returns:
        Shared cache bundle consumed by fold-local trainers.

    Raises:
        ValueError: If required cache inputs are missing or dimension-incompatible.
    """
    from sl_dl_model.bags import (
        build_gwps_bags,
        load_bags_npz,
        state_checkpoint_input_dim,
    )
    from sl_dl_model.gene_embeddings import load_esm2_embeddings

    if config.esm2_npz is None:
        raise ValueError("state_dl producer requires config.esm2_npz")

    esm = load_esm2_embeddings(config.esm2_npz)
    expected_input_dim = state_checkpoint_input_dim(config)
    if config.bags_npz is not None and Path(config.bags_npz).exists():
        bags = load_bags_npz(config.bags_npz)
        if expected_input_dim is not None and bags.input_dim != expected_input_dim:
            msg = (
                f"bags_npz {config.bags_npz} has input_dim={bags.input_dim}, "
                f"but STATE checkpoint expects input_dim={expected_input_dim}. "
                "Rebuild it once with "
                "`uv run python scripts/setup_exp08_assets.py bags` before "
                "launching the multi-rank training job."
            )
            raise ValueError(msg)
    else:
        logger.warning(
            "bags_npz is not set; the full gwps h5ad will be loaded into memory "
            "(%s). Pre-build the bags NPZ with `save_bags_npz` to avoid this.",
            config.gwps_h5ad,
        )
        bags = build_gwps_bags(config, rng_seed=config.seed)

    if expected_input_dim is not None and bags.input_dim != expected_input_dim:
        msg = (
            f"GWPS bags input_dim={bags.input_dim} does not match STATE "
            f"checkpoint input_dim={expected_input_dim}; rebuild bags_npz with "
            "checkpoint gene alignment"
        )
        raise ValueError(msg)

    return StateDlCaches(
        esm=esm,
        bags=bags,
        input_dim=bags.input_dim,
        output_dim=bags.input_dim,
    )
