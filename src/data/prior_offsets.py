"""The linear context prior's exported predictions, as the joint model reads them.

An export (``src.experiments.prior_export``) holds per-line residual predictions in
residual-SD units. The joint model adds ``value x residual SD`` to its head's output,
so the export must be on the same gene order and residual SD; ``checked_prior``
refuses one that is not, and ``check_prior_identity`` refuses a checkpoint trained
on another export.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriorOffsets:
    """A prior export.

    Attributes:
        identity: The export's run id and reference row; checkpoints record it.
        values: Lines x genes, residual-SD units.
        residual_scale: The per-gene residual SD the export was fitted with.
    """

    identity: dict[str, Any]
    values: pd.DataFrame
    residual_scale: pd.Series


def read_prior_offsets(path: Path) -> PriorOffsets:
    """Read ``prior.npz`` and ``prior.json`` from an export directory."""
    path = Path(path)
    record = json.loads((path / "prior.json").read_text())
    with np.load(path / "prior.npz") as payload:
        genes = [str(gene) for gene in payload["genes"]]
        values = pd.DataFrame(
            payload["values"],
            index=[str(model_id) for model_id in payload["lines"]],
            columns=genes,
        )
        scale = pd.Series(payload["residual_scale"], index=genes)
    return PriorOffsets(
        {"run_id": record["run_id"], "reference": record["reference"]}, values, scale
    )


def checked_prior(
    prior: PriorOffsets,
    *,
    genes: Sequence[str],
    residual_scale: pd.Series,
    lines: Collection[str],
) -> PriorOffsets:
    """The export, if it matches the joint model's genes, residual SD and lines."""
    if tuple(prior.values.columns) != tuple(genes):
        raise ValueError("the prior export's gene order differs from the gene panel")
    if not np.allclose(
        prior.residual_scale.to_numpy(),
        residual_scale.loc[list(genes)].to_numpy(),
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            "the prior export's residual scale differs from the joint model's"
        )
    missing = sorted(set(lines) - set(prior.values.index))
    if missing:
        raise ValueError(f"the prior export lacks lines {missing[:10]}")
    return prior


def check_prior_identity(
    recorded: Mapping[str, Any] | None, prior: PriorOffsets | None
) -> None:
    """Refuse a checkpoint whose recorded prior is not the configured one."""
    if recorded is None and prior is None:
        return
    if recorded is None:
        raise ValueError(
            "the checkpoint was trained without a prior but the config names "
            f"prior run {prior.identity['run_id']}"
        )
    if prior is None:
        raise ValueError(
            f"the checkpoint was trained on prior run {recorded['run_id']} but no "
            "prior is configured"
        )
    if dict(recorded) != prior.identity:
        raise ValueError(
            f"the checkpoint was trained on prior run {recorded['run_id']}, the "
            f"config's export is prior run {prior.identity['run_id']}"
        )
