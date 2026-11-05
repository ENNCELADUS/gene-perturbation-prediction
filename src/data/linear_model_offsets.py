"""The linear model's export (``hpc/run.sh linear-model-export``) as the joint model
reads it: per-line residual predictions in residual-SD units, which must share the
model's gene order and residual SD."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearModelOffsets:
    """A linear model export.

    Attributes:
        identity: The export's run id and reference row; checkpoints record it.
        values: Lines x genes, residual-SD units.
        residual_scale: The per-gene residual SD the export was fitted with.
    """

    identity: dict[str, Any]
    values: pd.DataFrame
    residual_scale: pd.Series


def read_linear_model_offsets(path: Path) -> LinearModelOffsets:
    """Read ``linear_model.npz`` and ``linear_model.json`` from an export directory."""
    path = Path(path)
    record = json.loads((path / "linear_model.json").read_text())
    with np.load(path / "linear_model.npz") as payload:
        genes = [str(gene) for gene in payload["genes"]]
        values = pd.DataFrame(
            payload["values"],
            index=[str(model_id) for model_id in payload["lines"]],
            columns=genes,
        )
        scale = pd.Series(payload["residual_scale"], index=genes)
    return LinearModelOffsets(
        {"run_id": record["run_id"], "reference": record["reference"]}, values, scale
    )


def checked_linear_model(
    linear_model: LinearModelOffsets,
    *,
    genes: Sequence[str],
    residual_scale: pd.Series,
    lines: Collection[str],
) -> LinearModelOffsets:
    """The export, if it matches the joint model's genes, residual SD and lines."""
    if tuple(linear_model.values.columns) != tuple(genes):
        raise ValueError(
            "the linear model export's gene order differs from the gene panel"
        )
    if not np.allclose(
        linear_model.residual_scale.to_numpy(),
        residual_scale.loc[list(genes)].to_numpy(),
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            "the linear model export's residual scale differs from the joint model's"
        )
    missing = sorted(set(lines) - set(linear_model.values.index))
    if missing:
        raise ValueError(f"the linear model export lacks lines {missing[:10]}")
    return linear_model


def check_linear_model_identity(
    recorded: Mapping[str, Any] | None, linear_model: LinearModelOffsets | None
) -> None:
    """Refuse a checkpoint whose recorded linear model is not the configured one."""
    if recorded is None and linear_model is None:
        return
    if recorded is None:
        raise ValueError(
            "the checkpoint was trained without a linear model but the config names "
            f"linear_model run {linear_model.identity['run_id']}"
        )
    if linear_model is None:
        raise ValueError(
            "the checkpoint was trained on linear model run "
            f"{recorded['run_id']} but no linear_model is configured"
        )
    if dict(recorded) != linear_model.identity:
        raise ValueError(
            f"the checkpoint was trained on linear model run {recorded['run_id']}, the "
            f"config's export is linear_model run {linear_model.identity['run_id']}"
        )
