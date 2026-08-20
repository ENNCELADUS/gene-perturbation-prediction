"""Configuration for the Exp13 Stage 1 response-model trainer.

``docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`` §7 pins the
per-anchor response metrics, held-out gene split, and four-line objective weights
before Stage 1 trains.

Following ``src/ddgcn/config.py`` (the repo's raising-loader reference), every
YAML key is required and every unrecognized key -- at the top level, inside
``train:``, or inside ``objective:`` -- raises ``ValueError``. A misspelled key
must never silently fall back to a dataclass default (``CLAUDE.md``, "Silent
failures -- the dominant risk"). Stage 1 reports its registered metrics and
baselines without a separate pass/fail threshold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

_FLOAT32_MATMUL_PRECISIONS = (None, "highest", "high", "medium")


@dataclass(frozen=True)
class Stage1TrainConfig:
    """Hyperparameters for the Stage 1 response-model trainer.

    Every field here was previously a CLI flag on
    ``scripts/train_geneeffect_response_model.py``; the trainer's PATH flags
    (``--split-json``, ``--out-dir``, etc.) are unaffected and stay on the CLI.
    """

    max_epochs: int
    patience: int
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_bag: int = 128
    gene_batch_size: int = 32
    validation_gene_batch_size: int = 1
    grad_clip: float = 1.0
    train_seed: int = 20260818
    collator_seed: int = 20260818
    data_seed: int = 42
    heldout_seed: int = 13
    heldout_fraction: float = 0.1
    log_every: int = 50
    float32_matmul_precision: str | None = "high"
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False
    max_cells_per_gene: int | None = 128
    total_cells_per_line: int | None = None
    pert_dim: int = 2024
    max_esm2_drop_fraction: float = 0.10
    w_mean_delta: float = 1.0
    w_energy: float = 1.0

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.max_bag < 2:
            raise ValueError(
                "max_bag must be >= 2 -- a distributional loss needs >= 2 cells"
            )
        if self.gene_batch_size < 1:
            raise ValueError("gene_batch_size must be >= 1")
        if self.validation_gene_batch_size < 1:
            raise ValueError("validation_gene_batch_size must be >= 1")
        if not 0 < self.heldout_fraction < 1:
            raise ValueError("heldout_fraction must be in (0, 1)")
        if self.w_mean_delta < 0 or self.w_energy < 0:
            raise ValueError("loss weights must be non-negative")
        if self.w_mean_delta == 0 and self.w_energy == 0:
            raise ValueError("at least one of w_mean_delta, w_energy must carry weight")
        if self.float32_matmul_precision not in _FLOAT32_MATMUL_PRECISIONS:
            raise ValueError(
                "float32_matmul_precision must be one of "
                f"{_FLOAT32_MATMUL_PRECISIONS}, got {self.float32_matmul_precision!r}"
            )


@dataclass(frozen=True)
class Stage1ObjectiveConfig:
    """The Stage 1 metrics and four-anchor objective weights.

    ``anchor_weights`` is stored as a tuple of ``(model_id, weight)`` pairs
    rather than a plain ``dict`` so the frozen dataclass stays hashable; use
    ``dict(objective.anchor_weights)`` for lookup.
    """

    anchor_weights: tuple[tuple[str, float], ...]
    required_anchor_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        weights = dict(self.anchor_weights)
        if not weights:
            raise ValueError("anchor_weights must be non-empty")
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError(
                f"anchor_weights must all be > 0, got {dict(self.anchor_weights)}"
            )
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"anchor_weights must sum to 1.0, got {total}")
        if not self.required_anchor_metrics:
            raise ValueError("required_anchor_metrics must be non-empty")
        expected_metrics = {"mean_delta_mse", "energy_distance"}
        actual_metrics = set(self.required_anchor_metrics)
        if actual_metrics != expected_metrics:
            raise ValueError(
                "required_anchor_metrics must contain exactly "
                f"{sorted(expected_metrics)}, got {sorted(actual_metrics)}"
            )


@dataclass(frozen=True)
class Stage1Config:
    """A fully-loaded, validated Stage 1 config plus its source provenance."""

    train: Stage1TrainConfig
    objective: Stage1ObjectiveConfig
    source_path: Path
    source_sha256: str

    def objective_payload(self) -> dict[str, Any]:
        """The JSON-ready payload for ``stage1_objective.json``."""
        objective = self.objective
        return {
            "anchor_weights": dict(objective.anchor_weights),
            "required_anchor_metrics": list(objective.required_anchor_metrics),
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
        }


def _require_mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
    return raw


def _build_train_config(raw: Any, path: Path) -> Stage1TrainConfig:
    raw = _require_mapping(raw, f"{path}: 'train'")
    valid = {f.name for f in fields(Stage1TrainConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{path}: unknown key(s) in 'train': {sorted(unknown)}")
    missing = valid - set(raw)
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) in 'train': {sorted(missing)}"
        )
    return Stage1TrainConfig(**raw)


def _build_objective(raw: Any, path: Path) -> Stage1ObjectiveConfig:
    raw = _require_mapping(raw, f"{path}: 'objective'")
    valid = {f.name for f in fields(Stage1ObjectiveConfig)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{path}: unknown key(s) in 'objective': {sorted(unknown)}")
    missing = valid - set(raw)
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) in 'objective': {sorted(missing)}"
        )
    anchor_weights_raw = _require_mapping(
        raw["anchor_weights"], f"{path}: 'objective.anchor_weights'"
    )
    return Stage1ObjectiveConfig(
        anchor_weights=tuple(sorted(anchor_weights_raw.items())),
        required_anchor_metrics=tuple(raw["required_anchor_metrics"] or ()),
    )


def load_stage1_config(path: str | Path) -> Stage1Config:
    """Load and validate a Stage 1 config, raising on any deviation from spec §7.

    Raises:
        ValueError: An unknown key appears anywhere or a required key is missing.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes) or {}
    raw = _require_mapping(raw, f"{path}: top level")

    allowed_top = {"train", "objective"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level key(s): {sorted(unknown_top)}")
    missing_top = allowed_top - set(raw)
    if missing_top:
        raise ValueError(
            f"{path}: missing required top-level key(s): {sorted(missing_top)}"
        )

    train = _build_train_config(raw["train"], path)
    objective = _build_objective(raw["objective"], path)

    return Stage1Config(
        train=train,
        objective=objective,
        source_path=path,
        source_sha256=source_sha256,
    )
