"""Pre-registered configuration for the Exp13 Stage 1 response-model trainer.

``docs/specs/2026-08-17-exp13-geneeffect-residual-protocol.md`` §7 requires four
things pinned in ``configs/experiments/13_geneeffect_226/stage1_response.yaml``
*before* Stage 1 trains -- "record converged metrics and freeze" is not a gate:
per-anchor response metrics, a held-out gene set per anchor (``heldout_fraction``/
``heldout_seed`` in :class:`Stage1TrainConfig`), a required improvement margin
over a basal-copy prediction and over a null shuffle, and the four-line weighting
used to combine anchor losses.

Following ``src/ddgcn/config.py`` (the repo's raising-loader reference), every
YAML key is required and every unrecognized key -- at the top level, inside
``train:``, or inside ``freeze_thresholds:`` -- raises ``ValueError``. A
misspelled key must never silently fall back to a dataclass default
(``CLAUDE.md``, "Silent failures -- the dominant risk"). The two margin floats
additionally raise if their YAML value is ``null``: a run cannot start on
unpre-registered thresholds.
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
class Stage1FreezeThresholds:
    """The four items spec §7 requires pre-registered before Stage 1 trains.

    ``anchor_weights`` is stored as a tuple of ``(model_id, weight)`` pairs
    rather than a plain ``dict`` so the frozen dataclass stays hashable; use
    ``dict(thresholds.anchor_weights)`` for lookup.
    """

    anchor_weights: tuple[tuple[str, float], ...]
    required_anchor_metrics: tuple[str, ...]
    min_improvement_over_basal_copy: float | None
    min_improvement_over_null_shuffle: float | None

    def __post_init__(self) -> None:
        # A null margin no longer blocks the run: the key must still be
        # present, but leaving it unset means "not pre-registered", and the
        # run proceeds ungated. What a null must never do is read as a pass --
        # `gate_is_preregistered` is False, and the run's freeze_gate payload
        # records `evaluated: false` rather than a verdict it did not earn.
        for name in (
            "min_improvement_over_basal_copy",
            "min_improvement_over_null_shuffle",
        ):
            value = getattr(self, name)
            if value is not None and float(value) < 0:
                raise ValueError(f"{name} must be >= 0 when set, got {value}")
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

    @property
    def gate_is_preregistered(self) -> bool:
        """Whether both §7 margins were declared before the run.

        False means the run is ungated: its metrics are still computed and
        reported, but no §7 pass can be claimed from them.
        """
        return (
            self.min_improvement_over_basal_copy is not None
            and self.min_improvement_over_null_shuffle is not None
        )


@dataclass(frozen=True)
class Stage1Config:
    """A fully-loaded, validated Stage 1 config plus its source provenance."""

    train: Stage1TrainConfig
    thresholds: Stage1FreezeThresholds
    source_path: Path
    source_sha256: str

    @property
    def gate_is_preregistered(self) -> bool:
        """Whether this run is §7-gated; delegates to the thresholds."""
        return self.thresholds.gate_is_preregistered

    def freeze_thresholds_payload(self) -> dict[str, Any]:
        """The JSON-ready payload for ``stage1_freeze_thresholds.json`` (spec §10)."""
        t = self.thresholds
        return {
            "anchor_weights": dict(t.anchor_weights),
            "required_anchor_metrics": list(t.required_anchor_metrics),
            "min_improvement_over_basal_copy": t.min_improvement_over_basal_copy,
            "min_improvement_over_null_shuffle": t.min_improvement_over_null_shuffle,
            "gate_is_preregistered": self.gate_is_preregistered,
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


def _build_thresholds(raw: Any, path: Path) -> Stage1FreezeThresholds:
    raw = _require_mapping(raw, f"{path}: 'freeze_thresholds'")
    valid = {f.name for f in fields(Stage1FreezeThresholds)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) in 'freeze_thresholds': {sorted(unknown)}"
        )
    missing = valid - set(raw)
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) in 'freeze_thresholds': {sorted(missing)}"
        )
    anchor_weights_raw = _require_mapping(
        raw["anchor_weights"], f"{path}: 'freeze_thresholds.anchor_weights'"
    )
    return Stage1FreezeThresholds(
        anchor_weights=tuple(sorted(anchor_weights_raw.items())),
        required_anchor_metrics=tuple(raw["required_anchor_metrics"] or ()),
        min_improvement_over_basal_copy=raw["min_improvement_over_basal_copy"],
        min_improvement_over_null_shuffle=raw["min_improvement_over_null_shuffle"],
    )


def load_stage1_config(path: str | Path) -> Stage1Config:
    """Load and validate a Stage 1 config, raising on any deviation from spec §7.

    Raises:
        ValueError: An unknown key appears anywhere, a required key is missing,
            or a pre-registered margin threshold is ``null``.
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = yaml.safe_load(raw_bytes) or {}
    raw = _require_mapping(raw, f"{path}: top level")

    allowed_top = {"train", "freeze_thresholds"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"{path}: unknown top-level key(s): {sorted(unknown_top)}")
    missing_top = allowed_top - set(raw)
    if missing_top:
        raise ValueError(
            f"{path}: missing required top-level key(s): {sorted(missing_top)}"
        )

    train = _build_train_config(raw["train"], path)
    thresholds = _build_thresholds(raw["freeze_thresholds"], path)

    return Stage1Config(
        train=train,
        thresholds=thresholds,
        source_path=path,
        source_sha256=source_sha256,
    )
