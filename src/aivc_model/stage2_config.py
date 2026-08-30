"""Strict configuration contract for the Exp13 Stage 2 full-model run."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class Stage2SeedsConfig:
    train: int
    collator: int
    projection: int


@dataclass(frozen=True)
class Stage2FeaturesConfig:
    delta_proj_dim: int
    summary_dim: int
    q_sc_dim: int
    esm2_dim: int
    context_dim: int
    cells_per_context: int
    cell_set_len: int
    standardizer_fit_split: str

    def __post_init__(self) -> None:
        actual = (
            self.delta_proj_dim,
            self.summary_dim,
            self.q_sc_dim,
            self.esm2_dim,
            self.context_dim,
        )
        expected = (256, 6, 3, 1280, 5120)
        if actual != expected:
            raise ValueError(f"feature dimensions must be {expected}, got {actual}")
        if self.cells_per_context != 128:
            raise ValueError("cells_per_context must be 128")
        if self.cell_set_len != 64:
            raise ValueError("cell_set_len must be 64")
        if self.standardizer_fit_split != "train":
            raise ValueError("standardizer_fit_split must be 'train'")


@dataclass(frozen=True)
class Stage2LossConfig:
    huber_delta: float
    beta: float
    g_var_percentile: float
    minimum_observations: int

    def __post_init__(self) -> None:
        if self.huber_delta != 1.0:
            raise ValueError("huber_delta must be 1.0")
        if self.beta != 1.0:
            raise ValueError("beta must be 1.0")
        if self.g_var_percentile != 0.75:
            raise ValueError("g_var_percentile must be 0.75")
        if self.minimum_observations != 5:
            raise ValueError("minimum_observations must be 5")


@dataclass(frozen=True)
class Stage2WarmupConfig:
    optimizer: str
    learning_rate: float
    max_epochs: int
    patience: int
    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        expected = ("AdamW", 1e-3, 100, 10, 256, 2)
        actual = (
            self.optimizer,
            self.learning_rate,
            self.max_epochs,
            self.patience,
            self.hidden_dim,
            self.num_layers,
        )
        if actual != expected:
            raise ValueError(f"warmup settings must be {expected}, got {actual}")


@dataclass(frozen=True)
class Stage2JointConfig:
    conditions_per_rank: int
    genes_per_batch: int
    contexts_per_gene: int
    response_batch_size: int
    state_learning_rate: float
    esm_adapter_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    grad_clip: float
    max_epochs: int
    patience: int

    def __post_init__(self) -> None:
        if self.genes_per_batch * self.contexts_per_gene != self.conditions_per_rank:
            raise ValueError(
                "genes_per_batch * contexts_per_gene must equal conditions_per_rank"
            )
        expected = (256, 8, 32, 64, 1e-6, 1e-5, 1e-4, 0.01, 1.0, 10, 2)
        actual = (
            self.conditions_per_rank,
            self.genes_per_batch,
            self.contexts_per_gene,
            self.response_batch_size,
            self.state_learning_rate,
            self.esm_adapter_learning_rate,
            self.head_learning_rate,
            self.weight_decay,
            self.grad_clip,
            self.max_epochs,
            self.patience,
        )
        if actual != expected:
            raise ValueError(f"joint settings must be {expected}, got {actual}")


@dataclass(frozen=True)
class Stage2DistributedConfig:
    mixed_precision: str

    def __post_init__(self) -> None:
        if self.mixed_precision != "bf16":
            raise ValueError("distributed mixed_precision must be bf16")


@dataclass(frozen=True)
class Stage2LambdaCalibrationConfig:
    train_batches: int
    statistic: str
    clip_min: float
    clip_max: float

    def __post_init__(self) -> None:
        expected = (8, "median_gradient_norm_ratio", 1e-3, 1e3)
        actual = (self.train_batches, self.statistic, self.clip_min, self.clip_max)
        if actual != expected:
            raise ValueError(
                f"lambda calibration settings must be {expected}, got {actual}"
            )


@dataclass(frozen=True)
class Stage2SelectionConfig:
    metric: str
    direction: str
    record_response_metrics: bool
    response_hard_guard: bool

    def __post_init__(self) -> None:
        expected = ("validation_macro_per_gene_spearman", "maximize", True, False)
        actual = (
            self.metric,
            self.direction,
            self.record_response_metrics,
            self.response_hard_guard,
        )
        if actual != expected:
            raise ValueError(f"selection settings must be {expected}, got {actual}")


@dataclass(frozen=True)
class Stage2RunScopeConfig:
    model: str
    num_seeds: int
    ablations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.model != "full" or self.num_seeds != 1 or self.ablations:
            raise ValueError(
                "run_scope must select the full model, one seed, no ablations"
            )


@dataclass(frozen=True)
class Stage2PathsConfig:
    split: Path
    gene_effect: Path
    copy_prior: Path
    copy_prior_manifest: Path
    source_registry: Path
    tx1_registration: Path
    tx1_cache: Path
    q_sc_cache: Path
    stage1_checkpoint: Path
    stage1_config: Path
    esm2_embeddings: Path
    esm2_universe_manifest: Path
    esm2_provenance_manifest: Path
    esm2_uniprot_mapping_json: Path
    esm2_uniprot_mapping_csv: Path
    state_hparams: Path
    state_model_dir: Path
    cell_line_manifest: Path
    perturbseq_sources: Path
    response_cache: Path
    output_root: Path


@dataclass(frozen=True)
class Stage2Config:
    seeds: Stage2SeedsConfig
    features: Stage2FeaturesConfig
    loss: Stage2LossConfig
    warmup: Stage2WarmupConfig
    joint: Stage2JointConfig
    distributed: Stage2DistributedConfig
    lambda_calibration: Stage2LambdaCalibrationConfig
    selection: Stage2SelectionConfig
    run_scope: Stage2RunScopeConfig
    paths: Stage2PathsConfig
    source_path: Path
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        """Return the complete validated config and provenance as JSON-ready data."""
        return _paths_to_strings(asdict(self))


ConfigT = TypeVar("ConfigT")


def _require_mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
    return raw


def _build_section(cls: type[ConfigT], raw: Any, path: Path, name: str) -> ConfigT:
    raw = _require_mapping(raw, f"{path}: '{name}'")
    valid = {field.name for field in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{path}: unknown key(s) in '{name}': {sorted(unknown)}")
    missing = valid - set(raw)
    if missing:
        raise ValueError(
            f"{path}: missing required key(s) in '{name}': {sorted(missing)}"
        )
    if cls is Stage2RunScopeConfig:
        raw = {**raw, "ablations": tuple(raw["ablations"] or ())}
    if cls is Stage2PathsConfig:
        raw = {key: Path(value) for key, value in raw.items()}
    return cls(**raw)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value


def load_stage2_config(
    path: str | Path, *, validate_paths: bool = False
) -> Stage2Config:
    """Load the frozen Stage-2 contract, raising on missing or unknown keys."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = _require_mapping(yaml.safe_load(raw_bytes) or {}, f"{path}: top level")

    sections: tuple[tuple[str, type[Any]], ...] = (
        ("seeds", Stage2SeedsConfig),
        ("features", Stage2FeaturesConfig),
        ("loss", Stage2LossConfig),
        ("warmup", Stage2WarmupConfig),
        ("joint", Stage2JointConfig),
        ("distributed", Stage2DistributedConfig),
        ("lambda_calibration", Stage2LambdaCalibrationConfig),
        ("selection", Stage2SelectionConfig),
        ("run_scope", Stage2RunScopeConfig),
        ("paths", Stage2PathsConfig),
    )
    allowed = {name for name, _ in sections}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s): {sorted(unknown)}")
    missing = allowed - set(raw)
    if missing:
        raise ValueError(
            f"{path}: missing required top-level key(s): {sorted(missing)}"
        )

    built = {name: _build_section(cls, raw[name], path, name) for name, cls in sections}
    config = Stage2Config(
        **built,
        source_path=path,
        source_sha256=source_sha256,
    )
    if validate_paths:
        missing_paths = [
            str(value)
            for value in asdict(config.paths).values()
            if not Path(value).exists()
        ]
        if missing_paths:
            raise ValueError(f"configured paths do not exist: {sorted(missing_paths)}")
    return config
