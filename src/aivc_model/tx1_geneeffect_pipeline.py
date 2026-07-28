"""Phase D Task 5: the arm-config schema and preflight validation
(``--dry-run``'s job) for the launch script `scripts/train_tx1_geneeffect_head
.py`.

Every prior Phase D task built one library-level seam and tested it in
isolation (Task 1: line-role selection + DepMap loading; Task 2: forward-only
ST + predicted-response generation/cache; Task 3: head training; Task 4: the
Phase F predictions table). Nothing before this module actually WIRED them
together against a real, multi-line run -- that wiring is the sibling module
:mod:`aivc_model.tx1_geneeffect_pipeline_run` (split out purely for
CLAUDE.md's 600-line-per-file rule; this module has no dependency on it).

This module owns the config schema (:class:`Tx1GeneEffectHeadConfig`) and
every check ``--dry-run`` performs: config keys, path existence, widths,
roles, and gene coverage -- all without materializing a single cell matrix
(no call here ever reaches ``tx1_embed_cache.load_line_cache`` or
``tx1_predicted_response.generate_predicted_response*``).

Every field this pipeline actually reads is REQUIRED at parse time (via
:func:`_require`), not defaulted -- this repo's config loading is otherwise
entirely ``.get(key, default)`` everywhere else, which is exactly how a
misspelled key goes unnoticed (D10). A missing/misspelled key here raises
immediately, before ``--dry-run`` even starts its own checks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pandas as pd
import yaml

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH
from aivc_model.tx1_geneeffect_data import (
    LineRoleSplit,
    build_line_role_split,
    load_validation_line_registration,
)
from aivc_model.tx1_geneeffect_eval import (
    DEFAULT_PRIMARY_METHOD,
    K_SCHEDULE,
    load_slice,
)
from aivc_model.tx1_geneeffect_train import SELECTION_METRIC_NAME
from aivc_model.tx1_predicted_response import (
    ARM_HVG,
    ARM_TX1,
    resolve_device,
    resolve_genes_against_vocabulary,
)

_LOGGER = logging.getLogger(__name__)

_VALID_ARMS = frozenset({ARM_TX1, ARM_HVG})
_VALID_BACKENDS = frozenset({"linear_mock", "state_checkpoint"})

__all__ = [
    "ObjectiveConfig",
    "PhaseFConfig",
    "StateArmConfig",
    "TrainingArmConfig",
    "Tx1GeneEffectHeadConfig",
    "dry_run_summary",
    "load_pipeline_config",
    "phase_b_recorded_widths",
    "validate_config_shape",
    "validate_gene_coverage",
    "validate_roles",
    "validate_widths",
    "verify_depmap_hash",
    "verify_gene_vocabulary_authenticity",
]


# --------------------------------------------------------------------------
# Config schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StateArmConfig:
    """ST construction/loading and predicted-response generation settings."""

    backend: str
    hparams_checkpoint_path: Path | None
    st_checkpoint_path: Path | None
    known_perturbation_vectors: Path | None
    gene_vocabulary_path: Path | None
    input_dim: int
    output_dim: int
    pert_dim: int
    output_space: str | None
    cell_set_len: int
    response_window_macro_batch_size: int
    response_generation_seed: int
    #: "auto" (default) resolves to CUDA when available, else CPU
    #: (aivc_model.tx1_predicted_response.resolve_device); P1-5.
    device: str = "auto"


@dataclass(frozen=True)
class ObjectiveConfig:
    """D3: the registered objective's only knob, plus the recorded metric name."""

    lam: float
    selection_metric: str


@dataclass(frozen=True)
class TrainingArmConfig:
    """``Tx1GeneEffectHead`` training hyperparameters."""

    hidden: int
    moments: int
    learning_rate: float
    epochs: int
    seed: int


@dataclass(frozen=True)
class PhaseFConfig:
    """Settings for the D1 ``tx1_3b_st`` predictions table."""

    k_schedule: tuple[int, ...]
    method: str
    alpha: float
    residual: bool


@dataclass(frozen=True)
class Tx1GeneEffectHeadConfig:
    """One arm's full Phase D Task 5 configuration."""

    arm: str
    run_id: str
    output_dir: Path
    validation_lines_path: Path
    state: StateArmConfig
    objective: ObjectiveConfig
    training: TrainingArmConfig
    phase_f: PhaseFConfig


def _require(section: Mapping[str, object], key: str, section_name: str) -> object:
    """Raise a named ``ValueError`` for a missing config key (D10) rather
    than silently falling back to a default."""
    if key not in section:
        raise ValueError(f"config section {section_name!r} is missing key {key!r}")
    return section[key]


def _optional_path(value: object) -> Path | None:
    """An explicitly optional path field (unlike ``prepare._path_or_none``,
    this is used only for fields that are genuinely optional under some
    backend, documented as such at each call site)."""
    return Path(str(value)) if value else None


def load_pipeline_config(path: Path) -> Tx1GeneEffectHeadConfig:
    """Parse one arm's YAML config.

    Raises:
        ValueError: A required section/key is missing.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    run = raw.get("run", {})
    data = raw.get("data", {})
    state_raw = raw.get("state", {})
    objective_raw = raw.get("objective", {})
    training_raw = raw.get("training", {})
    phase_f_raw = raw.get("phase_f", {})

    state = StateArmConfig(
        backend=str(_require(state_raw, "backend", "state")),
        hparams_checkpoint_path=_optional_path(
            state_raw.get("hparams_checkpoint_path")
        ),
        st_checkpoint_path=_optional_path(state_raw.get("st_checkpoint_path")),
        known_perturbation_vectors=_optional_path(
            state_raw.get("known_perturbation_vectors")
        ),
        gene_vocabulary_path=_optional_path(state_raw.get("gene_vocabulary_path")),
        input_dim=int(_require(state_raw, "input_dim", "state")),
        output_dim=int(_require(state_raw, "output_dim", "state")),
        pert_dim=int(_require(state_raw, "pert_dim", "state")),
        output_space=state_raw.get("output_space"),
        cell_set_len=int(_require(state_raw, "cell_set_len", "state")),
        response_window_macro_batch_size=int(
            _require(state_raw, "response_window_macro_batch_size", "state")
        ),
        response_generation_seed=int(
            _require(state_raw, "response_generation_seed", "state")
        ),
        # Genuinely optional (P1-5): "auto" (the default) picks CUDA when
        # available, CPU otherwise (aivc_model.tx1_predicted_response.
        # resolve_device), mirroring exp05's own `train.device: str =
        # "auto"` convention. Unlike every `_require`d field above, a config
        # simply omitting this key is a legitimate, intended default -- not
        # the D10 misspelled-key hazard those fields guard against.
        device=str(state_raw.get("device", "auto")),
    )
    objective = ObjectiveConfig(
        lam=float(_require(objective_raw, "lam", "objective")),
        selection_metric=str(_require(objective_raw, "selection_metric", "objective")),
    )
    training = TrainingArmConfig(
        hidden=int(_require(training_raw, "hidden", "training")),
        moments=int(_require(training_raw, "moments", "training")),
        learning_rate=float(_require(training_raw, "learning_rate", "training")),
        epochs=int(_require(training_raw, "epochs", "training")),
        seed=int(_require(training_raw, "seed", "training")),
    )
    phase_f = PhaseFConfig(
        k_schedule=tuple(
            int(value) for value in _require(phase_f_raw, "k_schedule", "phase_f")
        ),
        method=str(_require(phase_f_raw, "method", "phase_f")),
        alpha=float(_require(phase_f_raw, "alpha", "phase_f")),
        residual=bool(_require(phase_f_raw, "residual", "phase_f")),
    )
    return Tx1GeneEffectHeadConfig(
        arm=str(_require(raw, "arm", "<root>")),
        run_id=str(_require(run, "run_id", "run")),
        output_dir=Path(str(_require(run, "output_dir", "run"))),
        validation_lines_path=Path(
            str(_require(data, "validation_lines_path", "data"))
        ),
        state=state,
        objective=objective,
        training=training,
        phase_f=phase_f,
    )


# --------------------------------------------------------------------------
# Validation (--dry-run's job; also run before any real training)
# --------------------------------------------------------------------------


def validate_config_shape(
    config: Tx1GeneEffectHeadConfig, *, check_paths: bool
) -> None:
    """Validate ``config``'s own field values (D10: config keys).

    ``check_paths=False`` skips filesystem existence checks -- for testing
    the checked-in configs' VALUES in an environment where the referenced
    gitignored checkpoint assets are not present (e.g. CI). The real CLI
    always uses ``check_paths=True``.

    Raises:
        ValueError: Any field is out of range, or (``check_paths``) a
            required path does not exist.
    """
    if config.arm not in _VALID_ARMS:
        raise ValueError(
            f"arm must be one of {sorted(_VALID_ARMS)}; got {config.arm!r}"
        )
    state = config.state
    if state.backend not in _VALID_BACKENDS:
        raise ValueError(
            f"state.backend must be one of {sorted(_VALID_BACKENDS)}; got "
            f"{state.backend!r}"
        )
    for field_name, value in (
        ("input_dim", state.input_dim),
        ("output_dim", state.output_dim),
        ("pert_dim", state.pert_dim),
        ("cell_set_len", state.cell_set_len),
        ("response_window_macro_batch_size", state.response_window_macro_batch_size),
    ):
        if value <= 0:
            raise ValueError(f"state.{field_name} must be positive; got {value}")
    if config.arm == ARM_TX1 and state.input_dim != EMBEDDING_WIDTH:
        raise ValueError(
            f"arm={ARM_TX1!r} requires state.input_dim={EMBEDDING_WIDTH} (the "
            f"Tx1 embedding width); got {state.input_dim}"
        )
    if config.objective.lam < 0:
        raise ValueError(f"objective.lam must be >= 0; got {config.objective.lam}")
    if config.objective.selection_metric != SELECTION_METRIC_NAME:
        raise ValueError(
            "objective.selection_metric must equal the metric selection "
            f"actually uses ({SELECTION_METRIC_NAME!r}); got "
            f"{config.objective.selection_metric!r} -- this is exactly the "
            "Wave 2 selection_metric/generation_loss mismatch this field "
            "exists to catch"
        )
    training = config.training
    if training.hidden <= 0:
        raise ValueError(f"training.hidden must be positive; got {training.hidden}")
    if training.moments != 2:
        raise ValueError(
            "training.moments must equal 2 for the fixed online mean plus "
            f"population-variance reducer; got {training.moments}"
        )
    if training.learning_rate <= 0:
        raise ValueError(
            f"training.learning_rate must be positive; got {training.learning_rate}"
        )
    if training.epochs <= 0:
        raise ValueError(f"training.epochs must be positive; got {training.epochs}")
    if list(config.phase_f.k_schedule) != list(K_SCHEDULE):
        raise ValueError(
            f"phase_f.k_schedule must equal the frozen contract {K_SCHEDULE}; "
            f"got {list(config.phase_f.k_schedule)}"
        )
    if config.phase_f.method != DEFAULT_PRIMARY_METHOD:
        raise ValueError(
            f"phase_f.method must equal {DEFAULT_PRIMARY_METHOD!r} (D1's fixed "
            f"contract); got {config.phase_f.method!r}"
        )
    if config.phase_f.alpha <= 0:
        raise ValueError(f"phase_f.alpha must be positive; got {config.phase_f.alpha}")
    if check_paths:
        _validate_configured_paths_exist(config)


def _validate_configured_paths_exist(config: Tx1GeneEffectHeadConfig) -> None:
    state = config.state
    # gene_vocabulary_path is required regardless of backend: gene-coverage
    # validation always needs it (dry-run and real-run alike).
    if state.gene_vocabulary_path is None or not state.gene_vocabulary_path.is_file():
        raise ValueError(
            f"state.gene_vocabulary_path not found: {state.gene_vocabulary_path}"
        )
    if state.backend != "state_checkpoint":
        return
    for label, path in (
        ("hparams_checkpoint_path", state.hparams_checkpoint_path),
        ("st_checkpoint_path", state.st_checkpoint_path),
    ):
        if path is None or not path.is_file():
            raise ValueError(f"state.{label} not found: {path}")
    if state.known_perturbation_vectors is not None and not (
        state.known_perturbation_vectors.is_file()
    ):
        raise ValueError(
            "state.known_perturbation_vectors not found: "
            f"{state.known_perturbation_vectors}"
        )


def phase_b_recorded_widths(tx1_cache_dir: Path) -> dict[str, int]:
    """Read one line's recorded ``embeddings.npy``/``hvg.npy`` widths from
    Phase B's cache-root ``manifest.json`` -- a cheap JSON read, never the
    arrays themselves (they are never opened here).

    Raises:
        ValueError: The manifest is missing, empty, or a recorded entry has
            no shape for one of the two arrays.
    """
    manifest_path = Path(tx1_cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"no Phase B cache manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    lines = manifest.get("lines") if isinstance(manifest, dict) else None
    if not lines:
        raise ValueError(f"Phase B cache manifest at {manifest_path} records no lines")
    arrays = next(iter(lines.values())).get("arrays", {})
    try:
        embeddings_shape = arrays["embeddings.npy"]["shape"]
        hvg_shape = arrays["hvg.npy"]["shape"]
    except KeyError as exc:
        raise ValueError(
            f"Phase B cache manifest entry at {manifest_path} is missing {exc} shape"
        ) from exc
    return {
        "embeddings_width": int(embeddings_shape[-1]),
        "hvg_width": int(hvg_shape[-1]),
    }


def validate_widths(config: Tx1GeneEffectHeadConfig, tx1_cache_dir: Path) -> None:
    """Cross-check ``config.state.input_dim`` against Phase B's own recorded
    array widths (never against the arrays themselves).

    Raises:
        ValueError: The configured width disagrees with the recorded one.
    """
    widths = phase_b_recorded_widths(tx1_cache_dir)
    expected = (
        widths["embeddings_width"] if config.arm == ARM_TX1 else widths["hvg_width"]
    )
    if config.state.input_dim != expected:
        raise ValueError(
            f"arm={config.arm!r} state.input_dim={config.state.input_dim} "
            f"disagrees with the Phase B cache's recorded width {expected}"
        )


def validate_roles(
    line_manifest_path: Path, validation_lines_path: Path
) -> tuple[pd.DataFrame, dict[str, object], LineRoleSplit]:
    """Load the manifest + validation registration and build the 28/5/9 split.

    Delegates entirely to Task 1's own guards (``load_line_manifest``,
    ``load_validation_line_registration``, ``build_line_role_split``) --
    D6/D12 are enforced there, not re-derived here.

    Returns:
        ``(manifest, validation_registration, split)``.
    """
    manifest = load_line_manifest(line_manifest_path)
    registration = load_validation_line_registration(
        validation_lines_path, manifest, line_manifest_path
    )
    split = build_line_role_split(manifest, registration)
    return manifest, registration, split


def validate_gene_coverage(
    slice_df: pd.DataFrame, gene_vocabulary_path: Path
) -> tuple[str, ...]:
    """Resolve the frozen slice's ``gene_symbol`` column against the ST
    training vocabulary, without constructing any model.

    ``gene_vocabulary_path`` is a plain JSON list of gene symbols (Phase C's
    exact training-time perturbation-adapter vocabulary; see
    :mod:`aivc_model.tx1_geneeffect_pipeline_run`'s module docstring for the
    open provenance gap -- no committed Phase C artifact is confirmed to
    hold this exact ordered list yet). A lightweight
    ``SimpleNamespace(genes=...)`` stands in for a real
    ``PerturbationVectorAdapter`` here: :func:`resolve_genes_against_vocabulary`
    only reads a ``.genes`` attribute, so this reuses the REAL resolution
    function without constructing any adapter/model.

    Raises:
        ValueError: The vocabulary file is missing or not a non-empty JSON
            list.
        UnknownPerturbationGeneError: One or more slice genes (after alias
            resolution) are outside the vocabulary.
    """
    vocabulary = json.loads(Path(gene_vocabulary_path).read_text())
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError(
            f"{gene_vocabulary_path} must contain a non-empty JSON list of gene symbols"
        )
    fake_perturbations = SimpleNamespace(genes=[str(gene) for gene in vocabulary])
    return resolve_genes_against_vocabulary(
        [str(gene) for gene in slice_df["gene_symbol"]], fake_perturbations
    )


def verify_gene_vocabulary_authenticity(
    gene_vocabulary_path: Path, st_checkpoint_path: Path | None, backend: str
) -> None:
    """Authenticate ``gene_vocabulary_path`` against Phase C's own recorded
    hash (Wave 3 Codex gate P1-3).

    :func:`validate_gene_coverage` cannot detect a vocabulary whose genes are
    merely reordered: ``PerturbationVectorAdapter`` keys its trainable
    "missing" parameters by construction-time list POSITION, not gene
    identity (fix rounds 5/6), so a reordered file passes gene-coverage AND
    ``load_state_dict`` (identical ``gN`` keys/shapes) while silently binding
    every positional weight to the wrong gene symbol.

    ``aivc_model.train._save_model_checkpoint`` emits ``gene_vocabulary.json``
    beside ``pytorch_model.bin`` and records its sha256 as
    ``gene_vocabulary_sha256`` in that checkpoint's sibling ``metadata.json``.
    This recomputes ``gene_vocabulary_path``'s sha256 and compares it against
    that recorded value -- any reordering, edit, or wrong-file substitution
    changes the digest and is rejected. A no-op when
    ``backend != "state_checkpoint"`` (``linear_mock`` has no real checkpoint
    to authenticate against).

    Args:
        gene_vocabulary_path: The configured ``state.gene_vocabulary_path``.
        st_checkpoint_path: The configured ``state.st_checkpoint_path``; its
            sibling ``metadata.json`` is read from the same directory.
        backend: ``config.state.backend``.

    Raises:
        ValueError: ``backend == "state_checkpoint"`` and: ``st_checkpoint_
            path`` is ``None``; the checkpoint has no sibling
            ``metadata.json``; that metadata has no ``gene_vocabulary_sha256``
            (predates this fix -- fail loudly rather than trust an
            unauthenticated file); or the vocabulary file's sha256 disagrees
            with the recorded one.
    """
    if backend != "state_checkpoint":
        return
    if st_checkpoint_path is None:
        raise ValueError(
            "state.st_checkpoint_path is required to authenticate "
            "state.gene_vocabulary_path for backend=state_checkpoint"
        )
    metadata_path = Path(st_checkpoint_path).parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"no metadata.json beside checkpoint {st_checkpoint_path} "
            f"(expected {metadata_path}); cannot authenticate "
            "state.gene_vocabulary_path (P1-3). This checkpoint predates "
            "Phase C's gene_vocabulary.json/metadata.json emission fix, or "
            "is otherwise missing its sidecar metadata -- regenerate it "
            "with the current Phase C pipeline before using it in Phase D"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = (
        metadata.get("gene_vocabulary_sha256") if isinstance(metadata, dict) else None
    )
    if not expected:
        raise ValueError(
            f"{metadata_path} has no gene_vocabulary_sha256 -- this "
            "checkpoint predates Phase C's ordered-vocabulary emission fix "
            "(P1-3) and cannot be safely authenticated; regenerate the "
            "checkpoint with the current Phase C pipeline before using it "
            "here rather than assuming the configured "
            "state.gene_vocabulary_path is trustworthy"
        )
    observed = sha256_file(Path(gene_vocabulary_path))
    if observed != expected:
        raise ValueError(
            f"state.gene_vocabulary_path {gene_vocabulary_path} sha256 "
            f"{observed} does not match checkpoint {st_checkpoint_path}'s "
            f"recorded gene_vocabulary_sha256 {expected} ({metadata_path}); "
            "refusing to load -- a mismatched or reordered vocabulary "
            "silently binds trainable missing-gene weights to the wrong "
            "gene symbol (P1-3)"
        )


def verify_depmap_hash(
    depmap_gene_effect_path: Path, phase_a_registration: Mapping[str, object]
) -> None:
    """Standalone D5 content-hash check (no CSV parse), for a cheap dry run.

    Raises:
        ValueError: The file's sha256 does not match the registered hash.
    """
    expected = str(phase_a_registration["sources"]["depmap_gene_effect"]["sha256"])
    observed = sha256_file(Path(depmap_gene_effect_path))
    if observed != expected:
        raise ValueError(
            "DepMap GeneEffect source failed content-hash resolution (D5): "
            f"expected sha256 {expected}, got {observed} at "
            f"{depmap_gene_effect_path}"
        )


def dry_run_summary(
    config: Tx1GeneEffectHeadConfig,
    *,
    line_manifest_path: Path,
    phase_a_dir: Path,
    tx1_cache_dir: Path,
    depmap_gene_effect_path: Path,
) -> dict[str, object]:
    """Validate every input this pipeline needs, without materializing a
    single cell matrix: config keys, path existence, widths, roles, and gene
    coverage. Never calls ``load_line_cache``/``generate_predicted_response*``.

    Raises:
        ValueError: Any check fails (config shape, paths, widths, roles,
            DepMap hash).
        UnknownPerturbationGeneError: A slice gene is outside the ST
            vocabulary.
    """
    validate_config_shape(config, check_paths=True)
    validate_widths(config, tx1_cache_dir)
    _manifest, _registration, split = validate_roles(
        line_manifest_path, config.validation_lines_path
    )
    slice_df = load_slice(Path(phase_a_dir) / "differentially_essential_slice.csv")
    resolved_genes = validate_gene_coverage(slice_df, config.state.gene_vocabulary_path)
    verify_gene_vocabulary_authenticity(
        config.state.gene_vocabulary_path,
        config.state.st_checkpoint_path,
        config.state.backend,
    )
    phase_a_registration = json.loads(
        (Path(phase_a_dir) / "phase_a_registration.json").read_text()
    )
    verify_depmap_hash(depmap_gene_effect_path, phase_a_registration)
    return {
        "arm": config.arm,
        "run_id": config.run_id,
        "state_backend": config.state.backend,
        "state_device": str(resolve_device(config.state.device)),
        "state_input_dim": config.state.input_dim,
        "state_output_dim": config.state.output_dim,
        "objective_lam": config.objective.lam,
        "selection_metric": config.objective.selection_metric,
        "k_schedule": list(config.phase_f.k_schedule),
        "n_train_lines": len(split.train_model_ids),
        "n_validation_lines": len(split.validation_model_ids),
        "n_test_lines": len(split.test_model_ids),
        "n_anchor_lines": len(split.anchor_model_ids),
        "n_slice_genes": int(len(slice_df)),
        "n_resolved_genes": len(resolved_genes),
        "run_dir": str(Path(config.output_dir) / "runs" / config.run_id),
    }
