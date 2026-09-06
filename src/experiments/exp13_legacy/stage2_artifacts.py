"""Fail-closed lifecycle and integrity checks for Exp13 Stage 2 runs.

This module deliberately knows nothing about model architecture or training.  It
owns the small but load-bearing filesystem contract shared by the Stage 2 runner:
fresh run directories, atomic JSON writes, terminal sentinels, and verification
that every required output belongs to the same completed run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

REQUIRED_STAGE2_OUTPUTS: tuple[str, ...] = (
    "config_snapshot.json",
    "cell_line_geneeffect_226_split.json",
    "stage1_objective.json",
    "esm2_gene_universe_manifest.json",
    "esm2_provenance_manifest.json",
    "esm2_uniprot_mapping.json",
    "esm2_uniprot_mapping.csv",
    "g_var_manifest.json",
    "feature_schema.json",
    "projection.npz",
    "standardizer.npz",
    "residual_targets.npz",
    "response_targets/lineage.json",
    "backbone_load_report.json",
    "lambda_calibration.json",
    "feature_generation.json",
    "checkpoint_selection.json",
    "geneeffect_residual_predictions.csv",
    "geneeffect_residual_metrics.json",
    "run_manifest.json",
    "condition_features/stage1_frozen/manifest.json",
    "condition_features/stage2_selected/manifest.json",
    "warmup/training/best/head.pt",
    "warmup/training/best/metadata.json",
    "warmup/training/train_log.csv",
    "joint/training/best/e2e_state.pt",
    "joint/training/best/metadata.json",
    "joint/training/train_log.csv",
    "model_package/e2e_state.pt",
    "model_package/model_manifest.json",
)

_STAGE2_METHODS = frozenset(
    {
        "e2e_full",
        "gene_mean",
        "copy_prior",
        "nearest_line[z_c]",
        "context_pca_ridge[z_c]",
    }
)
_BASELINE_METHODS = _STAGE2_METHODS - {"e2e_full"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically replace *path* with canonical, newline-terminated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Stage2RunLayout:
    """Resolved paths for one immutable Stage 2 run directory."""

    root: Path

    @property
    def warmup(self) -> Path:
        return self.root / "warmup"

    @property
    def joint(self) -> Path:
        return self.root / "joint"

    @property
    def condition_features(self) -> Path:
        return self.root / "condition_features"

    @property
    def model_package(self) -> Path:
        return self.root / "model_package"

    @property
    def complete(self) -> Path:
        return self.root / "complete.json"

    @property
    def failure(self) -> Path:
        return self.root / "failure.json"


def prepare_run_dir(path: Path, *, resume: bool = False) -> Stage2RunLayout:
    """Create a fresh run directory or validate an explicitly resumed one.

    A terminal run is never resumable: reusing it could place new checkpoints
    beside old predictions.  An incomplete run may be resumed only when it has
    a run manifest and no failure marker; failures require a new run id so the
    evidence from the failed attempt stays intact.
    """
    root = Path(path)
    if not root.exists():
        root.mkdir(parents=True)
    elif not resume:
        raise FileExistsError(
            f"run directory {root} already exists; choose a fresh run id or use "
            "resume=True for a verified incomplete run"
        )
    elif (root / "complete.json").exists():
        raise ValueError(f"completed run {root} cannot be resumed")
    elif (root / "failure.json").exists():
        raise ValueError(f"failed run {root} cannot be resumed; choose a fresh run id")
    elif not (root / "run_manifest.json").is_file():
        raise ValueError(
            f"cannot resume {root}: run_manifest.json is absent, so run identity "
            "cannot be authenticated"
        )
    layout = Stage2RunLayout(root=root)
    for directory in (
        layout.warmup,
        layout.joint,
        layout.condition_features,
        layout.model_package,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def _require_artifacts(root: Path, relative_paths: Sequence[str]) -> None:
    """Require every declared output to exist as a nonempty file."""
    root = Path(root)
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Stage 2 artifact is missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"required Stage 2 artifact is empty: {path}")


def _read_run_id(root: Path) -> str:
    manifest_path = Path(root) / "run_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"run manifest is missing or unreadable: {exc}") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_manifest.json must contain a non-empty run_id")
    return run_id


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"required JSON artifact is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"required JSON artifact must contain an object: {path}")
    return payload


def _load_checkpoint_state(path: Path, label: str) -> Mapping[str, object]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise ValueError(f"{label} checkpoint cannot be loaded: {path}") from exc
    if (
        not isinstance(state, dict)
        or not state
        or any(not isinstance(key, str) or not key for key in state)
        or any(not isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError(f"{label} checkpoint must be a nonempty tensor state dict")
    for key, value in state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} checkpoint tensor is nonfinite: {key}")
    return state


def _require_equal_checkpoint_states(
    selected: Mapping[str, object], packaged: Mapping[str, object]
) -> None:
    import torch

    if set(selected) != set(packaged):
        raise ValueError("packaged checkpoint state keys differ from selected joint")
    for key in selected:
        left = selected[key]
        right = packaged[key]
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not torch.equal(left, right)
        ):
            raise ValueError(
                f"packaged checkpoint tensor differs from selected joint: {key}"
            )


def _verify_selected_checkpoint(
    root: Path,
    *,
    training_dir: str,
    filename: str,
    selection: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    checkpoint = root / training_dir / "best" / filename
    metadata_path = checkpoint.parent / "metadata.json"
    metadata = _read_json_object(metadata_path)
    selection_name = "validation_macro_per_gene_spearman"
    if (
        metadata.get("selection_name") != selection_name
        or metadata.get("selection_direction") != "maximize"
    ):
        raise ValueError(
            f"selected checkpoint metric contract mismatch: {metadata_path}"
        )
    if metadata.get("epoch") != selection.get("best_epoch"):
        raise ValueError(f"selected checkpoint epoch mismatch: {metadata_path}")
    metadata_metric = metadata.get("metric_value")
    selection_metric = selection.get("best_metric")
    if (
        isinstance(metadata_metric, bool)
        or isinstance(selection_metric, bool)
        or not isinstance(metadata_metric, (int, float))
        or not isinstance(selection_metric, (int, float))
        or not math.isfinite(float(metadata_metric))
        or not math.isfinite(float(selection_metric))
        or not math.isclose(
            float(metadata_metric),
            float(selection_metric),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"selected checkpoint metric mismatch: {metadata_path}")
    history_path = root / training_dir / "train_log.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = [
        row for row in rows if row.get("epoch") == str(metadata.get("epoch"))
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            f"selected checkpoint epoch is absent from training history: {history_path}"
        )
    try:
        history_metric = float(selected_rows[0][selection_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"training history has invalid selection metric: {history_path}"
        ) from exc
    if not math.isfinite(history_metric) or not math.isclose(
        history_metric,
        float(metadata_metric),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"training history selected metric mismatch: {history_path}")
    return metadata, _load_checkpoint_state(checkpoint, f"selected {training_dir}")


def _verify_lambda_calibration(payload: Mapping[str, object]) -> None:
    lambda_dep = payload.get("lambda_dep")
    if (
        isinstance(lambda_dep, bool)
        or not isinstance(lambda_dep, (int, float))
        or not math.isfinite(float(lambda_dep))
        or not 1e-3 <= float(lambda_dep) <= 1e3
    ):
        raise ValueError("lambda_dep must be finite and clipped to [1e-3, 1e3]")
    calibration_values: dict[str, list[float]] = {}
    for name in (
        "raw_ratios",
        "response_gradient_norms",
        "dependency_gradient_norms",
    ):
        values = payload.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in values
            )
        ):
            raise ValueError(
                f"lambda calibration {name} must contain 8 positive finite values"
            )
        calibration_values[name] = [float(value) for value in values]
    ratios = calibration_values["raw_ratios"]
    response_norms = calibration_values["response_gradient_norms"]
    dependency_norms = calibration_values["dependency_gradient_norms"]
    if any(
        not math.isclose(ratio, response / dependency, rel_tol=1e-12, abs_tol=1e-12)
        for ratio, response, dependency in zip(
            ratios, response_norms, dependency_norms, strict=True
        )
    ):
        raise ValueError("lambda calibration ratios do not match gradient norms")
    ordered = sorted(ratios)
    median_ratio = 0.5 * (ordered[3] + ordered[4])
    expected_lambda = min(1e3, max(1e-3, median_ratio))
    if not math.isclose(
        float(lambda_dep), expected_lambda, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("lambda_dep does not match the clipped median ratio")


def _verify_distributed_runtime(value: object) -> Mapping[str, object]:
    runtime = _require_mapping(value, "distributed_runtime")
    expected_fields = {
        "world_size",
        "mixed_precision",
        "ddp_static_graph",
        "ddp_find_unused_parameters",
        "conditions_per_rank",
        "global_conditions_per_step",
        "rank_topology",
    }
    if set(runtime) != expected_fields:
        raise ValueError("distributed_runtime fields mismatch")
    world_size = runtime.get("world_size")
    conditions_per_rank = runtime.get("conditions_per_rank")
    global_conditions = runtime.get("global_conditions_per_step")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in {2, 4}
    ):
        raise ValueError("distributed_runtime world_size must be 2 or 4")
    if (
        isinstance(conditions_per_rank, bool)
        or not isinstance(conditions_per_rank, int)
        or conditions_per_rank <= 0
    ):
        raise ValueError(
            "distributed_runtime conditions_per_rank must be a positive int"
        )
    if (
        isinstance(global_conditions, bool)
        or not isinstance(global_conditions, int)
        or global_conditions != world_size * conditions_per_rank
    ):
        raise ValueError(
            "distributed_runtime global_conditions_per_step is not derived from "
            "world_size and conditions_per_rank"
        )
    mixed_precision = runtime.get("mixed_precision")
    if not isinstance(mixed_precision, str) or not mixed_precision:
        raise ValueError("distributed_runtime mixed_precision must be nonempty")
    if runtime.get("ddp_static_graph") is not True:
        raise ValueError("distributed_runtime ddp_static_graph must be true")
    if runtime.get("ddp_find_unused_parameters") is not False:
        raise ValueError("distributed_runtime ddp_find_unused_parameters must be false")
    topology = runtime.get("rank_topology")
    if not isinstance(topology, list) or len(topology) != world_size:
        raise ValueError("distributed_runtime rank_topology does not match world_size")
    ranks = [record.get("rank") for record in topology if isinstance(record, dict)]
    if (
        len(ranks) != world_size
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
        or set(ranks) != set(range(world_size))
    ):
        raise ValueError("distributed_runtime rank_topology ranks are invalid")
    return runtime


def _verify_warmup_runtime(
    value: object,
    *,
    distributed_runtime: Mapping[str, object],
) -> Mapping[str, object]:
    runtime = _require_mapping(value, "warmup_runtime")
    expected_fields = {
        "world_size",
        "conditions_per_rank",
        "global_conditions_per_step",
        "optimizer_steps_per_epoch",
    }
    if set(runtime) != expected_fields:
        raise ValueError("warmup_runtime fields mismatch")
    for field in (
        "world_size",
        "conditions_per_rank",
        "global_conditions_per_step",
        "optimizer_steps_per_epoch",
    ):
        field_value = runtime.get(field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise ValueError(f"warmup_runtime {field} must be a positive int")
    for field in (
        "world_size",
        "conditions_per_rank",
        "global_conditions_per_step",
    ):
        if runtime[field] != distributed_runtime[field]:
            raise ValueError(f"warmup_runtime {field} differs from distributed_runtime")
    return runtime


def _verify_warmup_optimizer_steps(root: Path, expected: object) -> None:
    history_path = root / "warmup/training/train_log.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("optimizer_steps") != str(expected) for row in rows):
        raise ValueError("warmup train_log optimizer_steps differ from warmup_runtime")


def _string_vector(array: object, label: str) -> tuple[str, ...]:
    import numpy as np

    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"residual_targets.npz {label} must be a string vector")
    strings = tuple(value.astype(str).tolist())
    if (
        not strings
        or any(not item for item in strings)
        or len(set(strings)) != len(strings)
    ):
        raise ValueError(f"residual_targets.npz {label} must be nonempty and unique")
    return strings


def _verify_residual_target_artifact(
    root: Path,
    split_payload: Mapping[str, object],
    universe_payload: Mapping[str, object],
) -> tuple[
    dict[str, set[tuple[str, str]]],
    dict[tuple[str, str], float],
    dict[str, float],
]:
    import numpy as np

    from src.data.geneeffect import PINNED_SPLIT_SHA256

    split_path = root / "cell_line_geneeffect_226_split.json"
    if sha256_file(split_path) != PINNED_SPLIT_SHA256:
        raise ValueError("copied split does not match the pinned Exp13 authority")
    artifact_path = root / "residual_targets.npz"
    expected_arrays = {
        "gene_symbols",
        "model_ids",
        "residual_targets",
        "label_mask",
        "mu_train",
        "centering_model_ids",
    }
    try:
        with np.load(artifact_path, allow_pickle=False) as loaded:
            if set(loaded.files) != expected_arrays:
                raise ValueError("array names mismatch")
            genes = _string_vector(loaded["gene_symbols"], "gene_symbols")
            model_ids = _string_vector(loaded["model_ids"], "model_ids")
            centering_ids = _string_vector(
                loaded["centering_model_ids"], "centering_model_ids"
            )
            targets = np.asarray(loaded["residual_targets"])
            label_mask = np.asarray(loaded["label_mask"])
            mu_train = np.asarray(loaded["mu_train"])
    except Exception as exc:
        raise ValueError(f"residual_targets.npz is invalid: {exc}") from exc

    split_parts: dict[str, tuple[str, ...]] = {}
    for name in ("train", "val", "test"):
        values = split_payload.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"copied split must contain unique nonempty {name} IDs")
        split_parts[name] = tuple(values)
    expected_model_ids = (
        *split_parts["train"],
        *split_parts["val"],
        *split_parts["test"],
    )
    if model_ids != expected_model_ids:
        raise ValueError("residual-target ModelID order differs from the copied split")
    unlabeled = split_payload.get("unlabeled_train")
    if not isinstance(unlabeled, list) or any(
        item not in split_parts["train"] for item in unlabeled
    ):
        raise ValueError("copied split unlabeled_train is malformed")
    expected_centering = tuple(
        model_id for model_id in split_parts["train"] if model_id not in set(unlabeled)
    )
    if centering_ids != expected_centering:
        raise ValueError("residual-target centering ModelIDs mismatch")
    if universe_payload.get("scored_symbols") != list(genes):
        raise ValueError("residual-target genes differ from the scored universe")
    shape = (len(genes), len(model_ids))
    if targets.dtype != np.dtype(np.float32) or targets.shape != shape:
        raise ValueError("residual-target matrix must be float32 genes-by-ModelIDs")
    if label_mask.dtype != np.dtype(bool) or label_mask.shape != shape:
        raise ValueError("residual-target label mask must be bool genes-by-ModelIDs")
    if mu_train.dtype != np.dtype(np.float64) or mu_train.shape != (len(genes),):
        raise ValueError("residual-target mu_train must be ordered float64")
    if not np.isfinite(targets).all() or not np.isfinite(mu_train).all():
        raise ValueError("residual-target arrays must be finite")

    column_by_model = {model_id: index for index, model_id in enumerate(model_ids)}
    evaluation_keys = {
        split_name: {
            (model_id, gene)
            for model_id in split_parts[split_name]
            for gene_index, gene in enumerate(genes)
            if bool(label_mask[gene_index, column_by_model[model_id]])
        }
        for split_name in ("val", "test")
    }
    authoritative_residual = {
        (model_id, gene): float(targets[gene_index, column_by_model[model_id]])
        for split_name in ("val", "test")
        for model_id in split_parts[split_name]
        for gene_index, gene in enumerate(genes)
        if bool(label_mask[gene_index, column_by_model[model_id]])
    }
    authoritative_mu = {
        gene: float(mu_train[gene_index]) for gene_index, gene in enumerate(genes)
    }
    return evaluation_keys, authoritative_residual, authoritative_mu


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _verify_metric_scalar(recorded: object, actual: float, label: str) -> None:
    if math.isnan(actual):
        if recorded is not None:
            raise ValueError(f"{label} must be null because the metric is undefined")
        return
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or not math.isfinite(float(recorded))
        or not math.isclose(float(recorded), actual, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(f"{label} does not match recomputed predictions")


def _verify_metric_count(recorded: object, actual: int, label: str) -> None:
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, int)
        or recorded != actual
    ):
        raise ValueError(f"{label} does not match recomputed predictions")


def _verify_score_entry(
    entry: Mapping[str, object], score: object, *, label: str, e2e: bool
) -> None:
    if e2e:
        fields = {
            "macro_per_line_spearman": score.macro_per_line,
            "macro_per_gene_spearman": score.macro_per_gene,
        }
        counts = {
            "per_line_defined": score.n_lines,
            "per_gene_defined": score.n_genes,
            "per_line_undefined": score.n_line_undefined,
            "per_gene_undefined": score.n_gene_undefined,
        }
    else:
        fields = {
            "macro_per_line": score.macro_per_line,
            "macro_per_gene": score.macro_per_gene,
        }
        counts = {
            "n_lines": score.n_lines,
            "n_genes": score.n_genes,
            "n_line_undefined": score.n_line_undefined,
            "n_gene_undefined": score.n_gene_undefined,
        }
    for name, value in fields.items():
        _verify_metric_scalar(entry.get(name), value, f"{label}.{name}")
    for name, value in counts.items():
        _verify_metric_count(entry.get(name), value, f"{label}.{name}")
    if not e2e:
        return
    per_line = _require_mapping(entry.get("per_line_spearman"), f"{label}.per_line")
    expected_lines = {str(index) for index in score.per_line.index}
    if set(per_line) != expected_lines:
        raise ValueError(f"{label}.per_line_spearman membership mismatch")
    for model_id, value in score.per_line.items():
        _verify_metric_scalar(
            per_line[str(model_id)], float(value), f"{label}.per_line[{model_id}]"
        )


def _verify_predictions_and_metrics(
    root: Path,
    split_payload: Mapping[str, object],
    universe_payload: Mapping[str, object],
    expected_keys_by_split: Mapping[str, set[tuple[str, str]]],
    authoritative_residual: Mapping[tuple[str, str], float],
    authoritative_mu: Mapping[str, float],
    metrics: Mapping[str, object],
) -> float:
    import numpy as np
    import pandas as pd

    from src.eval.metrics import score_predictions

    split_members: dict[str, tuple[str, ...]] = {}
    for split_name in ("val", "test"):
        raw_members = split_payload.get(split_name)
        if (
            not isinstance(raw_members, list)
            or not raw_members
            or any(not isinstance(item, str) or not item for item in raw_members)
            or len(set(raw_members)) != len(raw_members)
        ):
            raise ValueError(
                f"copied split must contain unique nonempty {split_name} IDs"
            )
        split_members[split_name] = tuple(raw_members)
    if set(split_members["val"]) & set(split_members["test"]):
        raise ValueError("copied split val/test membership overlaps")
    raw_genes = universe_payload.get("scored_symbols")
    if (
        not isinstance(raw_genes, list)
        or not raw_genes
        or any(not isinstance(gene, str) or not gene for gene in raw_genes)
        or len(set(raw_genes)) != len(raw_genes)
        or universe_payload.get("scored_gene_count") != len(raw_genes)
    ):
        raise ValueError("scored-gene universe manifest is malformed")
    scored_genes = set(raw_genes)

    prediction_path = root / "geneeffect_residual_predictions.csv"
    try:
        frame = pd.read_csv(prediction_path)
    except Exception as exc:
        raise ValueError(
            f"GeneEffect predictions artifact is unreadable: {exc}"
        ) from exc
    required_columns = {
        "split",
        "method",
        "model_id",
        "gene_symbol",
        "gene_effect",
        "residual",
        "residual_prediction",
    }
    if set(frame.columns) != required_columns or frame.empty:
        raise ValueError(
            "GeneEffect predictions artifact has invalid columns or is empty"
        )
    identity_columns = ["split", "method", "model_id", "gene_symbol"]
    if frame[identity_columns].isna().any().any():
        raise ValueError("GeneEffect predictions contain missing identities")
    if frame.duplicated(identity_columns).any():
        raise ValueError("GeneEffect predictions contain duplicate evaluation keys")
    if set(frame["split"]) != {"val", "test"}:
        raise ValueError("GeneEffect predictions must contain exactly val and test")
    if set(frame["method"]) != _STAGE2_METHODS:
        raise ValueError("GeneEffect predictions do not contain the five fixed methods")
    numeric_columns = ["gene_effect", "residual", "residual_prediction"]
    try:
        numeric = frame[numeric_columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("GeneEffect truth/prediction columns must be numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("GeneEffect truth/prediction columns must be finite")
    frame[numeric_columns] = numeric
    for row in frame.itertuples(index=False):
        key = (str(row.model_id), str(row.gene_symbol))
        expected_residual = authoritative_residual.get(key)
        expected_mu = authoritative_mu.get(str(row.gene_symbol))
        if expected_residual is None or expected_mu is None:
            raise ValueError(
                f"prediction truth key is absent from residual targets: {key}"
            )
        if not math.isclose(
            float(row.residual),
            expected_residual,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"prediction residual differs from authoritative target: {key}"
            )
        if not math.isclose(
            float(row.gene_effect),
            expected_residual + expected_mu,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"prediction gene_effect differs from residual + mu_train: {key}"
            )

    baselines = _require_mapping(metrics.get("baselines"), "metrics.baselines")
    baseline_split = _require_mapping(baselines.get("split"), "metrics.baselines.split")
    slices = _require_mapping(baselines.get("slices"), "metrics.baselines.slices")
    validation_primary: float | None = None
    for split_name, expected_members in split_members.items():
        if baseline_split.get(split_name) != list(expected_members):
            raise ValueError(f"baseline metrics {split_name} membership mismatch")
        split_frame = frame.loc[frame["split"] == split_name]
        if set(split_frame["model_id"]) != set(expected_members):
            raise ValueError(
                f"prediction {split_name} membership differs from copied split"
            )
        if set(split_frame["gene_symbol"]) != scored_genes:
            raise ValueError(
                f"prediction {split_name} genes differ from the scored universe"
            )
        method_keys: dict[str, set[tuple[str, str]]] = {}
        reference_truth = None
        for method in sorted(_STAGE2_METHODS):
            method_frame = split_frame.loc[split_frame["method"] == method].copy()
            method_frame = method_frame.sort_values(
                ["model_id", "gene_symbol"], kind="stable"
            ).reset_index(drop=True)
            keys = set(
                method_frame[["model_id", "gene_symbol"]].itertuples(
                    index=False, name=None
                )
            )
            if not keys:
                raise ValueError(f"{split_name}/{method} has no evaluation rows")
            method_keys[method] = keys
            truth = method_frame[["model_id", "gene_symbol", "gene_effect", "residual"]]
            if reference_truth is None:
                reference_truth = truth
            elif not truth[["model_id", "gene_symbol"]].equals(
                reference_truth[["model_id", "gene_symbol"]]
            ) or not np.allclose(
                truth[["gene_effect", "residual"]].to_numpy(dtype=float),
                reference_truth[["gene_effect", "residual"]].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(f"truth values differ across methods in {split_name}")
        reference_keys = method_keys["e2e_full"]
        if any(keys != reference_keys for keys in method_keys.values()):
            raise ValueError(f"method evaluation keys differ in {split_name}")
        if reference_keys != expected_keys_by_split.get(split_name):
            raise ValueError(
                "method evaluation keys differ from residual label mask in "
                f"{split_name}"
            )

        e2e_frame = split_frame.loc[split_frame["method"] == "e2e_full"]
        e2e_score = score_predictions(
            e2e_frame, truth_col="residual", pred_col="residual_prediction"
        )
        metric_name = "validation" if split_name == "val" else "test"
        e2e_entry = _require_mapping(metrics.get(metric_name), f"metrics.{metric_name}")
        _verify_score_entry(
            e2e_entry, e2e_score, label=f"metrics.{metric_name}", e2e=True
        )
        if split_name == "val":
            validation_primary = float(e2e_score.macro_per_gene)

        slice_entry = _require_mapping(
            slices.get(split_name), f"metrics.baselines.slices.{split_name}"
        )
        method_entries = _require_mapping(
            slice_entry.get("methods"),
            f"metrics.baselines.slices.{split_name}.methods",
        )
        if set(method_entries) != _BASELINE_METHODS:
            raise ValueError(f"baseline metric methods mismatch in {split_name}")
        for method in sorted(_BASELINE_METHODS):
            method_frame = split_frame.loc[split_frame["method"] == method]
            score = score_predictions(
                method_frame, truth_col="residual", pred_col="residual_prediction"
            )
            entry = _require_mapping(
                method_entries[method], f"metrics.{split_name}.{method}"
            )
            _verify_score_entry(
                entry, score, label=f"metrics.{split_name}.{method}", e2e=False
            )
            if method == "gene_mean":
                if score.n_genes != 0 or entry.get("macro_per_gene") is not None:
                    raise ValueError("gene_mean per-gene metric must be undefined")
                coverage = _require_mapping(
                    entry.get("coverage"), f"metrics.{split_name}.gene_mean.coverage"
                )
                if coverage != {
                    "observed_rows": len(reference_keys),
                    "expected_rows": len(reference_keys),
                    "complete": True,
                }:
                    raise ValueError("gene_mean coverage metadata mismatch")
    if validation_primary is None or not math.isfinite(validation_primary):
        raise ValueError("validation primary metric must be finite")
    return validation_primary


def _load_npz_json_scalar(loaded: object, name: str, label: str) -> object:
    import numpy as np

    value = np.asarray(loaded[name])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} {name} must be a JSON string scalar")
    try:
        return json.loads(str(value.item()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} {name} is invalid JSON") from exc


def _verify_preprocessing_payloads(
    root: Path,
    *,
    feature_schema: Mapping[str, object],
    feature_generation: Mapping[str, object],
) -> None:
    import numpy as np

    from src.experiments.exp13_legacy.geneeffect_feature_store import (
        CONTEXT_WIDTH,
        GENE_WIDTH,
        Q_SC_WIDTH,
    )
    from src.model.features import (
        DELTA_WIDTH,
        FEATURE_SCHEMA,
        PROJECTION_WIDTH,
        SUMMARY_WIDTH,
        FixedSparseProjection,
    )

    if feature_schema != FEATURE_SCHEMA.to_dict():
        raise ValueError("feature schema artifact differs from the runtime schema")

    projection_path = root / "projection.npz"
    try:
        with np.load(projection_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"components", "metadata"}:
                raise ValueError("array names mismatch")
            components = np.asarray(loaded["components"])
            metadata = _load_npz_json_scalar(loaded, "metadata", "projection.npz")
    except Exception as exc:
        raise ValueError(f"projection.npz is invalid: {exc}") from exc
    if (
        components.dtype != np.dtype(np.float32)
        or components.shape != (PROJECTION_WIDTH, DELTA_WIDTH)
        or not np.isfinite(components).all()
        or not isinstance(metadata, dict)
    ):
        raise ValueError("projection.npz components or metadata are invalid")
    if metadata != feature_generation.get("projection"):
        raise ValueError("projection.npz differs from feature_generation")
    seed = metadata.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or metadata.get("algorithm") != "achlioptas_sparse_jl_v1"
        or metadata.get("input_width") != DELTA_WIDTH
        or metadata.get("output_width") != PROJECTION_WIDTH
    ):
        raise ValueError("projection metadata is operationally invalid")
    if not np.array_equal(components, FixedSparseProjection(seed=seed).components):
        raise ValueError("projection components differ from the declared generator")

    standardizer_path = root / "standardizer.npz"
    try:
        with np.load(standardizer_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"state"}:
                raise ValueError("array names mismatch")
            state = _load_npz_json_scalar(loaded, "state", "standardizer.npz")
    except Exception as exc:
        raise ValueError(f"standardizer.npz is invalid: {exc}") from exc
    if not isinstance(state, dict) or state != feature_generation.get("standardizer"):
        raise ValueError("standardizer.npz differs from feature_generation")
    blocks = state.get("blocks")
    expected_widths = {
        "delta_proj": PROJECTION_WIDTH,
        "s": SUMMARY_WIDTH,
        "q_sc": Q_SC_WIDTH,
        "e_g": GENE_WIDTH,
        "z_c": CONTEXT_WIDTH,
    }
    if state.get("version") != 1 or not isinstance(blocks, dict):
        raise ValueError("standardizer state is invalid")
    if set(blocks) != set(expected_widths):
        raise ValueError("standardizer blocks differ from the feature schema")
    for name, width in expected_widths.items():
        record = blocks[name]
        if not isinstance(record, dict):
            raise ValueError(f"standardizer block is invalid: {name}")
        mean = np.asarray(record.get("mean"), dtype=np.float64)
        scale = np.asarray(record.get("scale"), dtype=np.float64)
        constant = record.get("constant_columns")
        if (
            mean.shape != (width,)
            or scale.shape != (width,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not isinstance(constant, list)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < width
                for index in constant
            )
            or len(set(constant)) != len(constant)
        ):
            raise ValueError(f"standardizer block statistics are invalid: {name}")


def _verify_runner_contract(root: Path) -> None:
    """Verify cross-artifact identities emitted by the production runner."""
    json_names = (
        "config_snapshot.json",
        "cell_line_geneeffect_226_split.json",
        "stage1_objective.json",
        "esm2_gene_universe_manifest.json",
        "esm2_provenance_manifest.json",
        "esm2_uniprot_mapping.json",
        "g_var_manifest.json",
        "feature_schema.json",
        "backbone_load_report.json",
        "lambda_calibration.json",
        "feature_generation.json",
        "checkpoint_selection.json",
        "geneeffect_residual_metrics.json",
        "run_manifest.json",
        "model_package/model_manifest.json",
    )
    documents = {name: _read_json_object(root / name) for name in json_names}
    run = documents["run_manifest.json"]
    config = documents["config_snapshot.json"]
    universe = documents["esm2_gene_universe_manifest.json"]
    model_package = documents["model_package/model_manifest.json"]
    selection = documents["checkpoint_selection.json"]
    final_symbols = universe.get("scored_symbols")
    if (
        not isinstance(final_symbols, list)
        or not final_symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in final_symbols)
        or len(set(final_symbols)) != len(final_symbols)
    ):
        raise ValueError("ESM2 scored gene universe is invalid")
    metrics = documents["geneeffect_residual_metrics.json"]
    (
        expected_evaluation_keys,
        authoritative_residual,
        authoritative_mu,
    ) = _verify_residual_target_artifact(
        root,
        documents["cell_line_geneeffect_226_split.json"],
        documents["esm2_gene_universe_manifest.json"],
    )

    for phase in ("warmup", "joint"):
        if not isinstance(selection.get(phase), dict):
            raise ValueError(f"checkpoint_selection.json has invalid {phase!r} outcome")
    warmup_metadata, _ = _verify_selected_checkpoint(
        root,
        training_dir="warmup/training",
        filename="head.pt",
        selection=selection["warmup"],
    )
    joint_metadata, joint_state = _verify_selected_checkpoint(
        root,
        training_dir="joint/training",
        filename="e2e_state.pt",
        selection=selection["joint"],
    )
    distributed_runtime = _verify_distributed_runtime(run.get("distributed_runtime"))
    config_distributed = _require_mapping(
        config.get("distributed"), "config distributed"
    )
    config_joint = _require_mapping(config.get("joint"), "config joint")
    if distributed_runtime["mixed_precision"] != config_distributed.get(
        "mixed_precision"
    ) or distributed_runtime["conditions_per_rank"] != config_joint.get(
        "conditions_per_rank"
    ):
        raise ValueError("distributed_runtime differs from frozen config")
    if (
        _verify_distributed_runtime(model_package.get("distributed_runtime"))
        != distributed_runtime
    ):
        raise ValueError("model package/run distributed_runtime mismatch")
    warmup_runtime: Mapping[str, object] | None = None
    for phase, metadata in (
        ("warmup", warmup_metadata),
        ("joint", joint_metadata),
    ):
        provenance = _require_mapping(
            metadata.get("provenance"), f"{phase} checkpoint provenance"
        )
        if (
            _verify_distributed_runtime(provenance.get("distributed_runtime"))
            != distributed_runtime
        ):
            raise ValueError(f"{phase} checkpoint/run distributed_runtime mismatch")
        phase_warmup_runtime = _verify_warmup_runtime(
            provenance.get("warmup_runtime"),
            distributed_runtime=distributed_runtime,
        )
        if warmup_runtime is None:
            warmup_runtime = phase_warmup_runtime
        elif phase_warmup_runtime != warmup_runtime:
            raise ValueError("joint/warmup checkpoint warmup_runtime mismatch")
    assert warmup_runtime is not None
    _verify_warmup_optimizer_steps(root, warmup_runtime["optimizer_steps_per_epoch"])
    joint_provenance = _require_mapping(
        joint_metadata.get("provenance"), "joint checkpoint provenance"
    )
    if (
        joint_provenance.get("lambda_calibration_report")
        != documents["lambda_calibration.json"]
    ):
        raise ValueError("joint checkpoint lambda-calibration identity mismatch")
    _verify_lambda_calibration(documents["lambda_calibration.json"])
    expected_package = {
        "checkpoint": "e2e_state.pt",
        "projection": "../projection.npz",
        "standardizer": "../standardizer.npz",
        "feature_schema": "../feature_schema.json",
        "frozen_features": "../condition_features/stage1_frozen",
        "selected_features": "../condition_features/stage2_selected",
        "distributed_runtime": dict(distributed_runtime),
    }
    if model_package != expected_package:
        raise ValueError("model package operational manifest mismatch")
    packaged_state = _load_checkpoint_state(
        root / "model_package/e2e_state.pt", "packaged joint"
    )
    _require_equal_checkpoint_states(joint_state, packaged_state)
    _verify_preprocessing_payloads(
        root,
        feature_schema=documents["feature_schema.json"],
        feature_generation=documents["feature_generation.json"],
    )

    required_metric_sections = {"validation", "test", "baselines", "response"}
    if not required_metric_sections.issubset(metrics):
        raise ValueError("GeneEffect metrics artifact is missing required sections")
    response = _require_mapping(metrics.get("response"), "metrics.response")
    before_response = _require_mapping(
        response.get("before_stage2"), "metrics.response.before_stage2"
    )
    after_response = _require_mapping(
        response.get("after_stage2"), "metrics.response.after_stage2"
    )
    if before_response.get("input_lineage_status") != "historical_unverified_inputs":
        raise ValueError("before-stage2 response lineage status mismatch")
    if after_response.get("input_lineage_status") != "current_authenticated_inputs":
        raise ValueError("after-stage2 response lineage status mismatch")
    before_metrics = _require_mapping(
        before_response.get("metrics"), "metrics.response.before_stage2.metrics"
    )
    after_metrics = _require_mapping(
        after_response.get("metrics"), "metrics.response.after_stage2.metrics"
    )
    for label, value in (
        ("before_stage2.metrics.model_loss", before_metrics.get("model_loss")),
        ("after_stage2.metrics.model_loss", after_metrics.get("model_loss")),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"metrics.response.{label} must be finite")
    if response.get("comparison_status") != (
        "not_comparable_historical_input_lineage_incomplete"
    ):
        raise ValueError("metrics.response comparison status mismatch")
    if response.get("delta_reported") is not False:
        raise ValueError("metrics.response.delta_reported must be false")
    if response.get("hard_guard_applied") is not False:
        raise ValueError("metrics.response.hard_guard_applied must be false")
    validation_primary = _verify_predictions_and_metrics(
        root,
        documents["cell_line_geneeffect_226_split.json"],
        documents["esm2_gene_universe_manifest.json"],
        expected_evaluation_keys,
        authoritative_residual,
        authoritative_mu,
        metrics,
    )
    # BF16 forward noise can perturb near-tied ranks between online selection and
    # finalization; selected and packaged state tensors still match exactly above.
    if not math.isclose(
        float(joint_metadata["metric_value"]),
        validation_primary,
        rel_tol=1e-9,
        abs_tol=1e-4,
    ):
        raise ValueError(
            "selected joint checkpoint metric does not match recomputed validation"
        )


def mark_failure(layout: Stage2RunLayout, error: BaseException, *, phase: str) -> None:
    """Write one recoverable failure marker without deleting partial evidence."""
    if layout.complete.exists():
        raise ValueError("cannot mark an already completed run as failed")
    atomic_write_json(
        layout.failure,
        {
            "phase": str(phase),
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def mark_complete(
    layout: Stage2RunLayout,
    *,
    run_id: str,
    required_outputs: Sequence[str] = REQUIRED_STAGE2_OUTPUTS,
) -> Mapping[str, object]:
    """Verify and seal all outputs, then emit the sole completion sentinel."""
    if layout.failure.exists():
        raise ValueError(f"refusing to complete failed run {layout.root}")
    if layout.complete.exists():
        raise FileExistsError(f"completion sentinel already exists: {layout.complete}")
    manifest_run_id = _read_run_id(layout.root)
    if str(run_id) != manifest_run_id:
        raise ValueError(
            f"completion run_id {run_id!r} != run manifest {manifest_run_id!r}"
        )
    _require_artifacts(layout.root, required_outputs)
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        _verify_runner_contract(layout.root)
    payload: dict[str, object] = {
        "status": "complete",
        "run_id": str(run_id),
    }
    atomic_write_json(layout.complete, payload)
    return payload


def verify_complete_run(
    path: Path,
    *,
    required_outputs: Sequence[str] = REQUIRED_STAGE2_OUTPUTS,
) -> Mapping[str, object]:
    """Verify a terminal run and its required operational outputs."""
    root = Path(path)
    if (root / "failure.json").exists():
        raise ValueError(f"run {root} carries failure.json")
    complete_path = root / "complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"run {root} has no complete.json")
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not payload.get("run_id"):
        raise ValueError(f"invalid completion sentinel: {complete_path}")
    manifest_run_id = _read_run_id(root)
    if payload["run_id"] != manifest_run_id:
        raise ValueError(
            "completion sentinel run_id does not match run_manifest.json: "
            f"{payload['run_id']!r} != {manifest_run_id!r}"
        )
    _require_artifacts(root, required_outputs)
    if tuple(required_outputs) == REQUIRED_STAGE2_OUTPUTS:
        _verify_runner_contract(root)
    return payload
