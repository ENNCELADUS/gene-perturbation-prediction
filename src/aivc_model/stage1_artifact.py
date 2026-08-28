"""Compatibility/input seal for Stage 1 with incomplete historical lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from aivc_model.gene_embeddings import Esm2EmbeddingTable, load_esm2_embeddings
from aivc_model.stage1_config import load_stage1_config
from aivc_model.state_core import sha256_strings

_SCHEMA_VERSION = 1
_LEGACY_ESM_KEY = "perturbations.esm_matrix"
_VOCABULARY_HASH_KEY = "perturbations.gene_vocabulary_sha256"
_LEARNED_PREFIXES = ("state_adapter.", "perturbations.adapter.")
_DROPPED_PREFIXES = ("response_encoder.", "response_pooler.", "c_head.")
_DROPPED_EXACT_KEYS = frozenset({"control_expression_mean"})
_TRAINING_CODE_PROVENANCE_STATUS = "unavailable"
_TRAINING_CODE_PROVENANCE_REASON = (
    "historical_run_has_no_immutable_training_code_identity"
)
_TRAINING_DATA_PROVENANCE_STATUS = "incomplete"
_TRAINING_DATA_PROVENANCE_MISSING_IDENTITIES = (
    "cell_line_manifest",
    "tx1_basal_cache",
    "response_cache",
    "perturbseq_source_content",
)
_TRAINING_DATA_PROVENANCE_REASON = (
    "historical_run_manifest_does_not_hash_all_training_data_inputs"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_sha256",
        "stage1_genes",
        "stage1_gene_vocabulary_sha256",
        "esm2_artifact_sha256",
        "state_hparams_sha256",
        "compatibility_code_sha256",
        "training_code_provenance_status",
        "training_code_provenance_reason",
        "training_data_provenance_status",
        "training_data_provenance_missing_identities",
        "training_data_provenance_reason",
        "config_sha256",
        "source_sha256",
        "legacy_esm_matrix_sha256",
        "run_manifest_sha256",
        "checkpoint_metadata_sha256",
        "stage1_objective_sha256",
    }
)


@dataclass(frozen=True)
class Stage1ArtifactManifest:
    schema_version: int
    checkpoint_sha256: str
    stage1_genes: tuple[str, ...]
    stage1_gene_vocabulary_sha256: str
    esm2_artifact_sha256: str
    state_hparams_sha256: str
    compatibility_code_sha256: dict[str, str]
    training_code_provenance_status: str
    training_code_provenance_reason: str
    training_data_provenance_status: str
    training_data_provenance_missing_identities: tuple[str, ...]
    training_data_provenance_reason: str
    config_sha256: dict[str, str]
    source_sha256: dict[str, str]
    legacy_esm_matrix_sha256: str | None
    run_manifest_sha256: str
    checkpoint_metadata_sha256: str
    stage1_objective_sha256: str

    def write(self, path: Path) -> None:
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> Stage1ArtifactManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
            found = set(payload) if isinstance(payload, dict) else set()
            raise ValueError(
                "Stage-1 artifact manifest fields mismatch: "
                f"missing={sorted(_MANIFEST_FIELDS - found)}, "
                f"unexpected={sorted(found - _MANIFEST_FIELDS)}"
            )
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Stage-1 artifact schema: {payload['schema_version']!r}"
            )
        genes = tuple(str(gene).upper() for gene in payload["stage1_genes"])
        manifest = cls(
            schema_version=_SCHEMA_VERSION,
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            stage1_genes=genes,
            stage1_gene_vocabulary_sha256=str(payload["stage1_gene_vocabulary_sha256"]),
            esm2_artifact_sha256=str(payload["esm2_artifact_sha256"]),
            state_hparams_sha256=str(payload["state_hparams_sha256"]),
            compatibility_code_sha256=_read_hash_map(
                payload["compatibility_code_sha256"],
                "compatibility_code_sha256",
            ),
            training_code_provenance_status=str(
                payload["training_code_provenance_status"]
            ),
            training_code_provenance_reason=str(
                payload["training_code_provenance_reason"]
            ),
            training_data_provenance_status=str(
                payload["training_data_provenance_status"]
            ),
            training_data_provenance_missing_identities=tuple(
                str(identity)
                for identity in payload["training_data_provenance_missing_identities"]
            ),
            training_data_provenance_reason=str(
                payload["training_data_provenance_reason"]
            ),
            config_sha256=_read_hash_map(payload["config_sha256"], "config_sha256"),
            source_sha256=_read_hash_map(payload["source_sha256"], "source_sha256"),
            legacy_esm_matrix_sha256=(
                None
                if payload["legacy_esm_matrix_sha256"] is None
                else str(payload["legacy_esm_matrix_sha256"])
            ),
            run_manifest_sha256=str(payload["run_manifest_sha256"]),
            checkpoint_metadata_sha256=str(payload["checkpoint_metadata_sha256"]),
            stage1_objective_sha256=str(payload["stage1_objective_sha256"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.stage1_genes or len(set(self.stage1_genes)) != len(
            self.stage1_genes
        ):
            raise ValueError("Stage-1 gene vocabulary must be non-empty and unique")
        actual = sha256_strings(np.asarray(self.stage1_genes, dtype=object))
        if actual != self.stage1_gene_vocabulary_sha256:
            raise ValueError("Stage-1 gene vocabulary SHA256 mismatch")
        if (
            self.training_code_provenance_status
            != _TRAINING_CODE_PROVENANCE_STATUS
            or self.training_code_provenance_reason
            != _TRAINING_CODE_PROVENANCE_REASON
        ):
            raise ValueError(
                "Historical Stage-1 training-code provenance must be recorded as "
                "unavailable"
            )
        if (
            self.training_data_provenance_status
            != _TRAINING_DATA_PROVENANCE_STATUS
            or self.training_data_provenance_missing_identities
            != _TRAINING_DATA_PROVENANCE_MISSING_IDENTITIES
            or self.training_data_provenance_reason
            != _TRAINING_DATA_PROVENANCE_REASON
        ):
            raise ValueError(
                "Historical Stage-1 training-data provenance must remain incomplete "
                "with every missing identity recorded"
            )


@dataclass(frozen=True)
class Stage1ArtifactLoadReport:
    loaded_keys: tuple[str, ...]
    dropped_keys: tuple[str, ...]
    legacy_esm_matrix_authenticated: bool
    trainable: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_stage1_artifact(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    stage1_genes: Sequence[str],
    esm2_embeddings_path: Path,
    state_hparams_path: Path,
    run_manifest_path: Path,
    checkpoint_metadata_path: Path,
    stage1_objective_path: Path,
    compatibility_code_paths: Mapping[str, Path],
    config_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
) -> Stage1ArtifactManifest:
    """Seal compatible inputs while recording incomplete training lineage."""
    genes = tuple(str(gene).upper() for gene in stage1_genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Stage-1 gene vocabulary must be non-empty and unique")
    _require_asset_groups(compatibility_code_paths, config_paths, source_paths)
    table = load_esm2_embeddings(esm2_embeddings_path)
    state = _load_checkpoint_state(checkpoint_path)
    _validate_recorded_training_artifacts(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        run_manifest_path=run_manifest_path,
        stage1_objective_path=stage1_objective_path,
        state_hparams_path=state_hparams_path,
        esm2_embeddings_path=esm2_embeddings_path,
        config_paths=config_paths,
        source_paths=source_paths,
    )
    _authenticate_checkpoint_vocabulary(state, genes)
    legacy = state.get(_LEGACY_ESM_KEY)
    legacy_sha: str | None = None
    if legacy is not None:
        _authenticate_esm_matrix(legacy, genes, table, label="legacy checkpoint")
        legacy_sha = _sha256_tensor(legacy)
    manifest = Stage1ArtifactManifest(
        schema_version=_SCHEMA_VERSION,
        checkpoint_sha256=sha256_file(checkpoint_path),
        stage1_genes=genes,
        stage1_gene_vocabulary_sha256=sha256_strings(np.asarray(genes, dtype=object)),
        esm2_artifact_sha256=sha256_file(esm2_embeddings_path),
        state_hparams_sha256=sha256_file(state_hparams_path),
        compatibility_code_sha256=_hash_paths(compatibility_code_paths),
        training_code_provenance_status=_TRAINING_CODE_PROVENANCE_STATUS,
        training_code_provenance_reason=_TRAINING_CODE_PROVENANCE_REASON,
        training_data_provenance_status=_TRAINING_DATA_PROVENANCE_STATUS,
        training_data_provenance_missing_identities=(
            _TRAINING_DATA_PROVENANCE_MISSING_IDENTITIES
        ),
        training_data_provenance_reason=_TRAINING_DATA_PROVENANCE_REASON,
        config_sha256=_hash_paths(config_paths),
        source_sha256=_hash_paths(source_paths),
        legacy_esm_matrix_sha256=legacy_sha,
        run_manifest_sha256=sha256_file(run_manifest_path),
        checkpoint_metadata_sha256=sha256_file(checkpoint_metadata_path),
        stage1_objective_sha256=sha256_file(stage1_objective_path),
    )
    manifest.validate()
    manifest.write(manifest_path)
    return manifest


def load_stage1_artifact(
    model: nn.Module,
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    esm2_embeddings_path: Path,
    target_esm_embeddings_path: Path,
    target_esm_artifact_sha256: str,
    state_hparams_path: Path,
    run_manifest_path: Path,
    checkpoint_metadata_path: Path,
    stage1_objective_path: Path,
    compatibility_code_paths: Mapping[str, Path],
    config_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
    trainable: bool,
) -> Stage1ArtifactLoadReport:
    """Authenticate the sealed inputs and strictly restore learned Stage-1 keys."""
    manifest = Stage1ArtifactManifest.read(manifest_path)
    _require_asset_groups(compatibility_code_paths, config_paths, source_paths)
    _require_hash(checkpoint_path, manifest.checkpoint_sha256, "checkpoint")
    _require_hash(esm2_embeddings_path, manifest.esm2_artifact_sha256, "ESM-2")
    _require_hash(
        target_esm_embeddings_path,
        target_esm_artifact_sha256,
        "target-universe ESM-2",
    )
    _require_hash(state_hparams_path, manifest.state_hparams_sha256, "STATE hparams")
    _require_hash(run_manifest_path, manifest.run_manifest_sha256, "run manifest")
    _require_hash(
        checkpoint_metadata_path,
        manifest.checkpoint_metadata_sha256,
        "checkpoint metadata",
    )
    _require_hash(
        stage1_objective_path,
        manifest.stage1_objective_sha256,
        "Stage-1 objective",
    )
    _require_hash_maps(
        compatibility_code_paths,
        manifest.compatibility_code_sha256,
        "seal/load-time compatibility code",
    )
    _require_hash_maps(config_paths, manifest.config_sha256, "config")
    _require_hash_maps(source_paths, manifest.source_sha256, "source")

    table = load_esm2_embeddings(esm2_embeddings_path)
    target_table = load_esm2_embeddings(target_esm_embeddings_path)
    state = _load_checkpoint_state(checkpoint_path)
    _validate_recorded_training_artifacts(
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        run_manifest_path=run_manifest_path,
        stage1_objective_path=stage1_objective_path,
        state_hparams_path=state_hparams_path,
        esm2_embeddings_path=esm2_embeddings_path,
        config_paths=config_paths,
        source_paths=source_paths,
    )
    _authenticate_checkpoint_vocabulary(state, manifest.stage1_genes)
    legacy = state.get(_LEGACY_ESM_KEY)
    if manifest.legacy_esm_matrix_sha256 is None:
        if legacy is not None:
            raise ValueError("Unsealed legacy perturbations.esm_matrix is not allowed")
        legacy_authenticated = False
    else:
        if legacy is None:
            raise ValueError("Sealed legacy perturbations.esm_matrix is missing")
        _authenticate_esm_matrix(
            legacy, manifest.stage1_genes, table, label="legacy checkpoint"
        )
        if _sha256_tensor(legacy) != manifest.legacy_esm_matrix_sha256:
            raise ValueError("Legacy perturbations.esm_matrix SHA256 mismatch")
        legacy_authenticated = True

    _authenticate_target_matrix(model, target_table)
    expected_state = set(model.state_dict())
    target_fixed = expected_state - {
        key for key in expected_state if key.startswith(_LEARNED_PREFIXES)
    }
    if target_fixed != {_VOCABULARY_HASH_KEY}:
        raise ValueError(f"Unexpected destination fixed keys: {sorted(target_fixed)}")
    expected = expected_state - target_fixed
    checkpoint_learned = {key for key in state if key.startswith(_LEARNED_PREFIXES)}
    missing = sorted(expected - checkpoint_learned)
    unexpected_learned = sorted(checkpoint_learned - expected)
    dropped = sorted(key for key in state if _is_expected_drop(key))
    allowed = checkpoint_learned | set(dropped)
    if legacy is not None:
        allowed.add(_LEGACY_ESM_KEY)
    allowed.add(_VOCABULARY_HASH_KEY)
    unexpected = sorted(set(state) - allowed)
    if missing or unexpected_learned or unexpected or not expected:
        raise ValueError(
            "Incomplete Stage-1 artifact load: "
            f"missing={missing}, unexpected_learned={unexpected_learned}, "
            f"unexpected={unexpected}, loaded_count={len(expected) - len(missing)}"
        )
    learned_state = {key: state[key] for key in sorted(expected)}
    result = model.load_state_dict(learned_state, strict=False)
    if set(result.missing_keys) != target_fixed or result.unexpected_keys:
        raise ValueError(
            "Strict learned-key restoration failed: "
            f"missing={sorted(result.missing_keys)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    model.train(trainable)
    model.requires_grad_(trainable)
    return Stage1ArtifactLoadReport(
        loaded_keys=tuple(sorted(expected)),
        dropped_keys=tuple(dropped),
        legacy_esm_matrix_authenticated=legacy_authenticated,
        trainable=trainable,
    )


def _authenticate_target_matrix(model: nn.Module, table: Esm2EmbeddingTable) -> None:
    perturbations = getattr(model, "perturbations", None)
    genes = getattr(perturbations, "genes", None)
    matrix = getattr(perturbations, "esm_matrix", None)
    if genes is None or not isinstance(matrix, torch.Tensor):
        raise TypeError("Destination model must expose perturbations genes/esm_matrix")
    _authenticate_esm_matrix(matrix, tuple(genes), table, label="target model")


def _authenticate_checkpoint_vocabulary(
    state: Mapping[str, torch.Tensor], genes: Sequence[str]
) -> None:
    recorded = state.get(_VOCABULARY_HASH_KEY)
    legacy = state.get(_LEGACY_ESM_KEY)
    if recorded is None and legacy is None:
        raise ValueError(
            "Checkpoint has neither legacy ESM matrix nor gene-vocabulary hash"
        )
    if recorded is None:
        return
    expected = bytes.fromhex(
        sha256_strings(np.asarray(tuple(genes), dtype=object))
    )
    actual = bytes(recorded.detach().cpu().to(torch.uint8).tolist())
    if actual != expected:
        raise ValueError("Checkpoint gene-vocabulary SHA256 mismatch")


def _validate_recorded_training_artifacts(
    *,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    run_manifest_path: Path,
    stage1_objective_path: Path,
    state_hparams_path: Path,
    esm2_embeddings_path: Path,
    config_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
) -> None:
    run_root = checkpoint_path.parent.parent.resolve()
    if (
        checkpoint_metadata_path.parent.resolve() != checkpoint_path.parent.resolve()
        or run_manifest_path.parent.resolve() != run_root
        or stage1_objective_path.parent.resolve() != run_root
    ):
        raise ValueError("Stage-1 provenance files do not share the checkpoint run")
    metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_kind") != "best":
        raise ValueError("Only the selected best Stage-1 checkpoint may be sealed")
    comparisons = (
        (metadata.get("epoch"), run.get("best_epoch"), "best epoch"),
        (
            metadata.get("selection_metric"),
            run.get("selection_metric"),
            "selection metric",
        ),
        (
            metadata.get("metric_value"),
            run.get("best_metric_value"),
            "best metric value",
        ),
    )
    for metadata_value, run_value, label in comparisons:
        if metadata_value != run_value:
            raise ValueError(f"Checkpoint metadata/run manifest {label} mismatch")
    inputs = run.get("input_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("Stage-1 run manifest has no input_sha256 mapping")
    required_inputs = {
        "state_checkpoint": sha256_file(state_hparams_path),
        "esm2_embeddings": sha256_file(esm2_embeddings_path),
    }
    for name, digest in required_inputs.items():
        if inputs.get(name) != digest:
            raise ValueError(f"Stage-1 run manifest {name} SHA256 mismatch")
    if run.get("config_sha256") not in set(_hash_paths(config_paths).values()):
        raise ValueError("Stage-1 run manifest config SHA256 is not sealed")
    matching_configs = [
        path
        for path in config_paths.values()
        if sha256_file(path) == run.get("config_sha256")
    ]
    if len(matching_configs) != 1:
        raise ValueError("Exactly one sealed config must match the Stage-1 run")
    registered_objective = load_stage1_config(matching_configs[0]).objective_payload()
    recorded_objective = json.loads(stage1_objective_path.read_text(encoding="utf-8"))
    if recorded_objective != registered_objective:
        raise ValueError("Stage-1 objective does not match the authenticated config")
    source_hashes = set(_hash_paths(source_paths).values())
    for name in ("split_json", "perturbseq_sources"):
        if inputs.get(name) not in source_hashes:
            raise ValueError(f"Stage-1 run manifest {name} SHA256 is not sealed")


def _authenticate_esm_matrix(
    matrix: object,
    genes: Sequence[str],
    table: Esm2EmbeddingTable,
    *,
    label: str,
) -> None:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{label} ESM matrix must be a tensor")
    canonical = tuple(str(gene).upper() for gene in genes)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError(f"{label} gene vocabulary must be non-empty and unique")
    missing = [gene for gene in canonical if gene not in table.vectors_by_symbol]
    if missing:
        raise ValueError(f"{label} ESM matrix has unresolved genes: {missing[:10]}")
    expected = torch.as_tensor(
        np.vstack([table.vectors_by_symbol[gene] for gene in canonical]),
        dtype=torch.float32,
    )
    actual = matrix.detach().to(device="cpu", dtype=torch.float32)
    if actual.shape != expected.shape:
        raise ValueError(
            f"{label} ESM matrix shape mismatch: {tuple(actual.shape)} != "
            f"{tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        raise ValueError(f"{label} ESM matrix vector/order mismatch")


def _load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError("Stage-1 checkpoint must be a flat tensor state dict")
    return payload


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_hash_map(value: object, field: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not value
        or not all(
            isinstance(key, str) and isinstance(digest, str)
            for key, digest in value.items()
        )
    ):
        raise ValueError(f"{field} must be a non-empty string-to-SHA256 mapping")
    return dict(value)


def _require_asset_groups(*groups: Mapping[str, Path]) -> None:
    if any(not group for group in groups):
        raise ValueError(
            "compatibility code, config, and source artifact groups must be non-empty"
        )


def _hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch for {path}")


def _require_hash_maps(
    paths: Mapping[str, Path], expected: Mapping[str, str], label: str
) -> None:
    if set(paths) != set(expected):
        raise ValueError(
            f"{label} artifact names mismatch: expected={sorted(expected)}, "
            f"actual={sorted(paths)}"
        )
    for name, path in paths.items():
        _require_hash(path, expected[name], f"{label}:{name}")


def _is_expected_drop(key: str) -> bool:
    return key in _DROPPED_EXACT_KEYS or any(
        key.startswith(prefix) for prefix in _DROPPED_PREFIXES
    )
