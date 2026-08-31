"""Fail-closed orchestration boundary for the Exp13 Stage 2 E2E run.

The runner authenticates inputs, assembles response and dependency batches,
executes frozen-head warmup and joint tuning, evaluates only the selected
checkpoint, and seals the terminal model package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from aivc_model.gene_embeddings import load_esm2_embeddings
from aivc_model.esm2_provenance import load_and_authenticate_esm2_provenance
from aivc_model.geneeffect_data import (
    PINNED_COPY_PRIOR_SHA256,
    PINNED_SPLIT_SHA256,
    RAW_UMI_SEMANTICS,
    build_g_var,
    build_residual_data,
    build_scored_universe,
    load_exp13_split,
    load_geneeffect_long,
    load_source_registry,
    restrict_scored_universe_to_copy_prior,
    verify_q_sc_shards,
)
from aivc_model.geneeffect_e2e import (
    GeneEffectE2EModel,
    OnlineConditionBatch,
)
from aivc_model.distributed import require_distinct_devices, run_rank_zero_or_raise
from aivc_model.geneeffect_feature_store import (
    GeneEffectFrozenFeatureCache,
    GeneEffectFeatureStoreWriter,
    verify_geneeffect_feature_store,
)
from aivc_model.geneeffect_features import (
    FEATURE_SCHEMA,
    BlockStandardizer,
    FixedSparseProjection,
    compute_condition_features,
)
from aivc_model.geneeffect_head import GeneEffectResidualHead, moment_pool
from aivc_model.geneeffect_sampler import build_epoch_batches, shard_batches
from aivc_model.geneeffect_training import (
    OnlineSupervisedBatch,
    PrecomputedSupervisedBatch,
    ResponseSupervisionBatch,
    SupervisedMatrix,
    response_objective,
)
from aivc_model.stage1_artifact import sha256_file
from aivc_model.stage1_config import load_stage1_config
from aivc_model.stage2_artifacts import (
    Stage2RunLayout,
    atomic_write_json,
    mark_complete,
    mark_failure,
    prepare_run_dir,
)
from aivc_model.stage2_config import Stage2Config, load_stage2_config
from aivc_model.tx1_embed_cache import (
    authenticate_tx1_registration,
    load_hvg_gene_order,
    load_line_cache,
    verify_cache,
)
from aivc_model.tx1_response_data import (
    assemble_train_response_gene_bags,
    base_gene_name,
    referenced_source_paths,
)
from aivc_model.tx1_response_gene_bags_cache import (
    load_response_targets_cache,
    response_targets_fingerprint,
)
from aivc_model.response_training import predict_bags
from aivc_model.response_training import _state_window


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COPY_PRIOR_SCHEMA = "exp13-copy-prior-v1"
_COPY_PRIOR_DONOR = "ACH-000551"
_PINNED_GENE_EFFECT_SHA256 = (
    "e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e"
)
_ESM2_UNIVERSE_SCHEMA = "exp13-esm2-universes-v2"
_DDP_STATIC_GRAPH = True
_DDP_FIND_UNUSED_PARAMETERS = False


def _create_accelerator(mixed_precision: str) -> Accelerator:
    """Create the one process coordinator used for the whole Stage 2 run."""
    ddp = DistributedDataParallelKwargs(
        static_graph=_DDP_STATIC_GRAPH,
        find_unused_parameters=_DDP_FIND_UNUSED_PARAMETERS,
    )
    return Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp])


def _formal_distributed_runtime(
    accelerator: Accelerator, config: Stage2Config
) -> Mapping[str, object]:
    """Fail closed unless the launcher and auto-detected topology agree."""
    expected_world = int(accelerator.num_processes)
    if expected_world not in {2, 4}:
        raise RuntimeError(
            "formal Stage 2 requires an auto-detected 2- or 4-rank launch"
        )
    expected_precision = config.distributed.mixed_precision
    environment: dict[str, int] = {}
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is None:
            raise RuntimeError(f"formal Stage 2 requires launcher env {name}")
        try:
            environment[name] = int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"formal Stage 2 launcher env {name} is not an int"
            ) from exc
    expected_local = {
        "WORLD_SIZE": expected_world,
        "RANK": int(accelerator.process_index),
        "LOCAL_RANK": int(accelerator.local_process_index),
    }
    if environment != expected_local:
        raise RuntimeError(
            f"launcher env does not match Accelerator state: {environment} != "
            f"{expected_local}"
        )
    if accelerator.mixed_precision != expected_precision:
        raise RuntimeError(
            "Accelerator mixed precision does not match the frozen config: "
            f"{accelerator.mixed_precision} != {expected_precision}"
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError("formal Stage 2 requires one CUDA device per rank")
    local = {
        "rank": int(accelerator.process_index),
        "local_rank": int(accelerator.local_process_index),
        "device": str(accelerator.device),
        "device_name": torch.cuda.get_device_name(accelerator.device),
        "hostname": socket.gethostname(),
    }
    gathered: list[object | None] = [None] * expected_world
    torch.distributed.all_gather_object(gathered, local)
    if any(not isinstance(item, dict) for item in gathered):
        raise RuntimeError("distributed topology did not gather every rank record")
    topology = sorted(gathered, key=lambda item: int(item["rank"]))  # type: ignore[index]
    if [int(item["rank"]) for item in topology] != list(range(expected_world)):  # type: ignore[index]
        raise RuntimeError(
            f"distributed topology ranks are not exactly 0..{expected_world - 1}"
        )
    for field in ("local_rank", "device"):
        values = [item[field] for item in topology]  # type: ignore[index]
        if len(set(values)) != expected_world:
            raise RuntimeError(f"distributed topology {field} values are not unique")
    conditions_per_rank = config.joint.conditions_per_rank
    return {
        "world_size": expected_world,
        "mixed_precision": expected_precision,
        "ddp_static_graph": _DDP_STATIC_GRAPH,
        "ddp_find_unused_parameters": _DDP_FIND_UNUSED_PARAMETERS,
        "conditions_per_rank": conditions_per_rank,
        "global_conditions_per_step": expected_world * conditions_per_rank,
        "rank_topology": topology,
    }


def _run_all_ranks_or_raise(accelerator: Accelerator, label: str, action: Any) -> Any:
    """Run a local assembly action and fail every rank if any rank fails."""
    result: Any = None
    error: Exception | None = None
    try:
        result = action()
    except Exception as caught:
        error = caught
    if accelerator.num_processes > 1:
        failed = torch.tensor(
            [int(error is not None)], dtype=torch.int64, device=accelerator.device
        )
        failures = accelerator.gather(failed).detach().cpu().reshape(-1)
        if bool(failures.any()):
            summaries: list[object | None] = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(
                summaries,
                None if error is None else f"{type(error).__name__}: {error}",
            )
            raise RuntimeError(
                f"{label} failed on at least one rank: {tuple(summaries)}"
            ) from error
    elif error is not None:
        raise error
    return result


def _run_rank_zero_long_action(
    accelerator: Accelerator,
    label: str,
    status_path: Path,
    action: Any,
) -> None:
    """Coordinate a long rank-zero action without an idle collective timeout."""
    status_path = Path(status_path)
    if accelerator.is_main_process:
        try:
            action()
            payload = {"status": "passed"}
        except Exception as error:
            payload = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        atomic_write_json(status_path, payload)
    else:
        while not status_path.is_file():
            time.sleep(1.0)
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(
            f"{label} failed on rank zero: {payload.get('error')}\n"
            f"{payload.get('traceback')}"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        status_path.unlink()
    accelerator.wait_for_everyone()


def _verify_feature_store_for_run(
    state: "Stage2Preflight",
    data: "Stage2DependencyData",
    root: Path,
    *,
    stage: str,
    checkpoint_sha256: str,
    projection: FixedSparseProjection,
) -> Mapping[str, object]:
    expected_sources = _authenticated_raw_source_sha256(
        state, data.model_ids, verify_live=True
    )
    report = verify_geneeffect_feature_store(
        root,
        expected_stage=stage,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_feature_schema_sha256=FEATURE_SCHEMA.schema_hash,
        expected_projection_sha256=projection.components_hash,
        expected_source_sha256=expected_sources,
        expected_gene_embedding_source_sha256=sha256_file(
            state.config.paths.esm2_embeddings
        ),
        expected_model_ids=data.model_ids,
        expected_gene_symbols=data.genes,
    )
    if report.get("status") != "passed":
        raise RuntimeError(
            f"{stage} feature store verification failed: {report.get('discrepancies')}"
        )
    _authenticated_raw_source_sha256(state, data.model_ids, verify_live=True)
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


def _import_frozen_feature_store(
    state: "Stage2Preflight",
    data: "Stage2DependencyData",
    source: Path,
    target: Path,
    *,
    artifact_path: Path,
    checkpoint_sha256: str,
    projection: FixedSparseProjection,
) -> None:
    source = Path(source).resolve()
    target = Path(target)
    if source == target.resolve():
        raise ValueError("reused frozen feature store source and target must differ")
    if target.exists():
        raise FileExistsError(f"frozen feature store target already exists: {target}")
    if not source.is_dir():
        raise FileNotFoundError(
            f"reused frozen feature store is not a directory: {source}"
        )
    verify_kwargs = {
        "stage": "stage1_frozen",
        "checkpoint_sha256": checkpoint_sha256,
        "projection": projection,
    }
    _verify_feature_store_for_run(state, data, source, **verify_kwargs)
    source_manifest_sha256 = sha256_file(source / "manifest.json")
    staging = target.with_name(f".{target.name}.import-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"frozen feature store staging path exists: {staging}")
    try:
        shutil.copytree(source, staging, copy_function=os.link)
        _verify_feature_store_for_run(state, data, staging, **verify_kwargs)
        if sha256_file(source / "manifest.json") != source_manifest_sha256:
            raise ValueError("reused frozen feature store changed during import")
        staging.rename(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    atomic_write_json(
        artifact_path,
        {
            "source_path": str(source),
            "source_manifest_sha256": source_manifest_sha256,
            "imported_manifest_sha256": sha256_file(target / "manifest.json"),
            "method": "hardlink",
        },
    )


def _assert_configured_state_window(backbone: torch.nn.Module, expected: int) -> None:
    observed = _state_window(backbone)
    if observed != int(expected):
        raise ValueError(
            "loaded STATE cell_sentence_len does not match Stage 2 cell_set_len: "
            f"{observed} != {expected}"
        )


def _assert_joint_calibration_ready(model: GeneEffectE2EModel) -> None:
    """Require calibration to use the same train-mode boundary as joint steps."""
    if model.backbone_frozen:
        raise RuntimeError("lambda calibration requires an unfrozen backbone")
    if not model.training or not model.backbone.training or not model.head.training:
        raise RuntimeError(
            "lambda calibration requires model, backbone, and head in train mode"
        )
    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        raise RuntimeError(
            f"lambda calibration found non-trainable joint parameters: {frozen[:10]}"
        )


def _checked_calibration_closure(
    model: GeneEffectE2EModel,
    action: Callable[[], torch.Tensor],
    accelerator: Accelerator | None = None,
) -> Callable[[], torch.Tensor]:
    def checked() -> torch.Tensor:
        _assert_joint_calibration_ready(model)
        if accelerator is None:
            return action()
        with accelerator.autocast():
            return action()

    return checked


@dataclass(frozen=True)
class Stage2BundleSpec:
    """Configured paths required to load Stage 1."""

    stage1_config: Path
    cell_line_manifest: Path
    state_model_dir: Path
    perturbseq_sources: Path
    response_cache_dir: Path


@dataclass(frozen=True)
class Stage1CheckpointSpec:
    """Runtime inputs needed to restore the selected Stage 1 checkpoint."""

    stage1_genes: tuple[str, ...]
    checkpoint_sha256: str


@dataclass(frozen=True)
class Stage2Preflight:
    """Authenticated inputs needed before Stage 2 may allocate a GPU."""

    config: Stage2Config
    split: object
    universe: object
    residual_data: object
    variable_genes: object
    source_registry: object
    copy_prior: pd.Series
    stage1_checkpoint: Stage1CheckpointSpec
    bundle: Stage2BundleSpec
    report: Mapping[str, object]


@dataclass(frozen=True)
class ResponseAssembly:
    """Real Stage 1 anchor bags plus a deterministic epoch batch factory."""

    bags: object
    batch_factory: Any
    heldout_batch_factory: Any
    before_metrics: Mapping[str, object]
    batch_count: int
    train_records: tuple[Mapping[str, object], ...]
    heldout_records: tuple[Mapping[str, object], ...]
    dropped_records: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True)
class Stage2DependencyData:
    """In-memory identities plus lazily loaded basal and supervision arrays."""

    genes: tuple[str, ...]
    model_ids: tuple[str, ...]
    e_g: np.ndarray
    z_c: np.ndarray
    controls: Mapping[str, np.ndarray]
    basal_hvg: Mapping[str, np.ndarray]
    q_sc: Mapping[str, tuple[np.ndarray, np.ndarray]]
    hvg_indices: Mapping[str, int]
    targets: np.ndarray
    label_mask: np.ndarray
    g_var_mask: np.ndarray
    sampling: Mapping[str, Mapping[str, object]]
    residual_target_sha256: str
    centering_fit_model_ids_sha256: str
    mu_train_sha256: str


def _authenticated_target_esm2_sha256(state: Stage2Preflight) -> str:
    esm2_report = state.report.get("esm2")
    if not isinstance(esm2_report, Mapping):
        raise ValueError("preflight report is missing ESM2 authentication")
    expected = esm2_report.get("embedding_sha256")
    observed = sha256_file(state.config.paths.esm2_embeddings)
    if observed != expected:
        raise ValueError("target ESM2 embeddings changed after preflight")
    return observed


def _authenticated_raw_source_sha256(
    state: Stage2Preflight,
    model_ids: Sequence[str],
    *,
    verify_live: bool,
) -> dict[str, str]:
    """Return only the raw-source hashes authenticated during preflight."""
    tx1_report = state.report.get("tx1_cache")
    if not isinstance(tx1_report, Mapping):
        raise ValueError("preflight report is missing Tx1 cache authentication")
    raw = tx1_report.get("source_sha256")
    if not isinstance(raw, Mapping):
        raise ValueError("preflight Tx1 report is missing raw source SHA-256 bindings")
    expected = {str(key): str(value) for key, value in raw.items()}
    if set(expected) != set(model_ids):
        raise ValueError("preflight raw source SHA-256 membership mismatch")
    if verify_live:
        for model_id in model_ids:
            path = Path(state.source_registry.loc[model_id, "source_path"])
            if sha256_file(path) != expected[model_id]:
                raise ValueError(
                    f"raw source changed after preflight for {model_id}: {path}"
                )
    return {model_id: expected[model_id] for model_id in model_ids}


def load_stage2_bundle_spec(config: Stage2Config) -> Stage2BundleSpec:
    """Collect the Stage 1 paths already declared in the Stage 2 config."""
    bundle = Stage2BundleSpec(
        stage1_config=config.paths.stage1_config,
        cell_line_manifest=config.paths.cell_line_manifest,
        state_model_dir=config.paths.state_model_dir,
        perturbseq_sources=config.paths.perturbseq_sources,
        response_cache_dir=config.paths.response_cache,
    )
    for asset, label in (
        (bundle.stage1_config, "Stage 1 config"),
        (bundle.cell_line_manifest, "Stage 1 cell-line manifest"),
        (bundle.perturbseq_sources, "Stage 1 perturb-seq sources"),
    ):
        _require_file(asset, label)
    for asset, label in (
        (bundle.state_model_dir, "STATE model directory"),
        (bundle.response_cache_dir, "warm Stage 1 response cache"),
    ):
        _require_directory(asset, label)
    return bundle


def _require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or is not a file: {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or is not a directory: {path}")
    return path


def load_stage1_checkpoint_spec(
    config: Stage2Config,
    *,
    universe_manifest: Mapping[str, object],
    target_esm_symbols: tuple[str, ...],
) -> Stage1CheckpointSpec:
    """Load the Stage 1 vocabulary and observe the selected checkpoint identity."""
    checkpoint_path = _require_file(
        config.paths.stage1_checkpoint, "Stage 1 selected checkpoint"
    )
    _require_file(config.paths.state_hparams, "STATE hparams checkpoint")
    vocabulary = universe_manifest.get("stage1_vocabulary")
    raw_genes = vocabulary.get("symbols") if isinstance(vocabulary, Mapping) else None
    if (
        not isinstance(raw_genes, list)
        or not raw_genes
        or not all(isinstance(gene, str) and gene for gene in raw_genes)
    ):
        raise ValueError("ESM2 universe manifest lacks the Stage 1 gene vocabulary")
    stage1_genes = tuple(gene.upper() for gene in raw_genes)
    if len(set(stage1_genes)) != len(stage1_genes):
        raise ValueError("Stage 1 gene vocabulary must be unique")
    target_symbols = set(target_esm_symbols)
    missing = sorted(set(stage1_genes) - target_symbols)
    if missing:
        raise ValueError(
            "target-universe ESM2 artifact does not cover Stage 1 genes: "
            f"{missing[:10]}"
        )
    return Stage1CheckpointSpec(
        stage1_genes=stage1_genes,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def _copy_prior_symbols_sha256(symbols: Sequence[str]) -> str:
    payload = "".join(f"{symbol}\n" for symbol in symbols).encode()
    return hashlib.sha256(payload).hexdigest()


def _authenticate_copy_prior(
    config: Stage2Config, split: object, labels: pd.DataFrame
) -> pd.Series:
    """Authenticate the materialized train-side K562 copy-prior artifact."""
    manifest_path = _require_file(
        config.paths.copy_prior_manifest, "copy-prior manifest"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"copy-prior manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("copy-prior manifest root must be an object")
    if manifest.get("schema_version") != _COPY_PRIOR_SCHEMA:
        raise ValueError("copy-prior manifest schema_version mismatch")
    donor = manifest.get("donor")
    if donor != {
        "model_id": _COPY_PRIOR_DONOR,
        "split": "train",
        "unlabeled": False,
    }:
        raise ValueError("copy-prior manifest donor identity mismatch")
    if _COPY_PRIOR_DONOR not in split.train:
        raise ValueError("copy-prior donor is not a train member")
    if _COPY_PRIOR_DONOR in split.unlabeled_train:
        raise ValueError("copy-prior donor is an unlabeled train member")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("copy-prior manifest source metadata is missing")
    actual_source_sha256 = sha256_file(config.paths.gene_effect)
    if actual_source_sha256 != _PINNED_GENE_EFFECT_SHA256:
        raise ValueError("configured GeneEffect does not match pinned DepMap 26Q1")
    if source.get("sha256") != actual_source_sha256:
        raise ValueError("copy-prior manifest source SHA-256 mismatch")
    split_metadata = manifest.get("split")
    if not isinstance(split_metadata, dict):
        raise ValueError("copy-prior manifest split metadata is missing")
    actual_split_sha256 = sha256_file(config.paths.split)
    if actual_split_sha256 != PINNED_SPLIT_SHA256:
        raise ValueError("configured Exp13 split SHA-256 mismatch")
    if split_metadata.get("sha256") != actual_split_sha256:
        raise ValueError("copy-prior manifest split SHA-256 mismatch")

    output_metadata = manifest.get("output")
    if not isinstance(output_metadata, dict):
        raise ValueError("copy-prior manifest output metadata is missing")
    actual_output_sha256 = sha256_file(config.paths.copy_prior)
    if output_metadata.get("sha256") != actual_output_sha256:
        raise ValueError("copy-prior CSV SHA-256 mismatch")
    copy_frame = pd.read_csv(config.paths.copy_prior)
    if tuple(copy_frame.columns) != ("gene_symbol", "gene_effect"):
        raise ValueError(
            "copy_prior must contain exactly gene_symbol,gene_effect columns"
        )
    if (
        copy_frame["gene_symbol"].isna().any()
        or copy_frame["gene_symbol"].duplicated().any()
        or (copy_frame["gene_symbol"].astype(str) == "").any()
    ):
        raise ValueError("copy_prior gene_symbol values must be nonmissing and unique")
    copy_values = pd.to_numeric(copy_frame["gene_effect"], errors="coerce")
    if copy_values.isna().any() or not np.isfinite(copy_values.to_numpy()).all():
        raise ValueError("copy_prior gene_effect values must be finite numeric values")
    symbols = tuple(copy_frame["gene_symbol"].astype(str))
    if output_metadata.get("gene_symbols_sha256") != _copy_prior_symbols_sha256(
        symbols
    ):
        raise ValueError("copy-prior manifest gene-symbol coverage hash mismatch")
    counts = manifest.get("counts")
    drops = manifest.get("drop_reason_counts")
    if not isinstance(counts, dict) or not isinstance(drops, dict):
        raise ValueError("copy-prior manifest count metadata is missing")
    output_count = counts.get("output_gene_count")
    dropped_count = counts.get("dropped_gene_count")
    source_count = counts.get("source_gene_count")
    if (
        output_count != len(symbols)
        or not isinstance(dropped_count, int)
        or dropped_count < 0
        or source_count != len(symbols) + dropped_count
        or drops != {"missing_gene_effect": dropped_count}
    ):
        raise ValueError("copy-prior manifest counts do not match exact CSV coverage")

    donor_rows = labels.loc[labels["model_id"] == _COPY_PRIOR_DONOR]
    if donor_rows.empty:
        raise ValueError("copy-prior donor has no rows in pinned GeneEffect labels")
    if donor_rows["gene_symbol"].duplicated().any():
        raise ValueError("copy-prior donor labels contain duplicate gene symbols")
    finite_donor = donor_rows.loc[donor_rows["gene_effect"].notna()]
    expected_symbols = tuple(finite_donor["gene_symbol"].astype(str))
    if symbols != expected_symbols or actual_output_sha256 != PINNED_COPY_PRIOR_SHA256:
        raise ValueError("copy-prior CSV does not exactly match the pinned donor row")
    if source_count != len(donor_rows) or dropped_count != int(
        donor_rows["gene_effect"].isna().sum()
    ):
        raise ValueError(
            "copy-prior manifest counts disagree with the pinned donor row"
        )
    missing_symbols = tuple(
        donor_rows.loc[donor_rows["gene_effect"].isna(), "gene_symbol"].astype(str)
    )
    if manifest.get("donor_missing") != {
        "count": len(missing_symbols),
        "symbols": list(missing_symbols),
        "symbols_sha256": _copy_prior_symbols_sha256(missing_symbols),
    }:
        raise ValueError("copy-prior manifest donor-missing coverage mismatch")
    missing_symbols = tuple(
        donor_rows.loc[donor_rows["gene_effect"].isna(), "gene_symbol"].astype(str)
    )
    if manifest.get("donor_missing") != {
        "count": len(missing_symbols),
        "symbols": list(missing_symbols),
        "symbols_sha256": _copy_prior_symbols_sha256(missing_symbols),
    }:
        raise ValueError("copy-prior manifest donor-missing coverage mismatch")

    copy_prior = pd.Series(
        copy_values.to_numpy(dtype=float),
        index=pd.Index(symbols),
        name="gene_effect",
    )
    return copy_prior


def authenticate_target_esm2(
    config: Stage2Config,
    *,
    coverage_qualified_symbols: Sequence[str],
    candidate_symbols: Sequence[str],
    coverage_drop_report: Sequence[Mapping[str, object]],
    candidate_drop_report: Sequence[Mapping[str, object]],
    scored_symbols: Sequence[str],
    embedding_symbols: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate the generated ESM2 table, sidecar, and builder manifest."""
    provenance = load_and_authenticate_esm2_provenance(
        config.paths.esm2_provenance_manifest,
        config.paths.esm2_embeddings,
        expected_width=config.features.esm2_dim,
        mapping_json_path=config.paths.esm2_uniprot_mapping_json,
        mapping_csv_path=config.paths.esm2_uniprot_mapping_csv,
    )
    try:
        universe_manifest = json.loads(
            config.paths.esm2_universe_manifest.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid ESM2 universe manifest: {exc}") from exc
    if (
        not isinstance(universe_manifest, dict)
        or universe_manifest.get("schema_version") != _ESM2_UNIVERSE_SCHEMA
        or universe_manifest.get("status") != "authenticated_complete"
    ):
        raise ValueError("ESM2 universe manifest is not authenticated complete")
    upper_record = universe_manifest.get("coverage_qualified_upper_bound")
    candidate_record = universe_manifest.get("copy_prior_eligible_candidates")
    scored_record = universe_manifest.get("final_evaluated_universe")
    union_record = universe_manifest.get("embedding_union")
    if (
        not isinstance(upper_record, dict)
        or not isinstance(candidate_record, dict)
        or not isinstance(scored_record, dict)
        or not isinstance(union_record, dict)
    ):
        raise ValueError("ESM2 universe manifest is missing universe records")
    candidates = candidate_record.get("symbols")
    upper_symbols = upper_record.get("symbols")
    if (
        not isinstance(candidates, list)
        or not isinstance(upper_symbols, list)
        or any(not isinstance(symbol, str) for symbol in (*upper_symbols, *candidates))
        or candidates != sorted(set(candidates))
        or upper_symbols != sorted(set(upper_symbols))
        or not set(candidates).issubset(set(upper_symbols))
    ):
        raise ValueError("ESM2 universe manifest candidate symbols are invalid")
    for record, symbols, label in (
        (upper_record, upper_symbols, "coverage upper bound"),
        (candidate_record, candidates, "copy-prior candidates"),
    ):
        if record.get("count") != len(symbols) or record.get(
            "symbols_sha256"
        ) != _copy_prior_symbols_sha256(symbols):
            raise ValueError(f"ESM2 universe manifest {label} count/hash mismatch")
    if upper_record.get("drop_report") != list(coverage_drop_report):
        raise ValueError("ESM2 coverage-qualified drop report mismatch")
    if candidate_record.get("drop_report") != list(candidate_drop_report):
        raise ValueError("ESM2 copy-prior candidate drop report mismatch")
    if upper_symbols != list(coverage_qualified_symbols):
        raise ValueError("ESM2 coverage-qualified upper bound mismatch")
    if candidates != list(candidate_symbols):
        raise ValueError("ESM2 copy-prior candidate universe mismatch")
    expected_candidate_csv = "gene_symbol\n" + "".join(
        f"{symbol}\n" for symbol in candidates
    )
    if (
        candidate_record.get("csv_sha256")
        != hashlib.sha256(expected_candidate_csv.encode("utf-8")).hexdigest()
    ):
        raise ValueError("ESM2 candidate CSV identity mismatch")
    resolved = set(embedding_symbols)
    expected_scored = tuple(symbol for symbol in candidates if symbol in resolved)
    unresolved = tuple(symbol for symbol in candidates if symbol not in resolved)
    expected_record = {
        "symbols": list(expected_scored),
        "count": len(expected_scored),
        "symbols_sha256": _copy_prior_symbols_sha256(expected_scored),
        "unresolved_candidate_symbols": list(unresolved),
        "unresolved_candidate_count": len(unresolved),
    }
    if scored_record != expected_record or expected_scored != tuple(scored_symbols):
        raise ValueError("ESM2 universe manifest final evaluated record mismatch")
    if (
        universe_manifest.get("scored_symbols") != list(expected_scored)
        or universe_manifest.get("scored_gene_count") != len(expected_scored)
        or universe_manifest.get("coverage_thresholds")
        != {"train": 5, "val": 3, "test": 3}
    ):
        raise ValueError("ESM2 universe manifest top-level scoring metadata mismatch")
    artifact = provenance.get("embedding_artifact")
    if not isinstance(artifact, dict) or union_record.get("symbols") != artifact.get(
        "symbols"
    ):
        raise ValueError("ESM2 universe manifest embedding symbols mismatch")
    if not set(embedding_symbols).issubset(set(union_record["symbols"])):
        raise ValueError("loaded ESM2 symbols fall outside the embedding union")
    stage1_record = universe_manifest.get("stage1_vocabulary")
    if not isinstance(stage1_record, dict) or not isinstance(
        stage1_record.get("symbols"), list
    ):
        raise ValueError("ESM2 universe manifest Stage1 vocabulary is invalid")
    expected_union = sorted(set(candidates) | set(stage1_record["symbols"]))
    if union_record.get("symbols") != expected_union:
        raise ValueError("ESM2 embedding union is not candidates plus Stage1")
    expected_union_csv = "gene_symbol\n" + "".join(
        f"{symbol}\n" for symbol in expected_union
    )
    stage1_only = [
        symbol for symbol in stage1_record["symbols"] if symbol not in set(candidates)
    ]
    if (
        union_record.get("count") != len(expected_union)
        or union_record.get("symbols_sha256")
        != _copy_prior_symbols_sha256(expected_union)
        or union_record.get("csv_sha256")
        != hashlib.sha256(expected_union_csv.encode("utf-8")).hexdigest()
        or union_record.get("stage1_only_symbols") != stage1_only
        or union_record.get("stage1_only_count") != len(stage1_only)
        or union_record.get("drop_report") != []
    ):
        raise ValueError("ESM2 embedding-union provenance mismatch")
    verified_npz = union_record.get("verified_npz")
    provenance_record = union_record.get("provenance_manifest")
    if (
        not isinstance(verified_npz, dict)
        or verified_npz.get("artifact_sha256")
        != sha256_file(config.paths.esm2_embeddings)
        or verified_npz.get("resolved_count") != artifact.get("resolved_count")
        or verified_npz.get("vector_width") != artifact.get("vector_width")
        or union_record.get("requested_precompute_model")
        != provenance.get("requested_model_id")
        or union_record.get("model_identity_status")
        != "recorded_from_loaded_runtime_state"
        or union_record.get("uniprot_mapping")
        != {
            "isoform_policy": "canonical_reviewed_top_hit",
            "json_sha256": sha256_file(config.paths.esm2_uniprot_mapping_json),
            "csv_sha256": sha256_file(config.paths.esm2_uniprot_mapping_csv),
        }
    ):
        raise ValueError("ESM2 universe manifest NPZ authentication mismatch")
    if (
        not isinstance(provenance_record, dict)
        or provenance_record.get("sha256")
        != sha256_file(config.paths.esm2_provenance_manifest)
        or provenance_record.get("payload") != provenance
    ):
        raise ValueError("ESM2 universe manifest provenance authentication mismatch")
    input_sha256 = universe_manifest.get("input_sha256")
    if (
        not isinstance(input_sha256, dict)
        or input_sha256.get("split") != sha256_file(config.paths.split)
        or input_sha256.get("gene_effect") != sha256_file(config.paths.gene_effect)
        or input_sha256.get("copy_prior") != sha256_file(config.paths.copy_prior)
        or input_sha256.get("copy_prior_manifest")
        != sha256_file(config.paths.copy_prior_manifest)
    ):
        raise ValueError("ESM2 universe manifest input authentication mismatch")
    return universe_manifest, provenance


def _builder_drop_reports(
    coverage_upper_bound: object,
    candidates: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reconstruct the universe builder's exact ordered drop reports."""
    upper_drops: list[dict[str, object]] = []
    for row in coverage_upper_bound.coverage.itertuples(index=False):
        if bool(row.included):
            continue
        upper_drops.append(
            {
                "gene_symbol": str(row.gene_symbol),
                "finite_counts": {
                    "train": int(row.train_finite),
                    "val": int(row.val_finite),
                    "test": int(row.test_finite),
                },
                "reasons": str(row.drop_reason).split("|"),
            }
        )
    candidate_set = set(candidates.symbols)
    candidate_drops = [*upper_drops]
    candidate_drops.extend(
        {"gene_symbol": symbol, "reasons": ["copy_prior_missing"]}
        for symbol in coverage_upper_bound.symbols
        if symbol not in candidate_set
    )
    return upper_drops, candidate_drops


def preflight_stage2(config_path: Path) -> Stage2Preflight:
    """Validate all Stage 2 inputs without writing a run directory."""
    config = load_stage2_config(config_path)
    bundle = load_stage2_bundle_spec(config)
    for path, label in (
        (config.paths.split, "Exp13 split"),
        (config.paths.gene_effect, "DepMap GeneEffect"),
        (config.paths.source_registry, "raw-UMI source registry"),
        (config.paths.tx1_registration, "pinned Tx1 Phase-A registration"),
        (config.paths.esm2_embeddings, "target-universe ESM2 embeddings"),
        (config.paths.esm2_universe_manifest, "ESM2 universe manifest"),
        (config.paths.esm2_provenance_manifest, "ESM2 provenance manifest"),
        (config.paths.esm2_uniprot_mapping_json, "ESM2 UniProt mapping JSON"),
        (config.paths.esm2_uniprot_mapping_csv, "ESM2 UniProt mapping CSV"),
        (config.paths.copy_prior, "copy-prior baseline asset"),
        (config.paths.copy_prior_manifest, "copy-prior manifest"),
    ):
        _require_file(path, label)
    _require_directory(config.paths.tx1_cache, "Tx1 embedding cache")
    _require_directory(config.paths.q_sc_cache, "q_sc cache")

    split = load_exp13_split(config.paths.split)
    labels = load_geneeffect_long(config.paths.gene_effect, split)
    esm2 = load_esm2_embeddings(config.paths.esm2_embeddings)
    if esm2.dim != config.features.esm2_dim:
        raise ValueError(
            f"ESM2 width {esm2.dim} != configured width {config.features.esm2_dim}"
        )
    esm2_symbols = tuple(esm2.vectors_by_symbol)
    all_label_symbols = tuple(sorted(set(labels["gene_symbol"].astype(str))))
    coverage_upper_bound = build_scored_universe(labels, split, all_label_symbols)
    copy_prior = _authenticate_copy_prior(config, split, labels)
    candidates = restrict_scored_universe_to_copy_prior(
        coverage_upper_bound, copy_prior.index
    )
    coverage_drops, candidate_drops = _builder_drop_reports(
        coverage_upper_bound, candidates
    )
    universe = build_scored_universe(labels, split, esm2_symbols)
    if not universe.symbols:
        raise ValueError("ESM2/GeneEffect intersection produced no scored genes")
    universe = restrict_scored_universe_to_copy_prior(universe, copy_prior.index)
    if not universe.symbols:
        raise ValueError("copy-prior coverage produced no scored genes")
    copy_prior = copy_prior.reindex(universe.symbols)
    universe_manifest, esm2_provenance = authenticate_target_esm2(
        config,
        coverage_qualified_symbols=coverage_upper_bound.symbols,
        candidate_symbols=candidates.symbols,
        coverage_drop_report=coverage_drops,
        candidate_drop_report=candidate_drops,
        scored_symbols=universe.symbols,
        embedding_symbols=esm2_symbols,
    )
    residual_data = build_residual_data(labels, split, universe)
    variable_genes = build_g_var(residual_data, split, universe)
    if not variable_genes.symbols:
        raise ValueError("G_var is empty")
    registry = load_source_registry(config.paths.source_registry, split)
    missing_sources = [
        str(path) for path in registry["source_path"].map(Path) if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"raw-UMI registry source files are missing: {missing_sources[:10]}"
        )
    tx1_source_manifest, tx1_source_manifest_sha256 = authenticate_tx1_registration(
        config.paths.tx1_registration
    )
    expected_source_sha256 = {
        str(model_id): sha256_file(Path(str(row["source_path"])))
        for model_id, row in registry.iterrows()
    }
    tx1_report = dict(
        verify_cache(
            config.paths.tx1_cache,
            expected_model_ids=split.all_model_ids,
            expected_source_sha256=expected_source_sha256,
            expected_matrix_semantics=RAW_UMI_SEMANTICS,
            expected_tx1_source_manifest=tx1_source_manifest,
        )
    )
    if tx1_report.get("status") != "verified":
        raise ValueError(
            f"Tx1 cache verification failed: {tx1_report.get('discrepancies')}"
        )
    tx1_report["source_sha256"] = expected_source_sha256
    q_sc_report = verify_q_sc_shards(
        registry, config.paths.q_sc_cache, universe.symbols
    )
    if q_sc_report.get("status") != "passed":
        raise ValueError(
            f"q_sc cache verification failed: {q_sc_report.get('discrepancies')}"
        )
    stage1_checkpoint = load_stage1_checkpoint_spec(
        config,
        universe_manifest=universe_manifest,
        target_esm_symbols=esm2_symbols,
    )
    report: dict[str, object] = {
        "status": "passed",
        "config_sha256": config.source_sha256,
        "cell_lines": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "total": len(split.all_model_ids),
        },
        "scored_gene_count": len(universe.symbols),
        "g_var_gene_count": len(variable_genes.symbols),
        "copy_prior_sha256": sha256_file(config.paths.copy_prior),
        "copy_prior_manifest_sha256": sha256_file(config.paths.copy_prior_manifest),
        "esm2": {
            "embedding_sha256": sha256_file(config.paths.esm2_embeddings),
            "universe_manifest_sha256": sha256_file(
                config.paths.esm2_universe_manifest
            ),
            "provenance_manifest_sha256": sha256_file(
                config.paths.esm2_provenance_manifest
            ),
            "uniprot_mapping_json_sha256": sha256_file(
                config.paths.esm2_uniprot_mapping_json
            ),
            "uniprot_mapping_csv_sha256": sha256_file(
                config.paths.esm2_uniprot_mapping_csv
            ),
            "requested_model_id": esm2_provenance["requested_model_id"],
            "loaded_model_state_sha256": esm2_provenance["loaded_model"][
                "state_sha256"
            ],
            "loaded_model_config_sha256": esm2_provenance["loaded_model"][
                "config_sha256"
            ],
            "tokenizer_vocabulary_config_sha256": esm2_provenance["tokenizer"][
                "vocabulary_config_sha256"
            ],
        },
        "tx1_cache": tx1_report,
        "tx1_registration": {
            "registration_sha256": sha256_file(config.paths.tx1_registration),
            "source_manifest_sha256": tx1_source_manifest_sha256,
            "model_revision": tx1_source_manifest["model_revision"],
        },
        "q_sc_cache": q_sc_report,
        "stage1": {
            "checkpoint_sha256": stage1_checkpoint.checkpoint_sha256,
            "checkpoint_identity_status": "observed_not_authenticated",
            "stage1_gene_count": len(stage1_checkpoint.stage1_genes),
            "vocabulary_source": "esm2_universe_manifest",
        },
    }
    return Stage2Preflight(
        config=config,
        split=split,
        universe=universe,
        residual_data=residual_data,
        variable_genes=variable_genes,
        source_registry=registry,
        copy_prior=copy_prior,
        stage1_checkpoint=stage1_checkpoint,
        bundle=bundle,
        report=report,
    )


def _response_records(
    bags: object, anchor_weights: Mapping[str, float]
) -> list[dict[str, object]]:
    control_batch = np.asarray(bags.control_batch).astype(str)
    controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_id in sorted(set(control_batch.tolist())):
        mask = control_batch == model_id
        controls[model_id] = (
            np.asarray(bags.control_input)[mask],
            np.asarray(bags.effective_control_target)[mask],
        )
    labels = tuple(str(value) for value in bags.genes)
    records: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        model_id = label.split("@", 1)[1]
        if model_id not in controls or model_id not in anchor_weights:
            raise ValueError(f"response bag {label!r} has no registered anchor weight")
        control_tx1, control_hvg = controls[model_id]
        records.append(
            {
                "record_id": label,
                "gene": base_gene_name(label),
                "model_id": model_id,
                "control_tx1": control_tx1,
                "observed_hvg": np.asarray(bags.effective_target_bags[index]),
                "control_hvg": control_hvg,
                "anchor_weight": float(anchor_weights[model_id]),
            }
        )
    if not records:
        raise ValueError("response assembly produced no gene/anchor records")
    return records


def _normalize_response_weights(records: list[dict[str, object]]) -> None:
    counts: dict[str, int] = {}
    for row in records:
        model_id = str(row["model_id"])
        counts[model_id] = counts.get(model_id, 0) + 1
    for row in records:
        model_id = str(row["model_id"])
        row["weight"] = float(row["anchor_weight"]) / counts[model_id]


def response_batch_to_device(
    batch: ResponseSupervisionBatch, device: torch.device
) -> ResponseSupervisionBatch:
    """Move every tensor in a response dataclass to the model device."""
    moved = ResponseSupervisionBatch(
        controls_tx1=tuple(value.to(device) for value in batch.controls_tx1),
        observed_hvg=tuple(value.to(device) for value in batch.observed_hvg),
        control_hvg=tuple(value.to(device) for value in batch.control_hvg),
        genes=batch.genes,
        objective_weights=batch.objective_weights.to(device),
        batch_weight=batch.batch_weight,
    )
    moved.validate()
    return moved


def assemble_response_supervision(state: Stage2Preflight) -> ResponseAssembly:
    """Load the warm Stage 1 response cache and expose joint-training batches."""
    stage1 = load_stage1_config(state.bundle.stage1_config)
    response_manifest = (
        state.bundle.response_cache_dir / "response_targets" / "manifest.json"
    )
    _require_file(response_manifest, "warm Stage 1 response-target cache manifest")
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=state.bundle.cell_line_manifest,
        tx1_cache_dir=state.config.paths.tx1_cache,
        hvg_state_model_dir=state.bundle.state_model_dir,
        perturbseq_sources_path=state.bundle.perturbseq_sources,
        max_cells_per_gene=stage1.train.max_cells_per_gene,
        total_cells_per_line=stage1.train.total_cells_per_line,
        control_cells_per_line=stage1.train.max_bag,
        response_cache_dir=state.bundle.response_cache_dir,
        seed=stage1.train.data_seed,
        expected_cache_model_ids=state.split.all_model_ids,
    )
    all_records = _response_records(bags, dict(stage1.objective.anchor_weights))
    run_root = state.config.paths.stage1_checkpoint.parent.parent
    run_manifest = json.loads((run_root / "run_manifest.json").read_text())
    sealed_genes = set(state.stage1_checkpoint.stage1_genes)
    unresolved_raw = run_manifest.get("esm2_unresolved_genes")
    if (
        not isinstance(unresolved_raw, list)
        or any(not isinstance(gene, str) or not gene for gene in unresolved_raw)
        or len(set(unresolved_raw)) != len(unresolved_raw)
    ):
        raise ValueError(
            "Stage 1 run manifest esm2_unresolved_genes must be unique strings"
        )
    unresolved = set(unresolved_raw)
    dropped_records = tuple(
        {
            "record_id": str(row["record_id"]),
            "gene": str(row["gene"]),
            "model_id": str(row["model_id"]),
            "reason": "stage1_esm2_unresolved",
        }
        for row in all_records
        if str(row["gene"]) not in sealed_genes and str(row["gene"]) in unresolved
    )
    undeclared = sorted(
        {
            str(row["gene"])
            for row in all_records
            if str(row["gene"]) not in sealed_genes
            and str(row["gene"]) not in unresolved
        }
    )
    if undeclared:
        raise ValueError(
            "response genes are not in the sealed Stage 1 vocabulary and were not "
            f"declared ESM2-unresolved: {undeclared[:10]}"
        )
    eligible_records = [row for row in all_records if str(row["gene"]) in sealed_genes]
    heldout_raw = run_manifest.get("heldout_genes")
    if not isinstance(heldout_raw, dict):
        raise ValueError("Stage 1 run manifest has no heldout_genes mapping")
    heldout = {
        str(model_id): {str(gene) for gene in genes}
        for model_id, genes in heldout_raw.items()
    }
    records = [
        row
        for row in eligible_records
        if str(row["gene"]) not in heldout.get(str(row["model_id"]), set())
    ]
    heldout_records = [
        row
        for row in eligible_records
        if str(row["gene"]) in heldout.get(str(row["model_id"]), set())
    ]
    if (
        not records
        or not heldout_records
        or len(records) + len(heldout_records) != len(eligible_records)
    ):
        raise ValueError("Stage 1 held-out response partition is empty or incomplete")
    expected_counts = {
        "n_train_batches": len(records),
        "n_heldout_batches": len(heldout_records),
    }
    observed_counts = {key: run_manifest.get(key) for key in expected_counts}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in observed_counts.values()
    ):
        raise ValueError(
            "Stage 1 run manifest response record counts must be non-negative integers"
        )
    if observed_counts != expected_counts:
        raise ValueError(
            "Stage 1 response record counts mismatch: "
            f"manifest={observed_counts}, assembled={expected_counts}"
        )
    _normalize_response_weights(records)
    _normalize_response_weights(heldout_records)
    batch_size = state.config.joint.response_batch_size

    def make_factory(source_records: list[dict[str, object]]):
        def batch_factory(epoch: int):
            generator = np.random.default_rng(state.config.seeds.train + int(epoch))
            order = generator.permutation(len(source_records)).tolist()
            for start in range(0, len(order), batch_size):
                selected = [
                    source_records[index] for index in order[start : start + batch_size]
                ]
                batch = ResponseSupervisionBatch(
                    controls_tx1=tuple(
                        torch.as_tensor(row["control_tx1"], dtype=torch.float32)
                        for row in selected
                    ),
                    observed_hvg=tuple(
                        torch.as_tensor(row["observed_hvg"], dtype=torch.float32)
                        for row in selected
                    ),
                    control_hvg=tuple(
                        torch.as_tensor(row["control_hvg"], dtype=torch.float32)
                        for row in selected
                    ),
                    genes=tuple(str(row["gene"]) for row in selected),
                    objective_weights=torch.tensor(
                        [float(row["weight"]) for row in selected], dtype=torch.float32
                    ),
                )
                yield batch

        return batch_factory

    before_metrics_path = run_root / "heldout_metrics.json"
    before_metrics = json.loads(
        _require_file(before_metrics_path, "Stage 1 held-out metrics").read_text()
    )
    before_loss = before_metrics.get("model_loss")
    if not isinstance(before_loss, (int, float)) or not np.isfinite(before_loss):
        raise ValueError("Stage 1 heldout_metrics model_loss must be a finite number")
    return ResponseAssembly(
        bags=bags,
        batch_factory=make_factory(records),
        heldout_batch_factory=make_factory(heldout_records),
        before_metrics=before_metrics,
        batch_count=(len(records) + batch_size - 1) // batch_size,
        train_records=tuple(records),
        heldout_records=tuple(heldout_records),
        dropped_records=dropped_records,
    )


def construct_stage2_backbone(
    state: Stage2Preflight,
    response: ResponseAssembly,
    *,
    model_cls: type[torch.nn.Module] | None = None,
):
    """Construct the target-universe backbone and strictly restore Stage 1."""
    if model_cls is None:
        from state.tx.models.state_transition import (
            StateTransitionPerturbationModel,
        )

        model_cls = StateTransitionPerturbationModel
    from aivc_model.tx1_predicted_response import (
        construct_stage2_model_from_stage1_artifact,
    )

    stage1 = load_stage1_config(state.bundle.stage1_config)
    target_genes = tuple(
        dict.fromkeys((*state.universe.symbols, *state.stage1_checkpoint.stage1_genes))
    )
    return construct_stage2_model_from_stage1_artifact(
        model_cls=model_cls,
        checkpoint_path=state.config.paths.stage1_checkpoint,
        hparams_checkpoint_path=state.config.paths.state_hparams,
        input_dim=int(response.bags.input_dim),
        output_dim=int(response.bags.effective_target_dim),
        pert_dim=stage1.train.pert_dim,
        target_genes=target_genes,
        target_esm_embeddings_path=state.config.paths.esm2_embeddings,
        trainable=False,
    )


def _paired_sample_indices(
    model_id: str, obs: Any, *, count: int
) -> tuple[np.ndarray, Mapping[str, object]]:
    barcodes = tuple(str(value) for value in obs.index)
    if not barcodes or len(set(barcodes)) != len(barcodes):
        raise ValueError(f"{model_id}: obs cell barcodes must be non-empty and unique")
    ranked = sorted(
        range(len(barcodes)),
        key=lambda index: hashlib.sha256(
            f"{model_id}|{barcodes[index]}".encode()
        ).digest(),
    )
    distinct = ranked[: min(count, len(ranked))]
    indices = list(distinct)
    while len(indices) < count:
        indices.append(distinct[(len(indices) - len(distinct)) % len(distinct)])
    return np.asarray(indices, dtype=np.int64), {
        "distinct_count": len(distinct),
        "sampled_count": count,
        "padding_fraction": (count - len(distinct)) / count,
        "selection": "sha256(model_id|cell_barcode)_ascending",
    }


def _verify_artifact_hash(path: Path, expected: str, *, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _verify_named_artifact_hashes(
    root: Path, expected: Mapping[str, object], *, label: str
) -> None:
    if set(expected) != {"embeddings.npy", "hvg.npy", "obs.parquet"}:
        raise ValueError(f"{label} artifact membership mismatch")
    for filename, digest in expected.items():
        _verify_artifact_hash(
            root / str(filename), str(digest), label=f"{label} {filename}"
        )


def _cache_identity_fields(state: Stage2Preflight) -> dict[str, object]:
    tx1 = state.report.get("tx1_cache")
    q_sc = state.report.get("q_sc_cache")
    registration = state.report.get("tx1_registration")
    if not all(isinstance(value, Mapping) for value in (tx1, q_sc, registration)):
        raise ValueError("preflight cache identity reports are missing")
    fields = {
        "tx1_registration_sha256": registration.get("registration_sha256"),
        "tx1_source_manifest_sha256": registration.get("source_manifest_sha256"),
        "tx1_cache_manifest_sha256": tx1.get("manifest_sha256"),
        "q_sc_cache_manifest_sha256": q_sc.get("manifest_sha256"),
    }
    return fields


def build_dependency_data(state: Stage2Preflight) -> Stage2DependencyData:
    """Load aligned 226-line basal, q_sc, ESM2 and residual supervision data."""
    genes = tuple(state.universe.symbols)
    model_ids = tuple(state.split.all_model_ids)
    _authenticated_target_esm2_sha256(state)
    esm2 = load_esm2_embeddings(state.config.paths.esm2_embeddings)
    e_g = np.stack([esm2.vectors_by_symbol[gene] for gene in genes]).astype(np.float32)
    hvg_order = tuple(
        str(value).upper()
        for value in load_hvg_gene_order(state.bundle.state_model_dir)
    )
    hvg_indices = {gene: index for index, gene in enumerate(hvg_order)}
    if len(hvg_order) != 2_000 or len(hvg_indices) != len(hvg_order):
        raise ValueError("STATE HVG order must contain exactly 2000 unique genes")

    controls: dict[str, np.ndarray] = {}
    basal_hvg: dict[str, np.ndarray] = {}
    q_sc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sampling: dict[str, Mapping[str, object]] = {}
    z_rows: list[np.ndarray] = []
    tx1_report = state.report.get("tx1_cache")
    q_sc_report = state.report.get("q_sc_cache")
    if not isinstance(tx1_report, Mapping) or not isinstance(q_sc_report, Mapping):
        raise ValueError("preflight cache authentication reports are missing")
    tx1_hashes = tx1_report.get("line_artifact_sha256")
    q_sc_hashes = q_sc_report.get("shard_sha256")
    if not isinstance(tx1_hashes, Mapping) or set(tx1_hashes) != set(model_ids):
        raise ValueError("preflight Tx1 line hash coverage is incomplete")
    if not isinstance(q_sc_hashes, Mapping) or set(q_sc_hashes) != set(model_ids):
        raise ValueError("preflight q_sc shard hash coverage is incomplete")
    for model_id in model_ids:
        line_hashes = tx1_hashes[model_id]
        if not isinstance(line_hashes, Mapping):
            raise ValueError(f"{model_id}: preflight Tx1 hashes are malformed")
        _verify_named_artifact_hashes(
            state.config.paths.tx1_cache / model_id,
            line_hashes,
            label=f"{model_id} Tx1 pre-load",
        )
        embeddings, hvg, obs = load_line_cache(state.config.paths.tx1_cache, model_id)
        if embeddings.shape[0] != hvg.shape[0] or embeddings.shape[0] != len(obs):
            raise ValueError(f"{model_id}: Tx1/HVG/obs row counts disagree")
        indices, sampling[model_id] = _paired_sample_indices(
            model_id,
            obs,
            count=state.config.features.cells_per_context,
        )
        controls[model_id] = np.array(embeddings[indices], dtype=np.float32, copy=True)
        basal_hvg[model_id] = np.array(hvg[indices], dtype=np.float32, copy=True)
        _verify_named_artifact_hashes(
            state.config.paths.tx1_cache / model_id,
            line_hashes,
            label=f"{model_id} Tx1 post-load",
        )
        if (
            not np.isfinite(controls[model_id]).all()
            or not np.isfinite(basal_hvg[model_id]).all()
        ):
            raise ValueError(f"{model_id}: sampled basal arrays are non-finite")
        if basal_hvg[model_id].shape[1] != 2_000:
            raise ValueError(f"{model_id}: basal HVG width is not 2000")
        z_rows.append(
            moment_pool(torch.from_numpy(controls[model_id]), moments=2).numpy()
        )
        q_sc_path = state.config.paths.q_sc_cache / f"{model_id}.npz"
        expected_q_sc_hash = str(q_sc_hashes[model_id])
        _verify_artifact_hash(
            q_sc_path, expected_q_sc_hash, label=f"{model_id} q_sc pre-load"
        )
        with np.load(q_sc_path, allow_pickle=False) as shard:
            symbols = tuple(shard["gene_symbols"].astype(str).tolist())
            if symbols != genes:
                raise ValueError(f"{model_id}: q_sc gene order mismatch")
            q_sc[model_id] = (
                np.array(shard["values"], dtype=np.float32, copy=True),
                np.array(shard["available"], dtype=bool, copy=True),
            )
        _verify_artifact_hash(
            q_sc_path, expected_q_sc_hash, label=f"{model_id} q_sc post-load"
        )
    _verify_artifact_hash(
        state.config.paths.tx1_cache / "manifest.json",
        str(tx1_report.get("manifest_sha256")),
        label="Tx1 cache manifest post-load",
    )
    _verify_artifact_hash(
        state.config.paths.q_sc_cache / "manifest.json",
        str(q_sc_report.get("manifest_sha256")),
        label="q_sc cache manifest post-load",
    )
    z_c = np.stack(z_rows).astype(np.float32)

    long = state.residual_data.targets.long
    pivot = long.pivot(index="gene_symbol", columns="model_id", values="residual")
    targets = pivot.reindex(index=genes, columns=model_ids).to_numpy(dtype=np.float32)
    label_mask = np.isfinite(targets)
    targets = np.nan_to_num(targets, nan=0.0)
    g_var_symbols = set(state.variable_genes.symbols)
    g_var_mask = np.asarray([gene in g_var_symbols for gene in genes], dtype=bool)
    target_digest = hashlib.sha256()
    target_digest.update("\n".join(genes).encode())
    target_digest.update("\n".join(model_ids).encode())
    target_digest.update(targets.tobytes())
    target_digest.update(label_mask.tobytes())
    center_digest = hashlib.sha256(
        "\n".join(state.split.supervised_train).encode()
    ).hexdigest()
    mu_values = state.residual_data.targets.gene_mean.reindex(genes).to_numpy(
        dtype=np.float64
    )
    mu_digest = hashlib.sha256()
    mu_digest.update("\n".join(genes).encode())
    mu_digest.update(mu_values.tobytes())
    return Stage2DependencyData(
        genes=genes,
        model_ids=model_ids,
        e_g=e_g,
        z_c=z_c,
        controls=controls,
        basal_hvg=basal_hvg,
        q_sc=q_sc,
        hvg_indices=hvg_indices,
        targets=targets,
        label_mask=label_mask,
        g_var_mask=g_var_mask,
        sampling=sampling,
        residual_target_sha256=target_digest.hexdigest(),
        centering_fit_model_ids_sha256=center_digest,
        mu_train_sha256=mu_digest.hexdigest(),
    )


def _online_conditions(
    data: Stage2DependencyData,
    pairs: Sequence[tuple[int, int]],
    *,
    device: torch.device | None = None,
) -> OnlineConditionBatch:
    device = device or torch.device("cpu")
    genes = tuple(data.genes[gene] for gene, _context in pairs)
    model_ids = tuple(data.model_ids[context] for _gene, context in pairs)
    q_values = np.stack(
        [data.q_sc[data.model_ids[context]][0][gene] for gene, context in pairs]
    )
    q_available = np.asarray(
        [data.q_sc[data.model_ids[context]][1][gene] for gene, context in pairs],
        dtype=bool,
    )
    indices = tuple(data.hvg_indices.get(gene) for gene in genes)
    return OnlineConditionBatch(
        controls_tx1=tuple(
            torch.from_numpy(data.controls[model]).to(device) for model in model_ids
        ),
        basal_hvg=tuple(
            torch.from_numpy(data.basal_hvg[model]).to(device) for model in model_ids
        ),
        genes=genes,
        model_ids=model_ids,
        q_sc=torch.from_numpy(np.nan_to_num(q_values, nan=0.0)).to(device),
        e_g=torch.from_numpy(np.stack([data.e_g[gene] for gene, _ in pairs])).to(
            device
        ),
        z_c=torch.from_numpy(np.stack([data.z_c[context] for _, context in pairs])).to(
            device
        ),
        q_sc_mask=torch.from_numpy(q_available).to(device),
        gene_in_hvg_panel=torch.tensor(
            [index is not None for index in indices], device=device
        ),
        own_gene_hvg_indices=indices,
        own_gene_shift_available=torch.tensor(
            [index is not None for index in indices], device=device
        ),
    )


def build_frozen_feature_store(
    state: Stage2Preflight,
    data: Stage2DependencyData,
    backbone: torch.nn.Module,
    root: Path,
    *,
    stage: str = "stage1_frozen",
    checkpoint_sha256: str | None = None,
) -> tuple[FixedSparseProjection, Mapping[str, object]]:
    """Generate one authenticated frozen-backbone shard for every cell line."""
    expected_sources = _authenticated_raw_source_sha256(
        state, data.model_ids, verify_live=True
    )
    projection = FixedSparseProjection(seed=state.config.seeds.projection)
    writer = GeneEffectFeatureStoreWriter(
        root,
        stage=stage,
        model_ids=data.model_ids,
        gene_symbols=data.genes,
        e_g=data.e_g,
        z_c=data.z_c,
        gene_embedding_source_sha256=_authenticated_target_esm2_sha256(state),
        feature_schema_sha256=FEATURE_SCHEMA.schema_hash,
        projection_sha256=projection.components_hash,
    )
    checkpoint_hash = checkpoint_sha256 or state.stage1_checkpoint.checkpoint_sha256
    device = next(backbone.parameters()).device
    backbone.eval()
    for model_id in data.model_ids:
        controls = torch.from_numpy(data.controls[model_id]).to(device)
        basal = torch.from_numpy(data.basal_hvg[model_id]).to(device)
        gene_count = len(data.genes)
        delta_proj = np.empty(
            (gene_count, projection.components.shape[0]), dtype=np.float32
        )
        summaries = np.empty((gene_count, 6), dtype=np.float32)
        hvg_panel_mask = np.empty(gene_count, dtype=bool)
        own_gene_shift_mask = np.empty(gene_count, dtype=bool)
        chunk_size = state.config.joint.response_batch_size
        with torch.no_grad():
            for start in range(0, len(data.genes), chunk_size):
                genes = data.genes[start : start + chunk_size]
                predicted = predict_bags(
                    backbone,
                    tuple(controls for _ in genes),
                    genes,
                    seed=state.config.seeds.collator,
                )
                for offset, (gene, prediction) in enumerate(
                    zip(genes, predicted, strict=True)
                ):
                    features = compute_condition_features(
                        prediction.float(),
                        basal.float(),
                        projection=projection,
                        gene_in_hvg_panel=gene in data.hvg_indices,
                        own_gene_hvg_index=data.hvg_indices.get(gene),
                        own_gene_available=gene in data.hvg_indices,
                    )
                    position = start + offset
                    delta_proj[position] = (
                        features.delta_proj.detach()
                        .to("cpu", dtype=torch.float32)
                        .numpy()
                    )
                    summaries[position] = (
                        features.s.detach().to("cpu", dtype=torch.float32).numpy()
                    )
                    hvg_panel_mask[position] = bool(features.hvg_panel_mask)
                    own_gene_shift_mask[position] = bool(features.own_gene_shift_mask)
                del features, prediction, predicted
        q_values, q_mask = data.q_sc[model_id]
        if (
            sha256_file(Path(state.source_registry.loc[model_id, "source_path"]))
            != (expected_sources[model_id])
        ):
            raise ValueError(f"raw source changed during generation for {model_id}")
        writer.write_shard(
            model_id,
            delta_proj=delta_proj,
            s=summaries,
            q_sc=np.nan_to_num(q_values, nan=0.0),
            q_sc_mask=q_mask,
            hvg_panel_mask=hvg_panel_mask,
            own_gene_shift_mask=own_gene_shift_mask,
            source_sha256=expected_sources[model_id],
            model_checkpoint_sha256=checkpoint_hash,
        )
    manifest = writer.finalize()
    _authenticated_raw_source_sha256(state, data.model_ids, verify_live=True)
    return projection, manifest


def fit_train_standardizer(
    store_root: Path,
    data: Stage2DependencyData,
    train_model_ids: Sequence[str],
) -> BlockStandardizer:
    """Fit exact streaming population statistics on train pairs only."""
    gene_positions = np.arange(len(data.genes))

    def batches():
        for model_id in train_model_ids:
            context_index = data.model_ids.index(model_id)
            with np.load(
                Path(store_root) / "shards" / f"{model_id}.npz",
                allow_pickle=False,
            ) as shard:
                yield {
                    "delta_proj": shard["delta_proj"],
                    "s": shard["s"],
                    "q_sc": np.nan_to_num(shard["q_sc"], nan=0.0),
                    "e_g": data.e_g[gene_positions],
                    "z_c": np.broadcast_to(
                        data.z_c[context_index],
                        (len(data.genes), data.z_c.shape[1]),
                    ),
                }

    return BlockStandardizer().fit_batches(batches())


def _supervision_from_index(
    data: Stage2DependencyData,
    batch_index: Any,
    *,
    device: torch.device | None = None,
) -> SupervisedMatrix:
    device = device or torch.device("cpu")
    gene_indices = tuple(row.gene_index for row in batch_index.rows)
    context_rows = tuple(row.context_indices for row in batch_index.rows)
    targets = np.asarray(
        [
            data.targets[gene, list(contexts)]
            for gene, contexts in zip(gene_indices, context_rows, strict=True)
        ],
        dtype=np.float32,
    )
    masks = np.asarray(
        [
            data.label_mask[gene, list(contexts)]
            & np.asarray(row.label_mask, dtype=bool)
            for gene, contexts, row in zip(
                gene_indices, context_rows, batch_index.rows, strict=True
            )
        ],
        dtype=bool,
    )
    return SupervisedMatrix(
        target=torch.from_numpy(targets).to(device),
        label_mask=torch.from_numpy(masks).to(device),
        g_var_mask=torch.from_numpy(data.g_var_mask[list(gene_indices)]).to(device),
        gene_symbols=tuple(data.genes[index] for index in gene_indices),
        context_model_ids_by_gene=tuple(
            tuple(data.model_ids[index] for index in contexts)
            for contexts in context_rows
        ),
        residual_target_sha256=data.residual_target_sha256,
        centering_fit_model_ids_sha256=data.centering_fit_model_ids_sha256,
    )


class GeneEffectSupervisionCache:
    """Device-resident residual supervision with immutable row identities."""

    def __init__(
        self,
        data: Stage2DependencyData,
        *,
        device: torch.device | str,
    ) -> None:
        expected = (len(data.genes), len(data.model_ids))
        if data.targets.shape != expected or data.label_mask.shape != expected:
            raise ValueError("supervision target/mask shape does not match identities")
        if data.g_var_mask.shape != (len(data.genes),):
            raise ValueError("g_var_mask shape does not match gene identities")
        target = torch.device(device)
        self.gene_symbols = tuple(data.genes)
        self.model_ids = tuple(data.model_ids)
        self.residual_target_sha256 = data.residual_target_sha256
        self.centering_fit_model_ids_sha256 = data.centering_fit_model_ids_sha256
        self._tensors = {
            "target": torch.tensor(data.targets, dtype=torch.float32, device=target),
            "label_mask": torch.tensor(
                data.label_mask, dtype=torch.bool, device=target
            ),
            "g_var_mask": torch.tensor(
                data.g_var_mask, dtype=torch.bool, device=target
            ),
        }
        self._closed = False

    def gather(self, batch_index: Any) -> SupervisedMatrix:
        """Gather one rectangular gene-major batch without moving supervision."""
        if self._closed:
            raise RuntimeError("supervision cache is closed")
        rows = tuple(batch_index.rows)
        if not rows:
            raise ValueError("supervision batch cannot be empty")
        widths = {len(row.context_indices) for row in rows}
        if len(widths) != 1:
            raise ValueError("supervision rows must have one shared context width")
        width = widths.pop()
        if width == 0 or any(len(row.label_mask) != width for row in rows):
            raise ValueError(
                "supervision row masks must match a positive context width"
            )
        device = self._tensors["target"].device
        gene_indices = tuple(int(row.gene_index) for row in rows)
        context_rows = tuple(
            tuple(int(index) for index in row.context_indices) for row in rows
        )
        genes = torch.tensor(gene_indices, dtype=torch.long, device=device)
        contexts = torch.tensor(context_rows, dtype=torch.long, device=device)
        sampling_mask = torch.tensor(
            tuple(tuple(bool(value) for value in row.label_mask) for row in rows),
            dtype=torch.bool,
            device=device,
        )
        return SupervisedMatrix(
            target=self._tensors["target"][genes[:, None], contexts],
            label_mask=(
                self._tensors["label_mask"][genes[:, None], contexts] & sampling_mask
            ),
            g_var_mask=self._tensors["g_var_mask"][genes],
            gene_symbols=tuple(self.gene_symbols[index] for index in gene_indices),
            context_model_ids_by_gene=tuple(
                tuple(self.model_ids[index] for index in context_indices)
                for context_indices in context_rows
            ),
            residual_target_sha256=self.residual_target_sha256,
            centering_fit_model_ids_sha256=self.centering_fit_model_ids_sha256,
        )

    def close(self) -> None:
        """Release all cache-owned tensor references."""
        self._tensors.clear()
        self._closed = True


def _epoch_batch_indices(
    data: Stage2DependencyData,
    model_ids: Sequence[str],
    config: Stage2Config,
    epoch: int,
    *,
    process_index: int = 0,
    num_processes: int = 1,
) -> tuple[Any, ...]:
    global_contexts = tuple(data.model_ids.index(model_id) for model_id in model_ids)
    local = build_epoch_batches(
        data.label_mask[:, global_contexts],
        data.g_var_mask,
        genes_per_batch=config.joint.genes_per_batch,
        contexts_per_gene=config.joint.contexts_per_gene,
        seed=config.seeds.train,
        epoch=epoch,
    )
    if num_processes <= 0 or not 0 <= process_index < num_processes:
        raise ValueError("invalid Accelerator process_index/num_processes")
    if num_processes > 1:
        local = shard_batches(local, rank=process_index, world_size=num_processes)
    remapped = []
    for batch in local:
        rows = tuple(
            SimpleNamespace(
                gene_index=row.gene_index,
                context_indices=tuple(
                    global_contexts[index] for index in row.context_indices
                ),
                label_mask=row.label_mask,
            )
            for row in batch.rows
        )
        remapped.append(
            SimpleNamespace(rows=rows, objective_weight=batch.objective_weight)
        )
    return tuple(remapped)


def _validation_batch_indices(
    data: Stage2DependencyData,
    validation_model_ids: Sequence[str],
    genes_per_batch: int,
) -> tuple[Any, ...]:
    contexts = tuple(
        data.model_ids.index(model_id) for model_id in validation_model_ids
    )
    batches = []
    for start in range(0, len(data.genes), genes_per_batch):
        genes = range(start, min(start + genes_per_batch, len(data.genes)))
        rows = tuple(
            SimpleNamespace(
                gene_index=gene,
                context_indices=contexts,
                label_mask=tuple(
                    bool(value) for value in data.label_mask[gene, list(contexts)]
                ),
            )
            for gene in genes
        )
        batches.append(SimpleNamespace(rows=rows, objective_weight=1.0))
    return tuple(batches)


def build_warmup_batch_factories(
    state: Stage2Preflight,
    data: Stage2DependencyData,
    cache: GeneEffectFrozenFeatureCache,
    supervision_cache: GeneEffectSupervisionCache,
    *,
    process_index: int,
    num_processes: int,
):
    """Return cache-backed warmup factories and their validation contract."""

    def precomputed_factory(model_ids: Sequence[str]):
        def factory(epoch: int):
            for index in _epoch_batch_indices(
                data,
                model_ids,
                state.config,
                epoch,
                process_index=process_index,
                num_processes=num_processes,
            ):
                pairs = tuple(
                    (row.gene_index, context)
                    for row in index.rows
                    for context in row.context_indices
                )
                batch = PrecomputedSupervisedBatch(
                    features=cache.gather(pairs),
                    supervision=supervision_cache.gather(index),
                    objective_weight=index.objective_weight,
                )
                yield batch

        return factory

    train_precomputed = precomputed_factory(state.split.supervised_train)
    validation_indices = _validation_batch_indices(
        data, state.split.val, state.config.joint.genes_per_batch
    )

    def precomputed_validation_factory():
        for index in validation_indices:
            pairs = tuple(
                (row.gene_index, context)
                for row in index.rows
                for context in row.context_indices
            )
            yield PrecomputedSupervisedBatch(
                features=cache.gather(pairs),
                supervision=supervision_cache.gather(index),
            )

    from aivc_model.geneeffect_training_loop import ResidualValidationMetric

    metric_kwargs = {
        "validation_model_ids": tuple(state.split.val),
        "split_sha256": sha256_file(state.config.paths.split),
        "gene_effect_sha256": sha256_file(state.config.paths.gene_effect),
        "mu_train_sha256": data.mu_train_sha256,
    }
    warmup_metric = ResidualValidationMetric(
        batch_factory=precomputed_validation_factory,
        batch_kind="precomputed",
        **metric_kwargs,
    )
    return train_precomputed, warmup_metric, validation_indices


def build_joint_batch_factories(
    state: Stage2Preflight,
    data: Stage2DependencyData,
    *,
    process_index: int,
    num_processes: int,
    device: torch.device | None = None,
):
    """Return online factories bound to the Accelerator rank topology."""
    if num_processes > 1:
        conditions = (
            state.config.joint.genes_per_batch * state.config.joint.contexts_per_gene
        )
        if conditions != 256 or state.config.joint.conditions_per_rank != 256:
            raise ValueError("multi-rank Stage 2 requires 256 conditions per rank")

    def online_factory(epoch: int, *, shard_for_rank: bool):
        indices = _epoch_batch_indices(
            data,
            state.split.supervised_train,
            state.config,
            epoch,
            process_index=process_index if shard_for_rank else 0,
            num_processes=num_processes if shard_for_rank else 1,
        )
        for index in indices:
            pairs = tuple(
                (row.gene_index, context)
                for row in index.rows
                for context in row.context_indices
            )
            yield OnlineSupervisedBatch(
                conditions=_online_conditions(data, pairs, device=device),
                supervision=_supervision_from_index(data, index, device=device),
                objective_weight=index.objective_weight,
            )

    def train_factory(epoch: int):
        return online_factory(epoch, shard_for_rank=num_processes > 1)

    def calibration_factory(epoch: int):
        return online_factory(epoch, shard_for_rank=False)

    validation_indices = _validation_batch_indices(
        data, state.split.val, state.config.joint.genes_per_batch
    )

    def online_validation_factory():
        for index in validation_indices:
            pairs = tuple(
                (row.gene_index, context)
                for row in index.rows
                for context in row.context_indices
            )
            yield OnlineSupervisedBatch(
                conditions=_online_conditions(data, pairs, device=device),
                supervision=_supervision_from_index(data, index, device=device),
            )

    from aivc_model.geneeffect_training_loop import ResidualValidationMetric

    joint_metric = ResidualValidationMetric(
        batch_factory=online_validation_factory,
        batch_kind="online",
        validation_model_ids=tuple(state.split.val),
        split_sha256=sha256_file(state.config.paths.split),
        gene_effect_sha256=sha256_file(state.config.paths.gene_effect),
        mu_train_sha256=data.mu_train_sha256,
    )
    return train_factory, calibration_factory, joint_metric, validation_indices


def _validation_provenance(
    data: Stage2DependencyData,
    validation_indices: Sequence[Any],
    validation_model_ids: Sequence[str],
) -> tuple[tuple[str, ...], str, str]:
    genes = tuple(
        data.genes[row.gene_index] for batch in validation_indices for row in batch.rows
    )
    target_digest = hashlib.sha256()
    target_digest.update("\n".join(genes).encode())
    target_digest.update("\n".join(validation_model_ids).encode())
    masks: list[bytes] = []
    for index in validation_indices:
        supervision = _supervision_from_index(data, index)
        target_digest.update(supervision.target.contiguous().numpy().tobytes())
        masks.append(supervision.label_mask.contiguous().numpy().tobytes())
    for value in masks:
        target_digest.update(value)
    gene_digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    return genes, target_digest.hexdigest(), gene_digest


def score_dependency_split(
    model: GeneEffectE2EModel,
    data: Stage2DependencyData,
    model_ids: Sequence[str],
    *,
    split_name: str,
    gene_batch_size: int,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Score one fixed split from the selected E2E weights."""
    from aivc_model.residual_metrics import per_gene_spearman, per_line_spearman

    rows: list[dict[str, object]] = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for model_id in model_ids:
            context = data.model_ids.index(model_id)
            for start in range(0, len(data.genes), gene_batch_size):
                gene_indices = range(
                    start, min(start + gene_batch_size, len(data.genes))
                )
                pairs = tuple((gene, context) for gene in gene_indices)
                prediction = (
                    model(_online_conditions(data, pairs, device=device))
                    .delta_hat.cpu()
                    .numpy()
                )
                for (gene, _), value in zip(pairs, prediction, strict=True):
                    if data.label_mask[gene, context]:
                        rows.append(
                            {
                                "split": split_name,
                                "model_id": model_id,
                                "gene_symbol": data.genes[gene],
                                "residual_truth": float(data.targets[gene, context]),
                                "residual_prediction": float(value),
                            }
                        )
    frame = pd.DataFrame(rows)
    per_gene = per_gene_spearman(
        frame, truth_col="residual_truth", pred_col="residual_prediction"
    )
    per_line = per_line_spearman(
        frame, truth_col="residual_truth", pred_col="residual_prediction"
    )
    return frame, {
        "macro_per_gene_spearman": float(per_gene.mean(skipna=True)),
        "macro_per_line_spearman": float(per_line.mean(skipna=True)),
        "per_gene_defined": int(per_gene.notna().sum()),
        "per_gene_undefined": int(per_gene.isna().sum()),
        "per_line_defined": int(per_line.notna().sum()),
        "per_line_undefined": int(per_line.isna().sum()),
        "per_line_spearman": {
            str(model_id): (None if pd.isna(value) else float(value))
            for model_id, value in per_line.items()
        },
    }


def run_registered_baselines(state: Stage2Preflight, data: Stage2DependencyData):
    """Run every protocol-registered baseline on the fixed Stage 2 split."""
    from aivc_model.residual_ladder import FixedSplit, run_r1_ladder

    labels = state.residual_data.targets.long[
        ["model_id", "gene_symbol", "gene_effect"]
    ]
    labels = labels.loc[labels["gene_symbol"].isin(data.genes)].copy()
    z_c = pd.DataFrame(
        data.z_c,
        index=pd.Index(data.model_ids, name="model_id"),
        columns=[f"z_{index}" for index in range(data.z_c.shape[1])],
    )
    result = run_r1_ladder(
        labels,
        {"z_c": z_c},
        state.copy_prior,
        seed=state.config.seeds.train,
        min_lines=state.config.loss.minimum_observations,
        outer="fixed",
        split=FixedSplit(
            train=tuple(state.split.supervised_train),
            val=tuple(state.split.val),
            test=tuple(state.split.test),
            unlabeled_train=tuple(state.split.unlabeled_train),
        ),
    )
    required = {
        "gene_mean",
        "copy_prior",
        "nearest_line[z_c]",
        "context_pca_ridge[z_c]",
    }
    for split_name in ("val", "test"):
        split_predictions = result.predictions.loc[
            result.predictions["slice"] == split_name
        ]
        observed = set(split_predictions["method"].astype(str))
        expected_keys = set(
            labels.loc[
                labels["model_id"].isin(getattr(state.split, split_name)),
                ["model_id", "gene_symbol"],
            ].itertuples(index=False, name=None)
        )
        if observed != required:
            raise RuntimeError(
                f"registered {split_name} baselines incomplete: "
                f"missing={sorted(required - observed)}, "
                f"unexpected={sorted(observed - required)}"
            )
        for method in sorted(required):
            method_keys = set(
                split_predictions.loc[
                    split_predictions["method"] == method,
                    ["model_id", "gene_symbol"],
                ].itertuples(index=False, name=None)
            )
            if method_keys != expected_keys:
                raise RuntimeError(
                    f"registered baseline {method!r} has incomparable "
                    f"{split_name} coverage: "
                    f"missing={len(expected_keys - method_keys)}, "
                    f"unexpected={len(method_keys - expected_keys)}"
                )
        gene_mean_summary = result.summary["slices"][split_name]["methods"]["gene_mean"]
        gene_mean_summary["evaluation_status"] = "not_evaluable_constant_prediction"
        gene_mean_summary["coverage"] = {
            "observed_rows": len(expected_keys),
            "expected_rows": len(expected_keys),
            "complete": True,
        }
    return result


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _json_metrics(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metrics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return None
        if not np.isfinite(number):
            raise ValueError(f"metric artifact contains infinite value: {number}")
        return number
    if isinstance(value, np.integer):
        return int(value)
    return value


def _response_metric_record(
    before_metrics: Mapping[str, object],
    after_model_loss: float,
    *,
    response_lineage_sha256: str,
    response_lineage_artifact_sha256: str,
) -> Mapping[str, object]:
    """Record both audits without implying a lineage-comparable improvement."""
    if not np.isfinite(after_model_loss):
        raise ValueError("current authenticated response model_loss must be finite")
    return {
        "comparison_status": ("not_comparable_historical_input_lineage_incomplete"),
        "delta_reported": False,
        "before_stage2": {
            "input_lineage_status": "historical_unverified_inputs",
            "metrics": dict(before_metrics),
        },
        "after_stage2": {
            "input_lineage_status": "current_authenticated_inputs",
            "metrics": {"model_loss": float(after_model_loss)},
            "response_lineage_sha256": response_lineage_sha256,
            "response_lineage_artifact_sha256": (response_lineage_artifact_sha256),
        },
        "hard_guard_applied": False,
    }


def _assert_complete_sentinel(layout: Stage2RunLayout, run_id: str) -> None:
    try:
        payload = json.loads(layout.complete.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("completion sentinel is missing or unreadable") from exc
    if payload.get("status") != "complete" or payload.get("run_id") != run_id:
        raise RuntimeError("completion sentinel does not match the active run")


def _atomic_write_strict_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_RESPONSE_CACHE_FILES = (
    "genes.npy",
    "manifest.json",
    "metadata.parquet",
    "offsets.npy",
    "target_cells.npy",
)


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_array_claim(value: object) -> Mapping[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    claim = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return claim


def _response_record_claim(
    row: Mapping[str, object], *, membership: str
) -> Mapping[str, object]:
    observed = np.asarray(row["observed_hvg"])
    if not np.isfinite(observed).all():
        raise ValueError(f"response record {row['record_id']!r} has non-finite targets")
    record_id = str(row["record_id"])
    gene = str(row["gene"])
    model_id = str(row["model_id"])
    if record_id != f"{gene}@{model_id}":
        raise ValueError(f"response record identity is malformed: {record_id!r}")
    anchor_weight = float(row["anchor_weight"])
    objective_weight = float(row["weight"])
    if (
        not np.isfinite(anchor_weight)
        or anchor_weight <= 0
        or not np.isfinite(objective_weight)
        or objective_weight <= 0
    ):
        raise ValueError(f"response record {record_id!r} has invalid weights")
    return {
        "record_id": record_id,
        "gene": gene,
        "model_id": model_id,
        "membership": membership,
        "anchor_weight": anchor_weight,
        "objective_weight": objective_weight,
        "control_tx1": _response_array_claim(row["control_tx1"]),
        "observed_hvg": _response_array_claim(observed),
        "observed_hvg_mask": _response_array_claim(np.isfinite(observed)),
        "control_hvg": _response_array_claim(row["control_hvg"]),
    }


def _write_response_lineage_artifact(
    layout: Stage2RunLayout,
    state: Stage2Preflight,
    response: ResponseAssembly,
) -> tuple[str, str]:
    """Pin exact response supervision and every source identity before training."""
    stage1 = load_stage1_config(state.bundle.stage1_config)
    cache_root = state.bundle.response_cache_dir / "response_targets"
    found_cache_files = tuple(
        sorted(path.name for path in cache_root.iterdir() if path.is_file())
    )
    if found_cache_files != _RESPONSE_CACHE_FILES:
        raise ValueError(
            "response-target cache membership mismatch: "
            f"expected={list(_RESPONSE_CACHE_FILES)}, found={list(found_cache_files)}"
        )
    source_paths = tuple(referenced_source_paths(state.bundle.perturbseq_sources))
    fingerprint_kwargs = {
        "cell_line_manifest_path": state.bundle.cell_line_manifest,
        "perturbseq_sources_path": state.bundle.perturbseq_sources,
        "referenced_source_paths": source_paths,
        "tx1_cache_manifest_path": state.config.paths.tx1_cache / "manifest.json",
        "checkpoint_var_dims_path": state.bundle.state_model_dir / "var_dims.pkl",
        "max_cells_per_gene": stage1.train.max_cells_per_gene,
        "total_cells_per_line": stage1.train.total_cells_per_line,
        "seed": stage1.train.data_seed,
        "genes": None,
    }
    expected_fingerprint = response_targets_fingerprint(**fingerprint_kwargs)
    cached_genes, cached_targets, _metadata = load_response_targets_cache(
        state.bundle.response_cache_dir, expected_fingerprint
    )
    all_records = (*response.train_records, *response.heldout_records)
    by_id = {str(row["record_id"]): row for row in all_records}
    if len(by_id) != len(all_records):
        raise ValueError("response record identities are not unique")
    dropped_records = [dict(row) for row in response.dropped_records]
    dropped_by_id = {str(row.get("record_id")): row for row in dropped_records}
    if len(dropped_by_id) != len(dropped_records) or set(dropped_by_id) & set(by_id):
        raise ValueError("dropped response record identities are not unique")
    for record_id, row in dropped_by_id.items():
        if (
            record_id != f"{row.get('gene')}@{row.get('model_id')}"
            or row.get("reason") != "stage1_esm2_unresolved"
        ):
            raise ValueError(f"dropped response record is malformed: {record_id!r}")
    cached_ids = tuple(str(value) for value in cached_genes.tolist())
    if len(cached_ids) != len(cached_targets) or set(cached_ids) != (
        set(by_id) | set(dropped_by_id)
    ):
        raise ValueError("response cache and assembled record membership disagree")
    for record_id, cached_target in zip(cached_ids, cached_targets, strict=True):
        if record_id in dropped_by_id:
            continue
        assembled = np.asarray(by_id[record_id]["observed_hvg"])
        if (
            assembled.dtype != np.asarray(cached_target).dtype
            or assembled.shape != np.asarray(cached_target).shape
            or not np.array_equal(assembled, cached_target, equal_nan=True)
        ):
            raise ValueError(
                f"response cache target differs from assembled record {record_id!r}"
            )

    cache_hashes_before = {
        filename: sha256_file(cache_root / filename)
        for filename in _RESPONSE_CACHE_FILES
    }
    train_claims = [
        _response_record_claim(row, membership="train")
        for row in response.train_records
    ]
    heldout_claims = [
        _response_record_claim(row, membership="heldout")
        for row in response.heldout_records
    ]
    for claims in (train_claims, heldout_claims):
        by_model: dict[str, list[Mapping[str, object]]] = {}
        for claim in claims:
            by_model.setdefault(str(claim["model_id"]), []).append(claim)
        for model_id, model_claims in by_model.items():
            anchor_weight = float(model_claims[0]["anchor_weight"])
            if any(
                float(claim["anchor_weight"]) != anchor_weight for claim in model_claims
            ) or not np.isclose(
                sum(float(claim["objective_weight"]) for claim in model_claims),
                anchor_weight,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    f"response objective weights do not preserve {model_id} anchor mass"
                )
    stage1_root = state.config.paths.stage1_checkpoint.parent.parent
    sources = {
        "cell_line_manifest_sha256": sha256_file(state.bundle.cell_line_manifest),
        "perturbseq_sources_sha256": sha256_file(state.bundle.perturbseq_sources),
        "referenced_source_sha256": {
            str(path): sha256_file(path) for path in sorted(source_paths, key=str)
        },
        "tx1_cache_manifest_sha256": sha256_file(
            state.config.paths.tx1_cache / "manifest.json"
        ),
        "state_var_dims_sha256": sha256_file(
            state.bundle.state_model_dir / "var_dims.pkl"
        ),
        "stage1_run_manifest_sha256": sha256_file(stage1_root / "run_manifest.json"),
        "stage1_heldout_metrics_sha256": sha256_file(
            stage1_root / "heldout_metrics.json"
        ),
    }
    membership_payload = [
        {"record_id": claim["record_id"], "membership": claim["membership"]}
        for claim in (*train_claims, *heldout_claims)
    ]
    target_payload = [
        {
            "record_id": claim["record_id"],
            "observed_hvg": claim["observed_hvg"],
            "observed_hvg_mask": claim["observed_hvg_mask"],
        }
        for claim in (*train_claims, *heldout_claims)
    ]
    weight_payload = [
        {
            "record_id": claim["record_id"],
            "anchor_weight": claim["anchor_weight"],
            "objective_weight": claim["objective_weight"],
        }
        for claim in (*train_claims, *heldout_claims)
    ]
    payload: dict[str, object] = {
        "schema_version": "exp13-response-lineage-v1",
        "response_cache_fingerprint": expected_fingerprint,
        "response_cache_files": cache_hashes_before,
        "source_identities": sources,
        "train_records": train_claims,
        "heldout_records": heldout_claims,
        "dropped_records": dropped_records,
        "dropped_records_sha256": _canonical_json_sha256(dropped_records),
        "record_membership_sha256": _canonical_json_sha256(membership_payload),
        "target_tensors_sha256": _canonical_json_sha256(target_payload),
        "objective_weights_sha256": _canonical_json_sha256(weight_payload),
    }
    payload["lineage_sha256"] = _canonical_json_sha256(payload)
    cache_hashes_after = {
        filename: sha256_file(cache_root / filename)
        for filename in _RESPONSE_CACHE_FILES
    }
    if cache_hashes_after != cache_hashes_before:
        raise ValueError("response-target cache changed while lineage was pinned")
    if response_targets_fingerprint(**fingerprint_kwargs) != expected_fingerprint:
        raise ValueError("response source identities changed while lineage was pinned")
    output_dir = layout.root / "response_targets"
    output_dir.mkdir()
    output_path = output_dir / "lineage.json"
    _atomic_write_strict_json(output_path, payload)
    artifact_sha256 = sha256_file(output_path)
    manifest_path = layout.root / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "status": "response_targets_pinned",
            "response_lineage_sha256": payload["lineage_sha256"],
            "response_lineage_artifact_sha256": artifact_sha256,
        }
    )
    atomic_write_json(manifest_path, run_manifest)
    return str(payload["lineage_sha256"]), artifact_sha256


def _assert_model_baseline_coverage(
    model_predictions: pd.DataFrame, baseline_predictions: pd.DataFrame
) -> None:
    key_columns = ["split", "model_id", "gene_symbol"]
    model_keys = set(model_predictions[key_columns].itertuples(index=False, name=None))
    baseline_keys = set(
        baseline_predictions[key_columns].itertuples(index=False, name=None)
    )
    if model_keys != baseline_keys:
        raise RuntimeError(
            "E2E and registered baselines have incomparable truth-mask coverage: "
            f"model_only={len(model_keys - baseline_keys)}, "
            f"baseline_only={len(baseline_keys - model_keys)}"
        )


def _pin_run_inputs(
    layout: Stage2RunLayout,
    run_id: str,
    state: Stage2Preflight,
    *,
    git_commit: str,
    distributed_runtime: Mapping[str, object],
) -> None:
    """Snapshot configured inputs and initialize the run manifest."""
    atomic_write_json(layout.root / "config_snapshot.json", state.config.snapshot())
    shutil.copy2(
        state.config.paths.split,
        layout.root / "cell_line_geneeffect_226_split.json",
    )
    shutil.copy2(
        state.config.paths.esm2_universe_manifest,
        layout.root / "esm2_gene_universe_manifest.json",
    )
    shutil.copy2(
        state.config.paths.esm2_provenance_manifest,
        layout.root / "esm2_provenance_manifest.json",
    )
    shutil.copy2(
        state.config.paths.esm2_uniprot_mapping_json,
        layout.root / "esm2_uniprot_mapping.json",
    )
    shutil.copy2(
        state.config.paths.esm2_uniprot_mapping_csv,
        layout.root / "esm2_uniprot_mapping.csv",
    )
    atomic_write_json(
        layout.root / "g_var_manifest.json", state.variable_genes.manifest
    )
    atomic_write_json(
        layout.root / "run_manifest.json",
        {
            "run_id": run_id,
            "status": "initialized",
            "git_commit": git_commit,
            "distributed_runtime": dict(distributed_runtime),
            "seeds": {
                "train": state.config.seeds.train,
                "collator": state.config.seeds.collator,
                "projection": state.config.seeds.projection,
            },
            "cells_per_context": state.config.features.cells_per_context,
            "cell_set_len": state.config.features.cell_set_len,
            "projection_seed": state.config.seeds.projection,
            "preflight": state.report,
        },
    )


def _write_residual_targets_artifact(
    layout: Stage2RunLayout,
    state: Stage2Preflight,
    data: Stage2DependencyData,
) -> str:
    """Persist the exact residual/mask contract needed for terminal recomputation."""
    mu_train = state.residual_data.targets.gene_mean.reindex(data.genes).to_numpy(
        dtype=np.float64
    )
    if mu_train.shape != (len(data.genes),) or not np.isfinite(mu_train).all():
        raise ValueError("mu_train must be finite and aligned to the scored genes")
    mu_digest = hashlib.sha256()
    mu_digest.update("\n".join(data.genes).encode())
    mu_digest.update(mu_train.tobytes())
    if mu_digest.hexdigest() != data.mu_train_sha256:
        raise ValueError("mu_train values do not match the in-memory provenance hash")
    target_digest = hashlib.sha256()
    target_digest.update("\n".join(data.genes).encode())
    target_digest.update("\n".join(data.model_ids).encode())
    target_digest.update(data.targets.tobytes())
    target_digest.update(data.label_mask.tobytes())
    if target_digest.hexdigest() != data.residual_target_sha256:
        raise ValueError("residual targets do not match the in-memory provenance hash")
    centering_model_ids = tuple(state.split.supervised_train)
    centering_digest = hashlib.sha256(
        "\n".join(centering_model_ids).encode()
    ).hexdigest()
    if centering_digest != data.centering_fit_model_ids_sha256:
        raise ValueError(
            "centering ModelIDs do not match the in-memory provenance hash"
        )

    path = layout.root / "residual_targets.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                gene_symbols=np.asarray(data.genes),
                model_ids=np.asarray(data.model_ids),
                residual_targets=np.asarray(data.targets, dtype=np.float32),
                label_mask=np.asarray(data.label_mask, dtype=bool),
                mu_train=mu_train,
                centering_model_ids=np.asarray(centering_model_ids),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(path)
    manifest_path = layout.root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "residual_targets_pinned",
            "split_sha256": PINNED_SPLIT_SHA256,
            "residual_targets_artifact_sha256": digest,
            "residual_target_sha256": data.residual_target_sha256,
            "centering_fit_model_ids_sha256": (data.centering_fit_model_ids_sha256),
            "mu_train_sha256": data.mu_train_sha256,
        }
    )
    atomic_write_json(manifest_path, manifest)
    return digest


def _finalize_selected_run(
    *,
    layout: Stage2RunLayout,
    run_id: str,
    state: Stage2Preflight,
    data: Stage2DependencyData,
    response: ResponseAssembly,
    model: GeneEffectE2EModel,
    response_loss: Any,
    device: torch.device,
    projection: FixedSparseProjection,
    standardizer: BlockStandardizer,
    feature_manifest: Mapping[str, object],
    warmup: Mapping[str, object],
    joint: Mapping[str, object],
    git_commit: str,
    distributed_runtime: Mapping[str, object],
    response_lineage_sha256: str,
    response_lineage_artifact_sha256: str,
) -> None:
    """Evaluate, package, and seal one selected model on rank zero only."""
    baseline_result = run_registered_baselines(state, data)
    selected_checkpoint = layout.joint / "training" / "best" / "e2e_state.pt"
    selected_checkpoint_sha256 = sha256_file(selected_checkpoint)
    _final_projection, final_feature_manifest = build_frozen_feature_store(
        state,
        data,
        model.backbone,
        layout.condition_features / "stage2_selected",
        stage="stage2_selected",
        checkpoint_sha256=selected_checkpoint_sha256,
    )
    _verify_feature_store_for_run(
        state,
        data,
        layout.condition_features / "stage2_selected",
        stage="stage2_selected",
        checkpoint_sha256=selected_checkpoint_sha256,
        projection=projection,
    )
    val_predictions, val_metrics = score_dependency_split(
        model,
        data,
        state.split.val,
        split_name="val",
        gene_batch_size=state.config.joint.response_batch_size,
    )
    test_predictions, test_metrics = score_dependency_split(
        model,
        data,
        state.split.test,
        split_name="test",
        gene_batch_size=state.config.joint.response_batch_size,
    )
    model_predictions = pd.concat(
        [val_predictions, test_predictions], ignore_index=True
    ).rename(columns={"residual_truth": "residual"})
    model_predictions.insert(1, "method", "e2e_full")
    label_lookup = state.residual_data.targets.long.set_index(
        ["model_id", "gene_symbol"]
    )["gene_effect"]
    model_predictions["gene_effect"] = [
        float(label_lookup.loc[(row.model_id, row.gene_symbol)])
        for row in model_predictions.itertuples()
    ]
    baseline_predictions = baseline_result.predictions.rename(
        columns={"slice": "split"}
    )
    _assert_model_baseline_coverage(model_predictions, baseline_predictions)
    prediction_columns = [
        "split",
        "method",
        "model_id",
        "gene_symbol",
        "gene_effect",
        "residual",
        "residual_prediction",
    ]
    predictions = pd.concat(
        [
            model_predictions[prediction_columns],
            baseline_predictions[prediction_columns],
        ],
        ignore_index=True,
    )
    prediction_path = layout.root / "geneeffect_residual_predictions.csv"
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    predictions.to_csv(prediction_tmp, index=False)
    prediction_tmp.replace(prediction_path)
    heldout_sum = 0.0
    heldout_weight = 0.0
    with torch.no_grad():
        for cpu_batch in response.heldout_batch_factory(0):
            batch = response_batch_to_device(cpu_batch, device)
            objective = response_objective(
                model.backbone,
                batch,
                loss_fn=response_loss,
                collator_seed=state.config.seeds.collator,
            )
            heldout_sum += float(objective.weighted_sum)
            heldout_weight += float(objective.weight_sum)
    if heldout_weight <= 0:
        raise ValueError("selected response held-out metric has zero weight")
    metrics_payload = {
        "selection_metric": state.config.selection.metric,
        "undefined_metric_encoding": {
            "json_value": None,
            "status": "not_evaluable_constant_prediction",
        },
        "validation": val_metrics,
        "test": test_metrics,
        "baselines": baseline_result.summary,
        "response": _response_metric_record(
            response.before_metrics,
            heldout_sum / heldout_weight,
            response_lineage_sha256=response_lineage_sha256,
            response_lineage_artifact_sha256=response_lineage_artifact_sha256,
        ),
    }
    _atomic_write_strict_json(
        layout.root / "geneeffect_residual_metrics.json",
        _json_metrics(metrics_payload),
    )
    np.savez(
        layout.root / "projection.npz",
        components=projection.components,
        metadata=np.asarray(json.dumps(projection.metadata, sort_keys=True)),
    )
    np.savez(
        layout.root / "standardizer.npz",
        state=np.asarray(json.dumps(standardizer.to_state(), sort_keys=True)),
    )
    atomic_write_json(layout.root / "feature_schema.json", FEATURE_SCHEMA.to_dict())
    shutil.copy2(selected_checkpoint, layout.model_package / "e2e_state.pt")
    atomic_write_json(
        layout.model_package / "model_manifest.json",
        {
            "checkpoint": "e2e_state.pt",
            "projection": "../projection.npz",
            "standardizer": "../standardizer.npz",
            "feature_schema": "../feature_schema.json",
            "frozen_features": "../condition_features/stage1_frozen",
            "selected_features": "../condition_features/stage2_selected",
            "distributed_runtime": dict(distributed_runtime),
        },
    )
    atomic_write_json(
        layout.root / "checkpoint_selection.json",
        {"warmup": dict(warmup), "joint": dict(joint)},
    )
    atomic_write_json(
        layout.root / "feature_generation.json",
        {
            "feature_manifest": feature_manifest,
            "final_feature_manifest": final_feature_manifest,
            "projection": projection.metadata,
            "standardizer": standardizer.to_state(),
            "basal_sampling": data.sampling,
        },
    )
    atomic_write_json(
        layout.root / "run_manifest.json",
        {
            "run_id": run_id,
            "status": "artifacts_written",
            "git_commit": git_commit,
            "distributed_runtime": dict(distributed_runtime),
            "seeds": {
                "train": state.config.seeds.train,
                "collator": state.config.seeds.collator,
                "projection": state.config.seeds.projection,
            },
            "cells_per_context": state.config.features.cells_per_context,
            "cell_set_len": state.config.features.cell_set_len,
            "projection_seed": state.config.seeds.projection,
            "gene_universe": {
                "count": len(data.genes),
                "symbols": list(data.genes),
            },
            "preflight": state.report,
        },
    )


def run_full_stage2(
    config_path: Path,
    *,
    run_id: str,
    reuse_frozen_feature_store: Path | None = None,
) -> Path:
    """Assemble response supervision, construct Stage 1, and enter Stage 2."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must contain only letters, digits, '.', '_' and '-', and "
            "must start with a letter or digit"
        )
    launch_config = load_stage2_config(config_path)
    accelerator = _create_accelerator(launch_config.distributed.mixed_precision)
    require_distinct_devices(accelerator)
    distributed_runtime = _formal_distributed_runtime(accelerator, launch_config)
    state = _run_all_ranks_or_raise(
        accelerator,
        "Stage 2 preflight",
        lambda: preflight_stage2(config_path),
    )
    if state.config.snapshot() != launch_config.snapshot():
        raise RuntimeError("Stage 2 config changed between launch and preflight")
    layout = Stage2RunLayout(state.config.paths.output_root / run_id)
    run_rank_zero_or_raise(
        accelerator,
        "prepare Stage 2 run directory",
        lambda: prepare_run_dir(layout.root),
    )
    accelerator.wait_for_everyone()
    phase = "run_initialization"
    try:
        git_commit = _git_commit()
        run_rank_zero_or_raise(
            accelerator,
            "pin Stage 2 run inputs",
            lambda: _pin_run_inputs(
                layout,
                run_id,
                state,
                git_commit=git_commit,
                distributed_runtime=distributed_runtime,
            ),
        )
        accelerator.wait_for_everyone()
        torch.manual_seed(state.config.seeds.train)
        np.random.seed(state.config.seeds.train)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(state.config.seeds.train)
        phase = "training_assembly"
        response = _run_all_ranks_or_raise(
            accelerator,
            "assemble response supervision",
            lambda: assemble_response_supervision(state),
        )
        run_rank_zero_or_raise(
            accelerator,
            "pin response supervision lineage",
            lambda: _write_response_lineage_artifact(layout, state, response),
        )
        accelerator.wait_for_everyone()
        response_lineage_payload = json.loads(
            (layout.root / "response_targets" / "lineage.json").read_text(
                encoding="utf-8"
            )
        )
        response_lineage_sha256 = str(response_lineage_payload["lineage_sha256"])
        response_lineage_artifact_sha256 = sha256_file(
            layout.root / "response_targets" / "lineage.json"
        )
        data = _run_all_ranks_or_raise(
            accelerator,
            "assemble dependency data",
            lambda: build_dependency_data(state),
        )
        run_rank_zero_or_raise(
            accelerator,
            "pin residual target contract",
            lambda: _write_residual_targets_artifact(layout, state, data),
        )
        accelerator.wait_for_everyone()
        supervision_cache = _run_all_ranks_or_raise(
            accelerator,
            "load Stage 2 supervision cache",
            lambda: GeneEffectSupervisionCache(data, device=accelerator.device),
        )
        phase = "backbone_assembly"
        backbone, load_report = _run_all_ranks_or_raise(
            accelerator,
            "construct Stage 2 backbone",
            lambda: construct_stage2_backbone(state, response),
        )
        _assert_configured_state_window(backbone, state.config.features.cell_set_len)
        device = accelerator.device
        backbone.to(device)
        run_rank_zero_or_raise(
            accelerator,
            "write backbone load report",
            lambda: atomic_write_json(
                layout.root / "backbone_load_report.json",
                {
                    "checkpoint_sha256": load_report.checkpoint_sha256,
                    "loaded_keys": list(load_report.loaded_keys),
                    "dropped_keys": list(load_report.dropped_keys),
                    "legacy_esm_matrix_dropped": load_report.legacy_esm_matrix_dropped,
                    "trainable": load_report.trainable,
                    "response_batch_count": response.batch_count,
                    "backbone_type": type(backbone).__name__,
                    "state_window": _state_window(backbone),
                },
            ),
        )
        phase = "frozen_feature_generation"
        frozen_store = layout.condition_features / "stage1_frozen"
        projection = FixedSparseProjection(seed=state.config.seeds.projection)
        if reuse_frozen_feature_store is None:
            _run_rank_zero_long_action(
                accelerator,
                "generate frozen Stage 1 feature store",
                layout.root / ".stage1_feature_generation_status.json",
                lambda: build_frozen_feature_store(
                    state,
                    data,
                    backbone,
                    frozen_store,
                    checkpoint_sha256=load_report.checkpoint_sha256,
                ),
            )
        else:
            _run_rank_zero_long_action(
                accelerator,
                "import frozen Stage 1 feature store",
                layout.root / ".stage1_feature_import_status.json",
                lambda: _import_frozen_feature_store(
                    state,
                    data,
                    reuse_frozen_feature_store,
                    frozen_store,
                    artifact_path=layout.root / "frozen_feature_store_import.json",
                    checkpoint_sha256=load_report.checkpoint_sha256,
                    projection=projection,
                ),
            )
        _run_rank_zero_long_action(
            accelerator,
            "verify frozen Stage 1 feature store",
            layout.root / ".stage1_feature_verification_status.json",
            lambda: _verify_feature_store_for_run(
                state,
                data,
                frozen_store,
                stage="stage1_frozen",
                checkpoint_sha256=load_report.checkpoint_sha256,
                projection=projection,
            ),
        )
        feature_manifest = json.loads(
            (frozen_store / "manifest.json").read_text(encoding="utf-8")
        )
        standardizer = fit_train_standardizer(
            frozen_store, data, state.split.supervised_train
        )
        cache_scope = (*state.split.supervised_train, *state.split.val)
        frozen_cache = _run_all_ranks_or_raise(
            accelerator,
            "load frozen feature cache",
            lambda: GeneEffectFrozenFeatureCache.load(
                frozen_store,
                selected_model_ids=cache_scope,
                expected_gene_symbols=data.genes,
                expected_model_ids=data.model_ids,
                expected_stage="stage1_frozen",
                device=device,
            ),
        )
        local_warmup_indices = _epoch_batch_indices(
            data,
            state.split.supervised_train,
            state.config,
            0,
            process_index=int(accelerator.process_index),
            num_processes=int(accelerator.num_processes),
        )
        warmup_runtime = {
            "world_size": int(accelerator.num_processes),
            "conditions_per_rank": state.config.joint.conditions_per_rank,
            "global_conditions_per_step": (
                int(accelerator.num_processes) * state.config.joint.conditions_per_rank
            ),
            "optimizer_steps_per_epoch": len(local_warmup_indices),
        }
        head = GeneEffectResidualHead(
            hidden=state.config.warmup.hidden_dim,
            n_hidden_layers=state.config.warmup.num_layers,
        )
        model = GeneEffectE2EModel(
            backbone,
            head,
            projection,
            standardizer,
            collator_seed=state.config.seeds.collator,
        ).to(device)
        train_precomputed, warmup_metric, validation_indices = (
            build_warmup_batch_factories(
                state,
                data,
                frozen_cache,
                supervision_cache,
                process_index=int(accelerator.process_index),
                num_processes=int(accelerator.num_processes),
            )
        )
        from aivc_model.geneeffect_head import masked_geneeffect_residual_loss
        from aivc_model.geneeffect_training import (
            calibrate_lambda_dep,
            response_objective,
            warmup_step,
        )
        from aivc_model.geneeffect_training_loop import (
            CheckpointProvenance,
            TrainingProgressWriter,
            train_frozen_warmup,
            train_joint,
        )
        from aivc_model.response_training import ResponseLoss, ResponseLossWeights

        (
            validation_genes,
            validation_target_sha256,
            validation_gene_digest,
        ) = _validation_provenance(data, validation_indices, state.split.val)
        provenance = CheckpointProvenance(
            distributed_runtime=distributed_runtime,
            warmup_runtime=warmup_runtime,
            lambda_calibration_report=None,
            split_sha256=sha256_file(state.config.paths.split),
            gene_effect_sha256=sha256_file(state.config.paths.gene_effect),
            mu_train_sha256=data.mu_train_sha256,
            residual_target_sha256=data.residual_target_sha256,
            validation_target_sha256=validation_target_sha256,
            centering_fit_model_ids_sha256=data.centering_fit_model_ids_sha256,
            validation_model_ids=tuple(state.split.val),
            validation_gene_symbols=validation_genes,
            validation_gene_count=len(validation_genes),
            validation_gene_universe_sha256=validation_gene_digest,
        )
        model.freeze_backbone()
        phase = "frozen_head_warmup"
        progress = TrainingProgressWriter(
            layout.root / ".warmup_progress.json",
            {
                "world_size": warmup_runtime["world_size"],
                "conditions_per_rank": warmup_runtime["conditions_per_rank"],
                "global_conditions_per_step": warmup_runtime[
                    "global_conditions_per_step"
                ],
                "optimizer_steps_per_epoch": warmup_runtime[
                    "optimizer_steps_per_epoch"
                ],
            },
        )
        forward_head = None
        training_accelerator = None
        if accelerator.num_processes > 1:
            forward_head = accelerator.prepare(model.head)
            training_accelerator = accelerator

        def run_warmup() -> dict[str, object]:
            def bf16_warmup_step(*args, **kwargs):
                with accelerator.autocast():
                    return warmup_step(*args, **kwargs)

            return train_frozen_warmup(
                model,
                train_precomputed,
                warmup_metric,
                layout.warmup / "training",
                state.config,
                provenance,
                accelerator=training_accelerator,
                forward_head=forward_head,
                progress=progress,
                step_fn=bf16_warmup_step,
            )

        checkpoint_selection_warmup = _run_all_ranks_or_raise(
            accelerator,
            "train frozen GeneEffect head",
            run_warmup,
        )
        accelerator.wait_for_everyone()
        frozen_cache.close()
        supervision_cache.close()
        frozen_cache = None
        supervision_cache = None
        train_precomputed = None
        warmup_metric = None
        forward_head = None
        progress = None
        accelerator.free_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        online_factory, calibration_factory, joint_metric, _ = (
            build_joint_batch_factories(
                state,
                data,
                process_index=int(accelerator.process_index),
                num_processes=int(accelerator.num_processes),
                device=device,
            )
        )
        model.unfreeze_backbone()
        model.train()
        _assert_joint_calibration_ready(model)
        stage1_config = load_stage1_config(state.bundle.stage1_config)
        response_loss = ResponseLoss(
            ResponseLossWeights(
                mean_delta=stage1_config.train.w_mean_delta,
                energy=stage1_config.train.w_energy,
            )
        )

        def run_calibration() -> None:
            _assert_joint_calibration_ready(model)
            calibration_pairs = []
            dependency_batches = iter(calibration_factory(0))
            response_batches = iter(response.batch_factory(0))
            for _ in range(state.config.lambda_calibration.train_batches):
                dependency = next(dependency_batches)
                try:
                    response_batch = next(response_batches)
                except StopIteration:
                    response_batches = iter(response.batch_factory(0))
                    try:
                        response_batch = next(response_batches)
                    except StopIteration as exc:
                        raise ValueError(
                            "response batch factory produced no calibration batches"
                        ) from exc
                response_batch = response_batch_to_device(response_batch, device)

                def response_closure(batch=response_batch):
                    return response_objective(
                        model.backbone,
                        batch,
                        loss_fn=response_loss,
                        collator_seed=state.config.seeds.collator,
                    ).mean

                def dependency_closure(batch=dependency):
                    prediction = model(batch.conditions).delta_hat.reshape(
                        batch.supervision.shape
                    )
                    return masked_geneeffect_residual_loss(
                        prediction,
                        batch.supervision.target,
                        batch.supervision.label_mask,
                        batch.supervision.g_var_mask,
                        huber_delta=state.config.loss.huber_delta,
                        beta=state.config.loss.beta,
                    ).total

                calibration_pairs.append(
                    (
                        _checked_calibration_closure(
                            model, response_closure, accelerator
                        ),
                        _checked_calibration_closure(
                            model, dependency_closure, accelerator
                        ),
                    )
                )
            calibration = calibrate_lambda_dep(
                calibration_pairs,
                tuple(model.backbone.parameters()),
                clip_min=state.config.lambda_calibration.clip_min,
                clip_max=state.config.lambda_calibration.clip_max,
            )
            atomic_write_json(
                layout.root / "lambda_calibration.json",
                {
                    "lambda_dep": calibration.lambda_dep,
                    "raw_ratios": list(calibration.raw_ratios),
                    "response_gradient_norms": list(
                        calibration.response_gradient_norms
                    ),
                    "dependency_gradient_norms": list(
                        calibration.dependency_gradient_norms
                    ),
                },
            )

        phase = "lambda_calibration"
        _run_rank_zero_long_action(
            accelerator,
            "calibrate dependency loss weight",
            layout.root / ".lambda_calibration_status.json",
            run_calibration,
        )
        calibration_payload = json.loads(
            (layout.root / "lambda_calibration.json").read_text(encoding="utf-8")
        )
        joint_provenance = CheckpointProvenance(
            distributed_runtime=provenance.distributed_runtime,
            warmup_runtime=provenance.warmup_runtime,
            lambda_calibration_report=calibration_payload,
            split_sha256=provenance.split_sha256,
            gene_effect_sha256=provenance.gene_effect_sha256,
            mu_train_sha256=provenance.mu_train_sha256,
            residual_target_sha256=provenance.residual_target_sha256,
            validation_target_sha256=provenance.validation_target_sha256,
            centering_fit_model_ids_sha256=(provenance.centering_fit_model_ids_sha256),
            validation_model_ids=provenance.validation_model_ids,
            validation_gene_symbols=provenance.validation_gene_symbols,
            validation_gene_count=provenance.validation_gene_count,
            validation_gene_universe_sha256=(
                provenance.validation_gene_universe_sha256
            ),
        )

        def response_factory(epoch: int):
            for batch in response.batch_factory(epoch):
                yield response_batch_to_device(batch, device)

        phase = "joint_training"
        forward_model = None
        training_accelerator = None
        if accelerator.num_processes > 1:
            forward_model = accelerator.prepare(model)
            training_accelerator = accelerator
        joint = _run_all_ranks_or_raise(
            accelerator,
            "joint Stage 2 training",
            lambda: train_joint(
                model,
                online_factory,
                response_factory,
                joint_metric,
                layout.joint / "training",
                state.config,
                joint_provenance,
                response_loss_fn=response_loss,
                lambda_dep=float(calibration_payload["lambda_dep"]),
                accelerator=training_accelerator,
                forward_model=forward_model,
            ),
        )
        accelerator.wait_for_everyone()
        phase = "selected_checkpoint_evaluation_and_packaging"
        _run_rank_zero_long_action(
            accelerator,
            "evaluate and package selected Stage 2 model",
            layout.root / ".finalization_status.json",
            lambda: _finalize_selected_run(
                layout=layout,
                run_id=run_id,
                state=state,
                data=data,
                response=response,
                model=model,
                response_loss=response_loss,
                device=device,
                projection=projection,
                standardizer=standardizer,
                feature_manifest=feature_manifest,
                warmup=checkpoint_selection_warmup,
                joint=joint,
                git_commit=git_commit,
                distributed_runtime=distributed_runtime,
                response_lineage_sha256=response_lineage_sha256,
                response_lineage_artifact_sha256=(response_lineage_artifact_sha256),
            ),
        )
        phase = "completion_seal"
        _run_rank_zero_long_action(
            accelerator,
            "seal completed Stage 2 run",
            layout.root / ".completion_seal_status.json",
            lambda: mark_complete(layout, run_id=run_id),
        )
        _run_all_ranks_or_raise(
            accelerator,
            "verify completion sentinel",
            lambda: _assert_complete_sentinel(layout, run_id),
        )
        accelerator.wait_for_everyone()
        return layout.root
    except BaseException as exc:
        if accelerator.is_main_process and not layout.complete.exists():
            if not layout.failure.exists():
                mark_failure(layout, exc, phase=phase)
        raise


__all__ = [
    "Stage1CheckpointSpec",
    "Stage2BundleSpec",
    "Stage2Preflight",
    "ResponseAssembly",
    "assemble_response_supervision",
    "construct_stage2_backbone",
    "load_stage1_checkpoint_spec",
    "load_stage2_bundle_spec",
    "preflight_stage2",
    "run_full_stage2",
]
