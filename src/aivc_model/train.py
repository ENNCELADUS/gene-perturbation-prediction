"""Train the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__ is None or __package__ == "":
    src_root = str(Path(__file__).resolve().parents[1])
    if src_root in sys.path:
        sys.path.remove(src_root)
    sys.path.insert(0, src_root)

from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import broadcast_object_list, set_seed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aivc_model.model import (
    AivcModel,
    Esm2PerturbationAdapter,
    FixedGMMFeatureizer,
    LossWeights,
    MLPHead,
    PerturbationVectorAdapter,
    StateForwardAdapter,
    fit_fixed_gmm,
    load_state_model,
)
from aivc_model.distributed import (
    assert_all_ranks_stepped,
    require_exact_world_size,
    run_rank_zero_or_raise,
)
from aivc_model.gene_splits import (
    CANONICAL_GENE_COUNT,
    FoldSpec,
    load_canonical_outer_manifest,
)
from aivc_model.prepare import (
    AivcConfig,
    ArtifactAuthority,
    ExternalGeneBags,
    GeneBags,
    GeneSplit,
    ResponseEncoderConfig,
    SealedGeneBags,
    encode_batch_labels,
    fit_linear_projector,
    load_config,
    load_external_gene_bags,
    load_gene_bags,
    load_perturbation_vectors,
    load_state_batch_lookup,
    make_cell_set_chunks,
    make_gene_split,
    project_gene_bags_with_frozen_scvi,
    with_cached_scvi_teacher_latents,
    _sha256_strings,
)
from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM
from dependency_baseline.metrics import ranking_metrics, regression_metrics
from sl_dl_model.gene_embeddings import (
    load_esm2_embeddings,
    require_complete_esm_coverage,
)

_LOGGER = logging.getLogger(__name__)
_SCVI_CACHE_WAIT_TIMEOUT_SECONDS = 24 * 60 * 60
_SCVI_CACHE_POLL_SECONDS = 30.0
_METRIC_KEYS = (
    "total_loss",
    "hvg_mean_delta",
    "hvg_energy",
    "latent_mean_delta",
    "latent_energy",
    "pred_c",
    "obs_c",
    "occupancy",
    "pred_rank",
    "control_fallback_count",
    "n_chunks",
)
_MODEL_PER_GENE_KEYS = {
    "total_loss": "per_gene_total_loss",
    "hvg_mean_delta": "per_gene_hvg_mean_delta",
    "hvg_energy": "per_gene_hvg_energy",
    "latent_mean_delta": "per_gene_latent_mean_delta",
    "latent_energy": "per_gene_latent_energy",
    "pred_c": "per_gene_pred_c",
    "obs_c": "per_gene_obs_c",
    "occupancy": "per_gene_occupancy",
    "pred_rank": "per_gene_pred_rank",
}
_PREDICTION_COLUMNS = [
    "perturbation_gene",
    "y_true",
    "y_pred",
    "y_obs_anchor",
    "control_fallback_count",
    "n_chunks",
]
_FINAL_PREDICTION_COLUMNS = [
    "perturbation_gene",
    "y_true",
    "y_pred",
    "n_chunks",
]
_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class PredictionRequest:
    """A final label-prediction scope and its optional observed response source."""

    genes: tuple[str, ...]
    observed_response: GeneBags | None


@dataclass(frozen=True)
class ObservedBOracleFit:
    """Frozen observed-response-to-label oracle selected on inner validation."""

    model: torch.nn.Linear
    best_epoch: int
    checkpoint_sha256: str
    val_mse: float


@dataclass(frozen=True)
class _InputTensorCache:
    control_input: torch.Tensor
    input_bags: tuple[torch.Tensor, ...]
    latent_bags: tuple[torch.Tensor, ...]
    batch_bags: tuple[torch.Tensor | None, ...]
    control_batch: torch.Tensor | None
    estimated_bytes: int

    @classmethod
    def estimate_bytes(cls, data: GeneBags, batch_lookup: dict[str, int]) -> int:
        del batch_lookup
        total = _float32_array_bytes(data.control_input)
        total += sum(_float32_array_bytes(bag) for bag in data.input_bags)
        total += sum(_float32_array_bytes(bag) for bag in data.latent_bags)
        total += _label_array_bytes(data.control_batch)
        if data.batch_bags is not None:
            total += sum(_label_array_bytes(labels) for labels in data.batch_bags)
        return int(total)

    @classmethod
    def build(
        cls,
        data: GeneBags,
        batch_lookup: dict[str, int],
        device: torch.device,
    ) -> _InputTensorCache:
        batch_bags: tuple[torch.Tensor | None, ...]
        if data.batch_bags is None:
            batch_bags = tuple(None for _ in data.input_bags)
        else:
            batch_bags = tuple(
                _batch_tensor(labels, batch_lookup, device)
                for labels in data.batch_bags
            )
        return cls(
            control_input=torch.as_tensor(
                np.asarray(data.control_input, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            input_bags=tuple(
                torch.as_tensor(
                    np.asarray(bag, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                for bag in data.input_bags
            ),
            latent_bags=tuple(
                torch.as_tensor(
                    np.asarray(bag, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                for bag in data.latent_bags
            ),
            batch_bags=batch_bags,
            control_batch=_batch_tensor(data.control_batch, batch_lookup, device),
            estimated_bytes=cls.estimate_bytes(data, batch_lookup),
        )

    @classmethod
    def maybe_build(
        cls,
        data: GeneBags,
        batch_lookup: dict[str, int],
        device: torch.device,
        max_bytes: int,
        *,
        allow_cpu: bool = False,
    ) -> _InputTensorCache | None:
        if max_bytes <= 0 or (device.type != "cuda" and not allow_cpu):
            return None
        estimated_bytes = cls.estimate_bytes(data, batch_lookup)
        if estimated_bytes > max_bytes:
            return None
        return cls.build(data, batch_lookup, device)


def _float32_array_bytes(array: np.ndarray) -> int:
    return int(np.size(array)) * int(np.dtype(np.float32).itemsize)


def _label_array_bytes(labels: np.ndarray | None) -> int:
    if labels is None:
        return 0
    return int(np.size(labels)) * int(np.dtype(np.int64).itemsize)


def _gib_to_bytes(value: float) -> int:
    return int(float(value) * _BYTES_PER_GIB)


def _log_input_tensor_cache_state(
    cache: _InputTensorCache | None,
    estimated_bytes: int,
    max_bytes: int,
    device: torch.device,
) -> None:
    estimated_gib = estimated_bytes / _BYTES_PER_GIB
    max_gib = max_bytes / _BYTES_PER_GIB
    if cache is None:
        reason = "device is not cuda" if device.type != "cuda" else "cap exceeded"
        if max_bytes <= 0:
            reason = "cap is non-positive"
        _LOGGER.info(
            "AIVC input tensor cache disabled (%s; estimated %.2f GiB, cap %.2f GiB)",
            reason,
            estimated_gib,
            max_gib,
        )
        return
    _LOGGER.info(
        "AIVC input tensor cache enabled on %s (estimated %.2f GiB, cap %.2f GiB)",
        device,
        estimated_gib,
        max_gib,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STATE-ready AIVC A->B->C.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--materialize-scvi-cache-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--artifacts-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.materialize_scvi_cache_only:
        if args.artifacts_dir is None:
            parser.error(
                "--artifacts-dir is required with --materialize-scvi-cache-only"
            )
        _configure_cache_only_logging()
        _materialize_scvi_cache_only(config, args.artifacts_dir)
        return
    accelerator = _make_accelerator(config)
    _configure_logging(accelerator)
    paths = run_training(config, accelerator=accelerator, config_path=args.config)
    if accelerator.is_main_process:
        print(f"run dir: {paths['run_dir']}")
        print(f"train log: {paths['train_log']}")
        print(f"test metrics: {paths['test_metrics']}")


def run_training(
    config: AivcConfig,
    accelerator: Accelerator | None = None,
    config_path: Path | None = None,
    *,
    train_data: GeneBags | None = None,
    val_data: GeneBags | None = None,
    sealed_test: SealedGeneBags | None = None,
    external_override: ExternalGeneBags | None = None,
    fold_spec: FoldSpec | None = None,
    run_dir_override: Path | None = None,
    source_fingerprint: str | None = None,
    canonical_gene_order: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Run one train/val/test STATE-ready AIVC experiment."""
    audited_values = (
        train_data,
        val_data,
        sealed_test,
        fold_spec,
        run_dir_override,
        source_fingerprint,
        canonical_gene_order,
    )
    audited = any(value is not None for value in audited_values)
    if audited:
        if any(value is None for value in audited_values):
            raise ValueError("audited training requires every fold-scoped input")
    _require_authoritative_gene_batch_size(config)
    accelerator = accelerator or _make_accelerator(config)
    require_exact_world_size(accelerator, config.train.required_world_size)
    if audited:
        return _run_audited_training(
            config=config,
            train_data=train_data,
            val_data=val_data,
            sealed_test=sealed_test,
            external=external_override,
            fold_spec=fold_spec,
            run_dir=run_dir_override,
            source_fingerprint=source_fingerprint,
            canonical_gene_order=canonical_gene_order,
            accelerator=accelerator,
        )
    _configure_logging(accelerator)
    set_seed(config.train.seed)
    _configure_float32_matmul_precision(config)
    run_id = _resolve_run_id(config, accelerator)
    run_dir = config.data.output_dir / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    models_dir = run_dir / "models"
    if accelerator.is_main_process:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    data = load_gene_bags(config)
    split = make_gene_split(data.genes, data.y, config.split)
    external = load_external_gene_bags(
        config,
        data,
        artifacts_dir,
        project_scvi=config.projector.teacher != "scvi",
    )
    data, external = _with_rank_safe_scvi_teacher(
        config,
        data,
        split,
        external,
        artifacts_dir,
        accelerator,
        config_path=config_path,
    )

    projector_fit: tuple[np.ndarray, np.ndarray] | None = None
    featureizer: FixedGMMFeatureizer | None = None
    if accelerator.is_main_process:
        projector_fit = _fit_or_load_projector_cache(
            config,
            data,
            split,
            artifacts_dir,
        )
        featureizer = _fit_or_load_fixed_gmm_cache(
            config,
            data,
            split,
            artifacts_dir,
        )
    accelerator.wait_for_everyone()
    if projector_fit is None:
        projector_fit = _fit_or_load_projector_cache(
            config,
            data,
            split,
            artifacts_dir,
        )
    if featureizer is None:
        featureizer = _fit_or_load_fixed_gmm_cache(
            config,
            data,
            split,
            artifacts_dir,
        )
    projector_weight, projector_bias = projector_fit
    extra_genes = (
        tuple(str(gene) for gene in external.data.genes) if external is not None else ()
    )
    model = _build_model(
        config,
        data,
        featureizer,
        projector_weight,
        projector_bias,
        extra_genes=extra_genes,
        emit_checkpoint_output=accelerator.is_main_process,
    )
    batch_lookup = load_state_batch_lookup(config.state.model_dir)
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    evaluation_sets = {
        "internal_outer_test": PredictionRequest(
            genes=tuple(str(data.genes[index]) for index in split.test),
            observed_response=None,
        ),
    }
    if external is not None and config.external_test is not None:
        evaluation_sets[f"external:{config.external_test.name}"] = PredictionRequest(
            genes=tuple(str(gene) for gene in external.data.genes),
            observed_response=external.data,
        )
    train_loader = _gene_loader(
        split.train,
        shuffle=True,
        seed=config.train.seed,
        gene_batch_size=config.train.gene_batch_size,
        world_size=accelerator.num_processes,
    )
    val_loader = _gene_loader(
        split.val,
        shuffle=False,
        seed=config.train.seed,
        gene_batch_size=config.train.gene_batch_size,
        world_size=accelerator.num_processes,
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )
    evaluation_loaders = {
        scope: accelerator.prepare(
            _gene_loader(
                _prediction_indices(data, request),
                shuffle=False,
                seed=config.train.seed,
                gene_batch_size=config.train.gene_batch_size,
                world_size=accelerator.num_processes,
            )
        )
        for scope, request in evaluation_sets.items()
    }
    input_tensor_cache = _InputTensorCache.maybe_build(
        data,
        batch_lookup,
        accelerator.device,
        _gib_to_bytes(config.train.input_tensor_cache_max_gib),
    )
    if accelerator.is_main_process:
        _log_input_tensor_cache_state(
            input_tensor_cache,
            _InputTensorCache.estimate_bytes(data, batch_lookup),
            _gib_to_bytes(config.train.input_tensor_cache_max_gib),
            accelerator.device,
        )
    rng = np.random.default_rng(config.train.seed + accelerator.process_index)

    logs: list[dict[str, float]] = []
    best_val_spearman = -math.inf
    last_val_spearman = math.nan
    best_checkpoint_written = False
    for epoch in range(1, config.train.max_epochs + 1):
        weights = _loss_weights(config, epoch=epoch)
        _reset_peak_gpu_memory(accelerator.device)
        train_row = _run_epoch(
            model,
            data,
            train_loader,
            weights,
            optimizer,
            rng,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
            epoch=epoch,
            max_epochs=config.train.max_epochs,
            max_grad_norm=config.train.max_grad_norm,
            tensor_cache=input_tensor_cache,
        )
        train_row.pop("local_optimizer_steps")
        val_row, _val_predictions = _evaluate_prediction_only_final(
            model,
            data,
            val_loader,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
        )
        gpu_peak_memory_allocated_mb = _global_peak_gpu_memory_allocated_mb(accelerator)
        if accelerator.is_main_process:
            last_val_spearman = float(val_row.get("spearman", math.nan))
            row = {
                "epoch": epoch,
                "gpu_peak_memory_allocated_mb": gpu_peak_memory_allocated_mb,
                **_prefix(train_row, "train"),
                **_prefix(val_row, "val"),
            }
            logs.append(row)
            _write_csv_if_main(
                pd.DataFrame(logs),
                run_dir / "train_log.csv",
                accelerator,
            )
            should_save_best = (
                _is_better_metric(
                    last_val_spearman,
                    best_val_spearman,
                    mode="max",
                )
                or not best_checkpoint_written
            )
            if should_save_best:
                best_val_spearman = last_val_spearman
                best_checkpoint_written = True
                _save_model_checkpoint(
                    accelerator,
                    model,
                    models_dir / "best",
                    {
                        "checkpoint_kind": "best",
                        "epoch": epoch,
                        "selection_metric": "val_spearman",
                        "selection_mode": "max",
                        "metric_value": best_val_spearman,
                        "run_id": run_id,
                        "train_log": str(run_dir / "train_log.csv"),
                    },
                )
        accelerator.wait_for_everyone()

    test_rows = []
    prediction_frames = []
    for evaluation_scope, request in evaluation_sets.items():
        evaluation_data = request.observed_response or data
        test_row, scoped_predictions = _evaluate_prediction_only_final(
            model,
            evaluation_data,
            evaluation_loaders[evaluation_scope],
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
        )
        if accelerator.is_main_process:
            test_rows.append({"evaluation_scope": evaluation_scope, **test_row})
            scoped_predictions.insert(0, "evaluation_scope", evaluation_scope)
            if request.observed_response is not None:
                scoped_predictions = scoped_predictions.merge(
                    request.observed_response.metadata,
                    on="perturbation_gene",
                    how="left",
                    suffixes=("", "_metadata"),
                )
            prediction_frames.append(scoped_predictions)
    if accelerator.is_main_process:
        test_predictions = pd.concat(prediction_frames, ignore_index=True)
        unwrapped = accelerator.unwrap_model(model)
        test_predictions["perturbation_has_known_vector"] = test_predictions[
            "perturbation_gene"
        ].map(unwrapped.perturbations.has_known_vector)
        if external is not None:
            (artifacts_dir / "external_test_qa.json").write_text(
                json.dumps(external.qa, indent=2),
                encoding="utf-8",
            )
        _write_csv_if_main(
            pd.DataFrame(test_rows),
            run_dir / "test_metrics.csv",
            accelerator,
        )
        _write_csv_if_main(
            test_predictions,
            artifacts_dir / "test_predictions.csv",
            accelerator,
        )
        _write_split_artifact(data, split, artifacts_dir / "gene_splits.csv")
        _save_model_checkpoint(
            accelerator,
            model,
            models_dir / "final",
            {
                "checkpoint_kind": "final",
                "epoch": config.train.max_epochs,
                "selection_metric": "val_spearman",
                "selection_mode": "max",
                "metric_value": last_val_spearman,
                "best_metric_value": best_val_spearman,
                "run_id": run_id,
                "train_log": str(run_dir / "train_log.csv"),
                "test_metrics": str(run_dir / "test_metrics.csv"),
            },
        )
    accelerator.wait_for_everyone()
    return {
        "run_dir": run_dir,
        "train_log": run_dir / "train_log.csv",
        "test_metrics": run_dir / "test_metrics.csv",
    }


def _run_audited_training(
    config: AivcConfig,
    train_data: GeneBags,
    val_data: GeneBags,
    sealed_test: SealedGeneBags,
    external: ExternalGeneBags | None,
    fold_spec: FoldSpec,
    run_dir: Path,
    source_fingerprint: str,
    canonical_gene_order: tuple[str, ...],
    accelerator: Accelerator | None,
) -> dict[str, Path]:
    """Train one fold without exposing outer-test responses to fit code."""
    accelerator = accelerator or _make_accelerator(config)
    _configure_logging(accelerator)
    set_seed(config.train.seed + fold_spec.outer_fold)
    _configure_float32_matmul_precision(config)
    artifacts_dir = run_dir / "artifacts"
    models_dir = run_dir / "models"
    run_rank_zero_or_raise(
        accelerator,
        f"fold {fold_spec.outer_fold} directory creation",
        lambda: (
            artifacts_dir.mkdir(parents=True, exist_ok=True),
            models_dir.mkdir(parents=True, exist_ok=True),
        ),
    )
    authority = _fold_artifact_authority(
        config,
        fold_spec,
        source_fingerprint=source_fingerprint,
        canonical_split_sha256=_canonical_split_sha256(config),
    )

    external_genes = (
        tuple(str(gene) for gene in external.data.genes) if external is not None else ()
    )
    if train_data.access_recorder is None:
        raise ValueError("audited training lost its gene access recorder")
    model = _build_e2e_model(
        config,
        train_data,
        extra_genes=(*fold_spec.val_genes, *fold_spec.test_genes, *external_genes),
        canonical_gene_order=canonical_gene_order,
        emit_checkpoint_output=accelerator.is_main_process,
    )
    runtime_evidence = _runtime_evidence(
        model,
        train_data,
        canonical_gene_order,
    )
    batch_lookup = load_state_batch_lookup(config.state.model_dir)
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(model, config),
        weight_decay=config.train.weight_decay,
    )
    train_loader = _gene_loader(
        np.arange(len(train_data.genes), dtype=np.int64),
        shuffle=True,
        seed=config.train.seed + fold_spec.outer_fold,
        gene_batch_size=config.train.gene_batch_size,
        world_size=accelerator.num_processes,
    )
    val_loader = _gene_loader(
        np.arange(len(val_data.genes), dtype=np.int64),
        shuffle=False,
        seed=config.train.seed + fold_spec.outer_fold,
        gene_batch_size=config.train.gene_batch_size,
        world_size=accelerator.num_processes,
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )
    rng = np.random.default_rng(
        config.train.seed + fold_spec.outer_fold + accelerator.process_index
    )
    best_value = -math.inf
    best_epoch = 0
    logs = []
    for epoch in range(1, config.train.max_epochs + 1):
        fit_stages = (
            "adapter_fit",
            "state_fit",
            "response_encoder_fit",
            "gmm_fit",
            "c_head_fit",
            "gene_prompt_fit",
            "transition_supervision",
        )
        for stage in fit_stages:
            _authorize_data_access(train_data, stage)
        train_row = _run_epoch(
            model,
            train_data,
            train_loader,
            _loss_weights(config, epoch),
            optimizer,
            rng,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
            epoch=epoch,
            max_epochs=config.train.max_epochs,
            max_grad_norm=config.train.max_grad_norm,
        )
        for stage in fit_stages:
            train_data.record_access(stage)
        val_row, _ = _evaluate_prediction_only_final(
            model,
            val_data,
            val_loader,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
        )
        local_optimizer_steps = int(train_row.pop("local_optimizer_steps"))
        rank_optimizer_steps = assert_all_ranks_stepped(
            accelerator,
            local_optimizer_steps,
        )

        def write_epoch_outputs() -> None:
            nonlocal best_epoch, best_value
            value = float(val_row.get("spearman", math.nan))
            logs.append(
                {
                    "epoch": epoch,
                    "rank_optimizer_steps": rank_optimizer_steps,
                    **_prefix(train_row, "train"),
                    **_prefix(val_row, "val"),
                }
            )
            pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
            if _is_better_metric(value, best_value, mode="max") or best_epoch == 0:
                best_value = value
                best_epoch = epoch
                _save_model_checkpoint(
                    accelerator,
                    model,
                    models_dir / "best",
                    {
                        **authority.metadata(),
                        "checkpoint_kind": "best",
                        "epoch": epoch,
                        "selection_metric": "val_spearman",
                        "metric_value": value,
                        "outer_fold": fold_spec.outer_fold,
                        "selected_layer": None,
                        "best_epoch": epoch,
                        "state_checkpoint": str(config.state.checkpoint_path),
                        "esm2_npz": str(config.state.esm2_npz),
                    },
                )

        run_rank_zero_or_raise(
            accelerator,
            f"fold {fold_spec.outer_fold} epoch {epoch} output write",
            write_epoch_outputs,
        )

    checkpoint_dir = models_dir / "best"
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    state_dict = _load_authorized_model_checkpoint(
        checkpoint_dir,
        authority,
        map_location=accelerator.device,
    )
    selected_model = accelerator.unwrap_model(model)
    selected_model.load_state_dict(state_dict)
    selected_model.eval()
    selected_model.requires_grad_(False)
    checkpoint_frozen = True

    label_test = sealed_test.label_view(checkpoint_frozen=True)
    label_loader = accelerator.prepare(
        _gene_loader(
            np.arange(len(label_test.genes), dtype=np.int64),
            shuffle=False,
            seed=config.train.seed,
            gene_batch_size=config.train.gene_batch_size,
            world_size=accelerator.num_processes,
        )
    )
    internal_metrics, internal_predictions = _evaluate_prediction_only_final(
        model,
        label_test,
        label_loader,
        config.train.cell_set_len,
        accelerator,
        batch_lookup,
    )
    generation_data = sealed_test.open(
        "generation_quality_outer_test",
        checkpoint_frozen=checkpoint_frozen,
    )
    generation_loader = accelerator.prepare(
        _gene_loader(
            np.arange(len(generation_data.genes), dtype=np.int64),
            shuffle=False,
            seed=config.train.seed,
            gene_batch_size=config.train.gene_batch_size,
            world_size=accelerator.num_processes,
        )
    )
    response_metrics, response_predictions = _evaluate(
        model,
        generation_data,
        generation_loader,
        _loss_weights(config),
        rng,
        config.train.cell_set_len,
        accelerator,
        batch_lookup,
        pad_short=False,
    )
    oracle_data = sealed_test.open(
        "observed_b_shared_oracle_outer_test",
        checkpoint_frozen=checkpoint_frozen,
    )
    oracle_metrics, oracle_predictions = _evaluate_observed_b_shared(
        selected_model,
        oracle_data,
        accelerator.device,
    )
    return _write_audited_fold_outputs(
        config=config,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        model=selected_model,
        fold_spec=fold_spec,
        train_data=train_data,
        internal_metrics=internal_metrics,
        internal_predictions=internal_predictions,
        response_metrics=response_metrics,
        response_predictions=response_predictions,
        oracle_metrics=oracle_metrics,
        oracle_predictions=oracle_predictions,
        external=external,
        batch_lookup=batch_lookup,
        accelerator=accelerator,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
        source_fingerprint=source_fingerprint,
        runtime_evidence=runtime_evidence,
        authority=authority,
    )


def _fit_observed_b_oracle(
    config: AivcConfig,
    train_data: GeneBags,
    val_data: GeneBags,
    fold_spec: FoldSpec,
) -> ObservedBOracleFit:
    """Fit on train observed B and select one frozen epoch on validation B."""
    _authorize_data_access(train_data, "observed_b_oracle_fit")
    _authorize_data_access(val_data, "observed_b_oracle_selection")
    train_x = torch.as_tensor(
        _observed_b_features(train_data),
        dtype=torch.float32,
    )
    train_y = torch.as_tensor(train_data.y, dtype=torch.float32).reshape(-1, 1)
    val_x = torch.as_tensor(_observed_b_features(val_data), dtype=torch.float32)
    val_y = torch.as_tensor(val_data.y, dtype=torch.float32).reshape(-1, 1)
    torch.manual_seed(config.train.seed + fold_spec.outer_fold + 10_000)
    model = torch.nn.Linear(train_data.latent_dim, 1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    best_epoch = 0
    best_mse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.train.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(train_x), train_y).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(torch.nn.functional.mse_loss(model(val_x), val_y))
        if val_mse < best_mse:
            best_epoch = epoch
            best_mse = val_mse
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError("observed-B oracle requires at least one training epoch")
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    train_data.record_access("observed_b_oracle_fit")
    val_data.record_access("observed_b_oracle_selection")
    return ObservedBOracleFit(
        model=model,
        best_epoch=best_epoch,
        checkpoint_sha256=_module_sha256(model),
        val_mse=best_mse,
    )


def _authorize_data_access(data: GeneBags, stage: str) -> None:
    if data.access_recorder is None:
        raise ValueError("audited operation requires a GeneAccessRecorder")
    data.access_recorder.authorize(stage, tuple(str(gene) for gene in data.genes))


def _observed_b_features(data: GeneBags) -> np.ndarray:
    return np.vstack(
        [np.asarray(bag, dtype=np.float32).mean(axis=0) for bag in data.latent_bags]
    ).astype(np.float32)


def _evaluate_observed_b_oracle(
    oracle: ObservedBOracleFit,
    data: GeneBags,
) -> tuple[dict[str, float], pd.DataFrame]:
    with torch.no_grad():
        predictions = (
            oracle.model(torch.as_tensor(_observed_b_features(data)))
            .reshape(-1)
            .numpy()
        )
    y_true = np.asarray(data.y, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.float64)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
    frame = pd.DataFrame(
        {
            "perturbation_gene": [str(gene) for gene in data.genes],
            "y_true": y_true,
            "y_pred": y_pred,
            "n_chunks": np.ones(len(data.genes), dtype=float),
        }
    )
    return metrics, frame


def _evaluate_observed_b_shared(
    model: AivcModel,
    data: GeneBags,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate observed B through the frozen shared response encoder and C head."""
    control_expression = torch.as_tensor(
        np.asarray(data.control_input, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    predictions = []
    with torch.no_grad():
        for bag in data.input_bags:
            observed_expression = torch.as_tensor(
                np.asarray(bag, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            predictions.append(
                model.predict_c_from_response(
                    observed_expression,
                    control_expression,
                )
            )
    y_true = np.asarray(data.y, dtype=np.float64)
    y_pred = (
        torch.stack(predictions).detach().cpu().numpy().astype(np.float64, copy=False)
    )
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
    frame = pd.DataFrame(
        {
            "perturbation_gene": [str(gene) for gene in data.genes],
            "y_true": y_true,
            "y_pred": y_pred,
            "n_chunks": np.ones(len(data.genes), dtype=float),
        }
    )
    return metrics, frame


def _runtime_evidence(
    model: AivcModel,
    train_data: GeneBags,
    canonical_gene_order: tuple[str, ...],
) -> dict[str, object]:
    state_model = model.state_adapter.state_model
    state_input_dim = int(getattr(state_model, "input_dim", train_data.input_dim))
    state_output_dim = int(getattr(state_model, "output_dim", train_data.input_dim))
    state_pert_dim = int(getattr(state_model, "pert_dim", 0))
    feature_match_count = (
        int(len(train_data.feature_names))
        if train_data.feature_names is not None
        else int(train_data.input_dim)
    )
    if (
        state_input_dim != train_data.input_dim
        or feature_match_count != state_input_dim
    ):
        raise ValueError("STATE runtime input features do not match verified GeneBags")
    canonical = tuple(str(gene).upper() for gene in canonical_gene_order)
    evidence: dict[str, object] = {
        "state_input_dim": state_input_dim,
        "state_output_dim": state_output_dim,
        "state_pert_dim": state_pert_dim,
        "state_feature_match_count": feature_match_count,
        "esm_resolved_count": None,
        "esm_total_count": None,
        "esm_gene_order_sha256": None,
    }
    if isinstance(model.perturbations, Esm2PerturbationAdapter):
        resolved_order = tuple(model.perturbations.genes)
        canonical_resolved = resolved_order[: len(canonical)]
        if canonical_resolved != canonical:
            raise ValueError(
                "ESM-2 adapter order differs from canonical manifest order"
            )
        evidence.update(
            {
                "esm_resolved_count": len(canonical_resolved),
                "esm_total_count": len(canonical),
                "esm_gene_order_sha256": _ordered_gene_sha256(canonical_resolved),
            }
        )
    return evidence


def _ordered_gene_sha256(genes: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


def _canonical_split_sha256(config: AivcConfig) -> str:
    path = config.cv.outer_split_sha256_file
    if path is None:
        raise ValueError("audited training requires canonical split SHA-256 authority")
    text = path.read_text(encoding="utf-8")
    digest = text[:-1] if text.endswith("\n") else ""
    if (
        len(text) != 65
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "canonical split SHA-256 file must contain one lowercase digest and newline"
        )
    return digest


def _fold_artifact_authority(
    config: AivcConfig,
    fold_spec: FoldSpec,
    *,
    source_fingerprint: str,
    canonical_split_sha256: str | None = None,
) -> ArtifactAuthority:
    return ArtifactAuthority(
        source_fingerprint=source_fingerprint,
        canonical_split_sha256=(
            canonical_split_sha256
            if canonical_split_sha256 is not None
            else _canonical_split_sha256(config)
        ),
        outer_fold=fold_spec.outer_fold,
        fit_stage="inner_train",
        fit_genes=tuple(fold_spec.train_genes),
        train_genes=tuple(fold_spec.train_genes),
        val_genes=tuple(fold_spec.val_genes),
        test_genes=tuple(fold_spec.test_genes),
        selection_genes=tuple(fold_spec.val_genes),
    )


def _fit_audited_scvi_latents(
    config: AivcConfig,
    train_data: GeneBags,
    val_data: GeneBags,
    train_split: GeneSplit,
    artifacts_dir: Path,
    accelerator: Accelerator,
    authority: ArtifactAuthority,
) -> tuple[GeneBags, GeneBags]:
    if accelerator.is_main_process:
        train_data, _ = with_cached_scvi_teacher_latents(
            config,
            train_data,
            train_split,
            artifacts_dir,
            external=None,
            fit_teacher=True,
            log_fn=_LOGGER.info,
            authority=authority,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        train_data, _ = with_cached_scvi_teacher_latents(
            config,
            train_data,
            train_split,
            artifacts_dir,
            external=None,
            fit_teacher=False,
            log_fn=_LOGGER.info,
            authority=authority,
        )
    val_data = _project_audited_scvi_data(config, val_data, artifacts_dir)
    return train_data, val_data


def _project_audited_scvi_data(
    config: AivcConfig,
    data: GeneBags,
    artifacts_dir: Path,
) -> GeneBags:
    return project_gene_bags_with_frozen_scvi(
        config,
        data,
        artifacts_dir / "scvi_teacher_model",
    )


def _write_audited_fold_outputs(
    *,
    config: AivcConfig,
    run_dir: Path,
    artifacts_dir: Path,
    models_dir: Path,
    model: AivcModel,
    fold_spec: FoldSpec,
    train_data: GeneBags,
    internal_metrics: dict[str, float],
    internal_predictions: pd.DataFrame,
    response_metrics: dict[str, float],
    response_predictions: pd.DataFrame,
    oracle_metrics: dict[str, float],
    oracle_predictions: pd.DataFrame,
    external: ExternalGeneBags | None,
    batch_lookup: dict[str, int],
    accelerator: Accelerator,
    best_epoch: int,
    checkpoint_path: Path,
    source_fingerprint: str,
    runtime_evidence: dict[str, object],
    authority: ArtifactAuthority,
) -> dict[str, Path]:
    paths = _audited_output_paths(run_dir)
    external_metrics: dict[str, float] = {}
    external_predictions = pd.DataFrame()
    if external is not None:
        external_data = external.data
        external_loader = accelerator.prepare(
            _gene_loader(
                np.arange(len(external_data.genes), dtype=np.int64),
                shuffle=False,
                seed=config.train.seed,
                gene_batch_size=config.train.gene_batch_size,
                world_size=accelerator.num_processes,
            )
        )
        external_metrics, external_predictions = _evaluate_prediction_only_final(
            model,
            external_data,
            external_loader,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
        )

    def write_fold_outputs() -> None:
        metrics = _audited_metric_rows(
            fold_spec.outer_fold,
            internal_metrics,
            response_metrics,
            oracle_metrics,
            external_metrics,
            config.external_test.name if config.external_test is not None else None,
        )
        predictions = _audited_prediction_rows(
            fold_spec,
            internal_predictions,
            response_predictions,
            oracle_predictions,
            external_predictions,
            config.external_test.name if config.external_test is not None else None,
        )
        if train_data.access_recorder is None:
            raise ValueError("audited training lost its gene access recorder")
        access_audit = train_data.access_recorder.to_frame()
        external_qa = _external_qa_rows(external, fold_spec.outer_fold)
        metrics.to_csv(paths["fold_metrics"], index=False)
        predictions.to_csv(paths["predictions"], index=False)
        access_audit.to_csv(paths["fit_access_audit"], index=False)
        external_qa.to_csv(paths["external_alignment_qa"], index=False)
        paths["runtime_evidence"].write_text(
            json.dumps(runtime_evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fit_summary = {
            "adapter_sha256": _module_sha256(model.perturbations),
            "state_sha256": _module_sha256(model.state_adapter),
            "response_encoder_sha256": _module_sha256(model.response_encoder),
            "gmm_sha256": _module_sha256(model.response_pooler),
            "c_head_sha256": _module_sha256(model.c_head),
            "best_epoch": int(best_epoch),
            "checkpoint_sha256": _path_sha256(checkpoint_path),
            "source_fingerprint": source_fingerprint,
            **authority.metadata(),
        }
        paths["fit_audit_summary"].write_text(
            json.dumps(fit_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_fitted_artifact_metadata(
            artifacts_dir,
            authority,
            model,
        )
        _save_model_checkpoint(
            accelerator,
            model,
            models_dir / "final",
            {
                **authority.metadata(),
                "checkpoint_kind": "frozen_selected",
                "epoch": best_epoch,
                "outer_fold": fold_spec.outer_fold,
                "source_fingerprint": source_fingerprint,
                "selected_layer": None,
                "best_epoch": best_epoch,
                "state_checkpoint": str(config.state.checkpoint_path),
                "esm2_npz": str(config.state.esm2_npz),
            },
        )

    run_rank_zero_or_raise(
        accelerator,
        f"fold {fold_spec.outer_fold} final output write",
        write_fold_outputs,
    )
    return paths


def _write_fitted_artifact_metadata(
    artifacts_dir: Path,
    authority: ArtifactAuthority,
    model: AivcModel,
) -> None:
    common = authority.metadata()
    payloads = {
        "esm_adapter_fit": {
            **common,
            "kind": "esm_adapter",
            "artifact_sha256": _module_sha256(model.perturbations),
        },
        "state_fit": {
            **common,
            "kind": "state",
            "artifact_sha256": _module_sha256(model.state_adapter),
        },
        "response_encoder_fit": {
            **common,
            "kind": "response_encoder",
            "artifact_sha256": _module_sha256(model.response_encoder),
        },
        "gmm_fit": {
            **common,
            "kind": "trainable_gmm",
            "artifact_sha256": _module_sha256(model.response_pooler),
        },
        "c_head_fit": {
            **common,
            "kind": "c_head",
            "artifact_sha256": _module_sha256(model.c_head),
        },
    }
    for directory, payload in payloads.items():
        metadata_path = artifacts_dir / directory / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(metadata_path, payload)


def _audited_output_paths(run_dir: Path) -> dict[str, Path]:
    artifacts = run_dir / "artifacts"
    return {
        "run_dir": run_dir,
        "train_log": run_dir / "train_log.csv",
        "fold_metrics": artifacts / "fold_metrics.csv",
        "predictions": artifacts / "predictions.csv",
        "fit_access_audit": artifacts / "fit_access_audit.csv",
        "external_alignment_qa": artifacts / "external_alignment_qa.csv",
        "fit_audit_summary": run_dir / "fit_audit_summary.json",
        "runtime_evidence": run_dir / "runtime_evidence.json",
    }


def _audited_metric_rows(
    outer_fold: int,
    internal: dict[str, float],
    response: dict[str, float],
    oracle_response: dict[str, float],
    external: dict[str, float],
    external_name: str | None,
) -> pd.DataFrame:
    generation_keys = {
        key: value
        for key, value in response.items()
        if key
        in {
            "hvg_mean_delta",
            "hvg_energy",
            "latent_mean_delta",
            "latent_energy",
            "occupancy",
        }
    }
    oracle = dict(oracle_response)
    rows = [
        {
            "outer_fold": outer_fold,
            "evaluation_scope": "internal_outer_test",
            **internal,
        },
        {
            "outer_fold": outer_fold,
            "evaluation_scope": "generation_quality_outer_test",
            **generation_keys,
        },
        {
            "outer_fold": outer_fold,
            "evaluation_scope": "observed_b_shared_oracle_outer_test",
            **oracle,
        },
    ]
    if external_name is not None:
        rows.append(
            {
                "outer_fold": outer_fold,
                "evaluation_scope": f"external:{external_name}",
                **external,
            }
        )
    return pd.DataFrame(rows)


def _audited_prediction_rows(
    fold: FoldSpec,
    internal: pd.DataFrame,
    response: pd.DataFrame,
    oracle_response: pd.DataFrame,
    external: pd.DataFrame,
    external_name: str | None,
) -> pd.DataFrame:
    frames = []
    for scope, frame in (
        ("internal_outer_test", internal),
        ("generation_quality_outer_test", response),
    ):
        scoped = frame.copy()
        scoped.insert(0, "evaluation_scope", scope)
        scoped.insert(0, "inner_role", "outer_test")
        scoped.insert(0, "outer_fold", fold.outer_fold)
        frames.append(scoped)
    oracle = oracle_response.copy()
    if "y_obs_anchor" in oracle:
        oracle["y_pred"] = oracle["y_obs_anchor"]
    oracle.insert(0, "evaluation_scope", "observed_b_shared_oracle_outer_test")
    oracle.insert(0, "inner_role", "outer_test")
    oracle.insert(0, "outer_fold", fold.outer_fold)
    frames.append(oracle)
    if external_name is not None and not external.empty:
        scoped_external = external.copy()
        scoped_external.insert(0, "evaluation_scope", f"external:{external_name}")
        scoped_external.insert(0, "inner_role", "external_test")
        scoped_external.insert(0, "outer_fold", fold.outer_fold)
        frames.append(scoped_external)
    return pd.concat(frames, ignore_index=True)


def _external_qa_rows(
    external: ExternalGeneBags | None,
    outer_fold: int,
) -> pd.DataFrame:
    if external is None:
        return pd.DataFrame(columns=["outer_fold", "source_name"])
    rows = []
    for source in external.qa.get("sources", []):
        rows.append({"outer_fold": outer_fold, **dict(source)})
    if len(rows) != 3:
        raise ValueError("Adamson alignment QA must contain exactly three source rows")
    return pd.DataFrame(rows)


def _arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def _scvi_model_sha256(artifacts_dir: Path) -> str:
    model_dir = artifacts_dir / "scvi_teacher_model"
    model_payload = model_dir / "model.pt"
    return _path_sha256(model_payload if model_payload.exists() else model_dir)


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    paths = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    for item in paths:
        digest.update(str(item.relative_to(path.parent)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


class _GeneIndexDataset(Dataset[dict[str, int | bool]]):
    def __init__(self, indices: np.ndarray, is_padding: np.ndarray) -> None:
        if len(indices) != len(is_padding):
            msg = "indices and is_padding must have the same length"
            raise ValueError(msg)
        self._indices = [int(index) for index in indices]
        self._is_padding = [bool(value) for value in is_padding]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, int | bool]:
        return {
            "index": self._indices[index],
            "is_padding": self._is_padding[index],
        }


def _make_accelerator(config: AivcConfig) -> Accelerator:
    dataloader_config = DataLoaderConfiguration(
        even_batches=False,
        use_seedable_sampler=True,
        data_seed=config.train.seed,
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=True,
    )
    return Accelerator(
        cpu=config.train.device == "cpu",
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs],
    )


def _configure_logging(accelerator: Accelerator) -> None:
    level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("aivc_model").setLevel(level)


def _require_authoritative_gene_batch_size(config: AivcConfig) -> None:
    if config.train.gene_batch_size != 1:
        raise ValueError(
            "gene_batch_size must be 1 for authoritative exp05 training"
        )


def _configure_cache_only_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("aivc_model").setLevel(logging.INFO)


def _configure_float32_matmul_precision(config: AivcConfig) -> None:
    precision = config.train.float32_matmul_precision
    if precision is None:
        return
    allowed = {"highest", "high", "medium"}
    if precision not in allowed:
        msg = f"train.float32_matmul_precision must be one of {sorted(allowed)} or null"
        raise ValueError(msg)
    torch.set_float32_matmul_precision(precision)


def _resolve_run_id(config: AivcConfig, accelerator: Accelerator) -> str:
    run_id = config.train.run_id
    if accelerator.is_main_process and run_id is None:
        run_id = datetime.now(timezone.utc).strftime("state_aivc_%Y%m%dT%H%M%SZ")
    values: list[str | None] = [run_id]
    broadcast_object_list(values, from_process=0)
    if values[0] is None:
        msg = "Failed to resolve distributed run_id"
        raise RuntimeError(msg)
    return str(values[0])


def _with_rank_safe_scvi_teacher(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    external: Any,
    artifacts_dir: Path,
    accelerator: Accelerator,
    *,
    config_path: Path | None = None,
) -> tuple[GeneBags, Any]:
    if config.projector.teacher != "scvi":
        return data, external
    if accelerator.is_main_process:
        if config_path is None:
            _LOGGER.warning(
                "No config path supplied; materializing scVI cache in rank0 process"
            )
            with_cached_scvi_teacher_latents(
                config,
                data,
                split,
                artifacts_dir,
                external=external,
                fit_teacher=True,
                log_fn=_LOGGER.info,
            )
        else:
            _run_scvi_cache_subprocess(config_path, artifacts_dir)
        data, external = _wait_for_scvi_latent_cache(
            config,
            data,
            split,
            external,
            artifacts_dir,
            timeout_seconds=0.0,
            poll_seconds=0.0,
        )
    else:
        data, external = _wait_for_scvi_latent_cache(
            config,
            data,
            split,
            external,
            artifacts_dir,
        )
    accelerator.wait_for_everyone()
    return data, external


def _materialize_scvi_cache_only(config: AivcConfig, artifacts_dir: Path) -> None:
    """Materialize the scVI latent cache outside the Accelerate process group."""
    set_seed(config.train.seed)
    _configure_float32_matmul_precision(config)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data = load_gene_bags(config)
    split = make_gene_split(data.genes, data.y, config.split)
    external = load_external_gene_bags(
        config,
        data,
        artifacts_dir,
        project_scvi=False,
    )
    with_cached_scvi_teacher_latents(
        config,
        data,
        split,
        artifacts_dir,
        external=external,
        fit_teacher=True,
        log_fn=_LOGGER.info,
    )


def _run_scvi_cache_subprocess(config_path: Path, artifacts_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--materialize-scvi-cache-only",
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    _LOGGER.info("Starting isolated rank0 scVI latent cache materialization")
    subprocess.run(
        cmd,
        check=True,
        env=_isolated_scvi_cache_env(os.environ),
    )
    _LOGGER.info("Finished isolated rank0 scVI latent cache materialization")


def _isolated_scvi_cache_env(source: Mapping[str, str]) -> dict[str, str]:
    env = dict(source)
    distributed_keys = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NPROCS",
        "SLURM_NODEID",
        "SLURM_STEP_ID",
        "SLURM_STEP_NUM_TASKS",
    }
    for key in tuple(env):
        if (
            key in distributed_keys
            or key.startswith("TORCHELASTIC_")
            or key.startswith("ACCELERATE_")
        ):
            env.pop(key, None)
    visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        first_device = visible_devices.split(",", maxsplit=1)[0].strip()
        if first_device:
            env["CUDA_VISIBLE_DEVICES"] = first_device
    return env


def _wait_for_scvi_latent_cache(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    external: Any,
    artifacts_dir: Path,
    *,
    timeout_seconds: float = _SCVI_CACHE_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = _SCVI_CACHE_POLL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> tuple[GeneBags, Any]:
    started_at = monotonic_fn()
    last_error = "cache has not been checked yet"
    while True:
        try:
            return with_cached_scvi_teacher_latents(
                config,
                data,
                split,
                artifacts_dir,
                external=external,
                fit_teacher=False,
                log_fn=_LOGGER.info,
            )
        except FileNotFoundError as exc:
            last_error = str(exc)
        elapsed = monotonic_fn() - started_at
        if elapsed >= timeout_seconds:
            msg = (
                "Timed out waiting for rank0 scVI latent cache after "
                f"{elapsed:.1f}s at {artifacts_dir}: {last_error}"
            )
            raise TimeoutError(msg) from None
        sleep_fn(min(poll_seconds, timeout_seconds - elapsed))


def _fit_or_load_projector_cache(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    *,
    authority: ArtifactAuthority | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit or load the run-local ridge projector cache."""
    cache_dir = artifacts_dir / "ridge_projector_fit"
    metadata = _projector_cache_metadata(config, data, split, authority=authority)
    cached = _load_projector_cache(cache_dir, metadata)
    if cached is not None:
        return cached
    train_expr = np.vstack(
        [data.control_input, *[data.input_bags[i] for i in split.train]]
    )
    train_latent = np.vstack(
        [data.control_latent, *[data.latent_bags[i] for i in split.train]]
    )
    weight, bias = fit_linear_projector(
        train_expr,
        train_latent,
        config.projector.ridge_alpha,
    )
    _write_projector_cache(cache_dir, metadata, weight, bias)
    return weight, bias


def _fit_or_load_fixed_gmm_cache(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    artifacts_dir: Path,
    *,
    authority: ArtifactAuthority | None = None,
) -> FixedGMMFeatureizer:
    """Fit or load the run-local fixed GMM featureizer cache."""
    cache_dir = artifacts_dir / "fixed_gmm_fit"
    metadata = _fixed_gmm_cache_metadata(config, data, split, authority=authority)
    cached = _load_fixed_gmm_cache(cache_dir, metadata)
    if cached is not None:
        return cached
    featureizer = fit_fixed_gmm(
        tuple(data.latent_bags[i] for i in split.train),
        data.control_latent,
        n_components=config.gmm.n_components,
        covariance_floor=config.gmm.covariance_floor,
        random_state=config.train.seed,
        max_fit_cells=config.gmm.max_fit_cells,
    )
    _write_fixed_gmm_cache(cache_dir, metadata, featureizer, data.control_latent)
    return featureizer


def _load_projector_cache(
    cache_dir: Path,
    expected_metadata: dict[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    if not _cache_metadata_matches(cache_dir, expected_metadata):
        return None
    payload_path = cache_dir / "projector.npz"
    if not payload_path.exists():
        return None
    try:
        with np.load(payload_path) as payload:
            weight = np.asarray(payload["weight"], dtype=np.float32)
            bias = np.asarray(payload["bias"], dtype=np.float32)
    except (KeyError, OSError, ValueError):
        return None
    return weight, bias


def _load_fixed_gmm_cache(
    cache_dir: Path,
    expected_metadata: dict[str, object],
) -> FixedGMMFeatureizer | None:
    if not _cache_metadata_matches(cache_dir, expected_metadata):
        return None
    payload_path = cache_dir / "gmm.npz"
    if not payload_path.exists():
        return None
    try:
        with np.load(payload_path) as payload:
            means = np.asarray(payload["means"], dtype=np.float32)
            variances = np.asarray(payload["variances"], dtype=np.float32)
            weights = np.asarray(payload["weights"], dtype=np.float32)
            control_bag = np.asarray(payload["control_bag"], dtype=np.float32)
    except (KeyError, OSError, ValueError):
        return None
    return FixedGMMFeatureizer(means, variances, weights, control_bag)


def _cache_metadata_matches(
    cache_dir: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not (cache_dir / "COMPLETE").exists():
        return False
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata == expected_metadata
        and _artifact_metadata_is_authorized(metadata)
        and _artifact_metadata_is_authorized(expected_metadata)
    )


def _write_projector_cache(
    cache_dir: Path,
    metadata: dict[str, object],
    weight: np.ndarray,
    bias: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cache_dir / "COMPLETE"
    complete_path.unlink(missing_ok=True)
    _write_npz_atomic(
        cache_dir / "projector.npz",
        {
            "weight": np.asarray(weight, dtype=np.float32),
            "bias": np.asarray(bias, dtype=np.float32),
        },
    )
    _write_json_atomic(cache_dir / "metadata.json", metadata)
    _write_text_atomic(complete_path, "ok\n")


def _write_fixed_gmm_cache(
    cache_dir: Path,
    metadata: dict[str, object],
    featureizer: FixedGMMFeatureizer,
    control_bag: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cache_dir / "COMPLETE"
    complete_path.unlink(missing_ok=True)
    _write_npz_atomic(
        cache_dir / "gmm.npz",
        {
            "means": featureizer.means.detach().cpu().numpy(),
            "variances": featureizer.variances.detach().cpu().numpy(),
            "weights": featureizer.log_weights.detach().cpu().exp().numpy(),
            "control_bag": np.asarray(control_bag, dtype=np.float32),
        },
    )
    _write_json_atomic(cache_dir / "metadata.json", metadata)
    _write_text_atomic(complete_path, "ok\n")


def _projector_cache_metadata(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    *,
    authority: ArtifactAuthority | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "version": 2,
        "kind": "ridge_projector_fit",
        "seed": int(config.train.seed),
        "projector_config": {
            "teacher": str(config.projector.teacher),
            "latent_dim": int(config.projector.latent_dim),
            "ridge_alpha": float(config.projector.ridge_alpha),
            "trainable": bool(config.projector.trainable),
        },
        "train_indices": [int(index) for index in split.train],
        "train_genes": [str(data.genes[index]) for index in split.train],
        "primary": {
            "control_input": _array_cache_identity(data.control_input),
            "control_latent": _array_cache_identity(data.control_latent),
            "train_input_bags": [
                _array_cache_identity(data.input_bags[index]) for index in split.train
            ],
            "train_latent_bags": [
                _array_cache_identity(data.latent_bags[index]) for index in split.train
            ],
        },
    }
    if authority is not None:
        metadata.update(authority.metadata())
    return metadata


def _fixed_gmm_cache_metadata(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
    *,
    authority: ArtifactAuthority | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "version": 2,
        "kind": "fixed_gmm_fit",
        "seed": int(config.train.seed),
        "gmm_config": {
            "n_components": int(config.gmm.n_components),
            "covariance_floor": float(config.gmm.covariance_floor),
            "max_fit_cells": (
                None
                if config.gmm.max_fit_cells is None
                else int(config.gmm.max_fit_cells)
            ),
        },
        "train_indices": [int(index) for index in split.train],
        "train_genes": [str(data.genes[index]) for index in split.train],
        "primary": {
            "control_latent": _array_cache_identity(data.control_latent),
            "train_latent_bags": [
                _array_cache_identity(data.latent_bags[index]) for index in split.train
            ],
        },
    }
    if authority is not None:
        metadata.update(authority.metadata())
    return metadata


def _artifact_metadata_is_authorized(metadata: dict[str, object]) -> bool:
    authority_keys = {
        "source_fingerprint",
        "canonical_split_sha256",
        "outer_fold",
        "fit_stage",
        "fit_genes_sha256",
        "train_genes_sha256",
        "val_genes_sha256",
        "test_genes_sha256",
        "fit_genes",
        "train_genes",
        "val_genes",
        "test_genes",
        "selection_genes",
    }
    if "schema_version" not in metadata:
        return True
    if not authority_keys <= set(metadata):
        return False
    fit = tuple(str(gene) for gene in metadata["fit_genes"])
    train = tuple(str(gene) for gene in metadata["train_genes"])
    val_genes = tuple(str(gene) for gene in metadata["val_genes"])
    test_genes = tuple(str(gene) for gene in metadata["test_genes"])
    val = set(val_genes)
    test = set(test_genes)
    selection = {str(gene) for gene in metadata["selection_genes"]}
    return (
        fit == train
        and not set(train).intersection(test)
        and selection.issubset(val)
        and metadata["fit_genes_sha256"] == _sha256_strings(fit)
        and metadata["train_genes_sha256"] == _sha256_strings(train)
        and metadata["val_genes_sha256"] == _sha256_strings(val_genes)
        and metadata["test_genes_sha256"]
        == _sha256_strings(test_genes)
    )


def _array_cache_identity(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(value, encoding="utf-8")
    tmp_path.replace(path)


def _gene_loader(
    indices: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
    gene_batch_size: int,
    world_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    if gene_batch_size < 1:
        msg = "gene_batch_size must be at least 1"
        raise ValueError(msg)
    generator = torch.Generator()
    generator.manual_seed(seed)
    padded_indices, is_padding = _pad_gene_indices(
        indices,
        batch_size=gene_batch_size,
        world_size=world_size,
    )
    return DataLoader(
        _GeneIndexDataset(padded_indices, is_padding),
        batch_size=gene_batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _prediction_indices(
    internal_data: GeneBags,
    request: PredictionRequest,
) -> np.ndarray:
    data = request.observed_response or internal_data
    index_by_gene = {str(gene): index for index, gene in enumerate(data.genes)}
    missing = [gene for gene in request.genes if gene not in index_by_gene]
    if missing:
        raise ValueError(f"Prediction request contains unknown genes: {missing}")
    return np.asarray(
        [index_by_gene[gene] for gene in request.genes],
        dtype=np.int64,
    )


def _pad_gene_indices(
    indices: np.ndarray,
    *,
    batch_size: int,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)
    if world_size < 1:
        msg = "world_size must be at least 1"
        raise ValueError(msg)
    normalized = np.asarray(indices, dtype=np.int64)
    is_padding = np.zeros(len(normalized), dtype=bool)
    if len(normalized) == 0:
        return normalized, is_padding
    step_size = int(batch_size) * int(world_size)
    remainder = len(normalized) % step_size
    if remainder == 0:
        return normalized, is_padding
    pad_count = step_size - remainder
    repeats = int(math.ceil(pad_count / len(normalized)))
    padding = np.tile(normalized, repeats)[:pad_count]
    return (
        np.concatenate([normalized, padding]).astype(np.int64),
        np.concatenate([is_padding, np.ones(pad_count, dtype=bool)]),
    )


def _loss_weights(config: AivcConfig, epoch: int | None = None) -> LossWeights:
    b_loss_scale = _b_loss_anneal_scale(config, epoch)
    return LossWeights(
        latent_mean_delta=config.loss.latent_mean_delta_weight * b_loss_scale,
        latent_energy=config.loss.latent_energy_weight * b_loss_scale,
        hvg_mean_delta=config.loss.hvg_mean_delta_weight * b_loss_scale,
        hvg_energy=config.loss.hvg_energy_weight * b_loss_scale,
        pred_c=config.loss.pred_c_weight,
        obs_c=config.loss.obs_c_weight,
        occupancy=config.loss.occupancy_weight * b_loss_scale,
        pred_rank=config.loss.pred_rank_weight,
        pred_rank_tau=config.loss.pred_rank_tau,
        pred_rank_pair_margin=config.loss.pred_rank_pair_margin,
        pred_rank_pair_weight_clip=config.loss.pred_rank_pair_weight_clip,
    )


def _b_loss_anneal_scale(config: AivcConfig, epoch: int | None) -> float:
    epochs = int(config.loss.b_loss_anneal_epochs)
    final_fraction = float(config.loss.b_loss_anneal_final_fraction)
    if epoch is None or epochs <= 0:
        return 1.0
    if not 0.0 <= final_fraction <= 1.0:
        msg = "loss.b_loss_anneal_final_fraction must be between 0 and 1"
        raise ValueError(msg)
    if epochs <= 1:
        return final_fraction
    progress = min(max(int(epoch), 1), epochs) - 1
    return 1.0 - (1.0 - final_fraction) * (progress / float(epochs - 1))


def _build_e2e_model(
    config: AivcConfig,
    data: GeneBags,
    extra_genes: tuple[str, ...],
    canonical_gene_order: tuple[str, ...],
    emit_checkpoint_output: bool = True,
) -> AivcModel:
    expected_response_config = ResponseEncoderConfig(input_dim=2000, latent_dim=128)
    if config.response_encoder != expected_response_config:
        raise ValueError(
            "audited exp05 requires response_encoder input_dim=2000 and "
            "latent_dim=128"
        )
    pert_dim = config.state.pert_dim
    tokenizer = config.state.gene_tokenizer
    known_vectors: dict[str, np.ndarray] = {}
    canonical_genes: list[str] = []
    esm = None
    if tokenizer == "state_onehot":
        known_vectors = load_perturbation_vectors(
            config.state.known_perturbation_vectors
        )
        if pert_dim is None:
            pert_dim = _infer_pert_dim(known_vectors)
    elif tokenizer == "esm2":
        if pert_dim is None:
            raise ValueError("state.pert_dim is required for gene_tokenizer=esm2")
        canonical_genes = _canonical_esm2_genes(
            config,
            data,
            canonical_gene_override=canonical_gene_order,
        )
        if config.state.esm2_npz is None:
            raise ValueError("state.esm2_npz is required for gene_tokenizer=esm2")
        esm = load_esm2_embeddings(config.state.esm2_npz)
        require_complete_esm_coverage(canonical_genes, esm)
    else:
        raise ValueError(f"Unknown state.gene_tokenizer: {tokenizer}")
    output_dim = config.state.output_dim or data.input_dim
    state_model = load_state_model(
        backend=config.state.backend,
        checkpoint_path=config.state.checkpoint_path,
        input_dim=config.state.input_dim or data.input_dim,
        output_dim=output_dim,
        pert_dim=pert_dim,
        emit_checkpoint_output=emit_checkpoint_output,
    )
    if tokenizer == "esm2":
        if esm is None:
            raise AssertionError("ESM-2 table was not loaded")
        effective_pert_dim = _effective_state_pert_dim(state_model, pert_dim)
        all_genes = canonical_genes + sorted(
            set(str(gene).upper() for gene in extra_genes).difference(canonical_genes)
        )
        perturbations = Esm2PerturbationAdapter(
            all_genes,
            esm,
            adapter_hidden=config.state.esm2_adapter_hidden,
            pert_dim=effective_pert_dim,
        )
    else:
        perturbations = PerturbationVectorAdapter(
            sorted({*(str(gene) for gene in data.genes), *extra_genes}),
            known_vectors,
            pert_dim,
        )
    state_adapter = StateForwardAdapter(state_model)
    state_adapter.requires_grad_(True)
    response_encoder = ResponseEncoder(
        config.response_encoder.input_dim,
        config.response_encoder.latent_dim,
    )
    response_pooler = TrainableDiagonalGMM(
        latent_dim=config.response_encoder.latent_dim,
        n_components=config.gmm.n_components,
        covariance_floor=config.gmm.covariance_floor,
        init_scale=config.gmm.init_scale,
    )
    c_head = MLPHead(
        input_dim=response_pooler.output_dim,
        hidden_units=config.model.c_hidden_units,
        dropout=config.model.dropout,
    )
    return AivcModel(
        state_adapter=state_adapter,
        perturbations=perturbations,
        response_encoder=response_encoder,
        response_pooler=response_pooler,
        c_head=c_head,
        control_expression_mean=data.control_input.mean(axis=0).astype(np.float32),
    )


def _build_model(
    config: AivcConfig,
    data: GeneBags,
    featureizer: torch.nn.Module,
    projector_weight: np.ndarray,
    projector_bias: np.ndarray,
    extra_genes: tuple[str, ...] = (),
    emit_checkpoint_output: bool = True,
    canonical_gene_override: tuple[str, ...] | None = None,
) -> AivcModel:
    pert_dim = config.state.pert_dim
    tokenizer = config.state.gene_tokenizer
    known_vectors: dict[str, np.ndarray] = {}
    canonical_genes: list[str] = []
    esm = None
    if tokenizer == "state_onehot":
        known_vectors = load_perturbation_vectors(
            config.state.known_perturbation_vectors
        )
        if pert_dim is None:
            pert_dim = _infer_pert_dim(known_vectors)
    elif tokenizer == "esm2":
        if pert_dim is None:
            raise ValueError("state.pert_dim is required for gene_tokenizer=esm2")
        canonical_genes = _canonical_esm2_genes(
            config,
            data,
            canonical_gene_override=canonical_gene_override,
        )
        if config.state.esm2_npz is None:
            raise ValueError("state.esm2_npz is required for gene_tokenizer=esm2")
        esm = load_esm2_embeddings(config.state.esm2_npz)
        require_complete_esm_coverage(canonical_genes, esm)
    else:
        raise ValueError(f"Unknown state.gene_tokenizer: {tokenizer}")
    output_dim = config.state.output_dim or data.input_dim
    state_model = load_state_model(
        backend=config.state.backend,
        checkpoint_path=config.state.checkpoint_path,
        input_dim=config.state.input_dim or data.input_dim,
        output_dim=output_dim,
        pert_dim=pert_dim,
        emit_checkpoint_output=emit_checkpoint_output,
    )
    if tokenizer == "esm2":
        if esm is None:
            raise AssertionError("ESM-2 table was not loaded")
        effective_pert_dim = _effective_state_pert_dim(state_model, pert_dim)
        all_genes = canonical_genes + sorted(
            set(str(gene).upper() for gene in extra_genes).difference(canonical_genes)
        )
        perturbations = Esm2PerturbationAdapter(
            all_genes,
            esm,
            adapter_hidden=config.state.esm2_adapter_hidden,
            pert_dim=effective_pert_dim,
        )
    else:
        perturbations = PerturbationVectorAdapter(
            sorted({*(str(gene) for gene in data.genes), *extra_genes}),
            known_vectors,
            pert_dim,
        )
    response_encoder = ResponseEncoder(
        input_dim=int(projector_weight.shape[0]),
        latent_dim=int(projector_weight.shape[1]),
    )
    with torch.no_grad():
        response_encoder.linear.weight.copy_(
            torch.as_tensor(projector_weight.T, dtype=torch.float32)
        )
        response_encoder.linear.bias.copy_(
            torch.as_tensor(projector_bias, dtype=torch.float32)
        )
    if not (tokenizer == "esm2" or config.projector.trainable):
        response_encoder.requires_grad_(False)
    response_pooler = TrainableDiagonalGMM(
        latent_dim=int(featureizer.means.shape[1]),
        n_components=int(featureizer.means.shape[0]),
        covariance_floor=config.gmm.covariance_floor,
        init_scale=config.gmm.init_scale,
    )
    with torch.no_grad():
        response_pooler.means.copy_(featureizer.means)
        adjusted_variance = (
            featureizer.variances - config.gmm.covariance_floor
        ).clamp_min(1e-8)
        response_pooler.raw_variances.copy_(torch.log(torch.expm1(adjusted_variance)))
        response_pooler.mixture_logits.copy_(featureizer.log_weights)
    c_head = MLPHead(
        input_dim=response_pooler.output_dim,
        hidden_units=config.model.c_hidden_units,
        dropout=config.model.dropout,
    )
    state_adapter = StateForwardAdapter(state_model)
    if tokenizer == "esm2" or config.train.freeze_state:
        state_adapter.requires_grad_(False)
    return AivcModel(
        state_adapter=state_adapter,
        perturbations=perturbations,
        response_encoder=response_encoder,
        response_pooler=response_pooler,
        c_head=c_head,
        control_expression_mean=data.control_input.mean(axis=0).astype(np.float32),
    )


def _effective_state_pert_dim(state_model: torch.nn.Module, fallback: int) -> int:
    raw_dim = getattr(state_model, "pert_dim", None)
    if raw_dim is None:
        raise ValueError("Loaded STATE model must expose pert_dim")
    try:
        effective_dim = int(raw_dim)
    except (TypeError, ValueError) as error:
        raise ValueError("Loaded STATE model has an invalid pert_dim") from error
    if effective_dim != int(fallback):
        raise ValueError(
            "Loaded STATE model pert_dim "
            f"must equal configured repaired-path width {int(fallback)}; "
            f"got {effective_dim}"
        )
    return effective_dim


def _canonical_esm2_genes(
    config: AivcConfig,
    data: GeneBags,
    *,
    canonical_gene_override: tuple[str, ...] | None = None,
) -> list[str]:
    manifest_path = config.cv.outer_split_manifest
    sha256_path = config.cv.outer_split_sha256_file
    if manifest_path is None or sha256_path is None:
        raise ValueError(
            "ESM-2 tokenization requires the canonical outer manifest and SHA-256"
        )
    sha256_text = sha256_path.read_text(encoding="utf-8")
    if len(sha256_text) != 65 or not sha256_text.endswith("\n"):
        raise ValueError("outer split SHA-256 file must contain one digest and newline")
    expected_sha256 = sha256_text[:-1]
    requested_genes = canonical_gene_override or tuple(str(gene) for gene in data.genes)
    labels = pd.DataFrame(
        {"perturbation_gene": [str(gene).upper() for gene in requested_genes]}
    )
    manifest = load_canonical_outer_manifest(
        manifest_path,
        labels,
        expected_sha256,
    )
    canonical_genes = manifest["perturbation_gene"].tolist()
    data_genes = [str(gene).upper() for gene in requested_genes]
    if len(canonical_genes) != CANONICAL_GENE_COUNT or data_genes != canonical_genes:
        raise ValueError(
            "ESM-2 internal gene universe must remain exactly 9338 canonical rows"
        )
    return canonical_genes


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        msg = "AIVC model has no trainable parameters"
        raise ValueError(msg)
    return parameters


def _optimizer_parameter_groups(
    model: AivcModel,
    config: AivcConfig,
) -> list[dict[str, object]]:
    state_parameters = [
        parameter
        for parameter in model.state_adapter.parameters()
        if parameter.requires_grad
    ]
    state_ids = {id(parameter) for parameter in state_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in state_ids
    ]
    if not state_parameters or not other_parameters:
        raise ValueError("e2e exp05 requires trainable STATE and downstream parameters")
    return [
        {"params": state_parameters, "lr": config.train.state_learning_rate},
        {"params": other_parameters, "lr": config.train.learning_rate},
    ]


def _run_epoch(
    model: torch.nn.Module,
    data: GeneBags,
    indices: DataLoader[dict[str, torch.Tensor]],
    weights: LossWeights,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    cell_set_len: int,
    accelerator: Accelerator,
    batch_lookup: dict[str, int],
    *,
    epoch: int,
    max_epochs: int,
    max_grad_norm: float = 1.0,
    tensor_cache: _InputTensorCache | None = None,
) -> dict[str, float]:
    model.train()
    metric_sum = _empty_metric_sum(accelerator.device)
    metric_count = 0
    local_optimizer_steps = 0
    total = len(indices)
    iterator = tqdm(
        indices,
        desc=f"epoch {epoch}/{max_epochs}",
        total=total,
        miniters=max(1, math.ceil(total / 10)),
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for batch in iterator:
        optimizer.zero_grad(set_to_none=True)
        gene_indices = _batch_indices(batch)
        if not gene_indices:
            continue
        padding_flags = _batch_padding_flags(batch)
        valid_mask = torch.tensor(
            [not flag for flag in padding_flags],
            dtype=torch.bool,
            device=accelerator.device,
        )
        model_inputs, fallback_counts, n_chunks = _model_inputs_for_indices(
            data,
            gene_indices,
            rng,
            cell_set_len,
            accelerator.device,
            batch_lookup,
            pad_short=True,
            tensor_cache=tensor_cache,
        )
        model_inputs["gene_mask"] = valid_mask
        losses = model(weights=weights, **model_inputs)
        total_loss = losses["total"]
        if not isinstance(total_loss, torch.Tensor):
            msg = "Expected tensor loss for backward"
            raise TypeError(msg)
        accelerator.backward(total_loss)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        local_optimizer_steps += 1
        metric_tensor = _metric_tensor_from_losses(
            losses,
            fallback_counts,
            n_chunks,
        ).detach()
        if valid_mask.any():
            metric_sum = metric_sum + metric_tensor[valid_mask].sum(dim=0)
            metric_count += int(valid_mask.sum().item())
    metrics = _reduce_metric_mean(metric_sum, metric_count, accelerator)
    metrics["local_optimizer_steps"] = float(local_optimizer_steps)
    return metrics


def _evaluate(
    model: torch.nn.Module,
    data: GeneBags,
    indices: DataLoader[dict[str, torch.Tensor]],
    weights: LossWeights,
    rng: np.random.Generator,
    cell_set_len: int,
    accelerator: Accelerator,
    batch_lookup: dict[str, int],
    pad_short: bool,
    tensor_cache: _InputTensorCache | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    metric_sum = _empty_metric_sum(accelerator.device)
    metric_count = 0
    pred_tensors = []
    with torch.no_grad():
        for batch in indices:
            gene_indices = _batch_indices(batch)
            if not gene_indices:
                continue
            padding_flags = _batch_padding_flags(batch)
            valid_mask = torch.tensor(
                [not flag for flag in padding_flags],
                dtype=torch.bool,
                device=accelerator.device,
            )
            model_inputs, fallback_counts, n_chunks = _model_inputs_for_indices(
                data,
                gene_indices,
                rng,
                cell_set_len,
                accelerator.device,
                batch_lookup,
                pad_short=pad_short,
                tensor_cache=tensor_cache,
            )
            model_inputs["gene_mask"] = valid_mask
            losses = model(weights=weights, **model_inputs)
            metric_tensor = _metric_tensor_from_losses(
                losses,
                fallback_counts,
                n_chunks,
            ).detach()
            if valid_mask.any():
                metric_sum = metric_sum + metric_tensor[valid_mask].sum(dim=0)
                metric_count += int(valid_mask.sum().item())
            pred_tensors.append(
                _prediction_tensor_from_losses(
                    losses,
                    data,
                    gene_indices,
                    padding_flags,
                    fallback_counts,
                    n_chunks,
                    accelerator.device,
                )
            )
    summary = _reduce_metric_mean(metric_sum, metric_count, accelerator)
    predictions = _gather_predictions(pred_tensors, data, accelerator)
    if accelerator.is_main_process and not predictions.empty:
        y_true = predictions["y_true"].to_numpy(dtype=np.float64)
        y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
        summary.update(regression_metrics(y_true, y_pred))
        summary.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
        y_obs = predictions["y_obs_anchor"].to_numpy(dtype=np.float64)
        summary.update(_prefix(regression_metrics(y_true, y_obs), "obs"))
        summary.update(_prefix(ranking_metrics(y_true, y_obs, (-0.5, -1.0)), "obs"))
    return summary, predictions


def _evaluate_prediction_only_final(
    model: torch.nn.Module,
    data: GeneBags,
    indices: DataLoader[dict[str, torch.Tensor]],
    cell_set_len: int,
    accelerator: Accelerator,
    batch_lookup: dict[str, int],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate final y_pred from controls plus perturbation identity only."""
    model.eval()
    inference_model = accelerator.unwrap_model(model)
    inference_model.eval()
    control_cells, control_batch_indices = _final_prediction_control_tensors(
        data,
        cell_set_len,
        accelerator.device,
        batch_lookup,
    )
    pred_tensors = []
    with torch.no_grad():
        for batch in indices:
            gene_indices = _batch_indices(batch)
            if not gene_indices:
                continue
            padding_flags = _batch_padding_flags(batch)
            pred_tensors.append(
                _final_prediction_tensor(
                    inference_model,
                    data,
                    gene_indices,
                    padding_flags,
                    accelerator.device,
                    control_cells,
                    control_batch_indices,
                )
            )
    predictions = _gather_final_predictions(pred_tensors, data, accelerator)
    summary: dict[str, float] = {}
    if accelerator.is_main_process and not predictions.empty:
        y_true = predictions["y_true"].to_numpy(dtype=np.float64)
        y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
        summary.update(regression_metrics(y_true, y_pred))
        summary.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
    return summary, predictions


def _final_prediction_control_tensors(
    data: GeneBags,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cell_set_len < 1:
        msg = "cell_set_len must be at least 1"
        raise ValueError(msg)
    if data.control_input.shape[0] < 1:
        msg = "prediction-only final evaluation requires at least one control cell"
        raise ValueError(msg)
    control_cells = torch.as_tensor(
        np.asarray(data.control_input, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    control_batch_indices = _batch_tensor(data.control_batch, batch_lookup, device)
    return control_cells, control_batch_indices


def _final_prediction_tensor(
    model: torch.nn.Module,
    data: GeneBags,
    gene_indices: list[int],
    padding_flags: list[bool],
    device: torch.device,
    control_cells: torch.Tensor,
    control_batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    y_pred: list[torch.Tensor] = []
    n_chunks: list[float] = []
    for index in gene_indices:
        gene = str(data.genes[index])
        predicted_expression, _predicted_latent = model.predict_response(
            control_cells,
            gene,
            control_batch_indices,
        )
        y_pred.append(
            model.predict_c_from_response(predicted_expression, control_cells)
        )
        n_chunks.append(1.0)
    return torch.stack(
        [
            torch.as_tensor(gene_indices, dtype=torch.float32, device=device),
            torch.as_tensor(padding_flags, dtype=torch.float32, device=device),
            torch.as_tensor(
                [float(data.y[index]) for index in gene_indices],
                dtype=torch.float32,
                device=device,
            ),
            torch.stack(y_pred).reshape(-1).detach().to(dtype=torch.float32),
            torch.as_tensor(n_chunks, dtype=torch.float32, device=device),
        ],
        dim=1,
    )


def _gather_final_predictions(
    pred_tensors: list[torch.Tensor],
    data: GeneBags,
    accelerator: Accelerator,
) -> pd.DataFrame:
    if pred_tensors:
        local_predictions = torch.cat(pred_tensors, dim=0)
    else:
        local_predictions = torch.empty(
            (0, 5),
            dtype=torch.float32,
            device=accelerator.device,
        )
    gathered = accelerator.gather(local_predictions)
    if not accelerator.is_main_process:
        return pd.DataFrame(columns=_FINAL_PREDICTION_COLUMNS)
    gathered = gathered.detach().cpu()
    if gathered.numel() == 0:
        return pd.DataFrame(columns=_FINAL_PREDICTION_COLUMNS)
    gathered = gathered[gathered[:, 1] < 0.5]
    rows = [
        {
            "perturbation_gene": str(data.genes[int(row[0].item())]),
            "y_true": float(row[2].item()),
            "y_pred": float(row[3].item()),
            "n_chunks": float(row[4].item()),
        }
        for row in gathered
    ]
    return pd.DataFrame(rows, columns=_FINAL_PREDICTION_COLUMNS)


def _model_inputs_for_indices(
    data: GeneBags,
    indices: list[int],
    rng: np.random.Generator,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
    *,
    pad_short: bool,
    tensor_cache: _InputTensorCache | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    genes: list[str] = []
    control_chunk_groups: list[tuple[torch.Tensor, ...]] = []
    target_expression_groups: list[tuple[torch.Tensor, ...]] = []
    target_latent_groups: list[tuple[torch.Tensor, ...]] = []
    batch_index_groups: list[tuple[torch.Tensor | None, ...]] = []
    fallback_counts: list[float] = []
    n_chunks: list[float] = []
    y_values: list[float] = []
    for index in indices:
        gene = str(data.genes[index])
        chunks = make_cell_set_chunks(
            data,
            index,
            cell_set_len=cell_set_len,
            rng=rng,
            pad_short=pad_short,
            shuffle=True,
        )
        genes.append(gene)
        control_chunk_groups.append(
            tuple(
                _cached_or_numpy_rows(
                    tensor_cache.control_input if tensor_cache is not None else None,
                    data.control_input,
                    chunk.control_indices,
                    device,
                )
                for chunk in chunks
            )
        )
        target_expression_groups.append(
            tuple(
                _cached_or_numpy_rows(
                    (
                        tensor_cache.input_bags[index]
                        if tensor_cache is not None
                        else None
                    ),
                    data.input_bags[index],
                    chunk.target_indices,
                    device,
                )
                for chunk in chunks
            )
        )
        target_latent_groups.append(
            tuple(
                _cached_or_numpy_rows(
                    (
                        tensor_cache.latent_bags[index]
                        if tensor_cache is not None
                        else None
                    ),
                    data.latent_bags[index],
                    chunk.target_indices,
                    device,
                )
                for chunk in chunks
            )
        )
        batch_index_groups.append(
            tuple(
                _cached_or_encoded_labels(
                    (
                        tensor_cache.batch_bags[index]
                        if tensor_cache is not None
                        else None
                    ),
                    chunk.target_batch,
                    chunk.target_indices,
                    batch_lookup,
                    device,
                )
                for chunk in chunks
            )
        )
        fallback_counts.append(
            float(sum(chunk.control_fallback_count for chunk in chunks))
        )
        n_chunks.append(float(len(chunks)))
        y_values.append(float(data.y[index]))
    return (
        {
            "gene": tuple(genes),
            "control_chunks": tuple(control_chunk_groups),
            "target_expression_chunks": tuple(target_expression_groups),
            "target_latent_chunks": tuple(target_latent_groups),
            "batch_index_chunks": tuple(batch_index_groups),
            "y": torch.as_tensor(y_values, dtype=torch.float32, device=device),
        },
        torch.as_tensor(fallback_counts, dtype=torch.float32, device=device),
        torch.as_tensor(n_chunks, dtype=torch.float32, device=device),
    )


def _cached_or_numpy_rows(
    cached: torch.Tensor | None,
    array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if cached is not None:
        return _take_cached_rows(cached, indices)
    return torch.as_tensor(
        array[indices].astype(np.float32),
        dtype=torch.float32,
        device=device,
    )


def _cached_or_encoded_labels(
    cached: torch.Tensor | None,
    labels: np.ndarray | None,
    indices: np.ndarray,
    batch_lookup: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    if cached is not None:
        return _take_cached_rows(cached, indices)
    return _batch_tensor(labels, batch_lookup, device)


def _take_cached_rows(cached: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    index_tensor = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=cached.device,
    )
    return cached.index_select(0, index_tensor)


def _metric_tensor_from_losses(
    losses: Mapping[str, torch.Tensor],
    fallback_counts: torch.Tensor,
    n_chunks: torch.Tensor,
) -> torch.Tensor:
    values = []
    for key in _METRIC_KEYS:
        if key == "control_fallback_count":
            values.append(fallback_counts.reshape(-1))
        elif key == "n_chunks":
            values.append(n_chunks.reshape(-1))
        else:
            values.append(losses[_MODEL_PER_GENE_KEYS[key]].reshape(-1))
    return torch.stack(values, dim=1)


def _empty_metric_sum(device: torch.device) -> torch.Tensor:
    return torch.zeros(len(_METRIC_KEYS), dtype=torch.float32, device=device)


def _reduce_metric_mean(
    metric_sum: torch.Tensor,
    metric_count: int,
    accelerator: Accelerator,
) -> dict[str, float]:
    count = torch.tensor(
        float(metric_count),
        dtype=torch.float32,
        device=metric_sum.device,
    )
    global_sum = accelerator.reduce(metric_sum, reduction="sum")
    global_count = accelerator.reduce(count, reduction="sum")
    if not accelerator.is_main_process:
        return {}
    count_value = float(global_count.detach().cpu())
    if count_value <= 0.0:
        return {}
    values = (global_sum / count_value).detach().cpu().numpy()
    return {key: float(values[index]) for index, key in enumerate(_METRIC_KEYS)}


def _prediction_tensor_from_losses(
    losses: Mapping[str, torch.Tensor],
    data: GeneBags,
    gene_indices: list[int],
    padding_flags: list[bool],
    fallback_counts: torch.Tensor,
    n_chunks: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.as_tensor(gene_indices, dtype=torch.float32, device=device),
            torch.as_tensor(padding_flags, dtype=torch.float32, device=device),
            torch.as_tensor(
                [float(data.y[index]) for index in gene_indices],
                dtype=torch.float32,
                device=device,
            ),
            losses["pred_y"].reshape(-1).detach().to(dtype=torch.float32),
            losses["obs_y"].reshape(-1).detach().to(dtype=torch.float32),
            fallback_counts.reshape(-1).detach(),
            n_chunks.reshape(-1).detach(),
        ],
        dim=1,
    )


def _gather_predictions(
    pred_tensors: list[torch.Tensor],
    data: GeneBags,
    accelerator: Accelerator,
) -> pd.DataFrame:
    if pred_tensors:
        local_predictions = torch.cat(pred_tensors, dim=0)
    else:
        local_predictions = torch.empty(
            (0, 7),
            dtype=torch.float32,
            device=accelerator.device,
        )
    gathered = accelerator.gather(local_predictions)
    if not accelerator.is_main_process:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)
    gathered = gathered.detach().cpu()
    if gathered.numel() == 0:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)
    gathered = gathered[gathered[:, 1] < 0.5]
    rows = [
        {
            "perturbation_gene": str(data.genes[int(row[0].item())]),
            "y_true": float(row[2].item()),
            "y_pred": float(row[3].item()),
            "y_obs_anchor": float(row[4].item()),
            "control_fallback_count": float(row[5].item()),
            "n_chunks": float(row[6].item()),
        }
        for row in gathered
    ]
    return pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)


def _batch_indices(batch: Any) -> list[int]:
    if isinstance(batch, Mapping):
        return _batch_indices(batch["index"])
    if isinstance(batch, torch.Tensor):
        return [int(value.item()) for value in batch.reshape(-1)]
    if isinstance(batch, list | tuple):
        return [int(value) for value in batch]
    return [int(batch)]


def _batch_padding_flags(batch: Any) -> list[bool]:
    if isinstance(batch, Mapping):
        values = batch.get("is_padding")
        if values is None:
            return [False] * len(_batch_indices(batch))
        if isinstance(values, torch.Tensor):
            return [bool(value.item()) for value in values.reshape(-1)]
        if isinstance(values, list | tuple):
            return [bool(value) for value in values]
        return [bool(values)]
    return [False] * len(_batch_indices(batch))


def _batch_tensor(
    labels: np.ndarray | None,
    batch_lookup: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    batch_encoded = encode_batch_labels(labels, batch_lookup)
    if batch_encoded is None:
        return None
    return torch.as_tensor(batch_encoded, dtype=torch.long, device=device)


def _prefix(row: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _is_better_metric(value: float, best_value: float, *, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if not math.isfinite(best_value):
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    msg = f"Unknown selection mode: {mode}"
    raise ValueError(msg)


def _reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def _global_peak_gpu_memory_allocated_mb(accelerator: Accelerator) -> float:
    local_peak = math.nan
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        local_peak = torch.cuda.max_memory_allocated(accelerator.device) / (1024**2)
    local_peak_tensor = torch.tensor(
        [local_peak],
        dtype=torch.float32,
        device=accelerator.device,
    )
    gathered = accelerator.gather(local_peak_tensor).detach().cpu().tolist()
    if not accelerator.is_main_process:
        return math.nan
    finite_values = [float(value) for value in gathered if math.isfinite(float(value))]
    if not finite_values:
        return math.nan
    return max(finite_values)


def _write_csv_if_main(
    frame: pd.DataFrame,
    path: Path,
    accelerator: Accelerator,
) -> None:
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_model_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    path: Path,
    metadata: dict[str, object],
) -> None:
    if not accelerator.is_main_process:
        return
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(unwrapped.state_dict(), path / "pytorch_model.bin")
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_authorized_model_checkpoint(
    path: Path,
    authority: ArtifactAuthority,
    *,
    map_location: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Load audited best/final weights only after exact authority validation."""
    metadata_path = path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "audited checkpoint requires valid schema-v2 metadata"
        ) from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        raise ValueError("audited checkpoint requires valid schema-v2 metadata")
    expected = authority.metadata()
    if not _artifact_metadata_is_authorized(metadata) or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("checkpoint authority does not match the current fold")
    checkpoint_kind = metadata.get("checkpoint_kind")
    if checkpoint_kind not in {"best", "frozen_selected"}:
        raise ValueError("checkpoint authority has an invalid checkpoint kind")
    weights = torch.load(
        path / "pytorch_model.bin",
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(weights, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in weights.items()
    ):
        raise ValueError("audited checkpoint weights are not a state dictionary")
    return weights


def _write_split_artifact(data: GeneBags, split: GeneSplit, path: Path) -> None:
    rows = []
    for split_name, indices in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        for index in indices:
            rows.append(
                {
                    "split": split_name,
                    "perturbation_gene": str(data.genes[index]),
                    "y": float(data.y[index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _infer_pert_dim(known_vectors: dict[str, np.ndarray]) -> int:
    if not known_vectors:
        msg = "state.pert_dim is required when no known perturbation vectors are given"
        raise ValueError(msg)
    first = next(iter(known_vectors.values()))
    return int(first.shape[0])


if __name__ == "__main__":
    main()
