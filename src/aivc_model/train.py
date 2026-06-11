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
    ExpressionToLatentProjector,
    FixedGMMFeatureizer,
    LossWeights,
    MLPHead,
    PerturbationVectorAdapter,
    StateForwardAdapter,
    fit_fixed_gmm,
    load_state_model,
)
from aivc_model.prepare import (
    AivcConfig,
    GeneBags,
    GeneSplit,
    encode_batch_labels,
    fit_linear_projector,
    load_config,
    load_external_gene_bags,
    load_gene_bags,
    load_perturbation_vectors,
    load_state_batch_lookup,
    make_cell_set_chunks,
    make_gene_split,
    with_cached_scvi_teacher_latents,
)
from dependency_baseline.metrics import ranking_metrics, regression_metrics

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
) -> dict[str, Path]:
    """Run one train/val/test STATE-ready AIVC experiment."""
    accelerator = accelerator or _make_accelerator(config)
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
    eval_data = external.data if external is not None else data
    eval_indices = (
        np.arange(len(eval_data.genes), dtype=np.int64)
        if external is not None
        else split.test
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
    test_loader = _gene_loader(
        eval_indices,
        shuffle=False,
        seed=config.train.seed,
        gene_batch_size=config.train.gene_batch_size,
        world_size=accelerator.num_processes,
    )
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
    )
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
            tensor_cache=input_tensor_cache,
        )
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

    evaluation_scope = (
        f"external:{config.external_test.name}"
        if external is not None and config.external_test is not None
        else "internal_test"
    )
    test_row, test_predictions = _evaluate_prediction_only_final(
        model,
        eval_data,
        test_loader,
        config.train.cell_set_len,
        accelerator,
        batch_lookup,
    )
    if accelerator.is_main_process:
        test_row = {"evaluation_scope": evaluation_scope, **test_row}
        test_predictions.insert(0, "evaluation_scope", evaluation_scope)
        unwrapped = accelerator.unwrap_model(model)
        test_predictions["perturbation_has_known_vector"] = test_predictions[
            "perturbation_gene"
        ].map(unwrapped.perturbations.has_known_vector)
        if external is not None:
            test_predictions = test_predictions.merge(
                external.data.metadata,
                on="perturbation_gene",
                how="left",
                suffixes=("", "_metadata"),
            )
            (artifacts_dir / "external_test_qa.json").write_text(
                json.dumps(external.qa, indent=2),
                encoding="utf-8",
            )
        _write_csv_if_main(
            pd.DataFrame([test_row]),
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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit or load the run-local ridge projector cache."""
    cache_dir = artifacts_dir / "ridge_projector_fit"
    metadata = _projector_cache_metadata(config, data, split)
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
) -> FixedGMMFeatureizer:
    """Fit or load the run-local fixed GMM featureizer cache."""
    cache_dir = artifacts_dir / "fixed_gmm_fit"
    metadata = _fixed_gmm_cache_metadata(config, data, split)
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
    return metadata == expected_metadata


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
) -> dict[str, object]:
    return {
        "version": 1,
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


def _fixed_gmm_cache_metadata(
    config: AivcConfig,
    data: GeneBags,
    split: GeneSplit,
) -> dict[str, object]:
    return {
        "version": 1,
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


def _build_model(
    config: AivcConfig,
    data: GeneBags,
    featureizer: torch.nn.Module,
    projector_weight: np.ndarray,
    projector_bias: np.ndarray,
    extra_genes: tuple[str, ...] = (),
    emit_checkpoint_output: bool = True,
) -> AivcModel:
    pert_dim = config.state.pert_dim
    known_vectors = load_perturbation_vectors(config.state.known_perturbation_vectors)
    if pert_dim is None:
        pert_dim = _infer_pert_dim(known_vectors)
    output_dim = config.state.output_dim or data.input_dim
    state_model = load_state_model(
        backend=config.state.backend,
        checkpoint_path=config.state.checkpoint_path,
        input_dim=config.state.input_dim or data.input_dim,
        output_dim=output_dim,
        pert_dim=pert_dim,
        emit_checkpoint_output=emit_checkpoint_output,
    )
    perturbations = PerturbationVectorAdapter(
        sorted({*(str(gene) for gene in data.genes), *extra_genes}),
        known_vectors,
        pert_dim,
    )
    projector = ExpressionToLatentProjector(
        projector_weight,
        projector_bias,
        trainable=config.projector.trainable,
    )
    c_head = MLPHead(
        input_dim=featureizer.output_dim,
        hidden_units=config.model.c_hidden_units,
        dropout=config.model.dropout,
    )
    return AivcModel(
        state_adapter=StateForwardAdapter(state_model),
        perturbations=perturbations,
        projector=projector,
        featureizer=featureizer,
        c_head=c_head,
        control_expression_mean=data.control_input.mean(axis=0).astype(np.float32),
        control_latent_mean=data.control_latent.mean(axis=0).astype(np.float32),
        freeze_state=config.train.freeze_state,
    )


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        msg = "AIVC model has no trainable parameters"
        raise ValueError(msg)
    return parameters


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
    tensor_cache: _InputTensorCache | None = None,
) -> dict[str, float]:
    model.train()
    metric_sum = _empty_metric_sum(accelerator.device)
    metric_count = 0
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
        optimizer.step()
        metric_tensor = _metric_tensor_from_losses(
            losses,
            fallback_counts,
            n_chunks,
        ).detach()
        if valid_mask.any():
            metric_sum = metric_sum + metric_tensor[valid_mask].sum(dim=0)
            metric_count += int(valid_mask.sum().item())
    return _reduce_metric_mean(metric_sum, metric_count, accelerator)


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
        _pred_expression, predicted_latent = model.predict_bag(
            control_cells,
            gene,
            control_batch_indices,
        )
        y_pred.append(model.predict_c_from_latent(predicted_latent))
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
