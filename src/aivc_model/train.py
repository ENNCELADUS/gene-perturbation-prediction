"""Train the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
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
}
_PREDICTION_COLUMNS = [
    "perturbation_gene",
    "y_true",
    "y_pred",
    "y_obs_anchor",
    "control_fallback_count",
    "n_chunks",
]


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

    train_expr = np.vstack(
        [data.control_input, *[data.input_bags[i] for i in split.train]]
    )
    train_latent = np.vstack(
        [data.control_latent, *[data.latent_bags[i] for i in split.train]]
    )
    projector_weight, projector_bias = fit_linear_projector(
        train_expr,
        train_latent,
        config.projector.ridge_alpha,
    )
    featureizer = fit_fixed_gmm(
        tuple(data.latent_bags[i] for i in split.train),
        data.control_latent,
        n_components=config.gmm.n_components,
        covariance_floor=config.gmm.covariance_floor,
        random_state=config.train.seed,
        max_fit_cells=config.gmm.max_fit_cells,
    )
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
    weights = _loss_weights(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    rng = np.random.default_rng(config.train.seed + accelerator.process_index)

    logs: list[dict[str, float]] = []
    best_val_loss = math.inf
    last_val_loss = math.nan
    for epoch in range(1, config.train.max_epochs + 1):
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
        )
        val_row, _val_predictions = _evaluate(
            model,
            data,
            val_loader,
            weights,
            rng,
            config.train.cell_set_len,
            accelerator,
            batch_lookup,
            pad_short=True,
        )
        gpu_peak_memory_allocated_mb = _global_peak_gpu_memory_allocated_mb(accelerator)
        if accelerator.is_main_process:
            last_val_loss = float(val_row.get("total_loss", math.nan))
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
            if _is_better_loss(last_val_loss, best_val_loss):
                best_val_loss = last_val_loss
                _save_model_checkpoint(
                    accelerator,
                    model,
                    models_dir / "best",
                    {
                        "checkpoint_kind": "best",
                        "epoch": epoch,
                        "selection_metric": "val_total_loss",
                        "selection_mode": "min",
                        "metric_value": best_val_loss,
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
    test_row, test_predictions = _evaluate(
        model,
        eval_data,
        test_loader,
        weights,
        rng,
        config.train.cell_set_len,
        accelerator,
        batch_lookup,
        pad_short=False,
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
                "selection_metric": "val_total_loss",
                "selection_mode": "min",
                "metric_value": last_val_loss,
                "best_metric_value": best_val_loss,
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


def _loss_weights(config: AivcConfig) -> LossWeights:
    return LossWeights(
        latent_mean_delta=config.loss.latent_mean_delta_weight,
        latent_energy=config.loss.latent_energy_weight,
        hvg_mean_delta=config.loss.hvg_mean_delta_weight,
        hvg_energy=config.loss.hvg_energy_weight,
        pred_c=config.loss.pred_c_weight,
        obs_c=config.loss.obs_c_weight,
        occupancy=config.loss.occupancy_weight,
    )


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
    )


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
        model_inputs, fallback_counts, n_chunks = _model_inputs_for_indices(
            data,
            gene_indices,
            rng,
            cell_set_len,
            accelerator.device,
            batch_lookup,
            pad_short=True,
        )
        losses = model(weights=weights, **model_inputs)
        total_loss = losses["total"]
        if not isinstance(total_loss, torch.Tensor):
            msg = "Expected tensor loss for backward"
            raise TypeError(msg)
        accelerator.backward(total_loss)
        optimizer.step()
        metric_sum = metric_sum + _metric_tensor_from_losses(
            losses,
            fallback_counts,
            n_chunks,
        ).detach().sum(dim=0)
        metric_count += len(gene_indices)
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
            model_inputs, fallback_counts, n_chunks = _model_inputs_for_indices(
                data,
                gene_indices,
                rng,
                cell_set_len,
                accelerator.device,
                batch_lookup,
                pad_short=pad_short,
            )
            losses = model(weights=weights, **model_inputs)
            metric_tensor = _metric_tensor_from_losses(
                losses,
                fallback_counts,
                n_chunks,
            ).detach()
            valid_mask = torch.tensor(
                [not flag for flag in padding_flags],
                dtype=torch.bool,
                device=accelerator.device,
            )
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
    return summary, predictions


def _model_inputs_for_indices(
    data: GeneBags,
    indices: list[int],
    rng: np.random.Generator,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
    *,
    pad_short: bool,
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
                torch.as_tensor(
                    data.control_input[chunk.control_indices].astype(np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                for chunk in chunks
            )
        )
        target_expression_groups.append(
            tuple(
                torch.as_tensor(
                    data.input_bags[index][chunk.target_indices].astype(np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                for chunk in chunks
            )
        )
        target_latent_groups.append(
            tuple(
                torch.as_tensor(
                    data.latent_bags[index][chunk.target_indices].astype(np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                for chunk in chunks
            )
        )
        batch_index_groups.append(
            tuple(
                _batch_tensor(chunk.target_batch, batch_lookup, device)
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


def _loss_for_index(
    model: torch.nn.Module,
    data: GeneBags,
    index: int,
    weights: LossWeights,
    rng: np.random.Generator,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
    pad_short: bool,
) -> tuple[dict[str, float | torch.Tensor], float, float]:
    gene = str(data.genes[index])
    chunks = make_cell_set_chunks(
        data,
        index,
        cell_set_len=cell_set_len,
        rng=rng,
        pad_short=pad_short,
        shuffle=True,
    )
    control_chunks = tuple(
        torch.as_tensor(
            data.control_input[chunk.control_indices].astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        for chunk in chunks
    )
    target_expression_chunks = tuple(
        torch.as_tensor(
            data.input_bags[index][chunk.target_indices].astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        for chunk in chunks
    )
    target_latent_chunks = tuple(
        torch.as_tensor(
            data.latent_bags[index][chunk.target_indices].astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        for chunk in chunks
    )
    batch_index_chunks = tuple(
        _batch_tensor(chunk.target_batch, batch_lookup, device) for chunk in chunks
    )
    y = torch.tensor(float(data.y[index]), dtype=torch.float32, device=device)
    losses = model(
        gene=gene,
        control_chunks=control_chunks,
        target_expression_chunks=target_expression_chunks,
        target_latent_chunks=target_latent_chunks,
        batch_index_chunks=batch_index_chunks,
        y=y,
        weights=weights,
    )
    fallback_count = sum(chunk.control_fallback_count for chunk in chunks)
    row: dict[str, float | torch.Tensor] = {
        "total_tensor": losses["total"],
        "total_loss": float(losses["total"].detach().cpu()),
        "hvg_mean_delta": float(losses["hvg_mean_delta"].detach().cpu()),
        "hvg_energy": float(losses["hvg_energy"].detach().cpu()),
        "latent_mean_delta": float(losses["latent_mean_delta"].detach().cpu()),
        "latent_energy": float(losses["latent_energy"].detach().cpu()),
        "pred_c": float(losses["pred_c"].detach().cpu()),
        "obs_c": float(losses["obs_c"].detach().cpu()),
        "occupancy": float(losses["occupancy"].detach().cpu()),
        "control_fallback_count": float(fallback_count),
        "n_chunks": float(len(chunks)),
    }
    return (
        row,
        float(losses["pred_y"].detach().cpu()),
        float(losses["obs_y"].detach().cpu()),
    )


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


def _is_better_loss(value: float, best_value: float) -> bool:
    return math.isfinite(value) and value < best_value


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
