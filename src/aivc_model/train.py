"""Train the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list, gather_object, set_seed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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
    with_scvi_teacher_latents,
)
from dependency_baseline.metrics import ranking_metrics, regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STATE-ready AIVC A->B->C.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    accelerator = _make_accelerator(config)
    paths = run_training(config, accelerator=accelerator)
    if accelerator.is_main_process:
        print(f"run dir: {paths['run_dir']}")
        print(f"train log: {paths['train_log']}")
        print(f"test metrics: {paths['test_metrics']}")


def run_training(
    config: AivcConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Path]:
    """Run one train/val/test STATE-ready AIVC experiment."""
    accelerator = accelerator or _make_accelerator(config)
    set_seed(config.train.seed)
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
    data = _with_rank_safe_scvi_teacher(config, data, split, artifacts_dir, accelerator)
    external = load_external_gene_bags(config, data, artifacts_dir)

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
    train_loader = _gene_loader(split.train, shuffle=True, seed=config.train.seed)
    val_loader = _gene_loader(split.val, shuffle=False, seed=config.train.seed)
    test_loader = _gene_loader(eval_indices, shuffle=False, seed=config.train.seed)
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
    train_context = (
        accelerator.join_uneven_inputs([model])
        if accelerator.num_processes > 1
        else nullcontext()
    )
    with train_context:
        for epoch in range(1, config.train.max_epochs + 1):
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
            if accelerator.is_main_process:
                last_val_loss = float(val_row.get("total_loss", math.nan))
                row = {
                    "epoch": epoch,
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


class _GeneIndexDataset(Dataset[int]):
    def __init__(self, indices: np.ndarray) -> None:
        self._indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> int:
        return self._indices[index]


def _make_accelerator(config: AivcConfig) -> Accelerator:
    dataloader_config = DataLoaderConfiguration(
        even_batches=False,
        use_seedable_sampler=True,
        data_seed=config.train.seed,
    )
    return Accelerator(
        cpu=config.train.device == "cpu",
        dataloader_config=dataloader_config,
    )


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
    artifacts_dir: Path,
    accelerator: Accelerator,
) -> GeneBags:
    if config.projector.teacher != "scvi":
        return data
    if accelerator.is_main_process:
        data = with_scvi_teacher_latents(
            config,
            data,
            split,
            artifacts_dir,
            fit_teacher=True,
        )
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        data = with_scvi_teacher_latents(
            config,
            data,
            split,
            artifacts_dir,
            fit_teacher=False,
        )
    accelerator.wait_for_everyone()
    return data


def _gene_loader(indices: np.ndarray, *, shuffle: bool, seed: int) -> DataLoader[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _GeneIndexDataset(indices),
        batch_size=1,
        shuffle=shuffle,
        generator=generator,
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
    indices: DataLoader[int],
    weights: LossWeights,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    cell_set_len: int,
    accelerator: Accelerator,
    batch_lookup: dict[str, int],
) -> dict[str, float]:
    model.train()
    rows = []
    for index in indices:
        loss_row, _pred_y, _obs_y = _loss_for_index(
            model,
            data,
            _batch_index(index),
            weights,
            rng,
            cell_set_len,
            accelerator.device,
            batch_lookup,
            pad_short=True,
        )
        optimizer.zero_grad(set_to_none=True)
        total = loss_row["total_tensor"]
        if not isinstance(total, torch.Tensor):
            msg = "Expected tensor loss for backward"
            raise TypeError(msg)
        accelerator.backward(total)
        optimizer.step()
        rows.append({k: v for k, v in loss_row.items() if k != "total_tensor"})
    return _mean_rows(_gather_rows(rows, accelerator))


def _evaluate(
    model: torch.nn.Module,
    data: GeneBags,
    indices: DataLoader[int],
    weights: LossWeights,
    rng: np.random.Generator,
    cell_set_len: int,
    accelerator: Accelerator,
    batch_lookup: dict[str, int],
    pad_short: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    pred_rows = []
    with torch.no_grad():
        for index in indices:
            gene_index = _batch_index(index)
            loss_row, pred_y, obs_y = _loss_for_index(
                model,
                data,
                gene_index,
                weights,
                rng,
                cell_set_len,
                accelerator.device,
                batch_lookup,
                pad_short=pad_short,
            )
            rows.append({k: v for k, v in loss_row.items() if k != "total_tensor"})
            pred_rows.append(
                {
                    "perturbation_gene": str(data.genes[gene_index]),
                    "y_true": float(data.y[gene_index]),
                    "y_pred": float(pred_y),
                    "y_obs_anchor": float(obs_y),
                    "control_fallback_count": float(loss_row["control_fallback_count"]),
                    "n_chunks": float(loss_row["n_chunks"]),
                }
            )
    all_rows = _gather_rows(rows, accelerator)
    all_pred_rows = _gather_rows(pred_rows, accelerator)
    summary = _mean_rows(all_rows)
    prediction_columns = [
        "perturbation_gene",
        "y_true",
        "y_pred",
        "y_obs_anchor",
        "control_fallback_count",
        "n_chunks",
    ]
    predictions = pd.DataFrame(all_pred_rows, columns=prediction_columns)
    if accelerator.is_main_process and not predictions.empty:
        y_true = predictions["y_true"].to_numpy(dtype=np.float64)
        y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
        summary.update(regression_metrics(y_true, y_pred))
        summary.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
    return summary, predictions


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


def _batch_index(index: Any) -> int:
    if isinstance(index, torch.Tensor):
        return int(index.reshape(-1)[0].item())
    if isinstance(index, list | tuple):
        return int(index[0])
    return int(index)


def _batch_tensor(
    labels: np.ndarray | None,
    batch_lookup: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    batch_encoded = encode_batch_labels(labels, batch_lookup)
    if batch_encoded is None:
        return None
    return torch.as_tensor(batch_encoded, dtype=torch.long, device=device)


def _gather_rows(
    rows: list[dict[str, float] | dict[str, object]],
    accelerator: Accelerator,
) -> list[dict[str, Any]]:
    gathered = gather_object(rows)
    if not accelerator.is_main_process:
        return []
    return [row for row in gathered if isinstance(row, dict)]


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _prefix(row: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _is_better_loss(value: float, best_value: float) -> bool:
    return math.isfinite(value) and value < best_value


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
