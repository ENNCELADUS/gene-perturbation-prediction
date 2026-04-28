"""Train scGPT gene-score model with Hugging Face Accelerate."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.scgpt.data import GeneScoreDataset, collate_gene_score_batch
from src.scgpt.losses import GeneScoreLoss, GeneScoreLossConfig, build_gene_score_loss
from src.scgpt.model import GeneScoreModel
from src.utils.data import get_condition_splits, load_adata
from src.utils.distributed import disable_tqdm, is_primary_rank, log_primary_info
from src.utils.metrics import compute_gene_metrics
from src.utils.runtime import AccelerateRuntime

LOGGER = logging.getLogger(__name__)
VALIDATION_TOP_K_VALUES = [1, 5, 10]
TRAINING_STEP_FIELDS = [
    "Epoch",
    "Epoch Time",
    "Train Loss",
    "Val Loss",
    "Val Recall@1",
    "Val Recall@5",
    "Val Recall@10",
    "Val NDCG@1",
    "Val NDCG@5",
    "Val NDCG@10",
    "Val MRR",
    "GPU Max Allocated",
]


def run(config: dict) -> dict:
    """Run config-driven scGPT training."""
    runtime = AccelerateRuntime(config)
    adata = load_adata(config["data_config"]["h5ad_path"])
    split = get_condition_splits(config)
    pretrained_dir = Path(config["model_config"].get("pretrained_dir", "model/scGPT"))
    with (pretrained_dir / "vocab.json").open() as handle:
        vocab = json.load(handle)
    dataset = _build_dataset(
        adata=adata,
        conditions=split["train"],
        vocab=vocab,
        config=config,
    )
    validation_dataset = _build_dataset(
        adata=adata,
        conditions=split["validation"],
        vocab=vocab,
        config=config,
    )
    if len(validation_dataset) == 0:
        validation_dataset = None
    loader = _build_loader(
        dataset=dataset,
        vocab=vocab,
        n_genes=adata.n_vars,
        config=config,
        shuffle=True,
    )
    validation_loader = (
        _build_loader(
            dataset=validation_dataset,
            vocab=vocab,
            n_genes=adata.n_vars,
            config=config,
            shuffle=False,
        )
        if validation_dataset is not None
        else None
    )
    model = _build_model(config, adata.n_vars, dataset.gene_ids, runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training_config", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training_config", {}).get("weight_decay", 0.01)),
    )
    loss_fn = _build_loss(config)
    if validation_loader is None:
        model, optimizer, loader = runtime.prepare(model, optimizer, loader)
    else:
        model, optimizer, loader, validation_loader = runtime.prepare(
            model,
            optimizer,
            loader,
            validation_loader,
        )
    epochs = int(config.get("training_config", {}).get("epochs", 1))
    max_grad_norm = float(config.get("training_config", {}).get("max_grad_norm", 1.0))
    progress_disabled = disable_tqdm(config)
    step_log_path = _initialize_training_step_log(config)
    early_stopping = _build_early_stopping(config)
    checkpoint_state = _build_checkpoint_state(config, early_stopping)
    epochs_trained = 0
    for epoch_idx in range(epochs):
        epoch_number = epoch_idx + 1
        epochs_trained = epoch_number
        epoch_started_at = time.perf_counter()
        log_primary_info(
            LOGGER,
            "scGPT train epoch %d/%d started",
            epoch_number,
            epochs,
        )
        model.train()
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        n_batches = 0
        batch_iter = tqdm(
            loader,
            desc=f"scgpt train epoch {epoch_number}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=progress_disabled,
            leave=False,
        )
        for batch in batch_iter:
            optimizer.zero_grad()
            logits = _forward(model, batch)
            targets = batch["targets"].to(logits.device)
            loss = loss_fn(logits, targets)
            loss_value = float(loss.detach().cpu())
            batch_size = int(targets.shape[0])
            epoch_loss_sum += loss_value * batch_size
            epoch_example_count += batch_size
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss_value:.4f}")
            runtime.backward(loss)
            runtime.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        mean_loss = _mean_loss(
            runtime=runtime,
            loss_sum=epoch_loss_sum,
            example_count=epoch_example_count,
            device=runtime.device,
        )
        val_loss, val_metrics = _evaluate_validation(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            runtime=runtime,
            top_k_values=VALIDATION_TOP_K_VALUES,
        )
        epoch_time = time.perf_counter() - epoch_started_at
        gpu_max_allocated = _gpu_max_allocated_summary(runtime.device)
        log_primary_info(
            LOGGER,
            "scGPT train epoch %d/%d complete: batches=%d mean_loss=%.6f "
            "val_loss=%s gpu_max_allocated=%s",
            epoch_number,
            epochs,
            n_batches,
            mean_loss,
            _format_optional_loss(val_loss),
            gpu_max_allocated,
        )
        _append_training_step_log(
            path=step_log_path,
            epoch=epoch_number,
            epoch_time=epoch_time,
            train_loss=mean_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            gpu_max_allocated=gpu_max_allocated,
        )
        checkpoint_monitor_value = _monitor_value(
            monitor=str(checkpoint_state["monitor"]),
            train_loss=mean_loss,
            val_loss=val_loss,
        )
        _save_checkpoint_for_epoch(
            runtime=runtime,
            model=model,
            state=checkpoint_state,
            epoch=epoch_number,
            monitor_value=checkpoint_monitor_value,
        )
        monitor_value = _monitor_value(
            monitor=early_stopping["monitor"] if early_stopping else "",
            train_loss=mean_loss,
            val_loss=val_loss,
        )
        if early_stopping and _should_stop_early(early_stopping, monitor_value):
            log_primary_info(
                LOGGER,
                "scGPT early stopping triggered at epoch %d: monitor=%s best=%.6f",
                epoch_number,
                early_stopping["monitor"],
                early_stopping["best"],
            )
            break
    checkpoint_path = config["run_config"].get("save_checkpoint_path")
    if checkpoint_path and not bool(checkpoint_state["save_best_only"]):
        runtime.save_state_dict(model, checkpoint_path)
    elif checkpoint_path and not bool(checkpoint_state["saved"]):
        runtime.save_state_dict(model, checkpoint_path)
    return {
        "checkpoint_path": checkpoint_path,
        "n_train": len(dataset),
        "epochs_trained": epochs_trained,
        "best_epoch": checkpoint_state["best_epoch"],
        "best_monitor": checkpoint_state["monitor"],
        "best_monitor_value": checkpoint_state["best"],
    }


def _build_dataset(
    adata,
    conditions: list[str],
    vocab: dict,
    config: dict,
) -> GeneScoreDataset:
    return GeneScoreDataset(
        adata=adata,
        conditions=conditions,
        vocab=vocab,
        n_bins=int(config["model_config"].get("preprocess_binning", 51)),
        condition_key=config["data_config"].get("condition_key", "condition"),
        control_key=config["data_config"].get("control_key", "control"),
        n_control_samples=int(config["data_config"].get("control_n_samples", 8)),
        seed=int(config["run_config"].get("seed", 42)),
    )


def _build_loader(
    dataset: GeneScoreDataset,
    vocab: dict,
    n_genes: int,
    config: dict,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config.get("training_config", {}).get("batch_size", 32)),
        shuffle=shuffle,
        collate_fn=lambda batch: collate_gene_score_batch(batch, vocab, n_genes),
        num_workers=int(config["data_config"].get("num_workers", 0)),
    )


def _build_model(
    config: dict,
    n_genes: int,
    gene_ids,
    device: torch.device,
) -> GeneScoreModel:
    model_config = config["model_config"]
    pretrained_dir = Path(model_config.get("pretrained_dir", "model/scGPT"))
    return GeneScoreModel(
        n_genes=n_genes,
        checkpoint_path=pretrained_dir / "best_model.pt",
        vocab_path=pretrained_dir / "vocab.json",
        args_path=pretrained_dir / "args.json",
        score_gene_ids=gene_ids,
        freeze_encoder=bool(model_config.get("freeze_encoder", True)),
        freeze_layers_up_to=int(model_config.get("freeze_layers_up_to", 10)),
        score_mode=str(model_config.get("score_mode", "dot")),
        head_hidden_dim=int(model_config.get("head_hidden_dim", 512)),
        head_dropout=float(model_config.get("head_dropout", 0.2)),
        use_fast_transformer=bool(model_config.get("use_fast_transformer", False)),
        fast_transformer_backend=str(
            model_config.get("fast_transformer_backend", "flash")
        ),
        device=device,
    )


def _build_loss(config: dict) -> GeneScoreLoss:
    training_config = config.get("training_config", {})
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be a mapping")
    loss_config = training_config.get("loss")
    if loss_config is not None and not isinstance(loss_config, dict):
        raise ValueError("training_config.loss must be a mapping")
    return build_gene_score_loss(GeneScoreLossConfig.from_mapping(loss_config))


def _build_early_stopping(config: dict) -> dict[str, float | int | str] | None:
    training_config = config.get("training_config", {})
    if not isinstance(training_config, dict):
        return None
    early_stopping = training_config.get("early_stopping")
    if early_stopping is None:
        return None
    if not isinstance(early_stopping, dict):
        raise ValueError("training_config.early_stopping must be a mapping")
    patience = int(early_stopping.get("patience", 3))
    if patience < 1:
        raise ValueError("early_stopping.patience must be at least 1")
    monitor = str(early_stopping.get("monitor", "val_loss"))
    if monitor not in {"val_loss", "train_loss"}:
        raise ValueError("early_stopping.monitor must be 'val_loss' or 'train_loss'")
    return {
        "monitor": monitor,
        "patience": patience,
        "min_delta": float(early_stopping.get("min_delta", 0.0)),
        "best": float("inf"),
        "epochs_without_improvement": 0,
    }


def _build_checkpoint_state(
    config: dict,
    early_stopping: dict[str, float | int | str] | None,
) -> dict[str, float | int | str | bool | None]:
    run_config = config.get("run_config", {})
    if not isinstance(run_config, dict):
        raise ValueError("run_config must be a mapping")
    monitor = str(
        run_config.get(
            "save_best_monitor",
            early_stopping["monitor"] if early_stopping else "val_loss",
        )
    )
    if monitor not in {"val_loss", "train_loss"}:
        raise ValueError(
            "run_config.save_best_monitor must be 'val_loss' or 'train_loss'"
        )
    return {
        "save_best_only": bool(run_config.get("save_best_only", False)),
        "checkpoint_path": run_config.get("save_checkpoint_path"),
        "monitor": monitor,
        "best": float("inf"),
        "best_epoch": None,
        "saved": False,
    }


def _save_checkpoint_for_epoch(
    runtime: AccelerateRuntime,
    model: torch.nn.Module,
    state: dict[str, float | int | str | bool | None],
    epoch: int,
    monitor_value: float | None,
) -> None:
    if not bool(state["save_best_only"]) or monitor_value is None:
        return
    if monitor_value >= float(state["best"]):
        return

    checkpoint_path = state.get("checkpoint_path")
    if checkpoint_path is None:
        return
    runtime.save_state_dict(model, str(checkpoint_path))
    state["best"] = monitor_value
    state["best_epoch"] = epoch
    state["saved"] = True
    log_primary_info(
        LOGGER,
        "scGPT saved best checkpoint at epoch %d: monitor=%s value=%.6f",
        epoch,
        state["monitor"],
        monitor_value,
    )


def _monitor_value(
    monitor: str,
    train_loss: float,
    val_loss: float | None,
) -> float | None:
    if monitor == "train_loss":
        return train_loss
    if monitor == "val_loss":
        return val_loss
    return None


def _should_stop_early(
    state: dict[str, float | int | str],
    monitor_value: float | None,
) -> bool:
    if monitor_value is None:
        return False

    best = float(state["best"])
    min_delta = float(state["min_delta"])
    if monitor_value < best - min_delta:
        state["best"] = monitor_value
        state["epochs_without_improvement"] = 0
        return False

    state["epochs_without_improvement"] = int(
        state["epochs_without_improvement"]
    ) + 1
    return int(state["epochs_without_improvement"]) >= int(state["patience"])


def _evaluate_validation(
    model: torch.nn.Module,
    loader: DataLoader | None,
    loss_fn: GeneScoreLoss,
    runtime: AccelerateRuntime,
    top_k_values: list[int],
) -> tuple[float | None, dict[str, float | int]]:
    if loader is None:
        return None, {}

    model.eval()
    loss_sum = 0.0
    example_count = 0
    scores = []
    target_indices = []
    with torch.no_grad():
        for batch in loader:
            logits = _forward(model, batch)
            target_matrix = batch["targets"].to(logits.device)
            loss = loss_fn(logits, target_matrix)
            batch_size = int(target_matrix.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            example_count += batch_size
            gathered_logits = _gather_for_metrics(runtime, logits)
            gathered_targets = _gather_for_metrics(runtime, target_matrix)
            scores.extend(row.cpu().numpy() for row in gathered_logits)
            target_indices.extend(_target_indices_from_matrix(gathered_targets.cpu()))
    if example_count == 0:
        return None, {}
    val_loss = _mean_loss(
        runtime=runtime,
        loss_sum=loss_sum,
        example_count=example_count,
        device=runtime.device,
    )
    metrics = compute_gene_metrics(scores, target_indices, top_k_values)
    return val_loss, metrics


def _gather_for_metrics(
    runtime: AccelerateRuntime,
    tensor: torch.Tensor,
) -> torch.Tensor:
    gather_for_metrics = getattr(runtime, "gather_for_metrics", None)
    if callable(gather_for_metrics):
        return gather_for_metrics(tensor)
    return tensor


def _mean_loss(
    runtime: AccelerateRuntime,
    loss_sum: float,
    example_count: int,
    device: torch.device,
) -> float:
    if example_count == 0:
        return 0.0
    gather_for_metrics = getattr(runtime, "gather_for_metrics", None)
    if not callable(gather_for_metrics):
        return loss_sum / example_count

    stats = torch.tensor([loss_sum, float(example_count)], device=device)
    gathered = gather_for_metrics(stats).detach().cpu().flatten()
    total_loss = float(gathered[0::2].sum())
    total_examples = float(gathered[1::2].sum())
    return total_loss / total_examples if total_examples else 0.0


def _initialize_training_step_log(config: dict) -> Path | None:
    path = _training_step_log_path(config)
    if path is None or not is_primary_rank():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_STEP_FIELDS)
        writer.writeheader()
    return path


def _append_training_step_log(
    path: Path | None,
    epoch: int,
    epoch_time: float,
    train_loss: float,
    val_loss: float | None,
    val_metrics: dict[str, float | int],
    gpu_max_allocated: str,
) -> None:
    if path is None or not is_primary_rank():
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_STEP_FIELDS)
        writer.writerow(
            {
                "Epoch": epoch,
                "Epoch Time": f"{epoch_time:.3f}",
                "Train Loss": f"{train_loss:.6f}",
                "Val Loss": _format_optional_loss(val_loss),
                "Val Recall@1": _format_optional_metric(
                    val_metrics.get("recall@1")
                ),
                "Val Recall@5": _format_optional_metric(
                    val_metrics.get("recall@5")
                ),
                "Val Recall@10": _format_optional_metric(
                    val_metrics.get("recall@10")
                ),
                "Val NDCG@1": _format_optional_metric(val_metrics.get("ndcg@1")),
                "Val NDCG@5": _format_optional_metric(val_metrics.get("ndcg@5")),
                "Val NDCG@10": _format_optional_metric(
                    val_metrics.get("ndcg@10")
                ),
                "Val MRR": _format_optional_metric(val_metrics.get("mrr")),
                "GPU Max Allocated": gpu_max_allocated,
            }
        )


def _training_step_log_path(config: dict) -> Path | None:
    train_log_path = config.get("run_config", {}).get("train_log_path")
    if not train_log_path:
        return None
    return Path(str(train_log_path)).parent / "training_step.csv"


def _format_optional_loss(loss: float | None) -> str:
    return "" if loss is None else f"{loss:.6f}"


def _format_optional_metric(value: float | int | None) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _target_indices_from_matrix(targets: torch.Tensor) -> list[list[int]]:
    return [
        torch.nonzero(row > 0, as_tuple=False).flatten().tolist()
        for row in targets
    ]


def _gpu_max_allocated_summary(device: torch.device) -> str:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "not_available"

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    max_allocated_gb = torch.cuda.max_memory_allocated(device_index) / 1024**3
    return f"{max_allocated_gb:.3f}GB"


def _forward(model, batch: dict) -> torch.Tensor:
    return model(
        batch["genes"],
        batch["values"],
        batch["padding_mask"],
        control_gene_ids=batch["control_genes"],
        control_values=batch["control_values"],
        control_padding_mask=batch["control_padding_mask"],
        control_counts=batch["control_counts"],
    )
