"""Train the STATE-ready AIVC A->B->C pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

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
    paths = run_training(config)
    print(f"run dir: {paths['run_dir']}")
    print(f"train log: {paths['train_log']}")
    print(f"test metrics: {paths['test_metrics']}")


def run_training(config: AivcConfig) -> dict[str, Path]:
    """Run one train/val/test STATE-ready AIVC experiment."""
    _set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    data = load_gene_bags(config)
    split = make_gene_split(data.genes, data.y, config.split)
    run_dir = _run_dir(config)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data = with_scvi_teacher_latents(config, data, split, artifacts_dir)
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
    model.to(device)
    batch_lookup = load_state_batch_lookup(config.state.model_dir)
    weights = LossWeights(
        latent_mean_delta=config.loss.latent_mean_delta_weight,
        latent_energy=config.loss.latent_energy_weight,
        hvg_mean_delta=config.loss.hvg_mean_delta_weight,
        hvg_energy=config.loss.hvg_energy_weight,
        pred_c=config.loss.pred_c_weight,
        obs_c=config.loss.obs_c_weight,
        occupancy=config.loss.occupancy_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    rng = np.random.default_rng(config.train.seed)
    logs = []
    for epoch in range(1, config.train.max_epochs + 1):
        train_row = _run_epoch(
            model,
            data,
            split.train,
            weights,
            optimizer,
            rng,
            config.train.cell_set_len,
            device,
            batch_lookup,
        )
        val_row, _val_predictions = _evaluate(
            model,
            data,
            split.val,
            weights,
            rng,
            config.train.cell_set_len,
            device,
            batch_lookup,
            pad_short=True,
        )
        row = {"epoch": epoch, **_prefix(train_row, "train"), **_prefix(val_row, "val")}
        logs.append(row)
        pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
    eval_data = external.data if external is not None else data
    eval_indices = (
        np.arange(len(eval_data.genes), dtype=np.int64)
        if external is not None
        else split.test
    )
    evaluation_scope = (
        f"external:{config.external_test.name}"
        if external is not None and config.external_test is not None
        else "internal_test"
    )
    test_row, test_predictions = _evaluate(
        model,
        eval_data,
        eval_indices,
        weights,
        rng,
        config.train.cell_set_len,
        device,
        batch_lookup,
        pad_short=False,
    )
    test_row = {"evaluation_scope": evaluation_scope, **test_row}
    test_predictions.insert(0, "evaluation_scope", evaluation_scope)
    test_predictions["perturbation_has_known_vector"] = test_predictions[
        "perturbation_gene"
    ].map(model.perturbations.has_known_vector)
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
    pd.DataFrame([test_row]).to_csv(run_dir / "test_metrics.csv", index=False)
    test_predictions.to_csv(artifacts_dir / "test_predictions.csv", index=False)
    _write_split_artifact(data, split, artifacts_dir / "gene_splits.csv")
    return {
        "run_dir": run_dir,
        "train_log": run_dir / "train_log.csv",
        "test_metrics": run_dir / "test_metrics.csv",
    }


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
    model: AivcModel,
    data: GeneBags,
    indices: np.ndarray,
    weights: LossWeights,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
) -> dict[str, float]:
    model.train()
    rows = []
    for index in indices:
        loss_row, _pred_y, _obs_y = _loss_for_index(
            model,
            data,
            int(index),
            weights,
            rng,
            cell_set_len,
            device,
            batch_lookup,
            pad_short=True,
        )
        optimizer.zero_grad()
        loss_row["total_tensor"].backward()
        optimizer.step()
        rows.append({k: v for k, v in loss_row.items() if k != "total_tensor"})
    return _mean_rows(rows)


def _evaluate(
    model: AivcModel,
    data: GeneBags,
    indices: np.ndarray,
    weights: LossWeights,
    rng: np.random.Generator,
    cell_set_len: int,
    device: torch.device,
    batch_lookup: dict[str, int],
    pad_short: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    pred_rows = []
    with torch.no_grad():
        for index in indices:
            loss_row, pred_y, obs_y = _loss_for_index(
                model,
                data,
                int(index),
                weights,
                rng,
                cell_set_len,
                device,
                batch_lookup,
                pad_short=pad_short,
            )
            rows.append({k: v for k, v in loss_row.items() if k != "total_tensor"})
            pred_rows.append(
                {
                    "perturbation_gene": str(data.genes[index]),
                    "y_true": float(data.y[index]),
                    "y_pred": float(pred_y),
                    "y_obs_anchor": float(obs_y),
                    "control_fallback_count": float(loss_row["control_fallback_count"]),
                    "n_chunks": float(loss_row["n_chunks"]),
                }
            )
    summary = _mean_rows(rows)
    predictions = pd.DataFrame(pred_rows)
    if not predictions.empty:
        y_true = predictions["y_true"].to_numpy(dtype=np.float64)
        y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
        summary.update(regression_metrics(y_true, y_pred))
        summary.update(ranking_metrics(y_true, y_pred, (-0.5, -1.0)))
    return summary, predictions


def _loss_for_index(
    model: AivcModel,
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
    losses = model.losses_for_gene(
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


def _batch_tensor(
    labels: np.ndarray | None,
    batch_lookup: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    batch_encoded = encode_batch_labels(labels, batch_lookup)
    if batch_encoded is None:
        return None
    return torch.as_tensor(batch_encoded, dtype=torch.long, device=device)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _prefix(row: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


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


def _run_dir(config: AivcConfig) -> Path:
    run_id = config.train.run_id or datetime.now(timezone.utc).strftime(
        "state_aivc_%Y%m%dT%H%M%SZ"
    )
    run_dir = config.data.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _infer_pert_dim(known_vectors: dict[str, np.ndarray]) -> int:
    if not known_vectors:
        msg = "state.pert_dim is required when no known perturbation vectors are given"
        raise ValueError(msg)
    first = next(iter(known_vectors.values()))
    return int(first.shape[0])


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
