"""The readout head on the frozen joint backbone's cached features: the head
itself, its fit with ordinary early stopping, and its evaluation and export."""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.data.batches import FeatureBatch
from src.eval.accuracy import EvalResult, aggregate_geneeffect
from src.model.head import (
    GeneEffectBlockConfig,
    GeneEffectFeatureDims,
    GeneEffectMLP,
)
from src.model.normalization import BlockStandardizer
from src.training.checkpoint import TrainState, record_validation

ARMS = ("A0", "A1", "A2", "A3")


class FixedReadout(nn.Module):
    """Consume standardized features and optionally add per-gene PCA slopes."""

    def __init__(self, mlp: GeneEffectMLP, n_genes: int, explicit: bool):
        super().__init__()
        self.mlp = mlp
        self.slopes = nn.Parameter(torch.zeros(n_genes, 8)) if explicit else None

    def forward(self, features: FeatureBatch, gene_indices, contexts):
        response = self.mlp.blocks.use_s
        prediction = self.mlp(
            delta_proj=features.delta_proj if response else None,
            s=features.s if response else None,
            hvg_panel_mask=features.hvg_panel_mask if response else None,
            own_gene_shift_mask=features.own_gene_shift_mask if response else None,
            q_sc=features.q_sc,
            q_sc_mask=features.q_sc_mask,
            e_g=features.e_g,
            z_c=features.z_c,
        )
        if self.slopes is not None:
            prediction = prediction + (self.slopes[gene_indices] * contexts).sum(-1)
        return prediction

    def regularization(self):
        """Sum over eight slopes, mean over all covered genes, lambda 0.01."""
        if self.slopes is None:
            return next(self.parameters()).new_zeros(())
        return 0.01 * self.slopes.square().sum(dim=1).mean()


def make_readout(
    arm: str, dims: GeneEffectFeatureDims, n_genes: int, *, seed: int = 0
) -> FixedReadout:
    """Seeded canonical MLP; copy common columns when removing the response block."""
    if arm not in ARMS or n_genes < 1:
        raise ValueError("require A0-A3 and at least one training-covered gene")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        canonical = GeneEffectMLP(dims)
        if arm in {"A1", "A3"}:
            mlp = canonical
        else:
            mlp = GeneEffectMLP(
                dims, GeneEffectBlockConfig(use_delta_proj=False, use_s=False)
            )
            state = deepcopy(canonical.state_dict())
            offset = dims.delta_proj + dims.s + 2
            state["net.0.weight"] = state["net.0.weight"][:, offset:].clone()
            mlp.load_state_dict(state)
    return FixedReadout(mlp, n_genes, arm in {"A2", "A3"})


@dataclass(frozen=True)
class ReadoutSettings:
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-4

    def __post_init__(self):
        if min(self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError("epochs, patience and batch size must be positive")
        if not 0 <= self.learning_rate < float("inf"):
            raise ValueError("learning rate must be finite and nonnegative")


def _save(path, saved):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(saved, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_readout(
    cache,
    arm,
    out_dir,
    *,
    device="cpu",
    settings=ReadoutSettings(),
    standardizer=None,
    resume=None,
    head_seed=0,
):
    """Train one fresh paired head, then export its minimum-Huber checkpoint."""
    out_dir = Path(out_dir)
    scaler = cache.standardizer if standardizer is None else standardizer
    restored = None
    if resume is not None:
        if Path(resume).resolve() != (out_dir / "last.pt").resolve():
            raise ValueError("resume must use last.pt in the same arm directory")
        restored = torch.load(resume, map_location="cpu", weights_only=True)
        # Readout-head checkpoints written before head_seed existed were all seed 0;
        # read that legacy state as 0 rather than raising KeyError, and still
        # reject a resume that asks for a different seed.
        legacy_seed = restored["head_seed"] if "head_seed" in restored else 0
        if (
            restored["arm"] != arm
            or restored["settings"] != asdict(settings)
            or restored["cache_metadata"] != cache.metadata
            or restored["standardizer"] != scaler.to_state()
            or legacy_seed != head_seed
        ):
            raise ValueError("resume settings, scaler, arm, cache or head seed differ")
        scaler = BlockStandardizer.from_state(restored["standardizer"])
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "arm": arm,
        "settings": asdict(settings),
        "head_seed": head_seed,
        "data_seed": 0,
        "cache": str(cache.root.resolve()),
        "cache_metadata": cache.metadata,
        "training": "running",
        "evaluation": "not_started",
    }

    def status():
        (out_dir / "run.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    status()
    try:
        model = make_readout(arm, cache.dims, len(cache.genes), seed=head_seed).to(
            device
        )
        groups = [{"params": model.mlp.parameters(), "weight_decay": 0.01}]
        if model.slopes is not None:
            groups.append({"params": [model.slopes], "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(
            groups, lr=settings.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )
        state = (
            TrainState() if restored is None else TrainState(**restored["train_state"])
        )
        if restored is not None:
            model.load_state_dict(restored["model_state"])
            optimizer.load_state_dict(restored["optimizer"])
            # A failed checkpoint write may leave a metrics row beyond last.pt.
            path = out_dir / "metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(
                "".join(
                    json.dumps(row, allow_nan=False) + "\n"
                    for row in rows
                    if row["global_step"] <= state.global_step
                )
            )
        for epoch in range(state.next_epoch, settings.max_epochs):
            if state.bad_epochs >= settings.patience:
                break
            model.train()
            order = torch.randperm(
                len(cache), generator=torch.Generator().manual_seed(epoch)
            ).numpy()
            loss_sum, penalty_sum, updates = 0.0, 0.0, 0
            for start in range(0, len(order), settings.batch_size):
                indices = order[start : start + settings.batch_size]
                feature, genes, contexts, target = cache.batch(
                    "train", indices, device, standardizer=scaler
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(feature, genes, contexts)
                huber = F.huber_loss(prediction, target, delta=1.0)
                penalty = model.regularization()
                loss = huber + penalty
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite head training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                loss_sum += float(huber.detach()) * len(indices)
                penalty_sum += float(penalty.detach())
                updates += 1
                state.global_step += 1
            train = evaluate_head(
                model, cache, "train", device=device, standardizer=scaler
            )
            val = evaluate_head(model, cache, "val", device=device, standardizer=scaler)
            improved = record_validation(
                state, val.metrics, epoch, key="val_selective_spearman"
            )
            metrics = {
                "epoch": epoch + 1,
                "global_step": state.global_step,
                "train_rows": len(order),
                "optimizer_updates": updates,
                "update_huber": loss_sum / len(order),
                "update_l2": penalty_sum / updates,
                **train.metrics,
                **val.metrics,
            }
            with (out_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            saved = {
                "arm": arm,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_state": asdict(state),
                "settings": asdict(settings),
                "head_seed": head_seed,
                "cache_metadata": cache.metadata,
                "standardizer": scaler.to_state(),
                "context_scores": torch.from_numpy(cache.context_scores.copy()),
            }
            if improved:
                _save(out_dir / "best.pt", saved)
            _save(out_dir / "last.pt", saved)
            print(
                f"{arm} epoch={epoch + 1} step={state.global_step} "
                f"val_huber={val.metrics['val_geneeffect_loss']:.8f} "
                f"patience={state.bad_epochs}",
                flush=True,
            )
            if state.bad_epochs >= settings.patience:
                break
        record.update(
            training="completed",
            best_epoch=state.best_epoch + 1,
            stopped_epoch=state.next_epoch,
            global_step=state.global_step,
            stop_reason="patience"
            if state.bad_epochs >= settings.patience
            else "epoch_cap",
        )
        status()
    except Exception as exc:
        record.update(training="failed", error=f"{type(exc).__name__}: {exc}")
        status()
        raise
    return export_checkpoint(cache, out_dir / "best.pt", device=device)


def evaluate_head(
    model, cache, split, *, device="cpu", standardizer=None, batch_size=1024
):
    """Evaluate all finite cached rows, retaining the tail and undefined metrics."""
    if split not in {"train", "val"}:
        raise ValueError("P1-A evaluates train and val only")
    frame = cache.labels(split)
    predictions = np.empty(len(frame), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            indices = slice(start, start + batch_size)
            feature, genes, contexts, _ = cache.batch(
                split, indices, device, standardizer=standardizer
            )
            predictions[indices] = model(feature, genes, contexts).float().cpu().numpy()
    frame["residual_prediction"] = predictions
    frame["geneeffect_prediction"] = (
        predictions.astype(float) + cache.splits[split]["gene_mean"]
    )
    metrics, lines, genes = aggregate_geneeffect(
        frame,
        model_ids=cache.metadata["split_lines"][split],
        genes=cache.genes,
        variable_genes=cache.variable_genes,
        selective_genes=cache.selective_genes,
    )
    return EvalResult(
        {f"{split}_{key}": value for key, value in metrics.items()},
        frame,
        lines,
        genes,
        pd.DataFrame(),
    )


def evaluate_readout(cache, saved, split, *, device="cpu"):
    """Restore a diagnostic head and its fitted scaler, without any fitting."""
    if saved["cache_metadata"] != cache.metadata:
        raise ValueError("checkpoint and feature cache identities differ")
    model = make_readout(saved["arm"], cache.dims, len(cache.genes)).to(device)
    model.load_state_dict(saved["model_state"])
    scaler = BlockStandardizer.from_state(saved["standardizer"])
    return evaluate_head(model, cache, split, device=device, standardizer=scaler)


def export_checkpoint(cache, checkpoint, *, device="cpu"):
    """Export/retry evaluation and persist its state without changing training."""
    checkpoint = Path(checkpoint)
    run_path = checkpoint.parent / "run.json"
    record = json.loads(run_path.read_text())
    record.update(evaluation="running", evaluation_checkpoint=checkpoint.stem)
    record.pop("error", None)

    def status():
        run_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    status()
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for split in ("train", "val"):
            result = evaluate_readout(cache, saved, split, device=device)
            export_readout(
                result, checkpoint.parent / "evaluation" / checkpoint.stem / split
            )
        record["evaluation"] = "completed"
        if checkpoint.stem == "best":
            record["best_step"] = saved["train_state"]["global_step"]
        status()
    except Exception as exc:
        record.update(evaluation="failed", error=f"{type(exc).__name__}: {exc}")
        status()
        raise
    return record


def export_readout(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(destination / "predictions.parquet", index=False)
    result.per_line.to_csv(destination / "per_line.csv", index=False)
    result.per_gene.to_csv(destination / "per_gene.csv", index=False)
    (destination / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, allow_nan=False) + "\n"
    )
