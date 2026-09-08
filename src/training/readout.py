"""Single-device, fixed-cache P1-A optimization with ordinary early stopping."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.eval.readout import evaluate_head, export_checkpoint
from src.model.readout import make_readout
from src.model.normalization import BlockStandardizer
from src.training.checkpoint import TrainState, record_validation


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
        if (
            restored["arm"] != arm
            or restored["settings"] != asdict(settings)
            or restored["cache_metadata"] != cache.metadata
            or restored["standardizer"] != scaler.to_state()
            or restored["head_seed"] != head_seed
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
            improved = record_validation(state, val.metrics, epoch)
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
