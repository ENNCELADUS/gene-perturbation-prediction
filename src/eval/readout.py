"""Evaluate cached P1-A heads with the production GeneEffect aggregation."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.eval.geneeffect import EvalResult, aggregate_geneeffect
from src.model.normalization import BlockStandardizer
from src.model.readout import make_readout


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
