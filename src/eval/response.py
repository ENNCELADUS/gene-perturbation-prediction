"""Per-condition response scores and equal-anchor aggregation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from torch import nn

from src.data.batches import ResponseBatch
from src.model.response import predict_bags, response_terms


def response_rows(model: nn.Module, batch: ResponseBatch) -> list[dict]:
    """Score held-out conditions using the unwrapped evaluation-only backbone."""
    predictions = predict_bags(
        model.backbone, batch.controls_tx1, batch.genes, seed=model.collator_seed
    )
    terms = {
        key: values.detach().cpu().tolist()
        for key, values in response_terms(predictions, batch).items()
    }
    return [
        {
            "model_id": model_id,
            "gene_symbol": gene,
            "mean_delta_mse": mean,
            "energy_distance": energy,
        }
        for model_id, gene, mean, energy in zip(
            batch.model_ids,
            batch.genes,
            terms["mean_delta_mse"],
            terms["energy_distance"],
            strict=True,
        )
    ]


def aggregate_response(
    frame: pd.DataFrame, anchors: Sequence[str]
) -> dict[str, float | int]:
    """Average each term within ModelID, then equally across four anchors."""
    if len(anchors) != 4 or len(set(anchors)) != 4:
        raise ValueError("response evaluation requires four distinct anchors")
    if frame.empty or set(frame["model_id"]) != set(anchors):
        raise ValueError("response evaluation must cover exactly the four anchors")
    if frame.duplicated(["model_id", "gene_symbol"]).any():
        raise ValueError("duplicate response condition rows after metric gathering")
    terms = ["mean_delta_mse", "energy_distance"]
    if not np.isfinite(frame[terms].to_numpy(dtype=float)).all():
        raise ValueError("non-finite response loss")
    means = frame.groupby("model_id")[terms].mean().mean()
    return {
        "response_mean_delta_mse": float(means["mean_delta_mse"]),
        "response_energy_distance": float(means["energy_distance"]),
        "response_loss": float(means.sum()),
        "response_conditions": len(frame),
        "response_anchors": len(anchors),
    }
