"""Evaluate scGPT gene-score model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.scgpt.data import GeneScoreDataset, collate_gene_score_batch
from src.scgpt.factory import build_gene_score_model
from src.scgpt.output import cardinality_logits_from_output, logits_from_output
from src.utils.data import get_condition_splits, load_adata
from src.utils.metrics import compute_cardinality_metrics, compute_gene_metrics
from src.utils.runtime import AccelerateRuntime


def run(config: dict) -> dict:
    """Run gene-ranking evaluation for a finetuned scGPT scorer."""
    runtime = AccelerateRuntime(config)
    adata = load_adata(config["data_config"]["h5ad_path"])
    split = get_condition_splits(config)
    pretrained_dir = Path(config["model_config"].get("pretrained_dir", "model/scGPT"))
    with (pretrained_dir / "vocab.json").open() as handle:
        vocab = json.load(handle)
    dataset = GeneScoreDataset(
        adata=adata,
        conditions=split["test"],
        vocab=vocab,
        n_bins=int(config["model_config"].get("preprocess_binning", 51)),
        condition_key=config["data_config"].get("condition_key", "condition"),
        control_key=config["data_config"].get("control_key", "control"),
        n_control_samples=int(config["data_config"].get("control_n_samples", 8)),
        seed=int(config["run_config"].get("seed", 42)),
    )
    model = _build_model(
        config,
        adata.n_vars,
        dataset.gene_ids,
        runtime.device,
        gene_names=getattr(dataset, "gene_names", None),
    )
    checkpoint_path = config["run_config"].get("load_checkpoint_path")
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training_config", {}).get("batch_size", 32)),
        shuffle=False,
        collate_fn=lambda batch: collate_gene_score_batch(batch, vocab, adata.n_vars),
        num_workers=int(config["data_config"].get("num_workers", 0)),
    )
    model, loader = runtime.prepare(model, loader)
    model.eval()
    scores = []
    targets = []
    cardinality_logits = []
    with torch.no_grad():
        for batch in loader:
            model_output = model(
                batch["genes"].to(runtime.device),
                batch["values"].to(runtime.device),
                batch["padding_mask"].to(runtime.device),
                control_gene_ids=batch["control_genes"].to(runtime.device),
                control_values=batch["control_values"].to(runtime.device),
                control_padding_mask=batch["control_padding_mask"].to(runtime.device),
                control_counts=batch["control_counts"],
            )
            logits = logits_from_output(model_output)
            gathered_logits = runtime.gather_for_metrics(logits)
            gathered_targets = runtime.gather_for_metrics(
                batch["targets"].to(logits.device)
            )
            scores.extend(row.cpu().numpy() for row in gathered_logits)
            targets.extend(_target_indices_from_matrix(gathered_targets.cpu()))
            batch_cardinality_logits = cardinality_logits_from_output(model_output)
            if batch_cardinality_logits is not None:
                gathered_cardinality = runtime.gather_for_metrics(
                    batch_cardinality_logits
                )
                cardinality_logits.extend(
                    row.cpu().numpy() for row in gathered_cardinality
                )
    top_k_values = config.get("evaluation_config", {}).get("top_k_values", [1, 5, 10])
    metrics = compute_gene_metrics(scores, targets, top_k_values)
    if cardinality_logits:
        metrics.update(compute_cardinality_metrics(cardinality_logits, targets))
    if runtime.is_main_process:
        _write_json(config["run_config"].get("eval_log_path"), {"metrics": metrics})
    runtime.wait_for_everyone()
    return {"metrics": metrics}


def _build_model(
    config: dict,
    n_genes: int,
    gene_ids,
    device: torch.device,
    gene_names: list[str] | None = None,
):
    return build_gene_score_model(
        config=config,
        n_genes=n_genes,
        gene_ids=gene_ids,
        device=torch.device(device),
        gene_names=gene_names,
    )


def _target_indices_from_matrix(targets: torch.Tensor) -> list[list[int]]:
    """Convert gathered multi-hot targets to index lists."""
    return [
        torch.nonzero(row > 0, as_tuple=False).flatten().tolist() for row in targets
    ]


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
