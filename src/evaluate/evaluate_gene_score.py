"""
Evaluation for Route B1 gene-level scoring.

Ranks candidate conditions using compositional gene scores and evaluates
Top-K metrics on test conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import load_perturb_data
from ..data.gene_score_collator import GeneScoreDataset, collate_gene_score_batch
from ..evaluate.metrics import compute_all_metrics, parse_condition_genes
from ..model.gene_score import GeneScoreModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Route B1 gene-score model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Finetuned gene-score model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/gene_score/eval_results.json",
        help="Output path",
    )
    parser.add_argument(
        "--top_k", type=int, nargs="+", default=[1, 5, 10, 20], help="Top-K values"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_condition_matrix(
    conditions: List[str], gene_name_to_idx: Dict[str, int]
) -> torch.Tensor:
    matrix = np.zeros((len(conditions), len(gene_name_to_idx)), dtype=np.float32)
    for i, cond in enumerate(conditions):
        genes = parse_condition_genes(cond)
        for g in genes:
            if g in gene_name_to_idx:
                matrix[i, gene_name_to_idx[g]] = 1.0
    return torch.from_numpy(matrix)


def aggregate_by_voting(per_cell_rankings: List[List[str]], k: int) -> List[str]:
    from collections import defaultdict

    vote_counts = defaultdict(int)
    for ranking in per_cell_rankings:
        for rank, condition in enumerate(ranking):
            weight = len(ranking) - rank
            vote_counts[condition] += weight

    sorted_conditions = sorted(vote_counts.items(), key=lambda x: -x[1])
    return [cond for cond, _ in sorted_conditions[:k]]


def mask_target_gene_values(
    genes: torch.Tensor,
    values: torch.Tensor,
    conditions: List[str],
    vocab: Dict[str, int],
    neutral_value: int = 0,
) -> torch.Tensor:
    masked_values = values.clone()
    for i, condition in enumerate(conditions):
        target_genes = parse_condition_genes(condition)
        if not target_genes:
            continue
        token_ids = [vocab[g] for g in target_genes if g in vocab and g != "<pad>"]
        if not token_ids:
            continue
        row_gene_ids = genes[i]
        row_mask = torch.zeros_like(row_gene_ids, dtype=torch.bool)
        for token_id in token_ids:
            row_mask |= row_gene_ids == token_id
        masked_values[i][row_mask] = neutral_value
    return masked_values


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.output == "results/gene_score/eval_results.json":
        logging_cfg = config.get("logging", {})
        base_dir = logging_cfg.get("output_dir", "results")
        exp_name = logging_cfg.get(
            "experiment_name", config.get("model", {}).get("encoder", "experiment")
        )
        args.output = str(Path(base_dir) / exp_name / "eval_results.json")

    print("=" * 60)
    print("Route B1 Gene-Score Evaluation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    use_amp = torch.cuda.is_available()

    print("\n[1/4] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    test_conditions = dataset.test_conditions
    all_conditions = dataset.all_conditions
    print(f"  - Test conditions: {len(test_conditions)}")
    print(f"  - Candidate conditions: {len(all_conditions)}")

    print("\n[2/4] Loading model...")
    pretrained_dir = Path(config["model"].get("pretrained_dir", "model/scGPT"))
    n_genes = dataset.adata.n_vars
    model = GeneScoreModel(
        n_genes=n_genes,
        checkpoint_path=pretrained_dir / "best_model.pt",
        vocab_path=pretrained_dir / "vocab.json",
        args_path=pretrained_dir / "args.json",
        freeze_encoder=True,
        freeze_layers_up_to=10,
        device=device,
        head_hidden_dim=config.get("head", {}).get("hidden_dim", 512),
        head_dropout=config.get("head", {}).get("dropout", 0.2),
    )
    model.load(args.checkpoint, map_location=device)
    model.eval()

    print("\n[3/4] Preparing evaluation data...")
    test_dataset = GeneScoreDataset(
        adata=dataset.test_adata,
        conditions=test_conditions,
        vocab=model.backbone.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        collate_fn=lambda batch: collate_gene_score_batch(
            batch, model.backbone.vocab, n_genes
        ),
        num_workers=0,
    )

    gene_name_to_idx = {g: i for i, g in enumerate(dataset.adata.var_names.tolist())}
    condition_matrix = build_condition_matrix(all_conditions, gene_name_to_idx).to(
        device
    )

    print("\n[4/4] Scoring and ranking...")
    eval_config = config.get("evaluate", {})
    mask_target_genes = eval_config.get("mask", False)
    if mask_target_genes:
        print("  - Masking target gene expression (anti-cheat) enabled")

    per_condition_rankings: Dict[str, List[List[str]]] = {
        cond: [] for cond in test_conditions
    }

    for batch in tqdm(test_loader, desc="Scoring"):
        genes = batch["genes"].to(device)
        values = batch["values"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        conditions = batch["conditions"]

        if mask_target_genes:
            values = mask_target_gene_values(
                genes=genes,
                values=values,
                conditions=conditions,
                vocab=model.backbone.vocab,
                neutral_value=0,
            )

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                gene_scores = model(genes, values, padding_mask)  # (batch, n_genes)

        condition_scores = torch.matmul(
            gene_scores, condition_matrix.T.to(gene_scores.dtype)
        )
        top_k = max(args.top_k)
        top_indices = torch.topk(condition_scores, k=top_k, dim=1).indices

        for i, cond in enumerate(conditions):
            ranking = [all_conditions[idx] for idx in top_indices[i].tolist()]
            per_condition_rankings[cond].append(ranking)

    aggregated_predictions = []
    ground_truth = []
    for condition, rankings in per_condition_rankings.items():
        if not rankings:
            continue
        aggregated = aggregate_by_voting(rankings, k=max(args.top_k))
        aggregated_predictions.append(aggregated)
        ground_truth.append(condition)

    metrics = compute_all_metrics(
        predictions=aggregated_predictions,
        ground_truth=ground_truth,
        top_k_values=args.top_k,
        candidate_pool=None,
        include_macro=True,
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
