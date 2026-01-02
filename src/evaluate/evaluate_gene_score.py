"""
Evaluation for Route B1 gene-level scoring.

Ranks genes directly per query and evaluates Top-K gene metrics
against the ground-truth target gene set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import yaml

import numpy as np
import hashlib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import load_perturb_data
from ..data.gene_score_collator import GeneScoreDataset, collate_gene_score_batch
from ..evaluate.metrics import parse_condition_genes
from ..model.gene_score import GeneScoreModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Route B1 gene-score model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Finetuned gene-score model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/gene_score/eval_results.json",
        help="Output path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=None,
        help="Top-K values (overrides config evaluation.top_k_values)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_gene_metrics(
    scores: List[np.ndarray],
    targets: List[List[int]],
    top_k_values: List[int],
) -> Dict[str, float]:
    metrics = {f"hit@{k}": 0.0 for k in top_k_values}
    metrics.update({f"exact_hit@{k}": 0.0 for k in top_k_values})
    metrics.update({f"recall@{k}": 0.0 for k in top_k_values})
    metrics.update({f"ndcg@{k}": 0.0 for k in top_k_values})
    mrr_sum = 0.0
    n_queries = 0

    for score_vec, target_genes in zip(scores, targets):
        if not target_genes:
            continue
        n_queries += 1
        target_set = set(target_genes)
        ranking = np.argsort(-score_vec)
        positions = np.empty_like(ranking)
        positions[ranking] = np.arange(1, len(ranking) + 1)
        min_rank = int(positions[list(target_set)].min())
        mrr_sum += 1.0 / min_rank

        for k in top_k_values:
            topk = ranking[:k]
            topk_set = set(topk.tolist())
            overlap = len(topk_set & target_set)
            metrics[f"hit@{k}"] += 1.0 if overlap > 0 else 0.0
            metrics[f"exact_hit@{k}"] += 1.0 if target_set.issubset(topk_set) else 0.0
            metrics[f"recall@{k}"] += overlap / len(target_set)

            dcg = 0.0
            for rank, gene_idx in enumerate(topk, start=1):
                if gene_idx in target_set:
                    dcg += 1.0 / np.log2(rank + 1)
            idcg = sum(
                1.0 / np.log2(rank + 1)
                for rank in range(1, min(len(target_set), k) + 1)
            )
            metrics[f"ndcg@{k}"] += dcg / idcg if idcg > 0 else 0.0

    if n_queries == 0:
        metrics = {k: 0.0 for k in metrics}
        metrics["mrr"] = 0.0
        metrics["n_queries"] = 0
        return metrics

    for key in list(metrics.keys()):
        metrics[key] /= n_queries
    metrics["mrr"] = mrr_sum / n_queries
    metrics["n_queries"] = n_queries
    return metrics


def mask_target_gene_values(
    genes: torch.Tensor,
    values: torch.Tensor,
    conditions: List[str],
    vocab: Dict[str, int],
    mask_k: int,
    target_gene_pool: List[str],
    neutral_value: int = 0,
) -> torch.Tensor:
    if mask_k <= 0:
        return values

    masked_values = values.clone()
    for i, condition in enumerate(conditions):
        target_genes = sorted(parse_condition_genes(condition))
        if not target_genes:
            continue
        effective_k = max(mask_k, len(target_genes))

        extra_genes = select_extra_mask_genes(
            condition=condition,
            target_genes=target_genes,
            target_gene_pool=target_gene_pool,
            extra_k=effective_k - len(target_genes),
        )
        mask_genes = target_genes + extra_genes
        token_ids = [vocab[g] for g in mask_genes if g in vocab and g != "<pad>"]
        if not token_ids:
            continue
        row_gene_ids = genes[i]
        row_mask = torch.zeros_like(row_gene_ids, dtype=torch.bool)
        for token_id in token_ids:
            row_mask |= row_gene_ids == token_id
        masked_values[i][row_mask] = neutral_value
    return masked_values


def select_extra_mask_genes(
    condition: str,
    target_genes: List[str],
    target_gene_pool: List[str],
    extra_k: int,
) -> List[str]:
    if extra_k <= 0:
        return []

    pool = [g for g in target_gene_pool if g not in target_genes]
    if not pool:
        return []

    seed = int(hashlib.md5(condition.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    sample_k = min(extra_k, len(pool))
    return rng.choice(pool, size=sample_k, replace=False).tolist()


def build_target_gene_pool(conditions: List[str]) -> List[str]:
    pool = set()
    for cond in conditions:
        pool.update(parse_condition_genes(cond))
    return sorted(pool)


def main():
    args = parse_args()
    config = load_config(args.config)
    eval_config = config.get("evaluation", {})
    top_k_values = (
        args.top_k
        if args.top_k is not None
        else eval_config.get("top_k_values", [1, 5, 8, 10, 20, 40])
    )

    if args.output == "results/gene_score/eval_results.json":
        if eval_config.get("output_path"):
            args.output = eval_config["output_path"]
        else:
            logging_cfg = config.get("logging", {})
            base_dir = logging_cfg.get("output_dir", "results/gene_score")
            args.output = str(Path(base_dir) / "eval_results.json")

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
    print(f"  - Test conditions: {len(test_conditions)}")

    if args.checkpoint is None:
        base_dir = config.get("logging", {}).get("output_dir", "results/gene_score")
        args.checkpoint = eval_config.get(
            "checkpoint_path", str(Path(base_dir) / "best_model.pt")
        )

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
        score_mode=config.get("head", {}).get("score_mode", "dot"),
        head_hidden_dim=config.get("head", {}).get("hidden_dim", 512),
        head_dropout=config.get("head", {}).get("dropout", 0.2),
    )
    print(f"Loading finetuned checkpoint from {args.checkpoint}")
    model.load(args.checkpoint, map_location=device)
    model.eval()

    print("\n[3/4] Preparing evaluation data...")
    test_dataset = GeneScoreDataset(
        adata=dataset.adata,
        conditions=test_conditions,
        vocab=model.backbone.vocab,
        n_bins=config["model"].get("preprocess_binning", 51),
        match_keys=config["data"].get("control_match_keys"),
        n_control_samples=config["data"].get("control_n_samples", 8),
    )
    model.set_score_gene_ids(test_dataset.gene_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("training", {}).get("batch_size", 32),
        shuffle=False,
        collate_fn=lambda batch: collate_gene_score_batch(
            batch, model.backbone.vocab, n_genes
        ),
        num_workers=0,
    )

    print("\n[4/4] Scoring and ranking...")
    mask_k = eval_config.get("mask", 0)
    if isinstance(mask_k, bool):
        mask_k = int(mask_k)
    if mask_k > 0:
        print(f"  - Masking {mask_k} genes per query (anti-cheat) enabled")
    target_gene_pool = build_target_gene_pool(dataset.test_conditions)
    if "gene_name" in dataset.adata.var.columns:
        eval_gene_names = dataset.adata.var["gene_name"].tolist()
    else:
        eval_gene_names = dataset.adata.var_names.tolist()
    gene_name_to_idx = {g: i for i, g in enumerate(eval_gene_names)}
    all_scores: List[np.ndarray] = []
    all_targets: List[List[int]] = []

    for batch in tqdm(test_loader, desc="Scoring"):
        genes = batch["genes"].to(device)
        values = batch["values"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        control_genes = batch["control_genes"].to(device)
        control_values = batch["control_values"].to(device)
        control_padding_mask = batch["control_padding_mask"].to(device)
        control_counts = batch["control_counts"]
        conditions = batch["conditions"]

        if mask_k > 0:
            values = mask_target_gene_values(
                genes=genes,
                values=values,
                conditions=conditions,
                vocab=model.backbone.vocab,
                mask_k=mask_k,
                target_gene_pool=target_gene_pool,
                neutral_value=0,
            )

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                gene_scores = model(
                    genes,
                    values,
                    padding_mask,
                    control_gene_ids=control_genes,
                    control_values=control_values,
                    control_padding_mask=control_padding_mask,
                    control_counts=control_counts,
                    control_chunk_size=config.get("training", {}).get(
                        "control_chunk_size", 0
                    ),
                    control_no_grad=True,
                )  # (batch, n_genes)

        for i, condition in enumerate(conditions):
            target_genes = parse_condition_genes(condition)
            if not target_genes:
                continue
            target_indices = [
                gene_name_to_idx[g] for g in target_genes if g in gene_name_to_idx
            ]
            if not target_indices:
                continue
            all_scores.append(gene_scores[i].detach().cpu().numpy())
            all_targets.append(target_indices)

    metrics = compute_gene_metrics(
        scores=all_scores,
        targets=all_targets,
        top_k_values=top_k_values,
    )
    output_metrics = {"mrr": metrics.get("mrr", 0.0)}
    for k in top_k_values:
        output_metrics[f"exact_hit@{k}"] = metrics.get(f"exact_hit@{k}", 0.0)
        output_metrics[f"relevant_hit@{k}"] = metrics.get(f"hit@{k}", 0.0)
        output_metrics[f"recall@{k}"] = metrics.get(f"recall@{k}", 0.0)
        output_metrics[f"ndcg@{k}"] = metrics.get(f"ndcg@{k}", 0.0)
    results = {
        "metrics": output_metrics,
        "n_evaluated": len(all_targets),
    }

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for k in top_k_values:
        exact = results["metrics"].get(f"exact_hit@{k}", 0)
        relevant = results["metrics"].get(f"relevant_hit@{k}", 0)
        recall = results["metrics"].get(f"recall@{k}", 0)
        ndcg = results["metrics"].get(f"ndcg@{k}", 0)
        print(
            f"  Hit@{k}: exact={exact:.3f}, relevant={relevant:.3f}, recall={recall:.3f}, ndcg={ndcg:.3f}"
        )
    print(f"  MRR: {results['metrics'].get('mrr', 0):.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
