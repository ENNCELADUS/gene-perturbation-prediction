"""
Evaluation for PCA + kNN gene-scoring baseline.

Requires AnnData:
- obs['condition'] with perturbation labels
- obs['control'] with control indicator
- var['gene_name'] optional (defaults to var_names)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from scipy import sparse
from scipy.spatial.distance import cdist

from ..data import load_perturb_data
from ..evaluate.evaluate_gene_score import compute_gene_metrics
from ..evaluate.metrics import parse_condition_genes
from ..model.pca_knn import PcaKnnGeneScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCA+kNN baseline")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--output",
        type=str,
        default="results/pca_knn/eval_results.json",
        help="Output path for results JSON",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_cells_for_condition(dataset, condition: str):
    """
    Get cells for a condition, handling format variations.

    The split uses normalized conditions (e.g., 'ATL1') but adata may use
    original format (e.g., 'ATL1+ctrl' or 'ctrl+ATL1').
    """
    adata = dataset.adata
    cond_col = dataset.condition_col

    mask = adata.obs[cond_col] == condition
    if mask.sum() > 0:
        return adata[mask].copy()

    genes = parse_condition_genes(condition)
    if not genes:
        return adata[mask].copy()

    possible_formats = [condition]
    if len(genes) == 1:
        gene = list(genes)[0]
        possible_formats.extend([f"{gene}+ctrl", f"ctrl+{gene}"])
    elif len(genes) == 2:
        g1, g2 = sorted(genes)
        possible_formats.extend([f"{g1}+{g2}", f"{g2}+{g1}"])

    for fmt in possible_formats:
        mask = adata.obs[cond_col] == fmt
        if mask.sum() > 0:
            return adata[mask].copy()

    return adata[adata.obs[cond_col] == "__NOMATCH__"].copy()


def build_condition_profiles(
    dataset, conditions: List[str]
) -> Tuple[np.ndarray, List[str]]:
    profiles = []
    valid_conditions = []

    for condition in conditions:
        cond_adata = get_cells_for_condition(dataset, condition)
        if cond_adata.n_obs == 0:
            continue
        X = cond_adata.X
        if sparse.issparse(X):
            X = X.toarray()
        profile = np.mean(X, axis=0)
        profiles.append(profile)
        valid_conditions.append(condition)

    if not profiles:
        return np.empty((0, dataset.adata.n_vars)), []

    return np.vstack(profiles), valid_conditions


def build_label_matrix(
    conditions: List[str],
    gene_name_to_idx: Dict[str, int],
    n_genes: int,
) -> np.ndarray:
    labels = np.zeros((len(conditions), n_genes), dtype=np.float32)
    for i, condition in enumerate(conditions):
        genes = parse_condition_genes(condition)
        if not genes:
            continue
        indices = [gene_name_to_idx[g] for g in genes if g in gene_name_to_idx]
        if indices:
            labels[i, indices] = 1.0
    return labels


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


def apply_mask_to_profiles(
    profiles: np.ndarray,
    conditions: List[str],
    gene_name_to_idx: Dict[str, int],
    mask_k: int,
    target_gene_pool: List[str],
) -> np.ndarray:
    if mask_k <= 0:
        return profiles

    masked = np.array(profiles, copy=True)
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
        for gene in mask_genes:
            idx = gene_name_to_idx.get(gene)
            if idx is None:
                continue
            masked[i, idx] = 0.0
    return masked


def apply_global_mask(
    profiles: np.ndarray,
    gene_name_to_idx: Dict[str, int],
    mask_genes: List[str],
) -> np.ndarray:
    if not mask_genes:
        return profiles

    masked = np.array(profiles, copy=True)
    indices = [gene_name_to_idx[g] for g in mask_genes if g in gene_name_to_idx]
    if indices:
        masked[:, indices] = 0.0
    return masked


def compute_neighbor_indices(
    train_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    metric: str,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        train_norm = train_emb / (
            np.linalg.norm(train_emb, axis=1, keepdims=True) + eps
        )
        query_norm = query_emb / (
            np.linalg.norm(query_emb, axis=1, keepdims=True) + eps
        )
        sims = query_norm @ train_norm.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        vals = np.take_along_axis(sims, idx, axis=1)
        return idx, vals

    if metric == "euclidean":
        dist = cdist(query_emb, train_emb, metric="euclidean")
        idx = np.argsort(dist, axis=1)[:, :k]
        vals = np.take_along_axis(dist, idx, axis=1)
        return idx, vals

    raise ValueError(f"Unknown metric: {metric}")


def compute_knn_scores(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    metric: str,
    weight_mode: str,
    weight_normalize: bool,
    clip_negative_similarity: bool,
    eps: float,
) -> List[np.ndarray]:
    k = min(k, train_emb.shape[0])
    if k <= 0:
        return []

    neighbor_idx, neighbor_vals = compute_neighbor_indices(
        train_emb=train_emb,
        query_emb=query_emb,
        k=k,
        metric=metric,
        eps=eps,
    )

    scores = []
    for row_idx in range(query_emb.shape[0]):
        idx = neighbor_idx[row_idx]
        vals = neighbor_vals[row_idx]

        if weight_mode == "uniform":
            weights = np.ones_like(vals)
        elif weight_mode == "distance":
            if metric == "cosine":
                distances = 1.0 - vals
            else:
                distances = vals
            weights = 1.0 / (distances + eps)
        elif weight_mode == "similarity":
            if metric == "cosine":
                weights = vals
            else:
                weights = -vals
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        if clip_negative_similarity:
            weights = np.maximum(weights, 0.0)

        if weight_normalize:
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones_like(weights) / len(weights)

        neighbor_labels = train_labels[idx]
        scores.append(weights @ neighbor_labels)

    return scores


def score_conditions(
    train_profiles: np.ndarray,
    train_labels: np.ndarray,
    query_profiles: np.ndarray,
    n_components: int,
    standardize: bool,
    random_state: int,
    k: int,
    metric: str,
    weight_mode: str,
    weight_normalize: bool,
    clip_negative_similarity: bool,
    eps: float,
) -> List[np.ndarray]:
    model = PcaKnnGeneScorer(
        n_components=n_components,
        standardize=standardize,
        random_state=random_state,
    )
    model.fit(train_profiles, train_labels)
    train_emb = model.train_embeddings
    query_emb = model.transform(query_profiles)

    return compute_knn_scores(
        train_emb=train_emb,
        train_labels=train_labels,
        query_emb=query_emb,
        k=k,
        metric=metric,
        weight_mode=weight_mode,
        weight_normalize=weight_normalize,
        clip_negative_similarity=clip_negative_similarity,
        eps=eps,
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("PCA + kNN Baseline Evaluation")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    print(f"  - Train conditions: {len(dataset.train_conditions)}")
    print(f"  - Val conditions: {len(dataset.val_conditions)}")
    print(f"  - Test conditions: {len(dataset.test_conditions)}")

    if "gene_name" in dataset.adata.var.columns:
        gene_names = dataset.adata.var["gene_name"].tolist()
    else:
        gene_names = dataset.adata.var_names.tolist()
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    print("\n[2/4] Building condition profiles...")
    train_profiles, train_conditions = build_condition_profiles(
        dataset, dataset.train_conditions
    )
    val_profiles, val_conditions = build_condition_profiles(
        dataset, dataset.val_conditions
    )
    test_profiles, test_conditions = build_condition_profiles(
        dataset, dataset.test_conditions
    )

    if train_profiles.shape[0] == 0:
        raise RuntimeError("No training profiles found; check condition labels.")

    train_labels = build_label_matrix(train_conditions, gene_name_to_idx, n_genes)

    baseline_cfg = config.get("baseline", {})
    eval_cfg = config.get("evaluation", {})
    top_k_values = eval_cfg.get("top_k_values", [1, 5, 10, 20, 40])
    mask_k = eval_cfg.get("mask", 0)
    if isinstance(mask_k, bool):
        mask_k = int(mask_k)
    mask_all_targets = bool(eval_cfg.get("mask_all_targets", False))
    mask_train = bool(baseline_cfg.get("mask_train", False))

    if mask_all_targets:
        print("  - Masking all target genes for anti-cheat evaluation")
        full_target_pool = build_target_gene_pool(dataset.all_conditions)
        if mask_train:
            train_profiles = apply_global_mask(
                train_profiles, gene_name_to_idx, full_target_pool
            )
        val_profiles = apply_global_mask(
            val_profiles, gene_name_to_idx, full_target_pool
        )
        test_profiles = apply_global_mask(
            test_profiles, gene_name_to_idx, full_target_pool
        )
    elif mask_k > 0:
        train_val_pool = build_target_gene_pool(
            dataset.train_conditions + dataset.val_conditions
        )
        test_pool = build_target_gene_pool(dataset.test_conditions)

        if mask_train:
            train_profiles = apply_mask_to_profiles(
                train_profiles,
                train_conditions,
                gene_name_to_idx,
                mask_k,
                train_val_pool,
            )

        val_profiles = apply_mask_to_profiles(
            val_profiles,
            val_conditions,
            gene_name_to_idx,
            mask_k,
            train_val_pool,
        )
        test_profiles = apply_mask_to_profiles(
            test_profiles,
            test_conditions,
            gene_name_to_idx,
            mask_k,
            test_pool,
        )

    n_components = baseline_cfg.get("n_components", 50)
    if isinstance(n_components, list):
        n_components_list = [int(x) for x in n_components]
    else:
        n_components_list = [int(n_components)]

    k_values = baseline_cfg.get("k_values", [1, 3, 5, 10])
    metric = baseline_cfg.get("metric", "cosine")
    weight_mode = baseline_cfg.get("weight_mode", "similarity")
    weight_normalize = baseline_cfg.get("weight_normalize", True)
    clip_negative_similarity = baseline_cfg.get("clip_negative_similarity", True)
    standardize = baseline_cfg.get("standardize", False)
    selection_metric = baseline_cfg.get("selection_metric", "mrr")
    random_state = baseline_cfg.get("random_state", 42)
    eps = float(baseline_cfg.get("eps", 1.0e-8))

    print("\n[3/4] Selecting hyperparameters on validation set...")
    best_score = -np.inf
    best_params = {}
    best_val_metrics = {}
    max_components = min(train_profiles.shape[0], train_profiles.shape[1])

    for n_comp in n_components_list:
        if n_comp > max_components:
            print(f"  - n_comp={n_comp} skipped (max={max_components} for train set)")
            continue
        for k in k_values:
            if k <= 0:
                continue
            val_scores = score_conditions(
                train_profiles=train_profiles,
                train_labels=train_labels,
                query_profiles=val_profiles,
                n_components=n_comp,
                standardize=standardize,
                random_state=random_state,
                k=k,
                metric=metric,
                weight_mode=weight_mode,
                weight_normalize=weight_normalize,
                clip_negative_similarity=clip_negative_similarity,
                eps=eps,
            )

            val_targets = []
            val_score_list = []
            for idx, condition in enumerate(val_conditions):
                target_genes = parse_condition_genes(condition)
                if not target_genes:
                    continue
                target_indices = [
                    gene_name_to_idx[g] for g in target_genes if g in gene_name_to_idx
                ]
                if not target_indices:
                    continue
                val_score_list.append(val_scores[idx])
                val_targets.append(target_indices)

            if not val_score_list:
                continue

            metrics = compute_gene_metrics(
                scores=val_score_list,
                targets=val_targets,
                top_k_values=top_k_values,
            )
            score = metrics.get(selection_metric, -np.inf)
            print(f"  - n_comp={n_comp}, k={k}: {selection_metric}={score:.4f}")

            if score > best_score:
                best_score = score
                best_params = {"n_components": n_comp, "k": k}
                best_val_metrics = metrics

    if not best_params:
        raise RuntimeError("No valid validation metrics; check splits and labels.")

    print("\nSelected params:")
    print(f"  - n_components={best_params['n_components']}")
    print(f"  - k={best_params['k']}")

    print("\n[4/4] Evaluating on test set...")
    test_scores = score_conditions(
        train_profiles=train_profiles,
        train_labels=train_labels,
        query_profiles=test_profiles,
        n_components=best_params["n_components"],
        standardize=standardize,
        random_state=random_state,
        k=best_params["k"],
        metric=metric,
        weight_mode=weight_mode,
        weight_normalize=weight_normalize,
        clip_negative_similarity=clip_negative_similarity,
        eps=eps,
    )

    test_targets = []
    test_score_list = []
    for idx, condition in enumerate(test_conditions):
        target_genes = parse_condition_genes(condition)
        if not target_genes:
            continue
        target_indices = [
            gene_name_to_idx[g] for g in target_genes if g in gene_name_to_idx
        ]
        if not target_indices:
            continue
        test_score_list.append(test_scores[idx])
        test_targets.append(target_indices)

    metrics = compute_gene_metrics(
        scores=test_score_list,
        targets=test_targets,
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
        "n_evaluated": len(test_targets),
        "best_params": best_params,
        "val_metrics": best_val_metrics,
    }

    output_path = Path(args.output)
    if args.output == "results/pca_knn/eval_results.json":
        if eval_cfg.get("output_path"):
            output_path = Path(eval_cfg["output_path"])
        else:
            logging_cfg = config.get("logging", {})
            base_dir = logging_cfg.get("output_dir", "results/pca_knn")
            output_path = Path(base_dir) / "eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Results Summary")
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
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
