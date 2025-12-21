#!/usr/bin/env python
"""
Main entry point for reverse perturbation prediction.

Usage:
    python -m src.main --config src/configs/pca.yaml
    python -m src.main --config src/configs/scgpt.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .data import load_perturb_data, ConditionSplitter
from .evaluate import (
    CellRetrievalEvaluator,
    ClassifierEvaluator,
    generate_error_report,
    generate_report,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reverse Perturbation Prediction Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., src/configs/pca.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override experiment name from config",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        choices=["in_dist", "unseen_combo", "unseen_gene"],
        help="Generalization track for condition-level evaluation",
    )
    parser.add_argument(
        "--error_analysis",
        action="store_true",
        help="Generate detailed error analysis report",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_encoder_kwargs(config: dict) -> dict:
    """Extract encoder-specific kwargs from config."""
    model_config = config["model"]
    encoder_type = model_config["encoder"]

    if encoder_type == "pca":
        return {"n_components": model_config.get("n_components", 50)}
    elif encoder_type == "scgpt":
        return {
            "checkpoint": model_config.get("checkpoint"),
            "freeze": model_config.get("freeze_encoder", True),
            "use_lora": model_config.get("use_lora", False),
            "lora_rank": model_config.get("lora_rank", 8),
        }
    elif encoder_type == "logreg":
        return {
            "C": model_config.get("C", 1.0),
            "max_iter": model_config.get("max_iter", 1000),
            "solver": model_config.get("solver", "lbfgs"),
        }
    else:
        return {}


def run_pipeline(config: dict, args) -> dict:
    """
    Run the reverse perturbation prediction pipeline.

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Results dictionary with metrics
    """
    # Setup output directory
    output_dir = args.output_dir or config["logging"]["output_dir"]
    experiment_name = args.experiment_name or config["logging"].get(
        "experiment_name", config["model"]["encoder"]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Reverse Perturbation Prediction - {config['model']['encoder'].upper()}")
    print("=" * 60)
    print(f"Output: {run_dir}")

    # Load data
    print("\n[1/4] Loading dataset...")
    split_config = config["split"]
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        split_path=split_config.get("output_path"),
        min_cells_per_condition=split_config.get("min_cells_per_condition", 50),
        query_fraction=split_config.get("query_fraction", 0.2),
        min_query_cells=split_config.get("min_query_cells", 10),
        seed=split_config.get("seed", 42),
    )

    # Print summary
    summary = dataset.summary()
    print(f"  - Total cells: {summary['n_cells']}")
    print(f"  - Genes: {summary['n_genes']}")
    print(f"  - Valid conditions: {summary['n_valid_conditions']}")
    print(f"  - Ref cells: {summary['n_ref_cells']}")
    print(f"  - Query cells: {summary['n_query_cells']}")
    print(f"  - Dropped conditions: {summary['n_dropped_conditions']}")

    # Save split artifact
    print("\n[2/4] Saving split artifact...")
    split_path = Path(
        split_config.get(
            "output_path", f"splits/cell_split_seed{split_config['seed']}.json"
        )
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.split.save(split_path)
    print(f"  - Saved to: {split_path}")

    # Optional: condition-level split for generalization tracks
    track = args.track or config.get("track")
    if track:
        cond_cfg = config.get("condition_split", {})
        splitter = ConditionSplitter(
            train_ratio=cond_cfg.get("train_ratio", 0.7),
            val_ratio=cond_cfg.get("val_ratio", 0.1),
            test_ratio=cond_cfg.get("test_ratio", 0.2),
            seed=cond_cfg.get("seed", split_config.get("seed", 42)),
        )
        if track == "unseen_gene":
            cond_split = splitter.split_unseen_gene(
                dataset.all_conditions,
                n_holdout_genes=cond_cfg.get("n_holdout_genes", 5),
            )
        else:
            cond_split = splitter.split(dataset.all_conditions, track=track)
        cond_out = cond_cfg.get(
            "output_path",
            f"data/norman/splits/condition_split_{track}_seed{cond_split.seed}.json",
        )
        cond_out_path = Path(cond_out)
        cond_out_path.parent.mkdir(parents=True, exist_ok=True)
        cond_split.save(cond_out_path)
        dataset.apply_condition_split(cond_split)
        config["track"] = track
        print(f"  - Condition split ({track}) saved to: {cond_out_path}")

    # Setup evaluator
    print(f"\n[3/4] Setting up {config['model']['encoder']} encoder...")
    encoder_kwargs = get_encoder_kwargs(config)
    library_config = config.get("library", {})
    eval_config = config.get("evaluate", {})
    confidence_config = config.get("confidence", {})
    query_config = config.get("query", {})
    encoder_type = config["model"]["encoder"]

    # Use ClassifierEvaluator for discriminative models, CellRetrievalEvaluator for embedding-based
    if encoder_type == "logreg":
        evaluator = ClassifierEvaluator(
            classifier_kwargs=encoder_kwargs,
            top_k=config["retrieval"]["top_k"],
            mask_perturbed=eval_config.get("mask_perturbed", True),
            query_mode=query_config.get("mode", "cell"),
            pseudo_bulk_config=query_config.get("pseudo_bulk", {}),
            candidate_source=query_config.get("candidate_source", "all"),
            query_split=query_config.get("query_split", "test"),
        )
        evaluator.setup(dataset)
        print(f"  - Classifier: {encoder_type}")
        print(f"  - Mode: discriminative (probability-based ranking)")
    else:
        evaluator = CellRetrievalEvaluator(
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            metric=config["retrieval"]["metric"],
            top_k=config["retrieval"]["top_k"],
            mask_perturbed=eval_config.get("mask_perturbed", True),
            library_type=library_config.get("type", "bootstrap"),
            n_prototypes=library_config.get("n_prototypes", 30),
            m_cells_per_prototype=library_config.get("m_cells_per_prototype", 50),
            library_seed=library_config.get("seed", 42),
            query_mode=query_config.get("mode", "cell"),
            pseudo_bulk_config=query_config.get("pseudo_bulk", {}),
            candidate_source=query_config.get("candidate_source", "all"),
            query_split=query_config.get("query_split", "test"),
        )
        evaluator.setup(dataset)
        print(f"  - Encoder: {encoder_type}")
        print(f"  - Library type: {library_config.get('type', 'bootstrap')}")
        print(f"  - Similarity: {config['retrieval']['metric']}")

    results = {"config": config, "summary": summary}

    # Evaluate
    print("\n[4/4] Evaluating...")
    needs_details = bool(
        args.error_analysis
        or confidence_config.get("enable", False)
        or eval_config.get("mask_ablation", False)
        or query_config.get("pseudo_bulk_curve", {}).get("enable", False)
    )

    if needs_details:
        metrics, details = evaluator.evaluate_with_details(
            dataset, confidence_config=confidence_config
        )
    else:
        metrics = evaluator.evaluate(dataset, confidence_config=confidence_config)
    results["metrics"] = metrics
    print("  Metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Evaluate with masking OFF (ablation) if requested
    if eval_config.get("mask_ablation", False):
        print("\n  Running mask OFF ablation...")
        if encoder_type == "logreg":
            evaluator_no_mask = ClassifierEvaluator(
                classifier_kwargs=encoder_kwargs,
                top_k=config["retrieval"]["top_k"],
                mask_perturbed=False,
                query_mode=query_config.get("mode", "cell"),
                pseudo_bulk_config=query_config.get("pseudo_bulk", {}),
                candidate_source=query_config.get("candidate_source", "all"),
                query_split=query_config.get("query_split", "test"),
            )
        else:
            evaluator_no_mask = CellRetrievalEvaluator(
                encoder_type=encoder_type,
                encoder_kwargs=encoder_kwargs,
                metric=config["retrieval"]["metric"],
                top_k=config["retrieval"]["top_k"],
                mask_perturbed=False,
                library_type=library_config.get("type", "bootstrap"),
                n_prototypes=library_config.get("n_prototypes", 30),
                m_cells_per_prototype=library_config.get("m_cells_per_prototype", 50),
                library_seed=library_config.get("seed", 42),
                query_mode=query_config.get("mode", "cell"),
                pseudo_bulk_config=query_config.get("pseudo_bulk", {}),
                candidate_source=query_config.get("candidate_source", "all"),
                query_split=query_config.get("query_split", "test"),
            )
        evaluator_no_mask.setup(dataset)
        metrics_no_mask = evaluator_no_mask.evaluate(
            dataset, confidence_config=confidence_config
        )
        results["metrics_mask_off"] = metrics_no_mask
        print("  Metrics (mask OFF):")
        for k, v in sorted(metrics_no_mask.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Optional pseudo-bulk performance curve
    pseudo_curve_cfg = query_config.get("pseudo_bulk_curve", {})
    if pseudo_curve_cfg.get("enable", False):
        curve = evaluator.evaluate_pseudo_bulk_curve(
            dataset,
            cells_per_bulk_values=pseudo_curve_cfg.get("cells_per_bulk_values", []),
            n_bulks=pseudo_curve_cfg.get("n_bulks", 5),
            seed=pseudo_curve_cfg.get("seed", 42),
            confidence_config=confidence_config,
        )
        results["pseudo_bulk_curve"] = curve

    # Optional error analysis report
    if args.error_analysis:
        print("\n  Generating error analysis report...")
        error_report = generate_error_report(
            details["predictions"], details["ground_truth"], k=1
        )
        results["error_analysis"] = error_report

    # Optional comparison report across runs
    report_cfg = config.get("report", {})
    if report_cfg.get("enable", False):
        result_dirs = report_cfg.get("result_dirs", [])
        if result_dirs:
            report_path = generate_report(
                result_dirs=result_dirs,
                output_dir=report_cfg.get("output_dir", "results/reports"),
                report_name=report_cfg.get("report_name", "comparison_report"),
            )
            results["comparison_report"] = str(report_path)

    # Save results
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to: {run_dir / 'metrics.json'}")
    print("=" * 60)

    return results


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    results = run_pipeline(config, args)
    return results


if __name__ == "__main__":
    main()
