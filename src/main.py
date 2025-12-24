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

from .data import load_perturb_data, NormanConditionSplitter
from .evaluate import (
    CellRetrievalEvaluator,
    ClassifierEvaluator,
    ScGPTClassifierEvaluator,
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
        "--finetune_checkpoint",
        type=str,
        default=None,
        help="Path to scGPT fine-tune checkpoint (retrieval head/LoRA)",
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
        checkpoint = model_config.get("checkpoint") or model_config.get(
            "pretrained_dir"
        )
        return {
            "checkpoint": checkpoint,
            "finetune_checkpoint": model_config.get("finetune_checkpoint"),
            "finetune_apply_head": model_config.get("finetune_apply_head", True),
            "finetune_apply_classifier": model_config.get(
                "finetune_apply_classifier", False
            ),
            "freeze": model_config.get("freeze_encoder", True),
            "use_lora": model_config.get("use_lora", False),
            "lora_rank": model_config.get("lora_rank", 8),
            "raw_layer_key": model_config.get("raw_layer_key"),
            "gene_alias_map_path": model_config.get("gene_alias_map_path"),
            "preprocess": model_config.get("preprocess", False),
            "preprocess_normalize_total": model_config.get(
                "preprocess_normalize_total", 1e4
            ),
            "preprocess_log1p": model_config.get("preprocess_log1p", True),
            "preprocess_binning": model_config.get("preprocess_binning"),
            "preprocess_filter_gene_by_counts": model_config.get(
                "preprocess_filter_gene_by_counts", False
            ),
            "preprocess_filter_cell_by_counts": model_config.get(
                "preprocess_filter_cell_by_counts", False
            ),
            "preprocess_subset_hvg": model_config.get("preprocess_subset_hvg", False),
            "preprocess_hvg_use_key": model_config.get("preprocess_hvg_use_key"),
            "preprocess_hvg_flavor": model_config.get(
                "preprocess_hvg_flavor", "seurat_v3"
            ),
            "preprocess_result_binned_key": model_config.get(
                "preprocess_result_binned_key", "X_binned"
            ),
            "preprocess_result_normed_key": model_config.get(
                "preprocess_result_normed_key", "X_normed"
            ),
            "preprocess_result_log1p_key": model_config.get(
                "preprocess_result_log1p_key", "X_log1p"
            ),
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
    if args.finetune_checkpoint:
        config["model"]["finetune_checkpoint"] = args.finetune_checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Reverse Perturbation Prediction - {config['model']['encoder'].upper()}")
    print("=" * 60)
    print(f"Output: {run_dir}")

    # Load data
    print("\n[1/4] Loading dataset...")
    cond_split_config = config.get("condition_split", {})

    # Check if split file exists and has matching settings
    split_path = Path(cond_split_config.get("output_path", ""))
    use_existing_split = False

    if split_path.exists():
        # Load existing split and check if settings match
        try:
            from .data import ConditionSplit

            existing_split = ConditionSplit.load(split_path)

            # Check if settings match
            expected_unseen_genes = int(
                105 * cond_split_config.get("unseen_gene_fraction", 0.15)
            )
            actual_unseen_genes = len(existing_split.unseen_genes)
            settings_match = (
                abs(actual_unseen_genes - expected_unseen_genes) <= 2
            )  # Allow small variation

            if settings_match:
                print(f"  ✓ Using existing split: {split_path}")
                print(f"    (unseen genes: {actual_unseen_genes}, matches config)")
                use_existing_split = True
            else:
                print(
                    f"  ⚠ Existing split settings mismatch (unseen genes: {actual_unseen_genes} vs expected ~{expected_unseen_genes})"
                )
                print(f"    Will generate new split...")
        except Exception as e:
            print(f"  ⚠ Could not validate existing split: {e}")
            print(f"    Will generate new split...")

    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=split_path if use_existing_split else None,
        unseen_gene_fraction=cond_split_config.get("unseen_gene_fraction", 0.15),
        seen_single_train_ratio=cond_split_config.get("seen_single_train_ratio", 0.9),
        combo_seen2_train_ratio=cond_split_config.get("combo_seen2_train_ratio", 0.7),
        combo_seen2_val_ratio=cond_split_config.get("combo_seen2_val_ratio", 0.15),
        min_cells_per_condition=cond_split_config.get("min_cells_per_condition", 50),
        min_cells_per_double=cond_split_config.get("min_cells_per_double", 30),
        seed=cond_split_config.get("seed", 42),
    )

    # Print summary
    summary = dataset.summary()
    print(f"  - Total cells: {summary['n_cells']}")
    print(f"  - Genes: {summary['n_genes']}")
    print(f"  - Total conditions: {summary['n_conditions']}")
    if summary.get("has_condition_split"):
        print(f"  - Train conditions: {summary['n_train_conditions']}")
        print(f"  - Val conditions: {summary['n_val_conditions']}")
        print(f"  - Test conditions: {summary['n_test_conditions']}")
        print(f"  - Unseen genes: {summary['n_unseen_genes']}")
        print(f"  - Test strata: {summary.get('test_strata_counts', {})}")

    # Save condition split artifact
    print("\n[2/4] Saving condition split artifact...")
    cond_split_config = config.get("condition_split", {})
    split_path = Path(
        cond_split_config.get(
            "output_path",
            f"data/norman/splits/norman_condition_split_seed{cond_split_config.get('seed', 42)}.json",
        )
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.condition_split.save(split_path)
    print(f"  - Saved to: {split_path}")

    # Print split summary
    if dataset.condition_split:
        from .data import NormanConditionSplitter

        splitter_temp = NormanConditionSplitter(seed=cond_split_config.get("seed", 42))
        split_summary = splitter_temp.summary(dataset.condition_split)
        print(f"  - Split summary: {split_summary}")

    # Setup evaluator
    print(f"\n[3/4] Setting up {config['model']['encoder']} encoder...")
    encoder_kwargs = get_encoder_kwargs(config)
    library_config = config.get("library", {})
    eval_config = config.get("evaluate", {})
    eval_mode = eval_config.get("mode", "retrieval")
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
    elif eval_mode == "classifier":
        evaluator = ScGPTClassifierEvaluator(
            encoder_kwargs=encoder_kwargs,
            top_k=config["retrieval"]["top_k"],
            mask_perturbed=eval_config.get("mask_perturbed", True),
            mask_layer_key=encoder_kwargs.get("raw_layer_key"),
            query_mode=query_config.get("mode", "cell"),
            pseudo_bulk_config=query_config.get("pseudo_bulk", {}),
            candidate_source=query_config.get("candidate_source", "all"),
            query_split=query_config.get("query_split", "test"),
        )
        evaluator.setup(dataset)
        print("  - Encoder: scgpt (classification head)")
        print("  - Mode: discriminative (probability-based ranking)")
    else:
        evaluator = CellRetrievalEvaluator(
            encoder_type=encoder_type,
            encoder_kwargs=encoder_kwargs,
            metric=config["retrieval"]["metric"],
            top_k=config["retrieval"]["top_k"],
            mask_perturbed=eval_config.get("mask_perturbed", True),
            mask_layer_key=encoder_kwargs.get("raw_layer_key"),
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
        elif eval_mode == "classifier":
            evaluator_no_mask = ScGPTClassifierEvaluator(
                encoder_kwargs=encoder_kwargs,
                top_k=config["retrieval"]["top_k"],
                mask_perturbed=False,
                mask_layer_key=encoder_kwargs.get("raw_layer_key"),
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
