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

from .data import load_norman_data
from .evaluate import RetrievalEvaluator
from .utils import save_config


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
        "--split",
        type=str,
        choices=["val", "test", "both"],
        default="both",
        help="Which split to evaluate on",
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
    run_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, run_dir / "config.yaml")

    print("=" * 60)
    print(f"Reverse Perturbation Prediction - {config['model']['encoder'].upper()}")
    print("=" * 60)
    print(f"Output: {run_dir}")

    # Load data
    print("\n[1/4] Loading Norman dataset...")
    dataset = load_norman_data(
        data_dir=config["data"]["path"],
        split=config["data"]["split"],
        seed=config["data"]["seed"],
    )
    print(f"  - Cells: {dataset.adata.n_obs}, Genes: {dataset.adata.n_vars}")
    print(f"  - Train: {len(dataset.conditions.get('train', []))} conditions")
    print(f"  - Val: {len(dataset.conditions.get('val', []))} conditions")
    print(f"  - Test: {len(dataset.conditions.get('test', []))} conditions")

    # Setup evaluator
    print(f"\n[2/4] Setting up {config['model']['encoder']} encoder...")
    encoder_kwargs = get_encoder_kwargs(config)
    evaluator = RetrievalEvaluator(
        encoder_type=config["model"]["encoder"],
        encoder_kwargs=encoder_kwargs,
        metric=config["retrieval"]["metric"],
        top_k=config["retrieval"]["top_k"],
        mask_perturbed=config["data"].get("mask_perturbed", True),
        pseudo_bulk=config["data"].get("pseudo_bulk", True),
    )
    evaluator.setup(dataset)
    print(f"  - Encoder config: {encoder_kwargs}")
    print(f"  - Similarity: {config['retrieval']['metric']}")

    results = {"config": config}

    # Evaluate on validation set
    if args.split in ["val", "both"]:
        print("\n[3/4] Evaluating on validation set...")
        val_metrics = evaluator.evaluate(dataset, split="val")
        results["val"] = val_metrics
        print("  Validation metrics:")
        for k, v in sorted(val_metrics.items()):
            print(f"    {k}: {v:.4f}")

    # Evaluate on test set
    if args.split in ["test", "both"]:
        print("\n[4/4] Evaluating on test set...")
        test_metrics = evaluator.evaluate(dataset, split="test")
        results["test"] = test_metrics
        print("  Test metrics:")
        for k, v in sorted(test_metrics.items()):
            print(f"    {k}: {v:.4f}")

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
