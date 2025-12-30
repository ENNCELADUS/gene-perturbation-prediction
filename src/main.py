#!/usr/bin/env python
"""
Main entry point for reverse perturbation prediction workflows.

Supports multiple modes:
- data: Load and prepare data splits (default)
- train/build_db/evaluate/full: Route A forward model pipeline
- route_b1_train/route_b1_eval/route_b1_full: Route B1 gene-score pipeline
- tga: Target-Gene Activation baseline evaluation

Usage:
    # Data preparation only
    python -m src.main --config src/configs/scgpt_discriminative.yaml --mode data

    # Route B1 training
    run_ddp -m src.main --config src/configs/scgpt_discriminative.yaml --mode route_b1_train

    # Route B1 evaluation
    python -m src.main --config src/configs/scgpt_discriminative.yaml --mode route_b1_eval

    # TGA baseline
    python -m src.main --config src/configs/tga.yaml --mode tga
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from .data import load_perturb_data, NormanConditionSplitter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Route A/B: Reverse Perturbation Prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., src/configs/scgpt_discriminative.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="data",
        choices=[
            "data",
            "train",
            "build_db",
            "evaluate",
            "full",
            "route_b1_train",
            "route_b1_eval",
            "route_b1_full",
            "tga",
        ],
        help="Operation mode: data, train, build_db, evaluate, or full",
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
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (for build_db and evaluate modes)",
    )
    parser.add_argument(
        "--reference_db",
        type=str,
        default=None,
        help="Reference database path (for evaluate mode)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (for smoke testing)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, args) -> dict:
    """
    Load dataset and save condition split (DATA mode).

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Results dictionary with dataset summary
    """
    # Setup output directory
    output_dir = args.output_dir or config["logging"]["output_dir"]
    experiment_name = args.experiment_name or config["logging"].get(
        "experiment_name", config["model"]["encoder"]
    )
    if args.mode == "data":
        run_dir = None
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / experiment_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Data Loading - {config['model']['encoder'].upper()}")
    print("=" * 60)
    if run_dir is not None:
        print(f"Output: {run_dir}")

    # Load data
    print("\n[1/2] Loading dataset...")
    cond_split_config = config.get("condition_split", {})

    # Check if split file exists
    split_path = Path(cond_split_config.get("output_path", ""))
    use_existing_split = False

    if split_path.exists():
        print(f"  ✓ Using existing split: {split_path}")
        use_existing_split = True

    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=split_path if use_existing_split else None,
        seen_single_train_ratio=cond_split_config.get("seen_single_train_ratio", 0.9),
        double_train_ratio=cond_split_config.get("double_train_ratio", 0.7),
        double_val_ratio=cond_split_config.get("double_val_ratio", 0.15),
        double_freq_bins=cond_split_config.get("double_freq_bins", 3),
        double_count_bins=cond_split_config.get("double_count_bins", 3),
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
        print(f"  - Test strata: {summary.get('test_strata_counts', {})}")

    # Save condition split artifact
    print("\n[2/2] Saving condition split artifact...")
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
        splitter_temp = NormanConditionSplitter(seed=cond_split_config.get("seed", 42))
        split_summary = splitter_temp.summary(dataset.condition_split)
        print(f"  - Split summary: {split_summary}")

    results = {"config": config, "summary": summary}

    if run_dir is not None:
        with open(run_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print(f"Summary saved to: {run_dir / 'summary.json'}")
        print("=" * 60)

    return results


def run_train(config: dict, args) -> dict:
    """Run forward model training."""
    from .train.train_forward import main as train_main
    import sys

    # Override sys.argv for the train script
    sys.argv = [
        "train_forward",
        "--config",
        args.config,
    ]
    if args.output_dir:
        sys.argv.extend(["--output_dir", args.output_dir])
    if args.max_steps:
        sys.argv.extend(["--max_steps", str(args.max_steps)])

    print("\n" + "=" * 60)
    print("MODE: TRAIN - Forward Model")
    print("=" * 60)

    train_main()

    return {"status": "training_complete"}


def run_build_db(config: dict, args) -> dict:
    """Build reference database."""
    from .train.build_reference import main as build_db_main
    import sys

    # Determine checkpoint path
    checkpoint = args.checkpoint or "results/forward/best_model.pt"
    output = "results/forward/reference_db.pkl"

    sys.argv = [
        "build_reference",
        "--config",
        args.config,
        "--checkpoint",
        checkpoint,
        "--output",
        output,
    ]

    print("\n" + "=" * 60)
    print("MODE: BUILD_DB - Reference Database")
    print("=" * 60)

    build_db_main()

    return {"status": "database_built", "path": output}


def run_evaluate(config: dict, args) -> dict:
    """Run retrieval evaluation."""
    from .evaluate.evaluate_retrieval import main as eval_main
    import sys

    # Determine reference db path
    reference_db = args.reference_db or "results/forward/reference_db.pkl"
    output = "results/forward/eval_results.json"

    sys.argv = [
        "evaluate_retrieval",
        "--config",
        args.config,
        "--reference_db",
        reference_db,
        "--output",
        output,
    ]

    print("\n" + "=" * 60)
    print("MODE: EVALUATE - Retrieval Performance")
    print("=" * 60)

    eval_main()

    return {"status": "evaluation_complete", "results_path": output}


def run_full_pipeline(config: dict, args) -> dict:
    """Run complete pipeline: train → build_db → evaluate."""
    print("\n" + "=" * 60)
    print("MODE: FULL - Complete Pipeline")
    print("=" * 60)

    # 1. Data preparation
    print("\n[STAGE 1/4] Data Preparation")
    run_pipeline(config, args)

    # 2. Training
    print("\n[STAGE 2/4] Training Forward Model")
    run_train(config, args)

    # 3. Build reference database
    print("\n[STAGE 3/4] Building Reference Database")
    run_build_db(config, args)

    # 4. Evaluation
    print("\n[STAGE 4/4] Evaluating Retrieval")
    results = run_evaluate(config, args)

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)

    return results


def run_route_b1_train(config: dict, args) -> dict:
    """Run Route B1 gene-score training."""
    from .train.train_gene_score import main as train_main
    import sys

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        logging_config = config.get("logging", {})
        base_dir = logging_config.get("output_dir", "results/gene_score")
        output_dir = Path(base_dir)

    sys.argv = [
        "train_gene_score",
        "--config",
        args.config,
        "--output_dir",
        str(output_dir),
    ]
    if args.max_steps:
        sys.argv.extend(["--max_steps", str(args.max_steps)])

    print("\n" + "=" * 60)
    print("MODE: ROUTE_B1_TRAIN - Gene-Score Model")
    print("=" * 60)

    train_main()

    return {"status": "route_b1_training_complete"}


def run_route_b1_eval(config: dict, args) -> dict:
    """Run Route B1 gene-score evaluation."""
    from .evaluate.evaluate_gene_score import main as eval_main
    import sys

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        logging_config = config.get("logging", {})
        base_dir = logging_config.get("output_dir", "results/gene_score")
        output_dir = Path(base_dir)

    checkpoint = args.checkpoint or str(output_dir / "best_model.pt")
    eval_config = config.get("evaluate", {})
    output = eval_config.get("output_path", str(output_dir / "eval_results.json"))

    sys.argv = [
        "evaluate_gene_score",
        "--config",
        args.config,
        "--checkpoint",
        checkpoint,
        "--output",
        output,
    ]

    print("\n" + "=" * 60)
    print("MODE: ROUTE_B1_EVAL - Gene-Score Evaluation")
    print("=" * 60)

    eval_main()

    return {"status": "route_b1_evaluation_complete", "results_path": output}


def run_route_b1_full(config: dict, args) -> dict:
    """Run Route B1 pipeline: train → evaluate."""
    print("\n" + "=" * 60)
    print("MODE: ROUTE_B1_FULL - Complete Pipeline")
    print("=" * 60)

    print("\n[STAGE 1/2] Training Gene-Score Model")
    run_route_b1_train(config, args)

    print("\n[STAGE 2/2] Evaluating Gene-Score Model")
    results = run_route_b1_eval(config, args)

    print("\n" + "=" * 60)
    print("ROUTE B1 PIPELINE COMPLETE")
    print("=" * 60)

    return results


def run_tga(config: dict, args) -> dict:
    """Run TGA baseline evaluation."""
    from .evaluate.evaluate_tga import main as tga_main
    import sys

    output = config.get("evaluation", {}).get(
        "output_path", "results/tgd/eval_results.json"
    )

    sys.argv = [
        "evaluate_tgd",
        "--config",
        args.config,
        "--output",
        output,
    ]

    print("\n" + "=" * 60)
    print("MODE: TGA - Target-Gene Activation Baseline")
    print("=" * 60)

    tga_main()

    return {"status": "tga_evaluation_complete", "results_path": output}


def main():
    """Main entry point with mode dispatch."""
    args = parse_args()
    config = load_config(args.config)

    # Dispatch based on mode
    if args.mode == "data":
        results = run_pipeline(config, args)
    elif args.mode == "train":
        results = run_train(config, args)
    elif args.mode == "build_db":
        results = run_build_db(config, args)
    elif args.mode == "evaluate":
        results = run_evaluate(config, args)
    elif args.mode == "full":
        results = run_full_pipeline(config, args)
    elif args.mode == "route_b1_train":
        results = run_route_b1_train(config, args)
    elif args.mode == "route_b1_eval":
        results = run_route_b1_eval(config, args)
    elif args.mode == "route_b1_full":
        results = run_route_b1_full(config, args)
    elif args.mode == "tga":
        results = run_tga(config, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return results


if __name__ == "__main__":
    main()
