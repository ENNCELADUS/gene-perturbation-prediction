import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import sys

# Ensure src can be imported if running from src directory or root
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
from src.model.wrapper import ScGPTWrapper
from src.model.baseline import BaselineWrapper
from src.data.loader import PerturbationDataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="src/configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model_type",
        default="scgpt",
        choices=["scgpt", "baseline"],
        help="Model type to run (scgpt or baseline)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create output dir
    base_output_dir = Path(config["paths"]["output_dir"])
    output_dir = base_output_dir / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ZeroShotPerturbation", log_file=output_dir / "run.log")
    logger.info(
        f"Starting Zero-Shot Perturbation Pipeline with {args.model_type} model"
    )
    logger.info(f"Results will be saved to: {output_dir}")

    # 2. Initialize Components
    if args.model_type == "baseline":
        logger.info("Initializing Baseline Model Wrapper...")
        model_wrapper = BaselineWrapper(config, logger)
    else:
        logger.info("Initializing scGPT Model Wrapper...")
        model_wrapper = ScGPTWrapper(config, logger)

    logger.info("Initializing Data Loader...")
    data_loader = PerturbationDataLoader(config, model_wrapper.vocab, logger)

    # 3. Main Inference Loop
    targets = data_loader.get_test_targets()
    results = []

    logger.info(f"Processing {len(targets)} target genes...")

    for target in targets:
        if target == config["inference"]["control_target_gene"]:
            continue

        logger.info(f"Predicting for target: {target}")

        # Prepare Input
        batch_data = data_loader.prepare_perturbation_batch(target)
        if batch_data is None:
            continue

        # Predict
        with torch.no_grad():
            # We need gene_ids for the model to know which token maps to which column
            # Assuming column order matches vocab mapping done in loader
            # We can construct gene_ids from test_adata.var['id_in_vocab']
            gene_ids = np.array(data_loader.test_adata.var["id_in_vocab"])

            pred_expression = model_wrapper.predict(
                batch_data,
                gene_ids=gene_ids,
                amp=False,  # Disable amp for stability in simple inference unless needed
            )

        # Get Ground Truth
        ground_truth_adata = data_loader.get_target_ground_truth(target)
        if isinstance(ground_truth_adata.X, np.ndarray):
            truth_expression = ground_truth_adata.X
        else:
            truth_expression = ground_truth_adata.X.toarray()

        # Calculate Metrics
        # We compare Mean Predicted vs Mean Truth (Pseudo-bulk comparison often used)
        pred_mean = pred_expression.cpu().numpy().mean(axis=0)
        truth_mean = truth_expression.mean(axis=0)

        metrics = compute_metrics(truth_mean, pred_mean)
        metrics["target_gene"] = target
        results.append(metrics)

        logger.info(
            f"Target {target} - MSE: {metrics['mse']:.4f}, Pearson: {metrics['pearson']:.4f}"
        )

    # 4. Save Results
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "perturbation_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Calculate Overall Mean
    if not results_df.empty:
        logger.info(f"Overall Mean MSE: {results_df['mse'].mean():.4f}")
        logger.info(f"Overall Mean Pearson: {results_df['pearson'].mean():.4f}")
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
