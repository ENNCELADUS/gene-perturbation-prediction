import yaml
import argparse
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import sys

# Ensure src can be imported if running from src directory or root
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import get_logger
from src.utils.de_metrics import (
    compute_de_comparison_metrics,
    compute_pds,
    compute_pseudobulk_delta,
)
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
        choices=["scgpt", "scgpt_finetuned", "baseline"],
        help="Model type to run (scgpt, scgpt_finetuned, or baseline)",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Override model directory path (default: use config)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Number of threads for hpdex DE analysis (-1 = all cores)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override model directory if specified
    if args.model_dir:
        config["paths"]["model_dir"] = args.model_dir
    elif args.model_type == "scgpt_finetuned":
        # Default finetuned model path
        config["paths"]["model_dir"] = "model/scGPT_finetuned"

    # Create output dir
    base_output_dir = Path(config["paths"]["output_dir"])
    output_dir = base_output_dir / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ZeroShotPerturbation", log_file=output_dir / "run.log")
    logger.info(f"Starting Perturbation Pipeline with {args.model_type} model")
    logger.info(f"Model directory: {config['paths']['model_dir']}")
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

    # Get control mean for PDS calculation
    control_mean = data_loader.get_control_mean()
    gene_names = data_loader.get_gene_names()

    # 3. Main Inference Loop
    targets = data_loader.get_test_targets()
    results = []

    # For PDS: accumulate deltas across all perturbations
    pred_deltas = {}
    truth_deltas = {}
    target_gene_indices = {}

    logger.info(f"Processing {len(targets)} target genes...")

    for target in targets:
        if target == config["inference"]["control_target_gene"]:
            continue

        logger.info(f"Predicting for target: {target}")

        # Get ground truth to determine n_cells
        ground_truth_adata = data_loader.get_target_ground_truth(target)
        n_cells = ground_truth_adata.n_obs

        # Cap n_cells to avoid OOM (process in smaller batches)
        max_cells = config["inference"].get("max_cells", 64)
        n_cells = min(n_cells, max_cells)

        if n_cells == 0:
            logger.warning(f"No ground truth cells for {target}, skipping.")
            continue

        # Prepare Input - sample same number of control cells as ground truth
        # Use target hash as seed for reproducibility
        seed = hash(target) % (2**32)
        batch_result = data_loader.prepare_perturbation_batch(
            target, n_cells=n_cells, seed=seed, return_control_expr=True
        )
        if batch_result is None:
            continue

        batch_data, control_expr = batch_result

        # Predict
        with torch.no_grad():
            gene_ids = np.array(data_loader.test_adata.var["id_in_vocab"])
            pred_expression = model_wrapper.predict(
                batch_data,
                gene_ids=gene_ids,
                amp=True,
            )

        # Convert predictions to numpy
        if isinstance(pred_expression, torch.Tensor):
            pred_expr = pred_expression.cpu().numpy()
        else:
            pred_expr = np.asarray(pred_expression)

        # Get ground truth expression
        if isinstance(ground_truth_adata.X, np.ndarray):
            truth_expr = ground_truth_adata.X
        else:
            truth_expr = ground_truth_adata.X.toarray()

        # Compute DE-based metrics
        try:
            metrics = compute_de_comparison_metrics(
                control_expr=control_expr,
                pred_expr=pred_expr,
                truth_expr=truth_expr,
                gene_names=gene_names,
                fdr_threshold=0.05,
                threads=args.threads,
            )
        except Exception as e:
            logger.warning(f"DE metrics failed for {target}: {e}")
            metrics = {
                "pearson_log2fc": np.nan,
                "mse_log2fc": np.nan,
                "des": np.nan,
                "n_de_truth": 0,
                "n_de_pred": 0,
            }

        metrics["target_gene"] = target
        metrics["n_cells"] = n_cells
        results.append(metrics)

        # Accumulate deltas for PDS
        pred_deltas[target] = compute_pseudobulk_delta(pred_expr, control_mean)
        truth_deltas[target] = compute_pseudobulk_delta(truth_expr, control_mean)

        # Get target gene index for PDS (exclude from distance)
        try:
            target_gene_indices[target] = data_loader.test_adata.var_names.get_loc(
                target
            )
        except KeyError:
            target_gene_indices[target] = -1

        logger.info(
            f"Target {target} - Pearson: {metrics['pearson_log2fc']:.4f}, "
            f"DES: {metrics['des']:.4f}, n_DE: {metrics['n_de_truth']}"
        )

        # Clean up GPU memory after each target
        del batch_data, pred_expression
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Compute PDS (global metric)
    if pred_deltas:
        mean_pds, pds_scores = compute_pds(
            pred_deltas, truth_deltas, target_gene_indices
        )
        # Add PDS to results
        for r in results:
            r["pds"] = pds_scores.get(r["target_gene"], np.nan)
    else:
        mean_pds = np.nan

    # 5. Save Results
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        "target_gene",
        "n_cells",
        "pearson_log2fc",
        "mse_log2fc",
        "des",
        "pds",
        "n_de_truth",
        "n_de_pred",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    csv_path = output_dir / "perturbation_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Calculate and log overall metrics
    if not results_df.empty:
        logger.info("=" * 50)
        logger.info("Overall Metrics:")
        logger.info(
            f"  Mean Pearson (log2FC): {results_df['pearson_log2fc'].mean():.4f}"
        )
        logger.info(f"  Mean MSE (log2FC):     {results_df['mse_log2fc'].mean():.4f}")
        logger.info(f"  Mean DES:              {results_df['des'].mean():.4f}")
        logger.info(f"  Mean PDS:              {mean_pds:.4f}")
        logger.info("=" * 50)
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
