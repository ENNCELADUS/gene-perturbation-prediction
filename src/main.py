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
    compute_mae_top2k,
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
    # Map model_type to output folder name
    output_folder_map = {
        "scgpt": "scgpt_zeroshot",
        "scgpt_finetuned": "scgpt_finetuned",
        "baseline": "baseline",
    }
    output_folder = output_folder_map.get(args.model_type, args.model_type)
    output_dir = base_output_dir / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ZeroShotPerturbation", log_file=output_dir / "run.log")
    logger.info(f"Starting Perturbation Pipeline with {args.model_type} model")
    logger.info(f"Model directory: {config['paths']['model_dir']}")
    logger.info(f"Results will be saved to: {output_dir}")

    # Get metrics config
    mae_top_k = config.get("metrics", {}).get("mae_top_k", 2000)

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

    # Get batch size for processing
    eval_batch_size = config["inference"].get("eval_batch_size", 16)

    for target in targets:
        if target == config["inference"]["control_target_gene"]:
            continue

        logger.info(f"Predicting for target: {target}")

        # Get ground truth to determine n_cells
        ground_truth_adata = data_loader.get_target_ground_truth(target)
        n_cells = ground_truth_adata.n_obs

        if n_cells == 0:
            logger.warning(f"No ground truth cells for {target}, skipping.")
            continue

        logger.info(f"Processing {n_cells} cells in batches of {eval_batch_size}")

        # Process all cells batch by batch
        # Use target hash as base seed for reproducibility
        base_seed = hash(target) % (2**32)
        gene_ids = np.array(data_loader.test_adata.var["id_in_vocab"])

        pred_expr_list = []
        control_expr_list = []

        for batch_start in range(0, n_cells, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, n_cells)
            batch_n_cells = batch_end - batch_start

            # Use different seed for each batch to sample different control cells
            batch_seed = base_seed + batch_start

            batch_result = data_loader.prepare_perturbation_batch(
                target, n_cells=batch_n_cells, seed=batch_seed, return_control_expr=True
            )
            if batch_result is None:
                continue

            batch_data, batch_control_expr = batch_result
            control_expr_list.append(batch_control_expr)

            # Predict
            with torch.no_grad():
                pred_expression = model_wrapper.predict(
                    batch_data,
                    gene_ids=gene_ids,
                    amp=True,
                )

            # Convert predictions to numpy
            if isinstance(pred_expression, torch.Tensor):
                batch_pred_expr = pred_expression.cpu().numpy()
            else:
                batch_pred_expr = np.asarray(pred_expression)

            pred_expr_list.append(batch_pred_expr)

            # Clean up GPU memory after each batch
            del batch_data, pred_expression
            gc.collect()
            torch.cuda.empty_cache()

        # Skip if no predictions were made
        if not pred_expr_list:
            logger.warning(f"No predictions generated for {target}, skipping.")
            continue

        # Concatenate all batch predictions
        pred_expr = np.concatenate(pred_expr_list, axis=0)
        control_expr = np.concatenate(control_expr_list, axis=0)

        # Get ground truth expression (use all cells)
        if isinstance(ground_truth_adata.X, np.ndarray):
            truth_expr = ground_truth_adata.X
        else:
            truth_expr = ground_truth_adata.X.toarray()

        # Compute DES metric
        try:
            de_metrics = compute_de_comparison_metrics(
                control_expr=control_expr,
                pred_expr=pred_expr,
                truth_expr=truth_expr,
                gene_names=gene_names,
                fdr_threshold=0.05,
                threads=args.threads,
            )
        except Exception as e:
            logger.warning(f"DE metrics failed for {target}: {e}")
            de_metrics = {
                "des": np.nan,
                "n_de_truth": 0,
                "n_de_pred": 0,
            }

        # Compute MAE_top2k metric
        mae = compute_mae_top2k(
            pred_expr=pred_expr,
            truth_expr=truth_expr,
            control_mean=control_mean,
            top_k=mae_top_k,
        )

        metrics = {
            "target_gene": target,
            "n_cells": n_cells,
            "des": de_metrics["des"],
            "mae_top2k": mae,
            "n_de_truth": de_metrics["n_de_truth"],
            "n_de_pred": de_metrics["n_de_pred"],
        }
        results.append(metrics)

        # Accumulate deltas for PDS
        pred_deltas[target] = compute_pseudobulk_delta(pred_expr, control_mean)
        truth_deltas[target] = compute_pseudobulk_delta(truth_expr, control_mean)

        # Get target gene index (not used in cosine-based PDS but kept for compatibility)
        try:
            target_gene_indices[target] = data_loader.test_adata.var_names.get_loc(
                target
            )
        except KeyError:
            target_gene_indices[target] = -1

        logger.info(
            f"Target {target} - DES: {metrics['des']:.4f}, "
            f"MAE_top2k: {metrics['mae_top2k']:.4f}, n_DE: {metrics['n_de_truth']}"
        )

    # 4. Compute PDS (global metric using cosine similarity ranking)
    if pred_deltas:
        npds, pds_ranks = compute_pds(pred_deltas, truth_deltas, target_gene_indices)
        # Add PDS rank to results
        for r in results:
            r["pds_rank"] = pds_ranks.get(r["target_gene"], np.nan)
    else:
        npds = np.nan

    # 5. Save Results
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        "target_gene",
        "n_cells",
        "pds_rank",
        "des",
        "mae_top2k",
        "n_de_truth",
        "n_de_pred",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    csv_path = output_dir / "perturbation_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Calculate and log overall metrics
    if not results_df.empty:
        mean_des = results_df["des"].mean()
        mean_mae = results_df["mae_top2k"].mean()

        logger.info("=" * 50)
        logger.info("Overall Metrics:")
        logger.info(f"  nPDS (normalized):     {npds:.4f}")
        logger.info(f"  Mean DES:              {mean_des:.4f}")
        logger.info(f"  Mean MAE_top2k:        {mean_mae:.4f}")
        logger.info("=" * 50)
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
