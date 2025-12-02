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
from src.model.zeroshot import ScGPTWrapper
from src.model.baseline import BaselineWrapper
from src.data.loader import PerturbationDataLoader


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="src/configs/finetune.yaml", help="Path to config file"
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
        # Use finetuned_model_dir for finetuned model evaluation
        config["paths"]["model_dir"] = config["paths"].get(
            "finetuned_model_dir", "model/scGPT_finetuned"
        )

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

    # Fit baseline model on training data (pre-compute mean expression)
    if args.model_type == "baseline":
        logger.info("Fitting baseline model on training data...")
        train_adata = data_loader.get_train_adata()
        model_wrapper.fit_from_adata(
            train_adata,
            condition_key="target_gene",  # Column in train_adata.obs
            control_key=config["inference"]["control_target_gene"],
        )

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
                "n_intersect": 0,
            }

        # Compute MAE_top2k metric
        mae = compute_mae_top2k(
            pred_expr=pred_expr,
            truth_expr=truth_expr,
            control_mean=control_mean,
            top_k=mae_top_k,
        )

        # Per-perturbation metrics per eval_metrics.md Section 4.3.2
        metrics = {
            "perturbation_id": target,
            "n_cells": n_cells,
            "des_k": de_metrics["des"],
            "mae_top2000_k": mae,
            "n_true_de": de_metrics["n_de_truth"],
            "n_pred_de": de_metrics["n_de_pred"],
            "n_intersect": de_metrics.get("n_intersect", 0),
        }
        results.append(metrics)

        # Accumulate deltas for PDS
        pred_deltas[target] = compute_pseudobulk_delta(pred_expr, control_mean)
        truth_deltas[target] = compute_pseudobulk_delta(truth_expr, control_mean)

        # Get target gene index (not used in cosine-based PDS, kept for compatibility)
        try:
            target_gene_indices[target] = data_loader.test_adata.var_names.get_loc(
                target
            )
        except KeyError:
            target_gene_indices[target] = -1

        logger.info(
            f"Target {target} - DES: {metrics['des_k']:.4f}, "
            f"MAE_top2k: {metrics['mae_top2000_k']:.4f}, n_DE: {metrics['n_true_de']}"
        )

    # 4. Compute PDS (global metric using cosine similarity ranking)
    n_perts = len(pred_deltas)
    if pred_deltas:
        pds_result = compute_pds(pred_deltas, truth_deltas, target_gene_indices)
        mean_rank = pds_result["mean_rank"]
        npds = pds_result["npds"]
        pds_ranks = pds_result["ranks"]
        cosine_self = pds_result["cosine_self"]

        # Add PDS rank and cosine_self to results per eval_metrics.md Section 4.3.2
        for r in results:
            pert_id = r["perturbation_id"]
            r["rank_Rk"] = pds_ranks.get(pert_id, np.nan)
            r["rank_Rk_norm"] = (
                pds_ranks.get(pert_id, np.nan) / n_perts if n_perts > 0 else np.nan
            )
            r["cosine_self"] = cosine_self.get(pert_id, np.nan)
    else:
        mean_rank = np.nan
        npds = np.nan

    # 5. Save Results
    results_df = pd.DataFrame(results)

    # Reorder columns per eval_metrics.md Section 4.3.2
    col_order = [
        "perturbation_id",
        "rank_Rk",
        "rank_Rk_norm",
        "cosine_self",
        "mae_top2000_k",
        "des_k",
        "n_true_de",
        "n_pred_de",
        "n_intersect",
        "n_cells",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    csv_path = output_dir / "perturbation_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Calculate and log overall metrics per eval_metrics.md Section 4.3.1
    if not results_df.empty:
        mean_des = results_df["des_k"].mean()
        mean_mae = results_df["mae_top2000_k"].mean()

        # Compute normalized sub-scores (same as validation)
        # s_pds = (N - MeanRank) / (N - 1)
        if n_perts > 1 and not np.isnan(mean_rank):
            s_pds = (n_perts - mean_rank) / (n_perts - 1)
        elif n_perts == 1 and not np.isnan(mean_rank):
            s_pds = 1.0 if mean_rank == 1 else 0.0
        else:
            s_pds = np.nan

        # s_mae = 1 / (1 + mae / tau), tau = median of per-perturbation MAE
        mae_values = results_df["mae_top2000_k"].dropna().values
        if len(mae_values) > 0 and not np.isnan(mean_mae):
            tau = float(np.median(mae_values))
            if tau > 0:
                s_mae = 1.0 / (1.0 + mean_mae / tau)
            else:
                s_mae = 1.0
        else:
            s_mae = np.nan

        # s_des = des (already 0-1)
        s_des = mean_des

        # s_geo = (s_pds * s_mae * s_des)^(1/3)
        if (
            not np.isnan(s_pds)
            and not np.isnan(s_mae)
            and not np.isnan(s_des)
            and s_pds > 0
            and s_mae > 0
            and s_des > 0
        ):
            s_geo = (s_pds * s_mae * s_des) ** (1 / 3)
        else:
            s_geo = np.nan

        l_geo = 1.0 - s_geo if not np.isnan(s_geo) else np.nan

        logger.info("=" * 50)
        logger.info("Final Test Metrics (per eval_metrics.md Section 4.3.1):")
        logger.info(f"  pds_mean_rank:         {mean_rank:.2f}")
        logger.info(f"  pds_nrank:             {npds:.4f}")
        logger.info(f"  mae_top2000:           {mean_mae:.4f}")
        logger.info(f"  des:                   {mean_des:.4f}")
        logger.info("-" * 50)
        logger.info("Normalized Sub-Scores:")
        logger.info(f"  s_pds:                 {s_pds:.4f}")
        logger.info(f"  s_mae:                 {s_mae:.4f}")
        logger.info(f"  s_des:                 {s_des:.4f}")
        logger.info("-" * 50)
        logger.info("Composite Metrics:")
        logger.info(f"  s_geo:                 {s_geo:.4f}")
        logger.info(f"  l_geo:                 {l_geo:.4f}")
        logger.info("=" * 50)
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
