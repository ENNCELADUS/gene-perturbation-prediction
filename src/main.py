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
    compute_overall_score,
    compute_pds,
    compute_pseudobulk_delta,
)
from src.model.zeroshot import ScGPTWrapper
from src.model.baseline import BaselineWrapper
from src.model.random_forest import RandomForestWrapper
from src.model.ridge import RidgeWrapper
from src.model.linear import LinearWrapper
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
        choices=[
            "scgpt",
            "scgpt_finetuned",
            "baseline",
            "linear",
            "ridge",
            "random_forest",
        ],
        help="Model type to run (scgpt, scgpt_finetuned, baseline, or linear)",
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
    elif args.model_type == "linear":
        # Use linear_model_dir for linear model evaluation
        config["paths"]["linear_model_dir"] = config["paths"].get(
            "linear_model_dir", "model/linear_regression"
        )

    # Create output dir
    base_output_dir = Path(config["paths"]["output_dir"])
    # Map model_type to output folder name
    output_folder_map = {
        "scgpt": "scgpt_zeroshot",
        "scgpt_finetuned": "scgpt_finetuned",
        "baseline": "baseline",
        "linear": "linear",
        "ridge": "ridge",
        "random_forest": "random_forest",
    }
    output_folder = output_folder_map.get(args.model_type, args.model_type)

    # If config already points to a model-specific folder, use it directly; otherwise append folder
    if base_output_dir.name in output_folder_map.values():
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / output_folder

    log_dir = Path(config["paths"].get("log_dir", output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ZeroShotPerturbation", log_file=log_dir / "run.log")
    logger.info(f"Starting Perturbation Pipeline with {args.model_type} model")
    logger.info(f"Model directory: {config['paths']['model_dir']}")
    logger.info(f"Results will be saved to: {output_dir}")

    # Get metrics config
    mae_top_k = config.get("metrics", {}).get("mae_top_k", 2000)

    # 2. Initialize Components
    if args.model_type == "baseline":
        logger.info("Initializing Baseline Model Wrapper...")
        model_wrapper = BaselineWrapper(config, logger)
    elif args.model_type in ("ridge", "random_forest"):
        logger.info("Initializing Ridge Model Wrapper...")
        model_wrapper = RidgeWrapper(config, logger)
    elif args.model_type == "random_forest":
        logger.info("Initializing Random Forest Model Wrapper...")
        model_wrapper = RandomForestWrapper(config, logger)
    elif args.model_type == "linear":
        logger.info("Initializing Linear Model Wrapper...")
        model_wrapper = LinearWrapper(config, logger)
        linear_model_dir = Path(config["paths"]["linear_model_dir"])
        model_wrapper.load_model(linear_model_dir)
    else:
        logger.info("Initializing scGPT Model Wrapper...")
        model_wrapper = ScGPTWrapper(config, logger)

    logger.info("Initializing Data Loader...")
    data_loader = PerturbationDataLoader(config, model_wrapper.vocab, logger)

    # Fit baseline/ridge models on training data (pre-compute parameters)
    if args.model_type in ("baseline", "ridge", "random_forest"):
        logger.info(f"Fitting {args.model_type} model on training data...")
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
        i = 0
        for batch_start in range(0, n_cells, eval_batch_size):
            i += 1
            batch_end = min(batch_start + eval_batch_size, n_cells)
            batch_n_cells = batch_end - batch_start

            # Use different seed for each batch to sample different control cells
            # batch_seed = base_seed + batch_start
            batch_seed = base_seed + batch_start - (i % 6) * eval_batch_size

            batch_result = data_loader.prepare_perturbation_batch(
                target, n_cells=batch_n_cells, seed=batch_seed, return_control_expr=True
            )
            if batch_result is None:
                continue

            batch_data, batch_control_expr = batch_result
            control_expr_list.append(batch_control_expr)

            # Predict
            with torch.no_grad():
                if args.model_type == "ridge" or args.model_type == "random_forest":
                    pred_expression = model_wrapper.predict(
                        batch_data,
                        gene_ids=gene_ids,
                        amp=True,
                        target_gene=target,
                    )
                else:
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
        pds_nrank = pds_result["npds"]
        pds_ranks = pds_result["ranks"]
        cosine_self = pds_result["cosine_self"]

        # Add PDS rank, cosine_self, and per-perturbation overall_score
        for r in results:
            pert_id = r["perturbation_id"]
            r["rank_Rk"] = pds_ranks.get(pert_id, np.nan)
            r["rank_Rk_norm"] = (
                pds_ranks.get(pert_id, np.nan) / n_perts if n_perts > 0 else np.nan
            )
            r["cosine_self"] = cosine_self.get(pert_id, np.nan)

            # Compute per-perturbation overall_score using centralized function
            des_k = r["des_k"]
            mae_k = r["mae_top2000_k"]
            rank_norm_k = r["rank_Rk_norm"]

            # pds_k uses normalized rank (lower is better per docs/eval_metrics.md)
            pds_k = rank_norm_k

            # Use centralized function for scaled metrics and overall score
            score_result = compute_overall_score(pds_k, mae_k, des_k)
            r["overall_score"] = score_result["overall_score"]
    else:
        pds_nrank = np.nan
        # Add overall_score as NaN for results without PDS
        for r in results:
            r["overall_score"] = np.nan

    # 5. Save Results
    results_df = pd.DataFrame(results)

    # Reorder columns: de_metrics and overall_score first, other details last
    col_order = [
        "perturbation_id",
        # Three DE metrics (per-perturbation) and overall_score
        "des_k",
        "mae_top2000_k",
        "rank_Rk_norm",  # PDS per-perturbation (normalized rank)
        "overall_score",
        # Other details
        "rank_Rk",
        "cosine_self",
        "n_true_de",
        "n_pred_de",
        "n_intersect",
        "n_cells",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    csv_path = output_dir / "perturbation_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Log overall metrics using "Score of Averages" approach per eval_metrics.md §4.2.2
    if not results_df.empty:
        mean_des = results_df["des_k"].mean()
        mean_mae = results_df["mae_top2000_k"].mean()

        pds = pds_nrank

        # Compute overall_score from meaned metrics (NOT: average of per-perturbation scores)
        score_result = compute_overall_score(pds, mean_mae, mean_des)
        overall_score = score_result["overall_score"]

        logger.info("=" * 50)
        logger.info("Final Test Metrics (per eval_metrics.md Section 4.2.2):")
        logger.info("-" * 50)
        logger.info("Three DE Metrics (raw, averaged):")
        logger.info(f"  pds_nrank:             {pds:.4f}")
        logger.info(f"  mae:                   {mean_mae:.4f}")
        logger.info(f"  des:                   {mean_des:.4f}")
        logger.info("-" * 50)
        logger.info("Overall Score (0-100, from meaned metrics):")
        logger.info(f"  overall_score:         {overall_score:.2f}")
        logger.info("=" * 50)
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
