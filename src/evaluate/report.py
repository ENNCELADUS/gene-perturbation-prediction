"""
Comparison report generator for reverse perturbation prediction.

Generates:
- Consolidated comparison tables across models
- Confusion matrices
- Visualization plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


def aggregate_results(result_dirs: List[Path]) -> pd.DataFrame:
    """
    Aggregate metrics from multiple experiment runs.

    Args:
        result_dirs: List of result directories containing metrics.json

    Returns:
        DataFrame with columns for model, metrics, and mask status
    """
    rows = []

    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        metrics_file = result_dir / "metrics.json"

        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found, skipping")
            continue

        with open(metrics_file) as f:
            data = json.load(f)

        # Extract config info
        config = data.get("config", {})
        model_config = config.get("model", {})
        encoder_type = model_config.get("encoder", "unknown")
        exp_name = config.get("logging", {}).get("experiment_name")
        if not exp_name:
            exp_name = result_dir.parent.name
        elif exp_name == encoder_type and result_dir.parent.name != exp_name:
            exp_name = result_dir.parent.name

        # Extract mask ON metrics
        if "metrics" in data:
            row = {
                "experiment": exp_name,
                "encoder": encoder_type,
                "run_dir": str(result_dir),
                "mask": True,
            }
            row.update(data["metrics"])
            rows.append(row)

        # Extract mask OFF metrics if present
        if "metrics_mask_off" in data:
            row = {
                "experiment": exp_name,
                "encoder": encoder_type,
                "run_dir": str(result_dir),
                "mask": False,
            }
            row.update(data["metrics_mask_off"])
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def generate_comparison_table(
    results: pd.DataFrame,
    metrics: List[str] = None,
    format: str = "markdown",
) -> str:
    """
    Generate a comparison table across models.

    Args:
        results: DataFrame from aggregate_results
        metrics: List of metric columns to include (default: common retrieval metrics)
        format: Output format ('markdown', 'latex', 'csv')

    Returns:
        Formatted table string
    """
    if results.empty:
        return "No results to compare."

    if metrics is None:
        # Default retrieval metrics
        metrics = [
            "exact_hit@1",
            "exact_hit@5",
            "relevant_hit@1",
            "relevant_hit@5",
            "mrr",
            "ndcg@5",
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results.columns]

    # Prefer experiment name when it distinguishes runs using the same encoder.
    label_col = "experiment" if results["experiment"].nunique() > 1 else "encoder"
    # Pivot for comparison: rows = (label, mask), columns = metrics
    index_cols = [label_col, "mask"]
    display_df = results[index_cols + available_metrics].copy()
    display_df = display_df.sort_values(index_cols, ascending=[True, False])

    # Format floats
    for col in available_metrics:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )

    if format == "markdown":
        return display_df.to_markdown(index=False)
    elif format == "latex":
        return display_df.to_latex(index=False)
    elif format == "csv":
        return display_df.to_csv(index=False)
    else:
        return str(display_df)


def compute_per_condition_metrics(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-condition (macro) metrics.

    Args:
        predictions: List of top-K predictions per query
        ground_truth: List of true labels
        k: K value for Hit@K

    Returns:
        Dict mapping condition -> {hits, total, accuracy}
    """
    condition_stats = defaultdict(lambda: {"hits": 0, "total": 0})

    for preds, true in zip(predictions, ground_truth):
        condition_stats[true]["total"] += 1
        if true in preds[:k]:
            condition_stats[true]["hits"] += 1

    # Compute accuracy per condition
    for cond, stats in condition_stats.items():
        stats["accuracy"] = (
            stats["hits"] / stats["total"] if stats["total"] > 0 else 0.0
        )

    return dict(condition_stats)


def compute_macro_micro_summary(
    predictions: List[List[str]],
    ground_truth: List[str],
    top_k_values: List[int] = [1, 5],
) -> Dict[str, float]:
    """
    Compute micro (per-cell) and macro (per-condition) metrics.

    Args:
        predictions: List of top-K predictions per query
        ground_truth: List of true labels
        top_k_values: K values for metrics

    Returns:
        Dict with micro_hit@K and macro_hit@K for each K
    """
    metrics = {}

    for k in top_k_values:
        # Micro: average over all queries (cells)
        micro_correct = sum(
            1 for preds, true in zip(predictions, ground_truth) if true in preds[:k]
        )
        metrics[f"micro_hit@{k}"] = (
            micro_correct / len(ground_truth) if ground_truth else 0.0
        )

        # Macro: average per-condition accuracy
        condition_stats = compute_per_condition_metrics(predictions, ground_truth, k)
        macro_acc = np.mean([s["accuracy"] for s in condition_stats.values()])
        metrics[f"macro_hit@{k}"] = float(macro_acc)

    return metrics


def generate_report(
    result_dirs: List[Path | str],
    output_dir: Path | str,
    report_name: str = "comparison_report",
) -> Path:
    """
    Generate a full comparison report.

    Args:
        result_dirs: List of result directories
        output_dir: Output directory for report
        report_name: Base name for report files

    Returns:
        Path to generated report
    """
    result_dirs = [Path(d) for d in result_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate results
    results_df = aggregate_results(result_dirs)

    if results_df.empty:
        print("No results found to aggregate")
        return output_dir

    # Save aggregated CSV
    csv_path = output_dir / f"{report_name}.csv"
    results_df.to_csv(csv_path, index=False)

    # Generate markdown table
    table_md = generate_comparison_table(results_df, format="markdown")

    # Generate report markdown
    report_md = f"""# Reverse Perturbation Prediction - Model Comparison

## Overview

Comparison of {len(results_df["experiment"].unique())} experiments across {len(result_dirs)} runs.

## Results Table

{table_md}

## Notes

- **mask=True**: Anti-cheat masking enabled (perturbed gene expression zeroed)
- **mask=False**: No masking (includes potential leakage signal)
- **experiment/encoder**: Uses logging.experiment_name when multiple runs share an encoder; otherwise shows encoder
- **exact_hit@K**: Fraction where true condition is in top-K predictions
- **relevant_hit@K**: Fraction where any top-K prediction shares a gene with ground truth
- **mrr**: Mean Reciprocal Rank
- **ndcg@K**: Normalized Discounted Cumulative Gain at K
"""

    report_path = output_dir / f"{report_name}.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"Report generated: {report_path}")
    print(f"CSV data: {csv_path}")

    return report_path
