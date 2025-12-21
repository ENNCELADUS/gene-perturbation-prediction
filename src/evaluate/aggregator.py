"""
Experiment management for multi-model comparison.

Provides:
- Run aggregation across experiments
- Leaderboard generation
- Multi-seed comparison
"""

from __future__ import annotations

import json
import fnmatch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


@dataclass
class RunInfo:
    """Information about a single experiment run."""

    run_dir: Path
    experiment_name: str
    encoder: str
    track: str = "default"
    seed: int = 42
    mask_on: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)


class RunAggregator:
    """
    Aggregates and compares experiment runs.

    Supports:
    - Collection of runs by pattern
    - Leaderboard generation
    - Multi-seed averaging
    """

    def __init__(self, base_dir: Path | str = "results"):
        """
        Initialize aggregator.

        Args:
            base_dir: Base directory containing experiment results
        """
        self.base_dir = Path(base_dir)
        self.runs: List[RunInfo] = []

    def collect_runs(self, pattern: str = "*") -> List[RunInfo]:
        """
        Collect runs matching pattern.

        Args:
            pattern: Glob pattern for run directories

        Returns:
            List of RunInfo objects
        """
        self.runs = []

        for metrics_file in self.base_dir.rglob("metrics.json"):
            run_dir = metrics_file.parent
            if not run_dir.is_dir():
                continue
            if not fnmatch.fnmatch(run_dir.name, pattern):
                continue

            with open(metrics_file) as f:
                data = json.load(f)

            config = data.get("config", {})
            model_config = config.get("model", {})

            # Extract mask ON metrics
            if "metrics" in data:
                run = RunInfo(
                    run_dir=run_dir,
                    experiment_name=config.get("logging", {}).get(
                        "experiment_name", run_dir.name
                    ),
                    encoder=model_config.get("encoder", "unknown"),
                    track=config.get("track", "default"),
                    seed=config.get("split", {}).get("seed", 42),
                    mask_on=True,
                    metrics=data["metrics"],
                    config=config,
                )
                self.runs.append(run)

            # Extract mask OFF metrics if present
            if "metrics_mask_off" in data:
                run = RunInfo(
                    run_dir=run_dir,
                    experiment_name=config.get("logging", {}).get(
                        "experiment_name", run_dir.name
                    ),
                    encoder=model_config.get("encoder", "unknown"),
                    track=config.get("track", "default"),
                    seed=config.get("split", {}).get("seed", 42),
                    mask_on=False,
                    metrics=data["metrics_mask_off"],
                    config=config,
                )
                self.runs.append(run)

        return self.runs

    def build_leaderboard(
        self,
        metric: str = "exact_hit@1",
        group_by: List[str] = None,
    ) -> pd.DataFrame:
        """
        Build leaderboard sorted by metric.

        Args:
            metric: Metric to sort by
            group_by: Columns to group by before averaging

        Returns:
            DataFrame with leaderboard
        """
        if not self.runs:
            return pd.DataFrame()

        rows = []
        for run in self.runs:
            row = {
                "encoder": run.encoder,
                "track": run.track,
                "seed": run.seed,
                "mask": run.mask_on,
                "run_dir": str(run.run_dir),
            }
            row.update(run.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        if group_by:
            # Average across seeds
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: "mean" for col in numeric_cols if col != "seed"}
            agg_dict["seed"] = "count"
            df = df.groupby(group_by).agg(agg_dict).reset_index()
            df = df.rename(columns={"seed": "n_seeds"})

        # Sort by metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def compare_across_seeds(
        self,
        encoder: Optional[str] = None,
        track: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare performance across random seeds.

        Args:
            encoder: Filter to specific encoder
            track: Filter to specific track

        Returns:
            DataFrame with mean and std across seeds
        """
        if not self.runs:
            return pd.DataFrame()

        runs = self.runs
        if encoder:
            runs = [r for r in runs if r.encoder == encoder]
        if track:
            runs = [r for r in runs if r.track == track]

        rows = []
        for run in runs:
            row = {
                "encoder": run.encoder,
                "track": run.track,
                "seed": run.seed,
                "mask": run.mask_on,
            }
            row.update(run.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Group and compute stats
        group_cols = ["encoder", "track", "mask"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != "seed"]

        stats = []
        for name, group in df.groupby(group_cols):
            row = dict(zip(group_cols, name))
            row["n_seeds"] = len(group)
            for col in numeric_cols:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"] = group[col].std()
            stats.append(row)

        return pd.DataFrame(stats)

    def generate_summary_report(
        self,
        output_path: Path | str,
        primary_metric: str = "exact_hit@1",
    ) -> Path:
        """
        Generate comprehensive summary report.

        Args:
            output_path: Path for report file
            primary_metric: Primary metric for ranking

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        leaderboard = self.build_leaderboard(
            metric=primary_metric, group_by=["encoder", "track", "mask"]
        )

        report = f"""# Experiment Comparison Report

## Summary

- Total runs collected: {len(self.runs)}
- Encoders: {", ".join(set(r.encoder for r in self.runs))}
- Tracks: {", ".join(set(r.track for r in self.runs))}

## Leaderboard (by {primary_metric})

{leaderboard.to_markdown(index=False) if not leaderboard.empty else "No results available."}

## Notes

- Results averaged across seeds when applicable
- mask=True: Anti-cheat masking enabled
- mask=False: No masking (baseline)
"""

        with open(output_path, "w") as f:
            f.write(report)

        return output_path
