"""Command-line interface for dependency baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vcc_dependency_baseline.config import load_config
from vcc_dependency_baseline.evaluation import run_cv, summarize_metrics
from vcc_dependency_baseline.features import build_features


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vcc-dep-baseline",
        description="Build and evaluate Replogle K562 B->C dependency baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-features")
    build_parser.add_argument("--config", required=True, type=Path)

    cv_parser = subparsers.add_parser("run-cv")
    cv_parser.add_argument("--config", required=True, type=Path)
    cv_parser.add_argument("--features", type=Path, default=None)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results-dir", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "build-features":
        config = load_config(args.config)
        paths = build_features(config)
        print(f"features: {paths.features_npz}")
        print(f"metadata: {paths.metadata_csv}")
        print(f"qa: {paths.qa_report_md}")
    elif args.command == "run-cv":
        config = load_config(args.config)
        paths = run_cv(config, args.features)
        print(f"fold metrics: {paths.fold_metrics_csv}")
        print(f"summary: {paths.summary_csv}")
        print(f"predictions: {paths.predictions_csv}")
    elif args.command == "summarize":
        fold_metrics = pd.read_csv(args.results_dir / "fold_metrics.csv")
        summary = summarize_metrics(fold_metrics)
        output_path = args.results_dir / "summary_metrics.csv"
        summary.to_csv(output_path, index=False)
        print(f"summary: {output_path}")


if __name__ == "__main__":
    main()
