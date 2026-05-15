"""Command-line interface for dependency baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from dependency_baseline.config import SelectionConfig
from dependency_baseline.config import load_config
from dependency_baseline.artifacts import organize_artifacts
from dependency_baseline.evaluation import fit_final, run_cv, summarize_results
from dependency_baseline.features import build_features
from dependency_baseline.viability_report import write_viability_axis_report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
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
    cv_parser.add_argument("--run-id", type=str, default=None)
    cv_parser.add_argument("--resume", action="store_true")
    cv_parser.add_argument("--scope", action="append", default=None)
    cv_parser.add_argument("--feature-set", action="append", default=None)
    cv_parser.add_argument("--model", action="append", default=None)
    cv_parser.add_argument("--fold", action="append", type=int, default=None)
    cv_parser.add_argument("--weighting", action="append", default=None)
    cv_parser.add_argument("--log-file", type=Path, default=None)

    final_parser = subparsers.add_parser("fit-final")
    final_parser.add_argument("--config", required=True, type=Path)
    final_parser.add_argument("--features", type=Path, default=None)
    final_parser.add_argument("--run-id", type=str, default=None)
    final_parser.add_argument("--feature-set", action="append", default=None)
    final_parser.add_argument("--model", action="append", default=None)
    final_parser.add_argument("--weighting", action="append", default=None)
    final_parser.add_argument("--log-file", type=Path, default=None)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results-dir", required=True, type=Path)

    organize_parser = subparsers.add_parser("organize-artifacts")
    organize_parser.add_argument("--results-dir", required=True, type=Path)
    organize_parser.add_argument("--logs-dir", type=Path, default=None)

    viability_report_parser = subparsers.add_parser("viability-axis-report")
    viability_report_parser.add_argument("--run-dir", required=True, type=Path)
    viability_report_parser.add_argument("--features", required=True, type=Path)
    viability_report_parser.add_argument("--output", required=True, type=Path)
    viability_report_parser.add_argument("--baseline-predictions", type=Path)

    args = parser.parse_args()
    if args.command == "build-features":
        config = load_config(args.config)
        paths = build_features(config)
        print(f"features: {paths.features_npz}")
        print(f"metadata: {paths.metadata_path}")
        print(f"qa: {paths.qa_report_md}")
    elif args.command == "run-cv":
        config = load_config(args.config)
        paths = run_cv(
            config,
            args.features,
            run_id=args.run_id,
            resume=args.resume,
            selection=_selection_from_args(args),
            command=tuple(sys.argv),
            config_path=args.config,
            log_file=args.log_file,
        )
        print(f"run dir: {paths.run_dir}")
        print(f"fold metrics: {paths.fold_metrics_path}")
        print(f"summary: {paths.summary_csv}")
        print(f"predictions: {paths.predictions_path}")
        print(f"model manifest: {paths.model_manifest_path}")
        print(f"top-k candidates: {paths.topk_candidates_path}")
        print(f"log: {paths.log_file}")
    elif args.command == "fit-final":
        config = load_config(args.config)
        paths = fit_final(
            config,
            args.features,
            run_id=args.run_id,
            selection=_selection_from_args(args),
            command=tuple(sys.argv),
            config_path=args.config,
            log_file=args.log_file,
        )
        print(f"run dir: {paths.run_dir}")
        print(f"final model manifest: {paths.final_model_manifest_path}")
        print(f"final rankings: {paths.final_rankings_path}")
        print(f"log: {paths.log_file}")
    elif args.command == "summarize":
        summary_path, ranking_summary_path = summarize_results(args.results_dir)
        print(f"summary: {summary_path}")
        if ranking_summary_path is not None:
            print(f"ranking summary: {ranking_summary_path}")
    elif args.command == "organize-artifacts":
        organize_artifacts(args.results_dir, args.logs_dir)
        print(f"organized: {args.results_dir}")
    elif args.command == "viability-axis-report":
        report_path = write_viability_axis_report(
            run_dir=args.run_dir,
            features_npz=args.features,
            output_path=args.output,
            baseline_predictions=args.baseline_predictions,
        )
        print(f"report: {report_path}")


def _selection_from_args(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        scopes=tuple(args.scope) if getattr(args, "scope", None) else None,
        features=tuple(args.feature_set)
        if getattr(args, "feature_set", None)
        else None,
        models=tuple(args.model) if getattr(args, "model", None) else None,
        folds=tuple(args.fold) if getattr(args, "fold", None) else None,
        weightings=tuple(args.weighting) if getattr(args, "weighting", None) else None,
    )


if __name__ == "__main__":
    main()
