"""CLI entrypoint: uv run python -m sl_benchmark_baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sl_benchmark_baseline.config import SLBaselineConfig
from sl_benchmark_baseline.evaluate import run_cv

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K562 SL-pair dependency-only baseline (CV1)."
    )
    defaults = SLBaselineConfig()
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--split-types", nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(defaults.folds))
    parser.add_argument(
        "--ranking-k", type=int, nargs="+", default=list(defaults.ranking_k)
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--bags-npz", type=Path, default=defaults.bags_npz)
    parser.add_argument(
        "--embedding-method", type=str, default=defaults.embedding_method
    )
    parser.add_argument(
        "--fallback-strategy",
        choices=("zero", "global_mean"),
        default=defaults.fallback_strategy,
    )
    parser.add_argument(
        "--include-coverage-flag",
        dest="include_coverage_flag",
        action="store_true",
        default=defaults.include_coverage_flag,
    )
    parser.add_argument(
        "--no-coverage-flag",
        dest="include_coverage_flag",
        action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline CV loop from CLI flags. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    config = SLBaselineConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        split_types=None if args.split_types is None else tuple(args.split_types),
        folds=tuple(args.folds),
        ranking_k=tuple(args.ranking_k),
        seed=args.seed,
        bags_npz=args.bags_npz,
        embedding_method=args.embedding_method,
        fallback_strategy=args.fallback_strategy,
        include_coverage_flag=args.include_coverage_flag,
    )
    summary = run_cv(config)
    logger.info("Wrote summary with %d rows to %s", len(summary), config.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
