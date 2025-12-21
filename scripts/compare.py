#!/usr/bin/env python
"""
CLI script for comparing results across multiple experiments.

Usage:
    python scripts/compare.py --results results/pca_* results/scgpt_* --output results/comparison/
    python scripts/compare.py --pattern "results/*_2024*" --output results/comparison/
"""

import argparse
import glob
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate.report import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Compare results across multiple experiments"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        help="List of result directories to compare",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Glob pattern to match result directories (alternative to --results). "
            "Repeat to provide multiple patterns."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for comparison report",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="comparison_report",
        help="Name of the report file (default: comparison_report)",
    )

    args = parser.parse_args()

    # Collect result directories
    result_dirs = []

    if args.results:
        for path in args.results:
            # Expand glob patterns in the arguments
            expanded = glob.glob(path)
            if expanded:
                result_dirs.extend(expanded)
            elif Path(path).exists():
                result_dirs.append(path)

    if args.pattern:
        for pattern in args.pattern:
            result_dirs.extend(glob.glob(pattern))

    # Remove duplicates and convert to Path
    result_dirs = list(set(result_dirs))
    result_dirs = [Path(d) for d in result_dirs if Path(d).is_dir()]

    if not result_dirs:
        print("Error: No valid result directories found")
        print("Provide directories via --results or --pattern")
        sys.exit(1)

    print(f"Found {len(result_dirs)} result directories:")
    for d in sorted(result_dirs):
        print(f"  - {d}")

    # Generate report
    output_dir = Path(args.output)
    report_path = generate_report(result_dirs, output_dir, args.name)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
