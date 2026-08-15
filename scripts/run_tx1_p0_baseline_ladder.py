#!/usr/bin/env python3
"""Run the Tx1 GeneEffect P0 source-line baseline ladder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aivc_model.tx1_p0_baselines import (
    P0BaselineConfig,
    run_p0_baseline_ladder,
    write_p0_baseline_artifacts,
)

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-plan",
        type=Path,
        required=True,
        help="JSON emitted by scripts/build_tx1_p0_validation.py.",
    )
    parser.add_argument(
        "--validation-policy",
        type=Path,
        required=True,
        help="Independent validation_policy.json authority.",
    )
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--line-context", type=Path)
    parser.add_argument("--copy-k562-prior", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lineage-alpha", type=float, default=0.5)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ladder and write its contracted artifacts."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_p0_baseline_ladder(
        manifest_path=args.line_manifest,
        phase_a_dir=args.phase_a_dir,
        validation_plan_path=args.validation_plan,
        validation_policy_path=args.validation_policy,
        gene_effect_path=args.gene_effect,
        context_path=args.line_context,
        copy_k562_path=args.copy_k562_prior,
        config=P0BaselineConfig(
            lineage_alpha=args.lineage_alpha,
            pca_components=args.pca_components,
            ridge_alpha=args.ridge_alpha,
        ),
    )
    write_p0_baseline_artifacts(result, args.out_dir)
    _LOGGER.info(
        "Wrote P0 results for %d lines and %d genes to %s",
        result.summary["n_lines"],
        result.summary["n_genes"],
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
