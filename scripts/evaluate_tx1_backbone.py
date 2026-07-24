#!/usr/bin/env python3
"""Run the frozen Phase-A GeneEffect evaluation and k=10 gate.

Consumes a tidy long-format predictions table (see
``aivc_model.tx1_geneeffect_eval`` for the schema) and the frozen Phase-A
manifest/slice/panels, then writes ``per_line.csv``, ``curve.csv``, and
``verdict.json`` to the output directory. Exits 0 if the gate passes, 1
otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aivc_model.tx1_geneeffect_eval import (
    DEFAULT_BASELINE_METHOD,
    DEFAULT_PRIMARY_METHOD,
    GATE_K,
    K_SCHEDULE,
    RHO_MIN,
    evaluate,
    load_manifest,
    load_panels,
    load_predictions,
    load_slice,
)

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list, or None to use ``sys.argv``.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to a tidy predictions CSV or parquet file.",
    )
    parser.add_argument(
        "--phase-a-dir",
        type=Path,
        default=Path("results/phase_a_tx1_20260724"),
        help="Directory holding the frozen manifest/slice/panels.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate; default is every method in --predictions.",
    )
    parser.add_argument(
        "--baseline-method",
        default=DEFAULT_BASELINE_METHOD,
        help="Paired-difference baseline method.",
    )
    parser.add_argument(
        "--primary-method",
        default=DEFAULT_PRIMARY_METHOD,
        help="Candidate method the gate verdict is computed for.",
    )
    parser.add_argument("--rho-min", type=float, default=RHO_MIN)
    parser.add_argument("--gate-k", type=int, default=GATE_K)
    parser.add_argument("--k-schedule", type=int, nargs="+", default=K_SCHEDULE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluator CLI end to end.

    Args:
        argv: Argument list, or None to use ``sys.argv``.

    Returns:
        Process exit code: 0 if the gate passes, 1 otherwise.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    manifest = load_manifest(args.phase_a_dir / "cell_line_manifest.csv")
    slice_df = load_slice(args.phase_a_dir / "differentially_essential_slice.csv")
    panels = load_panels(args.phase_a_dir / "k_label_panels.csv")
    predictions = load_predictions(args.predictions)

    methods = args.methods or sorted(predictions["method"].unique().tolist())
    _LOGGER.info("Evaluating methods=%s over k_schedule=%s", methods, args.k_schedule)

    result = evaluate(
        predictions=predictions,
        manifest=manifest,
        slice_df=slice_df,
        panels=panels,
        methods=methods,
        k_schedule=args.k_schedule,
        baseline_method=args.baseline_method,
        primary_method=args.primary_method,
        gate_k=args.gate_k,
        rho_min=args.rho_min,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result["per_line"].to_csv(args.out_dir / "per_line.csv", index=False)
    result["curve"].to_csv(args.out_dir / "curve.csv", index=False)
    (args.out_dir / "verdict.json").write_text(
        json.dumps(result["gate"], indent=2) + "\n"
    )

    verdict = result["gate"]
    _LOGGER.info(
        "Gate k=%d method=%s: macro_mean=%.4f ci=[%.4f, %.4f] rho_min=%.4f passes=%s",
        verdict["k"],
        verdict["method"],
        verdict["macro_mean"],
        verdict["ci_lo"],
        verdict["ci_hi"],
        verdict["rho_min"],
        verdict["passes"],
    )
    return 0 if verdict["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
