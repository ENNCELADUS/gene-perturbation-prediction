#!/usr/bin/env python3
"""Run prediction-first P0 evaluation on the test cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aivc_model.tx1_p0_frozen_test import (
    build_frozen_predictions,
    evaluate_predictions,
    write_predictions,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--phase-a-dir", type=Path, required=True)
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--gene-effect", type=Path, required=True)
    predict.add_argument("--cache-root", type=Path, required=True)
    predict.add_argument("--exposure-ledger", type=Path, required=True)
    predict.add_argument("--tx1-context", type=Path, required=True)
    predict.add_argument("--hvg-context", type=Path, required=True)
    predict.add_argument("--previous-tx1", type=Path, required=True)
    predict.add_argument("--previous-hvg", type=Path, required=True)
    predict.add_argument("--previous-tx1-manifest", type=Path, required=True)
    predict.add_argument("--previous-hvg-manifest", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-dir", type=Path, required=True)
    evaluate.add_argument("--phase-a-dir", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--gene-effect", type=Path, required=True)
    evaluate.add_argument("--exposure-ledger", type=Path, required=True)
    evaluate.add_argument("--expected-prediction-manifest-sha256", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the selected prediction or evaluation phase."""
    args = parse_args()
    if args.mode == "predict":
        predictions, metadata = build_frozen_predictions(
            phase_a_dir=args.phase_a_dir,
            manifest_path=args.manifest,
            raw_gene_effect_path=args.gene_effect,
            cache_root=args.cache_root,
            exposure_ledger_path=args.exposure_ledger,
            representation_paths={"hvg": args.hvg_context, "tx1": args.tx1_context},
            cache_arrays={
                "hvg": ("hvg.npy", "hvg_mean"),
                "tx1": ("embeddings.npy", "tx1_mean"),
            },
            comparator_paths={
                "previous_hvg": args.previous_hvg,
                "previous_tx1": args.previous_tx1,
            },
            comparator_manifest_paths={
                "previous_hvg": args.previous_hvg_manifest,
                "previous_tx1": args.previous_tx1_manifest,
            },
        )
        write_predictions(predictions, metadata, args.output_dir)
        return 0
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    per_line, summary, verdict = evaluate_predictions(
        prediction_dir=args.prediction_dir,
        phase_a_dir=args.phase_a_dir,
        manifest_path=args.manifest,
        raw_gene_effect_path=args.gene_effect,
        exposure_ledger_path=args.exposure_ledger,
        expected_prediction_manifest_sha256=(args.expected_prediction_manifest_sha256),
    )
    args.output_dir.mkdir(parents=True)
    per_line.to_csv(args.output_dir / "per_line.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
