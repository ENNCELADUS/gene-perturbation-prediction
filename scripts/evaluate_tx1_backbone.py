#!/usr/bin/env python3
"""Run the frozen Phase-A GeneEffect evaluation and k=10 gate.

Consumes a tidy long-format predictions table (see
``aivc_model.tx1_geneeffect_eval`` for the schema) and the frozen Phase-A
manifest/slice/panels, then writes ``per_line.csv``, ``curve.csv``, and a
gate verdict to the output directory. A run is FORMAL only when it uses
every frozen-contract default (``rho_min``, ``gate_k``, ``k_schedule``,
``primary_method``, ``baseline_method``, ``expected_test_lines``) AND
neither ``--allow-partial`` nor ``--skip-hash-check`` is passed: the verdict
is written to ``verdict.json`` with ``"formal": true``, and the process
exits 0 if the gate passes, 1 otherwise. Any CLI override of those frozen
values, or either bypass flag, downgrades the run to DIAGNOSTIC: the
verdict is written to ``verdict_diagnostic.json`` (never ``verdict.json``)
with ``"formal": false`` and a ``"reason"`` naming the trigger(s), and the
process always exits 2 so a diagnostic run can never be mistaken for a
passing formal verdict. This closes off overriding a frozen threshold (e.g.
``--rho-min``) as a way to relabel a diagnostic run as formal.
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
    verify_artifact_hashes,
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
    parser.add_argument(
        "--expected-test-lines",
        type=int,
        default=9,
        help="Registered held-out line count the formal gate must match "
        "(Phase A froze 9). Pass 0 to skip the count check.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Bypass frozen-contract validation (diagnostic only; never a "
        "formal verdict).",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Bypass frozen-artifact SHA-256 verification against "
        "phase_a_registration.json (diagnostic only, like --allow-partial; "
        "never a formal verdict).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _override_reasons(
    rho_min: float,
    gate_k: int,
    k_schedule: list[int],
    primary_method: str,
    baseline_method: str,
    expected_test_lines: int,
) -> list[str]:
    """Name every CLI argument that overrides a frozen Phase-A default.

    A CLI override of a frozen threshold or schedule (e.g. ``--rho-min``)
    must not be able to relabel a run as formal: any deviation from the
    registered defaults downgrades the run to diagnostic, same as
    ``--allow-partial``/``--skip-hash-check``.

    Args:
        rho_min: ``--rho-min`` value.
        gate_k: ``--gate-k`` value.
        k_schedule: ``--k-schedule`` value.
        primary_method: ``--primary-method`` value.
        baseline_method: ``--baseline-method`` value.
        expected_test_lines: ``--expected-test-lines`` value.

    Returns:
        Argument names that deviate from their frozen Phase-A default, in a
        fixed, deterministic order; empty if every value matches.
    """
    reasons = []
    if rho_min != RHO_MIN:
        reasons.append("rho_min")
    if gate_k != GATE_K:
        reasons.append("gate_k")
    if list(k_schedule) != K_SCHEDULE:
        reasons.append("k_schedule")
    if primary_method != DEFAULT_PRIMARY_METHOD:
        reasons.append("primary_method")
    if baseline_method != DEFAULT_BASELINE_METHOD:
        reasons.append("baseline_method")
    if expected_test_lines != 9:
        reasons.append("expected_test_lines")
    return reasons


def _diagnostic_reason(
    allow_partial: bool, skip_hash_check: bool, override_reasons: list[str]
) -> str | None:
    """Build the diagnostic-downgrade reason string for the verdict.

    Args:
        allow_partial: Whether ``--allow-partial`` was passed.
        skip_hash_check: Whether ``--skip-hash-check`` was passed.
        override_reasons: Frozen-contract defaults overridden on the CLI
            (see :func:`_override_reasons`).

    Returns:
        A human-readable reason naming every trigger (bypass flag and/or
        non-frozen override), or None if the run is formal.
    """
    bypassed = []
    if allow_partial:
        bypassed.append("contract validation bypassed")
    if skip_hash_check:
        bypassed.append("artifact hash verification bypassed")
    if override_reasons:
        bypassed.append(f"non-frozen overrides: {', '.join(override_reasons)}")
    if not bypassed:
        return None
    return f"partial/diagnostic run ({'; '.join(bypassed)})"


def main(argv: list[str] | None = None) -> int:
    """Run the evaluator CLI end to end.

    A run is FORMAL only when every frozen-contract default (``rho_min``,
    ``gate_k``, ``k_schedule``, ``primary_method``, ``baseline_method``,
    ``expected_test_lines``) is unmodified and neither ``--allow-partial``
    nor ``--skip-hash-check`` was passed. Any override or bypass flag
    downgrades the run to diagnostic; contract validation (``strict``) and
    the artifact hash check still run unless the corresponding bypass flag
    (``--allow-partial`` / ``--skip-hash-check``) was explicit, so an
    override-only diagnostic run is still fail-closed on a tampered artifact
    or a broken contract invariant.

    Args:
        argv: Argument list, or None to use ``sys.argv``.

    Returns:
        Process exit code: 0 if the gate passes and 1 otherwise for a
        formal run; 2 always for a diagnostic run (any non-frozen CLI
        override, and/or ``--allow-partial``/``--skip-hash-check``),
        regardless of ``passes``.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    override_reasons = _override_reasons(
        args.rho_min,
        args.gate_k,
        args.k_schedule,
        args.primary_method,
        args.baseline_method,
        args.expected_test_lines,
    )
    is_diagnostic = args.allow_partial or args.skip_hash_check or bool(override_reasons)
    # Hash verification is an integrity check independent of coverage: only
    # --skip-hash-check bypasses it. --allow-partial bypasses contract
    # validation (coverage/schedule) but must still confirm the artifacts are
    # the frozen ones.
    run_hash_check = not args.skip_hash_check
    strict = not args.allow_partial
    reason = _diagnostic_reason(
        args.allow_partial, args.skip_hash_check, override_reasons
    )

    if run_hash_check:
        # A formal verdict, and an override-only diagnostic run, must still
        # verify the frozen-artifact hashes; only --skip-hash-check bypasses
        # this.
        verify_artifact_hashes(args.phase_a_dir)

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
        strict=strict,
        expected_test_lines=(args.expected_test_lines or None),
    )

    verdict = dict(result["gate"])
    verdict["formal"] = not is_diagnostic
    if is_diagnostic:
        verdict["reason"] = reason
        verdict_path = args.out_dir / "verdict_diagnostic.json"
    else:
        verdict_path = args.out_dir / "verdict.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result["per_line"].to_csv(args.out_dir / "per_line.csv", index=False)
    result["curve"].to_csv(args.out_dir / "curve.csv", index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")

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
    if is_diagnostic:
        _LOGGER.warning(
            "PARTIAL/DIAGNOSTIC RUN: %s. This is NOT a formal verdict; see %s.",
            reason,
            verdict_path,
        )
        return 2
    return 0 if verdict["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
