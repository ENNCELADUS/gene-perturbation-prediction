#!/usr/bin/env python3
"""Run train-only P1 distribution-context ablations under outer LOLO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from aivc_model.tx1_geneeffect_eval import line_bootstrap_ci
from aivc_model.tx1_p0_inputs import build_p0_inputs, write_p0_inputs
from aivc_model.tx1_p0_representation import run_audit

PROTOCOL_ID = "tx1_geneeffect_p1_distribution_context_v1"
EXPECTED_REPRESENTATIONS = {
    "ccle_expression": "ccle_expression_context.csv",
    "hvg_mean": "hvg_context.csv",
    "hvg_moments": "hvg_moments_context.csv",
    "multiview": "multiview_context.csv",
    "tx1_mean": "tx1_context.csv",
    "tx1_moments": "tx1_moments_context.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(path: Path) -> dict[str, object]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": PROTOCOL_ID,
        "formal": False,
        "test_lines_excluded": True,
        "selection": "none_fixed_ablation",
        "pca_components": 8,
        "ridge_alpha": 1.0,
        "shuffle_seed": 20260804,
        "representations": EXPECTED_REPRESENTATIONS,
    }
    if policy != expected:
        raise ValueError("P1 policy differs from the frozen implementation contract")
    return policy


def _paired_comparisons(audit: Mapping[str, object]) -> list[dict[str, object]]:
    representations = audit["representations"]
    if not isinstance(representations, Mapping):
        raise ValueError("representation audit payload is malformed")
    comparisons: list[dict[str, object]] = []
    for candidate, reference in (
        ("tx1_moments", "tx1_mean"),
        ("hvg_moments", "hvg_mean"),
        ("multiview", "tx1_mean"),
        ("multiview", "hvg_mean"),
    ):
        candidate_payload = representations[candidate]
        reference_payload = representations[reference]
        if not isinstance(candidate_payload, Mapping) or not isinstance(
            reference_payload, Mapping
        ):
            raise ValueError("representation result is malformed")
        candidate_rows = candidate_payload["per_line"]
        reference_rows = reference_payload["per_line"]
        candidate_by_line = {
            str(row["model_id"]): float(row["ridge_rho"]) for row in candidate_rows
        }
        reference_by_line = {
            str(row["model_id"]): float(row["ridge_rho"]) for row in reference_rows
        }
        if set(candidate_by_line) != set(reference_by_line):
            raise ValueError("paired representation line coverage differs")
        paired = [
            candidate_by_line[model_id] - reference_by_line[model_id]
            for model_id in sorted(candidate_by_line)
        ]
        point, lo, hi = line_bootstrap_ci(paired)
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "delta_rho": point,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--validation-policy", type=Path, required=True)
    parser.add_argument("--p1-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = _load_policy(args.p1_policy)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.tmp-", dir=args.output_dir.parent
        )
    )
    try:
        result = build_p0_inputs(
            phase_a_dir=args.phase_a_dir,
            manifest_path=args.manifest,
            gene_effect_path=args.gene_effect,
            cache_root=args.cache_root,
            expression_path=args.expression,
        )
        inputs_dir = temporary / "inputs"
        write_p0_inputs(result, inputs_dir)
        representation_paths = {
            name: inputs_dir / filename
            for name, filename in EXPECTED_REPRESENTATIONS.items()
        }
        audit = run_audit(
            manifest_path=args.manifest,
            validation_plan_path=args.validation_plan,
            validation_policy_path=args.validation_policy,
            phase_a_dir=args.phase_a_dir,
            representation_paths=representation_paths,
            gene_effect_path=inputs_dir / "gene_effect_long.csv",
            shared_prior_path=inputs_dir / "copy_k562_prior.csv",
            pca_components=int(policy["pca_components"]),
            ridge_alpha=float(policy["ridge_alpha"]),
            shuffle_seed=int(policy["shuffle_seed"]),
        )
        payload = {
            **audit,
            "protocol_id": PROTOCOL_ID,
            "p1_policy_sha256": _sha256(args.p1_policy),
            "paired_comparisons": _paired_comparisons(audit),
            "claim_scope": "train_head outer-LOLO diagnostic only",
        }
        (temporary / "representation_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
