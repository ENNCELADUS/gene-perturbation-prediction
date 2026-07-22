"""Bridge A forward-inference CLI on the frozen exp05 checkpoint.

Wires the counterfactual co-dependency path (see ``aivc_model.bridge_a``
module docstring for the exact definition), validates the wiring, and runs a
small pilot over a handful of Horlbeck-covered K562 gene pairs. This script
never runs the full 83,028-pair candidate-pair sweep -- it only measures
per-forward timing and projects the full-run cost.

Two gates run before the pilot (both under ``--skip-smoke``): the CONTROL
gate reproduces ``c_hat[gene | control]`` against the frozen run's own
``predictions.csv`` at MACHINE PRECISION (``--panel-size``, matching
``config.train.eval_control_panel_size`` exactly -- a FIXED nominal control
panel, unrelated to the independent-N mechanism below); the PERTURBED-ARM
gate (``bridge_a_gates.assert_perturbed_arm_integrity``) exercises the
a-perturbed path itself (gene-specific indexing, non-identity vs. control,
shape/batch alignment, finite/non-constant outputs, basal-sensitivity,
deterministic replay) and hard-fails unconditionally if broken or
degenerate.

The pilot uses INDEPENDENT-N-matched panels (Finding A --
``aivc_model.bridge_a_independent``, driven by ``--max-windows`` and the
Horlbeck/Replogle K562 coverage table at ``--coverage-csv``, not a flat
nominal cell target), multi-seed replicates (``--seeds``) to report seed
variance, and both Finding-1 pooler reference-latent conventions
(``--reference-convention``).

Typical invocation (see ``parse_args`` for full defaults)::

    PYTHONPATH=src ./.venv-esm2/bin/python scripts/bridge_a_forward.py \\
        --device cuda --pilot-genes 12
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr

from aivc_model.bridge_a import (
    REFERENCE_CONVENTIONS,
    BridgeAContext,
    PanelSpec,
    load_bridge_a_context,
    run_smoke_gate,
    sample_genes,
)
from aivc_model.bridge_a_gates import (
    aggregate_seed_replicate_c_hat,
    aggregate_seed_replicate_pairs,
    assert_perturbed_arm_integrity,
    load_gi_lookup,
    project_full_sweep,
    score_pairs_against_gi,
)
from aivc_model.bridge_a_independent import (
    IndependentPanelSpec,
    bridge_a_pairs_from_independent_c_hat_table,
    compute_independent_c_hat_table,
)
from aivc_model.bridge_a_panels import DEFAULT_MAX_WINDOWS, load_k562_gene_coverage

_LOGGER = logging.getLogger("bridge_a_forward")

_DEFAULT_CONFIG = Path(
    "configs/experiments/05_aivc_a_to_b_to_c/state_esm2_gwps_fixed.yaml"
)
_DEFAULT_RUN_DIR = Path(
    "results/experiments/05_aivc_a_to_b_to_c/runs/exp05_fixed_k562_pool_v1"
)
_DEFAULT_CHECKPOINT = _DEFAULT_RUN_DIR / "models/best/pytorch_model.bin"
_DEFAULT_PREDICTIONS_CSV = _DEFAULT_RUN_DIR / "artifacts/predictions.csv"
_DEFAULT_CHECKPOINT_SHA256 = (
    "48097722f5742a459b86ba6153dd21f145ff1a0e30dafa80061c325c2d46b811"
)
_DEFAULT_UNIVERSE_DIR = Path(
    "results/experiments/05_aivc_a_to_b_to_c/bridge_a/universe"
)
_DEFAULT_UNIVERSE_GENES = _DEFAULT_UNIVERSE_DIR / "candidate_universe_genes.txt"
_DEFAULT_UNIVERSE_PAIRS = _DEFAULT_UNIVERSE_DIR / "candidate_universe_pairs.csv"
_DEFAULT_OUTPUT_DIR = Path("results/experiments/05_aivc_a_to_b_to_c/bridge_a/pilot")
_DEFAULT_COVERAGE_CSV = Path(
    "data/sl_dependency_v0/processed/horlbeck_2018/k562_gene_coverage.csv"
)
_DEFAULT_SEEDS = (41, 42, 43)
_DEFAULT_SMOKE_TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=_DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--panel-size",
        type=int,
        default=None,
        help="cells for the CONTROL gate's bit-exact reproduction only -- a "
        "FIXED nominal panel, unrelated to --max-windows "
        "(default: config eval_control_panel_size)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="Finding-A independent-N window-budget cap: "
        "w_a = min(n_a // cell_set_len, max_windows) for every a-perturbed "
        "panel in the perturbed-arm gate and the pilot",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=_DEFAULT_COVERAGE_CSV,
        help="per-gene observed Replogle cell counts driving the Finding-A "
        "window budget (bridge_a_panels.load_k562_gene_coverage)",
    )
    parser.add_argument(
        "--panel-macro-batch-windows",
        type=int,
        default=None,
        help="default: config.train.eval_window_macro_batch_size",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="seed for the CONTROL gate only (default: config seed + outer_fold); "
        "the pilot uses --seeds",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(_DEFAULT_SEEDS),
        help="deterministic multi-seed replicates for the pilot (>= 2 required)",
    )
    parser.add_argument(
        "--reference-convention",
        choices=(*REFERENCE_CONVENTIONS, "both"),
        default="both",
        help="Finding-1 fix: which pooler reference-latent convention(s) to run",
    )
    parser.add_argument(
        "--seed-variance-threshold",
        type=float,
        default=0.5,
        help="flag a pair when s_a seed std > threshold * |s_a seed mean|",
    )

    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--smoke-reference-csv", type=Path, default=_DEFAULT_PREDICTIONS_CSV
    )
    parser.add_argument("--smoke-genes", type=int, default=8)
    parser.add_argument(
        "--smoke-tolerance",
        type=float,
        default=_DEFAULT_SMOKE_TOLERANCE,
        help="max ABSOLUTE error tolerance for control reproduction (Finding 3a; "
        "observed machine precision is ~5e-17..8e-17)",
    )
    parser.add_argument(
        "--allow-smoke-mismatch",
        action="store_true",
        help="report but do not abort on a CONTROL tolerance miss; does NOT "
        "cover the perturbed-arm gate, which always hard-fails",
    )
    parser.add_argument(
        "--gate-genes",
        type=int,
        default=3,
        help="how many pilot genes to exercise in the perturbed-arm gate",
    )

    parser.add_argument(
        "--genes", default=None, help="explicit comma-separated gene list"
    )
    parser.add_argument("--pilot-genes", type=int, default=12)
    parser.add_argument(
        "--universe-genes-file", type=Path, default=_DEFAULT_UNIVERSE_GENES
    )
    parser.add_argument("--pairs-csv", type=Path, default=_DEFAULT_UNIVERSE_PAIRS)
    parser.add_argument("--gene-sample-seed", type=int, default=0)

    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _resolve_panel_spec(
    context: BridgeAContext,
    *,
    target_cells: int | None,
    macro_batch_windows: int | None,
    seed: int | None,
) -> PanelSpec:
    default_target = context.config.train.eval_control_panel_size
    default_macro_batch = context.config.train.eval_window_macro_batch_size
    return PanelSpec(
        cell_set_len=int(context.config.train.cell_set_len),
        target_cells=int(target_cells or default_target),
        macro_batch_windows=int(macro_batch_windows or default_macro_batch),
        seed=int(context.eval_seed if seed is None else seed),
    )


def _resolve_independent_panel_spec(
    context: BridgeAContext,
    *,
    macro_batch_windows: int | None,
    seed: int,
    max_windows: int,
) -> IndependentPanelSpec:
    default_macro_batch = context.config.train.eval_window_macro_batch_size
    return IndependentPanelSpec(
        cell_set_len=int(context.config.train.cell_set_len),
        macro_batch_windows=int(macro_batch_windows or default_macro_batch),
        seed=seed,
        max_windows=max_windows,
    )


def run_pilot(
    context: BridgeAContext,
    genes: list[str],
    coverage: dict[str, int],
    pairs_csv: Path,
    cell_set_len: int,
    macro_batch_windows: int,
    max_windows: int,
    seeds: list[int],
    reference_conventions: list[str],
    seed_variance_threshold: float,
    output_dir: Path,
) -> tuple[dict[str, object], list[float]]:
    """Compute s_A for every pair among ``genes`` under every (seed, convention).

    Finding A: every basal gene gets its OWN independent-N window budget
    (``max_windows``-capped); the matched control panel uses the SAME
    window count. Finding 1: run under every convention in
    ``reference_conventions``.

    Returns:
        A tuple of the pilot summary dict and the list of per-forward
        wall-clock seconds across every (seed, convention).
    """
    _LOGGER.info(
        "pilot: %d genes -> %d unordered pairs, seeds=%s, reference_conventions=%s, "
        "max_windows=%d",
        len(genes),
        len(genes) * (len(genes) - 1) // 2,
        seeds,
        reference_conventions,
        max_windows,
    )
    gi_lookup = load_gi_lookup(pairs_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_forward_seconds: list[float] = []
    by_convention: dict[str, object] = {}

    for convention in reference_conventions:
        per_seed_pairs: list[pd.DataFrame] = []
        per_seed_c_hat: list[pd.DataFrame] = []
        for seed in seeds:
            spec = IndependentPanelSpec(
                cell_set_len=cell_set_len,
                macro_batch_windows=macro_batch_windows,
                seed=seed,
                max_windows=max_windows,
            )
            c_hat_table, budgets = compute_independent_c_hat_table(
                context, genes, coverage, spec, reference_convention=convention
            )
            if not c_hat_table["c_hat"].apply(math.isfinite).all():
                raise RuntimeError(
                    f"pilot ({convention}, seed={seed}) produced a non-finite c_hat"
                )
            all_forward_seconds.extend(c_hat_table["forward_seconds"])
            per_seed_c_hat.append(c_hat_table.assign(seed=seed))
            pairs = bridge_a_pairs_from_independent_c_hat_table(
                c_hat_table, budgets, cell_set_len
            )
            per_seed_pairs.append(pairs.assign(seed=seed))

        pair_seed_summary = aggregate_seed_replicate_pairs(
            per_seed_pairs, seeds, seed_variance_threshold=seed_variance_threshold
        )
        c_hat_seed_summary = aggregate_seed_replicate_c_hat(per_seed_c_hat)
        pair_seed_summary = score_pairs_against_gi(pair_seed_summary, gi_lookup)

        raw_pairs = pd.concat(per_seed_pairs, ignore_index=True)
        raw_c_hat = pd.concat(per_seed_c_hat, ignore_index=True)
        frames = {
            f"pilot_pairs_raw_{convention}.csv": raw_pairs,
            f"pilot_pairs_{convention}.csv": pair_seed_summary,
            f"pilot_c_hat_raw_{convention}.csv": raw_c_hat,
            f"pilot_c_hat_seed_summary_{convention}.csv": c_hat_seed_summary,
        }
        for name, frame in frames.items():
            frame.to_csv(output_dir / name, index=False)

        scored = pair_seed_summary.dropna(subset=["gi_score"])
        if len(scored) >= 3:
            spearman = spearmanr(scored["s_a_seed_mean"], -scored["gi_score"])
            spearman_stat = float(spearman.statistic)
            spearman_pvalue = float(spearman.pvalue)
        else:
            spearman_stat = math.nan
            spearman_pvalue = math.nan

        s_a_mean = pair_seed_summary["s_a_seed_mean"]
        s_a_std_median = float(pair_seed_summary["s_a_seed_std"].median())
        c_hat_std_median = float(c_hat_seed_summary["c_hat_seed_std"].median())
        n_material = int(pair_seed_summary["seed_variance_material"].sum())
        n_high_effective_n = int(pair_seed_summary["high_effective_n"].sum())
        effective_n_values = pd.concat(
            [
                pair_seed_summary["effective_n_a_to_b"],
                pair_seed_summary["effective_n_b_to_a"],
            ]
        )
        convention_summary: dict[str, object] = {
            "n_pairs": int(len(pair_seed_summary)),
            "n_pairs_with_gi_score": int(len(scored)),
            "n_pairs_high_effective_n": n_high_effective_n,
            "effective_n_min": int(effective_n_values.min()),
            "effective_n_max": int(effective_n_values.max()),
            "s_a_seed_mean_min": float(s_a_mean.min()),
            "s_a_seed_mean_median": float(s_a_mean.median()),
            "s_a_seed_mean_max": float(s_a_mean.max()),
            "s_a_seed_std_median": s_a_std_median,
            "c_hat_seed_std_median": c_hat_std_median,
            "n_pairs_seed_variance_material": n_material,
            "seed_variance_threshold": seed_variance_threshold,
            "spearman_s_a_vs_negative_gi_score": spearman_stat,
            "spearman_pvalue": spearman_pvalue,
        }
        _LOGGER.info(
            "pilot (%s): n_pairs=%d n_scored=%d n_high_effective_n=%d "
            "effective_n=[%d,%d] spearman=%.4f (p=%.4f) seed_std_median=%.4f "
            "material_pairs=%d/%d [sanity only]",
            convention,
            convention_summary["n_pairs"],
            convention_summary["n_pairs_with_gi_score"],
            n_high_effective_n,
            convention_summary["effective_n_min"],
            convention_summary["effective_n_max"],
            spearman_stat,
            spearman_pvalue,
            s_a_std_median,
            n_material,
            convention_summary["n_pairs"],
        )
        by_convention[convention] = convention_summary

    summary: dict[str, object] = {
        "sanity_only_not_a_result": True,
        "genes": genes,
        "n_genes": len(genes),
        "seeds": seeds,
        "max_windows": max_windows,
        "by_convention": by_convention,
    }
    return summary, all_forward_seconds


def _resolve_genes(args: argparse.Namespace) -> list[str]:
    if args.genes:
        return sorted(
            {gene.strip().upper() for gene in args.genes.split(",") if gene.strip()}
        )
    if args.pilot_genes > 0:
        universe = [
            gene for gene in args.universe_genes_file.read_text().splitlines() if gene
        ]
        return sample_genes(universe, args.pilot_genes, args.gene_sample_seed)
    return []


def _run_gates(
    context: BridgeAContext,
    args: argparse.Namespace,
    genes: list[str],
    coverage: dict[str, int],
    conventions: list[str],
) -> tuple[dict[str, object], list[float]]:
    """Run the CONTROL and PERTURBED-ARM (Finding A/B) gates."""
    result: dict[str, object] = {}
    forward_seconds: list[float] = []

    smoke_spec = _resolve_panel_spec(
        context,
        target_cells=args.panel_size,
        macro_batch_windows=args.panel_macro_batch_windows,
        seed=args.eval_seed,
    )
    _LOGGER.info(
        "control smoke-gate panel spec: cell_set_len=%d target_cells=%d "
        "macro_batch_windows=%d seed=%d",
        smoke_spec.cell_set_len,
        smoke_spec.target_cells,
        smoke_spec.macro_batch_windows,
        smoke_spec.seed,
    )
    smoke_summary, smoke_seconds = run_smoke_gate(
        context,
        args.smoke_reference_csv,
        args.smoke_genes,
        args.gene_sample_seed,
        smoke_spec,
        args.output_dir,
    )
    forward_seconds.extend(smoke_seconds)
    result["smoke"] = {**smoke_summary, "tolerance": args.smoke_tolerance}
    if (
        smoke_summary["max_abs_error"] > args.smoke_tolerance
        and not args.allow_smoke_mismatch
    ):
        raise SystemExit(
            "Bridge A CONTROL smoke gate FAILED: max_abs_error="
            f"{smoke_summary['max_abs_error']:.3e} > tolerance="
            f"{args.smoke_tolerance:.3e}. The forward-inference wiring likely "
            "does not match predictions.csv's construction; debug before "
            "running the pilot. Pass --allow-smoke-mismatch to override "
            "(this override does NOT apply to the perturbed-arm gate below)."
        )

    if len(genes) < 3:
        _LOGGER.warning(
            "skipping perturbed-arm gate: need >= 3 genes, have %d", len(genes)
        )
        return result, forward_seconds

    gate_spec = _resolve_independent_panel_spec(
        context,
        macro_batch_windows=args.panel_macro_batch_windows,
        seed=args.seeds[0],
        max_windows=args.max_windows,
    )
    gate_genes = genes[: max(3, args.gate_genes)]
    gate_results: dict[str, object] = {}
    for convention in conventions:
        gate_results[convention] = assert_perturbed_arm_integrity(
            context, gate_genes, coverage, gate_spec, convention
        )
        _LOGGER.info("perturbed-arm gate (%s) PASSED", convention)
    result["perturbed_arm_gate"] = gate_results
    return result, forward_seconds


def main() -> None:
    """CLI entry point: gate then pilot the Bridge A forward path."""
    args = parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        # Some CUDA containers raise "invalid device argument" if peak-memory
        # stats are reset before the context is initialized; touch the device
        # first and treat the diagnostic reset as best-effort.
        if device.index is not None:
            torch.cuda.set_device(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError as exc:  # pragma: no cover - environment-specific
            _LOGGER.warning("could not reset CUDA peak-memory stats: %s", exc)
    context = load_bridge_a_context(
        args.config, args.checkpoint, args.checkpoint_sha256, device
    )
    coverage = load_k562_gene_coverage(args.coverage_csv)
    conventions = (
        list(REFERENCE_CONVENTIONS)
        if args.reference_convention == "both"
        else [args.reference_convention]
    )
    genes = _resolve_genes(args)

    all_forward_seconds: list[float] = []
    result: dict[str, object] = {
        "device": str(device),
        "seeds": list(args.seeds),
        "max_windows": args.max_windows,
        "reference_conventions": conventions,
    }

    if not args.skip_smoke:
        gate_result, gate_seconds = _run_gates(
            context, args, genes, coverage, conventions
        )
        result.update(gate_result)
        all_forward_seconds.extend(gate_seconds)

    if genes:
        default_macro_batch = context.config.train.eval_window_macro_batch_size
        pilot_summary, pilot_seconds = run_pilot(
            context,
            genes,
            coverage,
            args.pairs_csv,
            int(context.config.train.cell_set_len),
            int(args.panel_macro_batch_windows or default_macro_batch),
            args.max_windows,
            list(args.seeds),
            conventions,
            args.seed_variance_threshold,
            args.output_dir,
        )
        all_forward_seconds.extend(pilot_seconds)
        result["pilot"] = pilot_summary
    else:
        _LOGGER.info("no --genes/--pilot-genes selection; skipping the pilot")

    result["full_sweep_projection"] = project_full_sweep(
        all_forward_seconds, device, args.max_windows
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _LOGGER.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
