#!/usr/bin/env python3
"""Train the Stage 1 response model on the four Perturb-seq anchor lines.

Fits ``L_resp`` (``01-blueprint.md`` §4) on Tx1 basal input -> ST -> HVG
expression, with the ESM2 perturbation adapter supplying ``p_g``. The module
trained is :class:`ForwardOnlyStateModel`: the ST state adapter and the
perturbation adapter, and **no GeneEffect head**. No dependency label is read
anywhere in this script.

Guards this script applies before any parameter moves:

- every anchor line must be ``train`` in the frozen 226-line split
  (``benchmark_split.assert_fit_eligible``), so Stage 1 cannot fit on a line
  the dependency evaluation later holds out;
- perturbation genes are held out **per line** for the model's own
  generalization check (Exp13 spec §7), never globally;
- the collator seed is pinned and recorded, because Tx1's gene subsampling is
  otherwise unseeded above 2048 detected genes
  (``docs/results/exp13-stage0-tx1-input-representation.md``).

Hyperparameters and the Exp13 §7 freeze thresholds come from ``--config``
(``configs/experiments/13_geneeffect_226/stage1_response.yaml``), whose loader
raises on any unknown or missing key. Training runs under ``accelerate``, so
the device is auto-detected and a multi-rank launch is supported; selection is
by held-out ``L_resp`` weighted across the four anchors, with early stopping.

``--assemble-only`` is phase 1 of ``scripts/run_stage1_response_ddp.sh``: it
warms ``--response-cache-dir`` single-process and exits, because assembly runs
on every rank before any accelerator exists and peaked at 194.6 GB RSS in
Phase C. A multi-rank launch against a cold cache is refused outright.

The run writes ``stage1_freeze_thresholds.json`` (pinned before any parameter
moves), ``run_manifest.json`` (config hash, seeds, input hashes, held-out
genes, selection metric), ``training_history.json``, ``heldout_metrics.json``
and the ``best``/``final`` checkpoints, so a later Stage 2 can state which
backbone produced its features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from aivc_model.benchmark_split import assert_fit_eligible
from aivc_model.distributed import require_distinct_devices
from aivc_model.residual_ladder import FixedSplit
from aivc_model.response_training import (
    ResponseLoss,
    ResponseLossWeights,
    TrainingConfig,
    evaluate_response_model,
    make_accelerator,
    split_heldout_genes,
    train_response_model,
)
from aivc_model.stage1_config import Stage1Config, load_stage1_config
from aivc_model.tx1_response_gene_bags_cache import (
    _SCHEMA_VERSION as _RESPONSE_CACHE_SCHEMA_VERSION,
)
from aivc_model.tx1_response_data import base_gene_name

_LOGGER = logging.getLogger("train_geneeffect_response_model")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def esm2_resolvable_genes(path: Path) -> set[str]:
    """Symbols the ESM2 table can actually embed.

    The adapter raises on an unresolved gene rather than zero-filling it, so
    the caller must restrict the gene set up front. Doing it here -- and
    recording what was dropped -- keeps that restriction declared instead of
    turning into a silent partial run.
    """
    payload = np.load(path, allow_pickle=True)
    symbols = np.asarray(payload["symbols"], dtype=object)
    # The table carries a row per requested symbol but flags which ones ESM2
    # actually embedded; ``load_esm2_embeddings`` drops the rest, so counting
    # them here would hand the adapter genes it will raise on.
    resolved = np.asarray(payload["resolved"], dtype=bool)
    return {
        str(symbol).upper()
        for symbol, ok in zip(symbols, resolved, strict=True)
        if bool(ok)
    }


def load_split(path: Path) -> FixedSplit:
    """Load the frozen 226-line split as the fit-eligibility authority."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FixedSplit(
        train=tuple(payload["train"]),
        val=tuple(payload["val"]),
        test=tuple(payload["test"]),
        unlabeled_train=tuple(payload.get("unlabeled_train", ())),
    )


def build_batches(bags: object, control_by_line: dict[str, dict[str, np.ndarray]]):
    """Turn assembled ``GeneBags`` into per-(gene, line) training batches.

    Each bag's gene label is ``GENE@ACH-XXXXXX`` (``composite_gene_key``), so
    the line is recovered from the label rather than guessed, and the control
    cells paired with it are that line's own -- never a pool blended across
    lines, which would make the control mean meaningless.
    """
    batches = []
    for index, label in enumerate(bags.genes):
        label = str(label)
        model_id = label.split("@", 1)[1] if "@" in label else None
        if model_id is None or model_id not in control_by_line:
            raise ValueError(f"bag {label!r} does not name a known line")
        observed = bags.effective_target_bags[index]
        control = control_by_line[model_id]
        batches.append(
            {
                "gene": base_gene_name(label),
                "label": label,
                "model_id": model_id,
                "control": control["input"],
                "control_target": control["target"],
                "observed": observed,
            }
        )
    return batches


def split_control_by_line(bags: object) -> dict[str, dict[str, np.ndarray]]:
    """Group the concatenated control views back into per-line blocks."""
    control_batch = np.asarray(bags.control_batch).astype(str)
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for model_id in sorted(set(control_batch.tolist())):
        mask = control_batch == model_id
        grouped[model_id] = {
            "input": np.asarray(bags.control_input)[mask],
            "target": np.asarray(bags.effective_control_target)[mask],
        }
    return grouped


def _evaluate_freeze_gate(
    metrics: dict[str, object], stage1: "Stage1Config"
) -> dict[str, object]:
    """Compare the held-out result against the §7 margins pinned before the run.

    Both margins are required and both are checked. The null-shuffle arm
    reuses ``evaluate_response_model``'s per-line losses under a permuted
    gene-to-bag assignment, which is the comparison the spec names: a model
    that scores well only because the anchors' bags look alike would clear
    the basal-copy floor and fail here.

    Returns a payload recording every input to the decision, so a later
    reader can see WHY it passed rather than trusting a bare boolean.
    """
    model_loss = float(metrics["model_loss"])
    basal = float(metrics["basal_copy_loss"])
    null_shuffle = float(metrics["null_shuffle_loss"])
    basal_margin = basal - model_loss
    null_margin = null_shuffle - model_loss
    required_basal = float(stage1.thresholds.min_improvement_over_basal_copy)
    required_null = float(stage1.thresholds.min_improvement_over_null_shuffle)
    missing = [
        name
        for name in stage1.thresholds.required_anchor_metrics
        if name not in {"mean_delta_mse", "energy_distance"}
    ]
    if missing:
        raise ValueError(
            f"required_anchor_metrics names unknown metrics {missing}; "
            "this trainer reports mean_delta_mse and energy_distance"
        )
    return {
        "passed": basal_margin >= required_basal and null_margin >= required_null,
        "improvement_over_basal_copy": basal_margin,
        "min_improvement_over_basal_copy": required_basal,
        "improvement_over_null_shuffle": null_margin,
        "min_improvement_over_null_shuffle": required_null,
        "model_loss": model_loss,
        "basal_copy_loss": basal,
        "null_shuffle_loss": null_shuffle,
    }


def _resolve_cpu_flag(device: str) -> bool:
    """Whether to pin the Accelerator to CPU.

    ``auto`` lets Accelerate detect CUDA/MPS; ``cpu`` forces CPU even where a
    GPU exists. The old ``--device cuda`` default raised at ``.to()`` on any
    machine without CUDA rather than falling back, so it is gone.
    """
    return device == "cpu"


def _require_warm_cache_for_multirank(args: argparse.Namespace) -> None:
    """Refuse a multi-rank launch against a cold response cache.

    ``assemble_train_response_gene_bags`` runs on EVERY rank before any
    accelerator object exists -- there is no rank gating inside it. One
    process measured 194.6 GB peak RSS in Phase C, and on 2026-07-26 two
    concurrent cold arms at ~621-625 GB each killed a shared node. N cold
    ranks here would pay that peak simultaneously. This is a memory guard,
    not a race guard: rank-zero gating would not fix it, because every rank
    genuinely needs the assembled bags.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    if args.response_cache_dir is None:
        raise ValueError(
            f"--response-cache-dir is required for a {world_size}-rank launch: "
            "run --assemble-only single-process first "
            "(scripts/run_stage1_response_ddp.sh does this as phase 1)"
        )
    manifest_path = args.response_cache_dir / "response_targets" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"--response-cache-dir {args.response_cache_dir} has no "
            f"response_targets/manifest.json, but this is a {world_size}-rank "
            "launch: every rank would assemble the response bags at once "
            "(~194.6 GB each). Run --assemble-only single-process first."
        )
    # A non-empty directory is not evidence of a usable cache -- a partial or
    # schema-stale one would pass that test and send every rank down the
    # rebuild path this guard exists to prevent. Check the manifest the loader
    # actually reads. NOTE: this still cannot detect a FINGERPRINT mismatch
    # (different max_cells_per_gene / data_seed / sources), because computing
    # the expected fingerprint needs the assembled inputs. Passing --config to
    # both phases, as scripts/run_stage1_response_ddp.sh does, is what keeps
    # the two phases' fingerprints equal.
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded.get("schema_version") != _RESPONSE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"response-targets cache at {manifest_path.parent} has "
            f"schema_version {recorded.get('schema_version')!r}, expected "
            f"{_RESPONSE_CACHE_SCHEMA_VERSION!r}: every rank of this "
            f"{world_size}-rank launch would rebuild it at once. Re-run "
            "--assemble-only single-process first."
        )


def main(argv: list[str] | None = None) -> int:
    """Assemble the anchors, train the backbone, and write the run artifacts."""
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s")

    if not args.assemble_only and args.config is None:
        raise ValueError("--config is required unless --assemble-only is given")
    stage1 = load_stage1_config(args.config) if args.config is not None else None

    from aivc_model.tx1_response_data import assemble_train_response_gene_bags

    split = load_split(args.split_json)
    # Assembly knobs come from the config so phase 1 (--assemble-only) and
    # phase 2 warm/read the SAME cache fingerprint. Defaults here only apply
    # to a bare --assemble-only run with no --config.
    max_cells_per_gene = stage1.train.max_cells_per_gene if stage1 else 128
    total_cells_per_line = stage1.train.total_cells_per_line if stage1 else None
    data_seed = stage1.train.data_seed if stage1 else 42

    if not args.assemble_only:
        _require_warm_cache_for_multirank(args)

    _LOGGER.info("assembling response bags for the anchor lines")
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=args.cell_line_manifest,
        tx1_cache_dir=args.tx1_cache_dir,
        hvg_state_model_dir=args.state_model_dir,
        perturbseq_sources_path=args.perturbseq_sources,
        max_cells_per_gene=max_cells_per_gene,
        total_cells_per_line=total_cells_per_line,
        response_cache_dir=args.response_cache_dir,
        seed=data_seed,
    )
    control_by_line = split_control_by_line(bags)
    for model_id in sorted(control_by_line):
        # Stage 1 must not fit on a line the dependency split holds out.
        assert_fit_eligible(model_id, split)
    _LOGGER.info("anchor lines %s all fit-eligible", ", ".join(sorted(control_by_line)))

    if args.assemble_only:
        # Phase 1 stops here, deliberately before importing accelerate or
        # constructing a model: its whole job is to leave a warm cache.
        _LOGGER.info(
            "--assemble-only: response cache warm at %s, exiting without training",
            args.response_cache_dir,
        )
        return 0

    assert stage1 is not None  # guarded above
    torch.manual_seed(stage1.train.collator_seed)

    from aivc_model.tx1_predicted_response import construct_forward_only_model

    batches = build_batches(bags, control_by_line)
    genes_by_line: dict[str, list[str]] = {}
    for batch in batches:
        genes_by_line.setdefault(batch["model_id"], []).append(batch["gene"])
    heldout = split_heldout_genes(
        genes_by_line,
        fraction=stage1.train.heldout_fraction,
        seed=stage1.train.heldout_seed,
    )
    train_batches = [b for b in batches if b["gene"] not in heldout[b["model_id"]]]
    heldout_batches = [b for b in batches if b["gene"] in heldout[b["model_id"]]]
    _LOGGER.info(
        "%d train batches, %d held-out batches across %d lines",
        len(train_batches),
        len(heldout_batches),
        len(control_by_line),
    )

    resolvable = esm2_resolvable_genes(args.esm2_embeddings)
    wanted = {b["gene"] for b in batches}
    unresolved = sorted(wanted - resolvable)
    drop_fraction = len(unresolved) / len(wanted) if wanted else 0.0
    if drop_fraction > stage1.train.max_esm2_drop_fraction:
        raise ValueError(
            f"{len(unresolved)}/{len(wanted)} perturbation genes "
            f"({drop_fraction:.1%}) have no ESM2 embedding, above the "
            f"{stage1.train.max_esm2_drop_fraction:.1%} gate -- this usually "
            "means the wrong --esm2-embeddings file, not a coverage shortfall"
        )
    if unresolved:
        _LOGGER.info(
            "dropping %d/%d perturbation genes with no ESM2 embedding (%.1f%%)",
            len(unresolved),
            len(wanted),
            100 * drop_fraction,
        )
        train_batches = [b for b in train_batches if b["gene"] in resolvable]
        heldout_batches = [b for b in heldout_batches if b["gene"] in resolvable]
        batches = [b for b in batches if b["gene"] in resolvable]
        if not train_batches or not heldout_batches:
            raise ValueError("ESM2 filtering emptied the train or held-out set")

    # The manifest must report the sets actually trained on, not the pre-ESM2
    # ones: recording pre-filter gene names beside post-filter batch counts
    # would describe a run that never happened.
    heldout_scored: dict[str, list[str]] = {}
    for batch in heldout_batches:
        heldout_scored.setdefault(batch["model_id"], []).append(batch["gene"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Spec §10: the thresholds are pinned BEFORE any parameter moves, so a
    # run cannot retrofit the bar it was judged against.
    (args.out_dir / "stage1_freeze_thresholds.json").write_text(
        json.dumps(stage1.freeze_thresholds_payload(), indent=2) + "\n"
    )
    _LOGGER.info("pinned stage1_freeze_thresholds.json before training")

    config = TrainingConfig(
        max_epochs=stage1.train.max_epochs,
        patience=stage1.train.patience,
        learning_rate=stage1.train.learning_rate,
        weight_decay=stage1.train.weight_decay,
        max_bag=stage1.train.max_bag,
        grad_clip=stage1.train.grad_clip,
        seed=stage1.train.train_seed,
        log_every=stage1.train.log_every,
        ddp_static_graph=stage1.train.ddp_static_graph,
        ddp_find_unused_parameters=stage1.train.ddp_find_unused_parameters,
    )
    accelerator = make_accelerator(config, cpu=_resolve_cpu_flag(args.device))
    require_distinct_devices(accelerator)
    if stage1.train.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(stage1.train.float32_matmul_precision)
    logging.getLogger().setLevel(
        args.log_level if accelerator.is_main_process else "WARNING"
    )

    model = construct_forward_only_model(
        model_cls=_state_model_cls(),
        hparams_checkpoint_path=args.state_checkpoint,
        input_dim=int(bags.input_dim),
        output_dim=int(bags.effective_target_dim),
        pert_dim=stage1.train.pert_dim,
        genes=sorted({b["gene"] for b in batches}),
        esm2_embeddings_path=args.esm2_embeddings,
    )

    loss_fn = ResponseLoss(
        ResponseLossWeights(
            mean_delta=stage1.train.w_mean_delta, energy=stage1.train.w_energy
        )
    )
    anchor_weights = dict(stage1.thresholds.anchor_weights)
    history = train_response_model(
        model,
        train_batches,
        heldout_batches,
        anchor_weights=anchor_weights,
        out_dir=args.out_dir,
        config=config,
        loss_fn=loss_fn,
        accelerator=accelerator,
    )
    if not accelerator.is_main_process:
        return 0

    (args.out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, default=str) + "\n"
    )

    # Score the checkpoint that was SELECTED, not whatever weights the last
    # epoch left in memory. With early stopping those differ, and reporting
    # the final model's metrics beside a manifest naming an earlier best epoch
    # would describe a model nobody will ever load.
    selected = accelerator.unwrap_model(model)
    best_weights = args.out_dir / "best" / "pytorch_model.bin"
    load_result = selected.load_state_dict(
        torch.load(best_weights, map_location=accelerator.device), strict=True
    )
    _LOGGER.info(
        "reloaded best checkpoint (epoch %s) before scoring: %s",
        history["best_epoch"],
        load_result,
    )
    metrics = evaluate_response_model(
        selected, heldout_batches, loss_fn=loss_fn, device=accelerator.device
    )
    _LOGGER.info(
        "held-out loss %.6f vs basal-copy floor %.6f",
        metrics["model_loss"],
        metrics["basal_copy_loss"],
    )
    # Spec §7 is a GATE, not a record: state whether the run cleared the
    # margins that were pinned before it started. Recording the numbers and
    # leaving the comparison to a human is exactly the "record converged
    # metrics and freeze" the spec rules out.
    gate = _evaluate_freeze_gate(metrics, stage1)
    metrics["freeze_gate"] = gate
    _LOGGER.info(
        "freeze gate: %s (basal-copy margin %.6f vs required %.6f)",
        "PASS" if gate["passed"] else "FAIL",
        gate["improvement_over_basal_copy"],
        gate["min_improvement_over_basal_copy"],
    )
    (args.out_dir / "heldout_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_path": str(args.config),
                "config_sha256": stage1.source_sha256,
                "collator_seed": stage1.train.collator_seed,
                "data_seed": data_seed,
                "train_seed": stage1.train.train_seed,
                "heldout_seed": stage1.train.heldout_seed,
                "heldout_fraction": stage1.train.heldout_fraction,
                "heldout_genes": {k: sorted(v) for k, v in heldout_scored.items()},
                "anchor_lines": sorted(control_by_line),
                "anchor_weights": anchor_weights,
                "esm2_unresolved_genes": unresolved,
                "esm2_drop_fraction": drop_fraction,
                "n_train_batches": len(train_batches),
                "n_heldout_batches": len(heldout_batches),
                "input_dim": int(bags.input_dim),
                "output_dim": int(bags.effective_target_dim),
                "world_size": accelerator.num_processes,
                "device": str(accelerator.device),
                "mixed_precision": str(accelerator.mixed_precision),
                "selection_metric": history["selection_metric"],
                "best_epoch": history["best_epoch"],
                "best_metric_value": history["best_metric_value"],
                "stopped_early": history["stopped_early"],
                "loss_weights": {
                    "mean_delta": stage1.train.w_mean_delta,
                    "energy": stage1.train.w_energy,
                },
                "input_sha256": {
                    "split_json": _sha256(args.split_json),
                    "state_checkpoint": _sha256(args.state_checkpoint),
                    "esm2_embeddings": _sha256(args.esm2_embeddings),
                    "perturbseq_sources": _sha256(args.perturbseq_sources),
                },
            },
            indent=2,
        )
        + "\n"
    )
    _LOGGER.info("wrote %s", args.out_dir)
    return 0


def _state_model_cls():
    """Import the STATE model class lazily; it is not installed on the Mac."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    return StateTransitionPerturbationModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Only paths and run-mode switches live here. Every hyperparameter and
    every Exp13 spec §7 freeze threshold comes from ``--config``, so the
    values a run used are pinned in one tracked file instead of scattered
    across a shell invocation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--cell-line-manifest", type=Path, required=True)
    parser.add_argument("--tx1-cache-dir", type=Path, required=True)
    parser.add_argument("--state-model-dir", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--perturbseq-sources", type=Path, required=True)
    parser.add_argument("--response-cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help=(
            "Assemble the response bags into --response-cache-dir and exit 0 "
            "without importing accelerate or moving a parameter. Phase 1 of "
            "scripts/run_stage1_response_ddp.sh: assembly runs on every rank "
            "before any accelerator exists and peaked at 194.6 GB RSS in "
            "Phase C, so it must happen once, single-process, first."
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
