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

The run writes ``run_manifest.json`` (config, seeds, input hashes, held-out
genes), ``training_history.json`` and ``heldout_metrics.json`` next to the
checkpoint, so a later Stage 2 can state which backbone produced its features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from aivc_model.benchmark_split import assert_fit_eligible
from aivc_model.residual_ladder import FixedSplit
from aivc_model.response_training import (
    ResponseLoss,
    ResponseLossWeights,
    TrainingConfig,
    evaluate_response_model,
    split_heldout_genes,
    train_response_model,
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


def main(argv: list[str] | None = None) -> int:
    """Assemble the anchors, train the backbone, and write the run artifacts."""
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s")
    torch.manual_seed(args.collator_seed)

    from aivc_model.tx1_predicted_response import construct_forward_only_model
    from aivc_model.tx1_response_data import assemble_train_response_gene_bags

    split = load_split(args.split_json)

    _LOGGER.info("assembling response bags for the anchor lines")
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=args.cell_line_manifest,
        tx1_cache_dir=args.tx1_cache_dir,
        hvg_state_model_dir=args.state_model_dir,
        perturbseq_sources_path=args.perturbseq_sources,
        max_cells_per_gene=args.max_cells_per_gene,
        total_cells_per_line=args.total_cells_per_line,
        response_cache_dir=args.response_cache_dir,
        seed=args.data_seed,
    )
    control_by_line = split_control_by_line(bags)
    for model_id in sorted(control_by_line):
        # Stage 1 must not fit on a line the dependency split holds out.
        assert_fit_eligible(model_id, split)
    _LOGGER.info("anchor lines %s all fit-eligible", ", ".join(sorted(control_by_line)))

    batches = build_batches(bags, control_by_line)
    genes_by_line: dict[str, list[str]] = {}
    for batch in batches:
        genes_by_line.setdefault(batch["model_id"], []).append(batch["gene"])
    heldout = split_heldout_genes(
        genes_by_line, fraction=args.heldout_fraction, seed=args.heldout_seed
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
    if drop_fraction > args.max_esm2_drop_fraction:
        raise ValueError(
            f"{len(unresolved)}/{len(wanted)} perturbation genes "
            f"({drop_fraction:.1%}) have no ESM2 embedding, above the "
            f"{args.max_esm2_drop_fraction:.1%} gate -- this usually means the "
            "wrong --esm2-embeddings file, not a coverage shortfall"
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

    model = construct_forward_only_model(
        model_cls=_state_model_cls(),
        hparams_checkpoint_path=args.state_checkpoint,
        input_dim=int(bags.input_dim),
        output_dim=int(bags.effective_target_dim),
        pert_dim=args.pert_dim,
        genes=sorted({b["gene"] for b in batches}),
        esm2_embeddings_path=args.esm2_embeddings,
    )

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_bag=args.max_bag,
        seed=args.train_seed,
    )
    loss_fn = ResponseLoss(
        ResponseLossWeights(mean_delta=args.w_mean_delta, energy=args.w_energy)
    )
    history = train_response_model(
        model, train_batches, config=config, loss_fn=loss_fn, device=args.device
    )
    # Persist before scoring. Evaluation is cheap relative to training but can
    # still raise, and a metrics failure must not discard hours of fitting.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out_dir / "response_model.pt")
    (args.out_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, default=str) + "\n"
    )
    _LOGGER.info("checkpoint saved before evaluation")

    metrics = evaluate_response_model(
        model, heldout_batches, loss_fn=loss_fn, device=args.device
    )
    _LOGGER.info(
        "held-out loss %.6f vs basal-copy floor %.6f",
        metrics["model_loss"],
        metrics["basal_copy_loss"],
    )

    (args.out_dir / "heldout_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "collator_seed": args.collator_seed,
                "data_seed": args.data_seed,
                "train_seed": args.train_seed,
                "heldout_seed": args.heldout_seed,
                "heldout_fraction": args.heldout_fraction,
                "heldout_genes": {k: sorted(v) for k, v in heldout.items()},
                "anchor_lines": sorted(control_by_line),
                "esm2_unresolved_genes": unresolved,
                "esm2_drop_fraction": drop_fraction,
                "n_train_batches": len(train_batches),
                "n_heldout_batches": len(heldout_batches),
                "input_dim": int(bags.input_dim),
                "output_dim": int(bags.effective_target_dim),
                "loss_weights": {
                    "mean_delta": args.w_mean_delta,
                    "energy": args.w_energy,
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--cell-line-manifest", type=Path, required=True)
    parser.add_argument("--tx1-cache-dir", type=Path, required=True)
    parser.add_argument("--state-model-dir", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--perturbseq-sources", type=Path, required=True)
    parser.add_argument("--response-cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pert-dim", type=int, default=2024)
    parser.add_argument("--max-esm2-drop-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-bag", type=int, default=128)
    parser.add_argument("--max-cells-per-gene", type=int, default=128)
    parser.add_argument("--total-cells-per-line", type=int, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--w-mean-delta", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=1.0)
    parser.add_argument("--collator-seed", type=int, default=20260818)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=20260818)
    parser.add_argument("--heldout-seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
