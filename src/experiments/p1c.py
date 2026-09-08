"""P1-C CLI: fold preparation, native diagnostics, variant training and export."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.p1b import ResponseView, balanced_epoch
from src.eval.p1b import (
    aggregate,
    evaluate_rows,
    export_predictions,
    predictions,
    score_bag,
)
from src.experiments.geneeffect import _write_json
from src.experiments.p1b import loss_for_indices
from src.experiments.p1b_preparation import digest, open_bundle
from src.model.p1c import VARIANTS, INPUT_LAYOUT, build_variant, parameter_groups
from src.training.p1b import fit


MAX_EPOCHS = 50
PATIENCE = 5

_NATIVE_STATE_VARIANTS = ("V2-null", "V2", "V3")


def native_inputs(manifest):
    """Load the native STATE checkpoint's ``state_dict`` and the gene one-hot map.

    Verifies both files' digests against the manifest before loading either.
    """
    checkpoint_path = manifest["native_state_checkpoint"]
    if digest(checkpoint_path) != manifest["native_checkpoint_sha256"]:
        raise ValueError("native checkpoint identity changed")
    map_path = manifest["native_map"]
    if digest(map_path) != manifest["native_vocabulary_sha256"]:
        raise ValueError("native vocabulary identity changed")
    native_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    native_map = torch.load(map_path, map_location="cpu", weights_only=False)
    return native_state, native_map


def _read_fold(prepared, manifest):
    record = json.loads((Path(prepared) / "fold.json").read_text())
    if record["external"] != manifest["external"]:
        raise ValueError("fold.json disagrees with manifest")
    return record


def restrict_to_native(bundle, native_genes):
    """Shallow-copy ``bundle`` restricted to conditions the native model can predict."""
    bundle = dict(bundle)
    bundle["splits"] = {
        role: [i for i in indices if bundle["keys"][i][1] in native_genes]
        for role, indices in bundle["splits"].items()
    }
    bundle["panels"] = {
        key: panel
        for key, panel in bundle["panels"].items()
        if key.endswith("/native_common") or key.endswith("/native_all")
    }
    return bundle


def _load_template(prepared, manifest):
    path = Path(prepared) / "B-init.pt"
    if digest(path) != manifest["model_files"]["B-init"]:
        raise ValueError("prepared model identity changed")
    return torch.load(path, map_location="cpu", weights_only=False)


def init_check(backbone, view, bundle, variant, device):
    """Validate a freshly built variant's zero-effect (or zero-adapter) start."""
    tolerance = 1e-6 if torch.device(device).type == "cpu" else 1e-4
    if variant in ("V1", "V3"):
        max_deviation = 0.0
        for anchor in bundle["anchors"]:
            indices = [
                i for i in bundle["splits"]["val"] if bundle["keys"][i][0] == anchor
            ][:4]
            if not indices:
                raise ValueError(
                    "zero-effect initialisation check failed: "
                    f"anchor {anchor!r} has no validation conditions"
                )
            batch = view.batch(indices, device)
            with torch.no_grad():
                predicted = predictions(backbone, batch, device)
            for pred, control in zip(predicted, batch.control_hvg):
                if not torch.allclose(pred, control, atol=tolerance, rtol=0):
                    raise ValueError(
                        "zero-effect initialisation check failed: "
                        f"{variant} prediction differs from control_hvg beyond "
                        f"tolerance {tolerance} for anchor {anchor!r}"
                    )
                max_deviation = max(max_deviation, float((pred - control).abs().max()))
        return {
            "status": "passed",
            "max_abs_deviation": max_deviation,
            "tolerance": tolerance,
        }
    if variant in ("V2-null", "V2"):
        final = backbone.perturbations.adapter.net[-1]
        if not (
            torch.all(final.weight == 0).item() and torch.all(final.bias == 0).item()
        ):
            raise ValueError(
                "zero-effect initialisation check failed: "
                f"{variant} adapter final layer is not all zeros"
            )
        if backbone.batch_index != 0:
            raise ValueError(
                "zero-effect initialisation check failed: "
                f"{variant} must be built with batch_index=0"
            )
        return {"status": "passed", "check": "adapter_final_layer_zero"}
    if variant == "V0":
        return {"status": "not_applicable"}
    raise ValueError(f"no init check defined for variant {variant!r}")


def train_variant(prepared, runs, variant, lr, *, device, resume=False):
    runs = Path(runs)
    if (runs / "external_evaluation.json").exists():
        raise ValueError(
            "external evaluation has started: further adaptation prohibited"
        )
    input_layout = INPUT_LAYOUT[variant]
    bundle, view, manifest = open_bundle(prepared, input_layout=input_layout)
    fold = _read_fold(prepared, manifest)["fold"]
    template = _load_template(prepared, manifest)
    template_sha256 = manifest["model_files"]["B-init"]
    identity = f"{manifest['bundle_sha256']}:{variant}:{lr}:{template_sha256}"

    # Fail before anything is written: fit() itself would reject a resume-less
    # rerun or an identity-mismatched resume, but only after parameters.json
    # and init_check have already run against this run directory.
    if (runs / "last.pt").exists() and not resume:
        raise FileExistsError("run exists; use resume")
    if resume:
        existing = json.loads((runs / "training.json").read_text())
        if existing["identity"] != identity:
            raise ValueError("resume identity mismatch")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    kwargs = {}
    if variant in _NATIVE_STATE_VARIANTS:
        native_state, _ = native_inputs(manifest)
        kwargs["native_state"] = native_state
    backbone, report = build_variant(template, variant, **kwargs)
    backbone = backbone.to(device)

    runs.mkdir(parents=True, exist_ok=True)
    _write_json(runs / "parameters.json", report)
    check = init_check(backbone, view, bundle, variant, device)

    groups = parameter_groups(backbone, lr)
    state = fit(
        backbone,
        groups,
        lambda epoch: balanced_epoch(
            bundle["keys"], bundle["splits"]["train"], bundle["anchors"], epoch
        ),
        lambda indices: loss_for_indices(backbone, view, indices, device=device),
        lambda: aggregate(
            evaluate_rows(backbone, view, bundle["splits"]["val"], device=device)
        ),
        runs,
        identity=identity,
        resume=resume,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        train_evaluate=lambda: aggregate(
            evaluate_rows(backbone, view, bundle["splits"]["train"], device=device)
        ),
        count_exposure=lambda indices: Counter(bundle["keys"][i][0] for i in indices),
    )
    training = json.loads((runs / "training.json").read_text())
    training.update(
        variant=variant,
        fold=fold,
        lr=lr,
        init_check=check,
        input_layout=input_layout,
    )
    _write_json(runs / "training.json", training)
    return state


def evaluate_variant(prepared, runs, *, device, external=False):
    runs = Path(runs)
    training = json.loads((runs / "training.json").read_text())
    if training["status"] != "completed":
        raise ValueError("P1-C training is not completed")
    variant = training["variant"]
    input_layout = INPUT_LAYOUT[variant]
    bundle, view, manifest = open_bundle(prepared, input_layout=input_layout)
    _read_fold(prepared, manifest)
    if not training["identity"].startswith(f"{manifest['bundle_sha256']}:{variant}:"):
        raise ValueError("evaluation checkpoint identity mismatch")
    template = _load_template(prepared, manifest)
    kwargs = {}
    if variant in _NATIVE_STATE_VARIANTS:
        native_state, _ = native_inputs(manifest)
        kwargs["native_state"] = native_state
    backbone, _ = build_variant(template, variant, **kwargs)
    backbone = backbone.to(device)

    checkpoint_path = runs / "best.pt"
    checkpoint_hash = digest(checkpoint_path)
    backbone.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
            "model_state"
        ],
        strict=True,
    )

    if external:
        selection = {
            "checkpoint_sha256": checkpoint_hash,
            "variant": variant,
            "fold": training["fold"],
            "bundle": manifest["bundle_sha256"],
        }
        path = runs / "external_evaluation.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError("selected checkpoint changed")
        _write_json(path, selection)

    directory = runs / "evaluation" / ("external" if external else "internal")
    directory.mkdir(parents=True, exist_ok=True)
    status = {
        "bundle": manifest["bundle_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "variant": variant,
        "fold": training["fold"],
    }
    _write_json(directory / "evaluation.json", {**status, "status": "running"})
    try:
        roles = ["val", "external"] if external else ["train", "val"]
        frame, _ = export_predictions(
            backbone, view, roles, device=device, directory=directory
        )
        summary = frame.groupby(["role", "model_id", "panel", "method"])[
            ["mean_delta_mse", "energy_distance", "response_loss"]
        ].agg(["mean", "count"])
        summary.to_csv(directory / "summary.csv")
        _write_json(directory / "evaluation.json", {**status, "status": "completed"})
        return frame
    except Exception as exc:
        _write_json(
            directory / "evaluation.json",
            {**status, "status": "failed", "error": str(exc)},
        )
        raise


def _indices_with_roles(bundle, roles):
    """Flatten ``bundle["splits"][role]`` for each role, tagging each index with it."""
    return [(i, role) for role in roles for i in bundle["splits"][role]]


@torch.no_grad()
def _native_null_predictions(model, view, indexed, coordinates, device, batch_size=32):
    """Score every ``(index, role)`` condition with its gene replaced by
    ``"non-targeting"``."""
    model.eval()
    rows = []
    for start in range(0, len(indexed), batch_size):
        chunk = indexed[start : start + batch_size]
        selected = [i for i, _ in chunk]
        batch = view.batch(selected, device)
        genes = tuple("non-targeting" for _ in selected)
        outputs = predictions(model, batch, device, genes=genes)
        for (i, role), pred, observed, basal in zip(
            chunk, outputs, batch.observed_hvg, batch.control_hvg
        ):
            anchor, gene = view.bundle["keys"][i]
            rows.append(
                {
                    "role": role,
                    "model_id": anchor,
                    "gene": gene,
                    **score_bag(pred, observed, basal, coordinates),
                }
            )
    return pd.DataFrame(rows)


def evaluate_native(prepared, runs, batch_indices, *, device):
    runs = Path(runs)
    prepared = Path(prepared)
    bundle, view, manifest = open_bundle(
        prepared, input_layout=INPUT_LAYOUT["N-native"]
    )
    fold = _read_fold(prepared, manifest)["fold"]
    template = _load_template(prepared, manifest)
    native_state, native_map = native_inputs(manifest)
    native_checkpoint_hash = manifest["native_checkpoint_sha256"]
    native_genes = {str(gene) for gene in native_map}
    restricted_bundle = restrict_to_native(bundle, native_genes)
    restricted_view = ResponseView(
        restricted_bundle, view.cache, input_layout=INPUT_LAYOUT["N-native"]
    )
    coordinates = restricted_bundle["coordinates"]
    has_non_targeting = "non-targeting" in native_map

    for index in batch_indices:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        backbone, _ = build_variant(
            template,
            "N-native",
            native_state=native_state,
            native_map=native_map,
            batch_index=index,
        )
        backbone = backbone.to(device)
        name = "N-native" if index == 0 else f"N-native-b{index}"
        for external, roles in ((False, ("train", "val")), (True, ("val", "external"))):
            directory = (
                runs / "evaluation" / name / ("external" if external else "internal")
            )
            directory.mkdir(parents=True, exist_ok=True)
            status = {
                "bundle": manifest["bundle_sha256"],
                "checkpoint_sha256": native_checkpoint_hash,
                "variant": "N-native",
                "batch_index": index,
                "fold": fold,
            }
            _write_json(directory / "evaluation.json", {**status, "status": "running"})
            try:
                indexed = _indices_with_roles(restricted_bundle, roles)
                if index == 0:
                    frame, _ = export_predictions(
                        backbone,
                        restricted_view,
                        list(roles),
                        device=device,
                        directory=directory,
                    )
                    summary = frame.groupby(["role", "model_id", "panel", "method"])[
                        ["mean_delta_mse", "energy_distance", "response_loss"]
                    ].agg(["mean", "count"])
                    summary.to_csv(directory / "summary.csv")
                    if has_non_targeting:
                        null_frame = _native_null_predictions(
                            backbone, restricted_view, indexed, coordinates, device
                        )
                        null_frame.to_parquet(
                            directory / "native_null.parquet", index=False
                        )
                    else:
                        _write_json(
                            directory / "native_null.json",
                            {
                                "status": "unavailable",
                                "reason": "non-targeting absent from native_map",
                            },
                        )
                else:
                    indices = [i for i, _ in indexed]
                    frame = evaluate_rows(
                        backbone, restricted_view, indices, device=device
                    )
                    frame["role"] = [role for _, role in indexed]
                    frame.to_csv(directory / "summary.csv", index=False)
                _write_json(
                    directory / "evaluation.json", {**status, "status": "completed"}
                )
            except Exception as exc:
                _write_json(
                    directory / "evaluation.json",
                    {**status, "status": "failed", "error": str(exc)},
                )
                raise


def parser():
    import argparse

    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--fold", required=True)
    prepare.add_argument("--reference-manifest", type=Path, default=None)

    tier0 = commands.add_parser("tier0")
    tier0.add_argument("--p1b-runs", type=Path, required=True)
    tier0.add_argument("--p1b-prepared", type=Path, required=True)
    tier0.add_argument("--out-dir", type=Path, required=True)

    evaluate_native_parser = commands.add_parser("evaluate-native")
    evaluate_native_parser.add_argument("--prepared", type=Path, required=True)
    evaluate_native_parser.add_argument("--runs", type=Path, required=True)
    evaluate_native_parser.add_argument(
        "--batch-indices", type=int, nargs="+", default=[0]
    )
    evaluate_native_parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    train = commands.add_parser("train")
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--runs", type=Path, required=True)
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--lr", type=float, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prepared", type=Path, required=True)
    evaluate.add_argument("--runs", type=Path, required=True)
    evaluate.add_argument("--external", action="store_true")
    evaluate.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    compare = commands.add_parser("compare")
    compare.add_argument("--root", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)

    return result


def main(argv=None):
    import os

    args = parser().parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("P1-C requires one process per arm")
    if args.command == "prepare":
        from src.experiments.p1c_preparation import prepare_fold

        prepare_fold(
            args.checkpoint,
            args.fold,
            args.out_dir,
            reference_manifest=args.reference_manifest,
        )
    elif args.command == "tier0":
        from src.eval.p1c_tier0 import run_tier0

        run_tier0(args.p1b_runs, args.p1b_prepared, args.out_dir)
    elif args.command == "evaluate-native":
        evaluate_native(
            args.prepared, args.runs, args.batch_indices, device=args.device
        )
    elif args.command == "train":
        train_variant(
            args.prepared,
            args.runs,
            args.variant,
            args.lr,
            device=args.device,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate_variant(
            args.prepared, args.runs, device=args.device, external=args.external
        )
    else:
        from src.eval.p1c import summarize

        summarize(args.root, args.out_dir)


if __name__ == "__main__":
    main()
