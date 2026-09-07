"""P1-B preparation, response adaptation and independent diagnostic exports."""

from contextlib import nullcontext
import torch
from src.model.response import predict_bags, response_terms
from dataclasses import replace

from pathlib import Path
import json
from collections import Counter
import random
import numpy as np
from src.experiments.geneeffect import _write_json
from src.experiments.p1b_preparation import (
    prepare_bundle,
    open_bundle,
    load_state,
    digest,
)
from src.data.p1b import balanced_epoch
from src.eval.p1b import aggregate, evaluate_rows, export_predictions
from src.model.p1b import configure_parameters
from src.training.p1b import fit, stage2_eligible


def autocast(device):
    device = torch.device(device)
    return (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def loss_for_indices(model, view, indices, *, device):
    batch = view.batch(indices, device)
    with autocast(device):
        predicted = predict_bags(model, batch.controls_tx1, batch.genes, seed=0)
    coordinates = view.bundle["coordinates"]
    masked = replace(
        batch,
        observed_hvg=tuple(y[:, coordinates] for y in batch.observed_hvg),
        control_hvg=tuple(y[:, coordinates] for y in batch.control_hvg),
    )
    terms = response_terms(tuple(p[:, coordinates] for p in predicted), masked)
    return terms["mean_delta_mse"].mean() + terms["energy_distance"].mean()


def parser():
    import argparse

    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    for name in ("evaluate", "train-interface", "train-stage2", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--runs", type=Path, required=True)
        if name != "compare":
            command.add_argument(
                "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
            )
        if name.startswith("train"):
            command.add_argument("--resume", action="store_true")
        if name == "train-stage2":
            command.add_argument(
                "--arm", choices=["B-continue", "B-unfreeze"], required=True
            )
        if name == "evaluate":
            command.add_argument(
                "--state",
                choices=[
                    "B-native",
                    "B-init",
                    "B-joint",
                    "B-interface",
                    "B-continue",
                    "B-unfreeze",
                ],
                required=True,
            )
            command.add_argument(
                "--external",
                action="store_true",
                help="Freeze all adaptation decisions before exporting Jurkat",
            )
    return result


def main(argv=None):
    import os

    args = parser().parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("P1-B requires one process per arm")
    if args.command == "prepare":
        prepare_bundle(args.checkpoint, args.out_dir)
    elif args.command.startswith("train"):
        train_arm(
            args.prepared,
            args.runs,
            "B-interface" if args.command == "train-interface" else args.arm,
            device=args.device,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate_state(
            args.prepared,
            args.runs,
            args.state,
            device=args.device,
            external=args.external,
        )
    else:
        from src.eval.p1b_comparison import compare

        compare(args.prepared, args.runs)


def training_record(runs, arm):
    path = Path(runs) / arm / "training.json"
    record = json.loads(path.read_text())
    if record["status"] != "completed":
        raise ValueError(f"{arm} training is not completed")
    return record


def train_arm(prepared, runs, arm, *, device, resume=False):
    runs = Path(runs)
    if (runs / "external_evaluation.json").exists():
        raise ValueError(
            "external evaluation has started: further adaptation prohibited"
        )
    bundle, view, manifest = open_bundle(prepared)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = load_state(prepared, "B-init", manifest).to(device)
    source_hash = manifest["model_files"]["B-init"]
    if arm != "B-interface":
        decision = json.loads((runs / "stage2.json").read_text())
        if decision["bundle"] != manifest["bundle_sha256"] or not decision["eligible"]:
            raise ValueError("stage two not eligible for these inputs")
        training_record(runs, "B-interface")
        start = runs / "B-interface" / "best.pt"
        if digest(start) != decision["start_sha256"]:
            raise ValueError("interface checkpoint changed after stage decision")
        source_hash = digest(start)
        model.load_state_dict(
            torch.load(start, map_location="cpu", weights_only=False)["model_state"],
            strict=True,
        )
    groups, report = configure_parameters(
        model, bundle["loaded_keys"], unfreeze=arm == "B-unfreeze"
    )
    directory = runs / arm
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "parameters.json", report)
    identity = f"{manifest['bundle_sha256']}:{arm}:{source_hash}"
    # A full real 192-condition first update is the memory/finite-gradient check.
    # Any OOM is fatal; batch size never changes automatically.
    state = fit(
        model,
        groups,
        lambda epoch: balanced_epoch(
            bundle["keys"], bundle["splits"]["train"], bundle["anchors"], epoch
        ),
        lambda indices: loss_for_indices(model, view, indices, device=device),
        lambda: aggregate(
            evaluate_rows(model, view, bundle["splits"]["val"], device=device)
        ),
        directory,
        identity=identity,
        resume=resume,
        train_evaluate=lambda: aggregate(
            evaluate_rows(model, view, bundle["splits"]["train"], device=device)
        ),
        count_exposure=lambda indices: Counter(bundle["keys"][i][0] for i in indices),
    )
    if arm == "B-interface":
        history = json.loads((directory / "history.json").read_text())
        initial = history[0]["val"]["response_loss"]
        _write_json(
            runs / "stage2.json",
            {
                "bundle": manifest["bundle_sha256"],
                "initial_loss": initial,
                "best_loss": state["best_loss"],
                "eligible": stage2_eligible(initial, state["best_loss"]),
                "start_sha256": digest(directory / "best.pt"),
            },
        )
    return state


def freeze_external_selection(prepared, runs, manifest):
    runs = Path(runs)
    decision = json.loads((runs / "stage2.json").read_text())
    if decision["bundle"] != manifest["bundle_sha256"]:
        raise ValueError("stage decision belongs to different inputs")
    arms = ["B-interface"] + (
        ["B-continue", "B-unfreeze"] if decision["eligible"] else []
    )
    checkpoints = {}
    for arm in arms:
        record = training_record(runs, arm)
        if not record["identity"].startswith(
            manifest["bundle_sha256"] + ":" + arm + ":"
        ):
            raise ValueError("training record identity mismatch")
        checkpoints[arm] = digest(runs / arm / "best.pt")
    if checkpoints["B-interface"] != decision["start_sha256"]:
        raise ValueError("stage start checkpoint identity mismatch")
    selection = {
        "bundle": manifest["bundle_sha256"],
        "decision": decision,
        "checkpoints": checkpoints,
    }
    path = runs / "external_evaluation.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("selected checkpoints changed")
    _write_json(path, selection)
    return selection


def evaluate_state(prepared, runs, state, *, device, external=False):
    runs = Path(runs)
    bundle, view, manifest = open_bundle(prepared)
    if external:
        freeze_external_selection(prepared, runs, manifest)
    directory = runs / "evaluation" / state / ("external" if external else "internal")
    directory.mkdir(parents=True, exist_ok=True)
    if state == "B-native":
        _write_json(directory / "evaluation.json", manifest["native"])
        return
    model = load_state(
        prepared, state if state in ("B-init", "B-joint") else "B-init", manifest
    ).to(device)
    checkpoint_hash = manifest["model_files"].get(state)
    if state not in ("B-init", "B-joint"):
        record = training_record(runs, state)
        if not record["identity"].startswith(
            manifest["bundle_sha256"] + ":" + state + ":"
        ):
            raise ValueError("evaluation checkpoint identity mismatch")
        path = runs / state / "best.pt"
        checkpoint_hash = digest(path)
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state"],
            strict=True,
        )
    status = {
        "bundle": manifest["bundle_sha256"],
        "checkpoint_sha256": checkpoint_hash,
        "state": state,
    }
    _write_json(directory / "evaluation.json", {**status, "status": "running"})
    try:
        # External export includes source holdouts for paired effect differences.
        roles = ["val", "external"] if external else ["train", "val"]
        frame, _ = export_predictions(
            model, view, roles, device=device, directory=directory
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


if __name__ == "__main__":
    main()
