"""Six-arm response-model comparison in STATE's log expression space.

Leave-one-anchor-out over the four response anchors, as independent jobs (the
untrained arms, one per trained arm and held-out anchor, the summary) that each
write ``folds/<arm>__<anchor>.json`` once; ``verdicts.json`` is written last.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.io import write_json
from src.data.prepared import load_inputs, read_manifest
from src.experiments.joint.config import load_config
from src.experiments.joint.response_arms import (
    ARMS,
    SEED,
    TRAINED_ARMS,
    UNTRAINED_ARMS,
    VERDICTS,
    Arm,
    OneHotPerturbations,
    World,
    array,
    build_world,
    fold_indices,
    fold_path,
    gene_blind_shuffled,
    global_mean_effect,
    loss_ratio,
    losses,
    mlp,
    no_change,
    released_state,
    result,
    score_arm,
    state_joint,
    train,
    write_fold,
)
from src.model.state import load_released_state

HCT116 = "ACH-000971"


def _anchors(config: Mapping[str, Any], inputs: Any) -> tuple[str, ...]:
    if inputs is not None:
        return tuple(inputs.response_anchors)
    return tuple(read_manifest(Path(config["prepared_root"]))["response_anchors"])


def _released_state_factory(config: Mapping[str, Any]) -> Callable[[], nn.Module]:
    def build() -> nn.Module:
        return load_released_state(
            Path(config["paths"]["state_checkpoint"]),
            cell_set_len=int(config["model"]["cell_sentence_len"]),
        )

    return build


def _untrained_done(out_dir: Path, anchors: Sequence[str]) -> bool:
    return (out_dir / "sanity.json").is_file() and all(
        fold_path(out_dir, arm, a).is_file() for arm in UNTRAINED_ARMS for a in anchors
    )


def run_untrained_arms(
    world: World,
    config: Mapping[str, Any],
    out_dir: Path,
    state_factory: Callable[[], nn.Module],
) -> None:
    comparison = config["comparison"]
    batch_size, shuffles = int(comparison["batch_size"]), int(comparison["shuffles"])
    anchors = world.anchors
    out_dir.mkdir(parents=True, exist_ok=True)

    def pending(arm: str) -> list[str]:
        return [a for a in anchors if not fold_path(out_dir, arm, a).is_file()]

    def score(arm: Arm, a: str) -> dict[str, list[float | None]]:
        held_out, source = fold_indices(world, a)
        return score_arm(
            arm,
            world,
            held_out,
            source,
            position=anchors.index(a),
            shuffles=shuffles,
            batch_size=batch_size,
        )

    if pending("no_change"):
        # The basal bag is the prediction whatever the arm, fold or gene, so one
        # pass over every condition gives every fold's held-out and source losses.
        everything = np.concatenate([world.by_anchor[a] for a in anchors])
        loss = np.full(len(world.model_ids), np.nan)
        loss[everything] = losses(no_change(world), world, everything, batch_size)
        for a in pending("no_change"):
            held_out, source = fold_indices(world, a)
            shuffled = gene_blind_shuffled(loss[held_out], shuffles)
            write_fold(
                out_dir,
                world,
                "no_change",
                a,
                result(loss[held_out], shuffled, loss[source]),
            )

    # The released checkpoint is untrained, so its held-out score per anchor is
    # also the sanity line.
    if pending("released_state"):
        vocabulary = torch.load(
            Path(config["paths"]["state_model_dir"]) / "pert_onehot_map.pt",
            map_location="cpu",
            weights_only=False,
        )
        released = released_state(
            world, state_factory(), OneHotPerturbations(vocabulary)
        )
        for a in pending("released_state"):
            write_fold(out_dir, world, "released_state", a, score(released, a))
    if not (out_dir / "sanity.json").is_file():
        sanity = {}
        for a in anchors:
            fold = json.loads(fold_path(out_dir, "released_state", a).read_text())
            base = json.loads(fold_path(out_dir, "no_change", a).read_text())
            loss = array(fold["loss"])
            sanity[a] = {
                "ratio_to_no_change": _number(loss_ratio(loss, array(base["loss"]))),
                "covered_genes": int(np.isfinite(loss).sum()),
                "total_genes": len(loss),
            }
        write_json(
            out_dir / "sanity.json",
            {
                "checkpoint": str(config["paths"]["state_checkpoint"]),
                "statistic": "released STATE checkpoint, untrained: mean loss over the "
                "anchor's covered conditions / no-change mean over the same conditions",
                "anchors": sanity,
            },
        )

    for a in pending("global_mean_effect"):
        _, source = fold_indices(world, a)
        arm = global_mean_effect(world, source)
        write_fold(out_dir, world, "global_mean_effect", a, score(arm, a))


def run_trained_arm(
    world: World,
    inputs: Any,
    config: Mapping[str, Any],
    out_dir: Path,
    arm_name: str,
    anchor: str,
    state_factory: Callable[[], nn.Module],
) -> None:
    comparison = config["comparison"]
    held_out, source = fold_indices(world, anchor)
    base = array(
        json.loads(fold_path(out_dir, "no_change", anchor).read_text())["loss"]
    )
    pools = [world.by_anchor[b] for b in world.anchors if b != anchor]
    # Every job starts from the same fixed seed, so its result does not depend on
    # which jobs ran before it, in this process or another.
    torch.manual_seed(SEED)
    if arm_name == "state_joint":
        arm = state_joint(world, state_factory(), inputs, config)
    else:
        cells = world.basal if arm_name == "mlp_hvg" else world.tx1
        arm = mlp(world, cells, comparison["hidden"], comparison["learning_rate"])
    curve = train(arm, world, pools, held_out, base, comparison)
    result = score_arm(
        arm,
        world,
        held_out,
        source,
        position=world.anchors.index(anchor),
        shuffles=int(comparison["shuffles"]),
        batch_size=int(comparison["batch_size"]),
    )
    write_fold(out_dir, world, arm_name, anchor, result, [_number(r) for r in curve])


def run_untrained(
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> None:
    """Write every untrained arm's fold files and ``sanity.json``.

    Existing fold files are kept. ``state_factory`` returns a fresh released
    STATE model; by default it loads ``paths.state_checkpoint``.
    """
    out_dir = Path(out_dir)
    if _untrained_done(out_dir, _anchors(config, inputs)):
        return
    inputs = load_inputs(config) if inputs is None else inputs
    world = build_world(
        inputs, torch.device(device), int(config["comparison"]["batch_size"])
    )
    run_untrained_arms(
        world, config, out_dir, state_factory or _released_state_factory(config)
    )


def pending_trained_jobs(
    config: Mapping[str, Any], out_dir: Path, *, inputs: Any = None
) -> list[tuple[str, str]]:
    """``(arm, anchor)`` of every trained fold without a fold file, anchors outer."""
    out_dir = Path(out_dir)
    return [
        (arm, a)
        for a in _anchors(config, inputs)
        for arm in TRAINED_ARMS
        if not fold_path(out_dir, arm, a).is_file()
    ]


def run_trained_job(
    config: Mapping[str, Any],
    out_dir: Path,
    arm: str,
    anchor: str,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> None:
    """Train ``arm`` on the other anchors, score it on ``anchor``, write its fold file.

    Does nothing when the fold file exists. Needs the no-change fold file of
    ``anchor`` (written by :func:`run_untrained`) for the held-out curve.
    """
    out_dir = Path(out_dir)
    anchors = _anchors(config, inputs)
    if arm not in TRAINED_ARMS:
        raise ValueError(f"{arm!r} is not a trained arm; choose from {TRAINED_ARMS}")
    if anchor not in anchors:
        raise ValueError(f"{anchor!r} is not a response anchor; choose from {anchors}")
    if fold_path(out_dir, arm, anchor).is_file():
        return
    base = fold_path(out_dir, "no_change", anchor)
    if not base.is_file():
        raise FileNotFoundError(f"{base} is missing: run the untrained arms first")
    inputs = load_inputs(config) if inputs is None else inputs
    world = build_world(
        inputs, torch.device(device), int(config["comparison"]["batch_size"])
    )
    run_trained_arm(
        world,
        inputs,
        config,
        out_dir,
        arm,
        anchor,
        state_factory or _released_state_factory(config),
    )


def bootstrap(
    folds: Mapping[str, Mapping[str, Mapping[str, Any]]],
    anchors: Sequence[str],
    variants: Mapping[str, Sequence[str]],
    resamples: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Pooled ratio per resample, variant and arm.

    The bootstrap unit is the perturbed gene: each resample draws gene
    identities with replacement once, from every held-out gene of every fold,
    and weights each fold's conditions by how often their gene was drawn. A
    gene perturbed in several anchors therefore moves all its folds together,
    and every arm sees the same draw.
    """
    rng = np.random.default_rng(SEED)
    losses = {arm: {a: array(folds[arm][a]["loss"]) for a in anchors} for arm in ARMS}
    genes = {a: folds["no_change"][a]["genes"] for a in anchors}
    universe = sorted({gene for a in anchors for gene in genes[a]})
    index = {gene: i for i, gene in enumerate(universe)}
    positions = {a: np.array([index[g] for g in genes[a]], dtype=int) for a in anchors}
    samples = {v: {arm: np.empty(resamples) for arm in ARMS} for v in variants}
    for b in range(resamples):
        counts = np.bincount(
            rng.integers(0, len(universe), len(universe)), minlength=len(universe)
        )
        weights = {a: counts[positions[a]].astype(float) for a in anchors}
        for arm in ARMS:
            ratio = {
                a: _weighted_ratio(losses[arm][a], losses["no_change"][a], weights[a])
                for a in anchors
            }
            for variant, members in variants.items():
                samples[variant][arm][b] = np.mean([ratio[a] for a in members])
    return samples


def _weighted_ratio(
    loss: np.ndarray, no_change: np.ndarray, weights: np.ndarray
) -> float:
    """``loss_ratio`` with each condition counted ``weights`` times."""
    covered = np.isfinite(loss) & (weights > 0)
    if not covered.any():
        return math.nan
    w = weights[covered]
    return float((w * loss[covered]).sum() / (w * no_change[covered]).sum())


def _interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(x) for x in np.percentile(finite, [2.5, 97.5])]


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarise(
    config: Mapping[str, Any], out_dir: Path, *, inputs: Any = None
) -> pd.DataFrame:
    """Write ``comparison.csv``, ``curves.csv`` and, last, ``verdicts.json``.

    Reads every arm x fold file; raises naming the fold files that are missing.
    """
    out_dir = Path(out_dir)
    anchors = _anchors(config, inputs)
    comparison = config["comparison"]
    missing = [
        str(fold_path(out_dir, arm, a))
        for arm in ARMS
        for a in anchors
        if not fold_path(out_dir, arm, a).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"response comparison fold files missing: {missing}")
    folds = {
        arm: {a: json.loads(fold_path(out_dir, arm, a).read_text()) for a in anchors}
        for arm in ARMS
    }
    rows, curves = [], []
    for arm in ARMS:
        for a in anchors:
            fold = folds[arm][a]
            base = folds["no_change"][a]
            loss, shuffled = array(fold["loss"]), array(fold["shuffled_loss"])
            scored = np.isfinite(loss) & np.isfinite(shuffled)
            identity = math.nan
            if scored.any():
                own = loss[scored].mean()
                identity = (shuffled[scored].mean() - own) / own
            rows.append(
                {
                    "arm": arm,
                    "fold": a,
                    "held_out_ratio": loss_ratio(loss, array(base["loss"])),
                    "identity_share": float(identity),
                    "source_ratio": loss_ratio(
                        array(fold["source_loss"]), array(base["source_loss"])
                    ),
                    "n_conditions": len(loss),
                    "n_covered": int(np.isfinite(loss).sum()),
                }
            )
            for epoch, ratio in enumerate(fold["curve"] or ()):
                curves.append(
                    {"arm": arm, "fold": a, "epoch": epoch, "held_out_ratio": ratio}
                )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    pd.DataFrame(curves, columns=["arm", "fold", "epoch", "held_out_ratio"]).to_csv(
        out_dir / "curves.csv", index=False
    )

    variants = {
        "all_folds": list(anchors),
        "without_hct116": [a for a in anchors if a != HCT116],
    }
    samples = bootstrap(folds, anchors, variants, int(comparison["bootstrap"]))
    by_fold = table.set_index(["arm", "fold"])["held_out_ratio"]
    pooled, verdicts = {}, {}
    for variant, members in variants.items():
        point = {
            arm: float(np.mean([by_fold[(arm, a)] for a in members])) for arm in ARMS
        }
        pooled[variant] = {
            arm: {
                "ratio": _number(point[arm]),
                "interval": _interval(samples[variant][arm]),
            }
            for arm in ARMS
        }
        verdicts[variant] = {}
        for name, arm, reference in VERDICTS:
            low, high = _interval(samples[variant][arm] - samples[variant][reference])
            separated = low is not None and (low > 0 or high < 0)
            verdicts[variant][name] = {
                "difference": _number(point[arm] - point[reference]),
                "interval": [low, high],
                "label": "separated" if separated else "overlap",
            }
    write_json(
        out_dir / "verdicts.json",
        {
            "statistic": "held-out loss ratio to no-change, mean of per-fold ratios",
            "difference": "first arm minus second arm; below 0 favours the first",
            "folds": variants,
            "bootstrap": int(comparison["bootstrap"]),
            "seed": SEED,
            "pooled": pooled,
            "verdicts": verdicts,
        },
    )
    return table


def run_comparison(
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    device: str,
    inputs: Any = None,
    state_factory: Callable[[], nn.Module] | None = None,
) -> pd.DataFrame:
    """Run (or resume) the whole comparison in this process; the arm x fold table,
    read back without recomputing when ``verdicts.json`` exists."""
    out_dir = Path(out_dir)
    if (out_dir / "verdicts.json").is_file():
        return pd.read_csv(out_dir / "comparison.csv")
    inputs = load_inputs(config) if inputs is None else inputs
    state_factory = state_factory or _released_state_factory(config)
    jobs = pending_trained_jobs(config, out_dir, inputs=inputs)
    if jobs or not _untrained_done(out_dir, _anchors(config, inputs)):
        # One world for every job: the observed cells are read from the cache once.
        world = build_world(
            inputs, torch.device(device), int(config["comparison"]["batch_size"])
        )
        run_untrained_arms(world, config, out_dir, state_factory)
        for arm, anchor in jobs:
            run_trained_arm(world, inputs, config, out_dir, arm, anchor, state_factory)
    return summarise(config, out_dir, inputs=inputs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--untrained",
        action="store_true",
        help="only the untrained arms and sanity.json",
    )
    mode.add_argument(
        "--job",
        nargs=2,
        metavar=("ARM", "ANCHOR"),
        help="only one trained arm on one held-out anchor",
    )
    mode.add_argument(
        "--summarise",
        action="store_true",
        help="only the tables and verdicts from existing fold files",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.untrained:
        run_untrained(config, args.out_dir, device=args.device)
    elif args.job:
        run_trained_job(config, args.out_dir, *args.job, device=args.device)
    else:
        if args.summarise:
            table = summarise(config, args.out_dir)
        else:
            table = run_comparison(config, args.out_dir, device=args.device)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
