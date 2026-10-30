"""Build the SL split (multi-study SL benchmark design, decision 7) from the final
split and the SL benchmark's lists: the screened single-cell lines
already held out keep their side, and further screened labelled training lines move to
validation and test, drawn across lineages, until each side holds a fixed fraction
of the screened lines. Lines sharing a patient with a moved line are excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import SPLIT_SCHEMA_VERSION, FixedSplit, load_geneeffect_split

POLICY = (
    "The SL split: the final split with screened single-cell training lines moved to "
    "validation and test until each side holds {fraction:.0%} of the {scored} screened "
    "lines with a scored list (seed {seed}, drawn across lineages); the final split's "
    "held-out lines keep their side. Lines sharing a PatientID with a moved line are "
    "excluded. Stage one follows it."
)


def scored_lines(lists: pd.DataFrame, lines: set[str]) -> list[str]:
    """Single-cell lines with at least one list holding both classes."""
    both = (lists["n_positive"] > 0) & (lists["n_negative"] > 0)
    return sorted(set(lists.loc[both, "model_id"]) & lines)


def draw_order(candidates: pd.DataFrame, seed: int) -> list[str]:
    """Candidates interleaved across lineages: a random order within each lineage,
    then sorted by rank over lineage size, so every prefix spreads over lineages."""
    rng = np.random.default_rng(seed)
    frame = candidates.sort_values("model_id").assign(key=rng.random(len(candidates)))
    frame = frame.sort_values(["lineage", "key"])
    rank = frame.groupby("lineage").cumcount()
    size = frame.groupby("lineage")["model_id"].transform("size")
    frame = frame.assign(position=(rank + 0.5) / size)
    return frame.sort_values(["position", "key"])["model_id"].tolist()


def build_sl_split(
    final: dict,
    split: FixedSplit,
    table: pd.DataFrame,
    lists: pd.DataFrame,
    patients: pd.Series,
    *,
    fraction: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """The SL split's payload (without ``inputs``) and table. ``final`` is the final
    split's payload, ``table`` its table, and ``patients`` maps every ModelID it lists
    (single-cell and bulk) to its PatientID."""
    scored = scored_lines(lists, set(table["model_id"]))
    target = round(fraction * len(scored))
    count = {"val": len(set(scored) & set(split.val)),
             "test": len(set(scored) & set(split.test))}  # fmt: skip
    if max(count.values()) > target:
        raise ValueError(f"more held-out SL lines than the target {target}: {count}")
    held_patients = {patients[m] for m in (*split.val, *split.test)}
    candidates = table[
        table["model_id"].isin(scored)
        & table["model_id"].isin(split.supervised_train)
        & ~table["patient_id"].isin(held_patients)
    ]
    excluded = dict(final["excluded"])
    moved = {"val": [], "test": []}
    for model_id in draw_order(candidates[["model_id", "lineage"]], seed):
        if min(count.values()) >= target:
            break
        if model_id in excluded:
            continue
        side = "val" if count["val"] <= count["test"] else "test"
        moved[side].append(model_id)
        count[side] += 1
        for other in patients.index[patients == patients[model_id]]:
            if other != model_id and other not in excluded:
                excluded[other] = (
                    f"shares PatientID {patients[model_id]} with held-out SL line "
                    f"{model_id}"
                )
    gone = set(moved["val"]) | set(moved["test"]) | set(excluded)
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "policy": POLICY.format(fraction=fraction, scored=len(scored), seed=seed),
        "train": [m for m in split.train if m not in gone],
        "val": [*split.val, *moved["val"]],
        "test": [*split.test, *moved["test"]],
        "unlabeled_train": [m for m in split.unlabeled_train if m not in gone],
        "bulk_labelled": [m for m in final["bulk_labelled"] if m not in gone],
        "bulk_unlabelled": [m for m in final["bulk_unlabelled"] if m not in gone],
        "excluded": dict(sorted(excluded.items())),
    }
    order = [*payload["train"], *payload["val"], *payload["test"]]
    rows = table.set_index("model_id").loc[order].reset_index()[table.columns]
    rows["split"] = (
        ["train"] * len(payload["train"])
        + ["val"] * len(payload["val"])
        + ["test"] * len(payload["test"])
    )
    return payload, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    model = pd.read_csv(args.model, dtype=str)
    payload, table = build_sl_split(
        json.loads(args.final.read_text()),
        load_geneeffect_split(args.final),
        pd.read_csv(args.final.with_suffix(".csv"), dtype=str),
        pd.read_csv(args.lists),
        model.set_index("ModelID")["PatientID"],
        fraction=args.fraction,
        seed=args.seed,
    )
    payload["inputs"] = {
        "final_split": {"path": str(args.final)},
        "lists": {"path": str(args.lists)},
        "draw": {"fraction": args.fraction, "seed": args.seed},
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(
        f"{args.out}: train {len(payload['train'])}, val {len(payload['val'])}, "
        f"test {len(payload['test'])}; {len(payload['excluded'])} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
