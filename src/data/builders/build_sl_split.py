"""Build the SL split (multi-study SL benchmark design, decision 7) from the final
split's lines, for stage two only. The reviewed placements
(configs/benchmarks/sl_split_placements.csv) name every validation and test line; every
other line of the final split is a training line, whatever its stage-one side. A placed
line's screened patient siblings must be placed on its side; its other patient siblings
are excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.depmap import read_models
from src.data.splits import SPLIT_SCHEMA_VERSION, FixedSplit, load_geneeffect_split

POLICY = (
    "The SL split, for stage two only: {val} validation and {test} test lines "
    "placed by hand so that training, validation and test hold about 80%, 10% and "
    "10% of the kept SL pairs, across lineages; every other line of the final split "
    "is a training line. Lines of one patient are on one side; a held-out line's "
    "unscreened patient siblings are excluded. Each line's group (with_geneeffect: "
    "stage one's production fit reads its GeneEffect; without_geneeffect: it does "
    "not) stratifies the results."
)
GROUPS = ("with_geneeffect", "without_geneeffect")
SIDES = ("val", "test")


def scored_lines(lists: pd.DataFrame, lines: set[str]) -> list[str]:
    """Single-cell lines with at least one list holding both classes."""
    both = (lists["n_positive"] > 0) & (lists["n_negative"] > 0)
    return sorted(set(lists.loc[both, "model_id"]) & lines)


def read_placements(path: Path) -> dict[str, tuple[str, str]]:
    """The reviewed placements: line_id -> (side, reason), side val or test."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["line_id", "side", "reason"]:
        raise ValueError(f"{path}: columns must be line_id, side, reason")
    if set(frame["side"]) - set(SIDES) or (frame["reason"] == "").any():
        raise ValueError(f"{path}: every placement is val or test with a reason")
    if frame["line_id"].duplicated().any():
        raise ValueError(f"{path}: a line is placed twice")
    return {
        m: (s, r) for m, s, r in zip(frame["line_id"], frame["side"], frame["reason"])
    }


def build_sl_split(
    final: dict,
    split: FixedSplit,
    table: pd.DataFrame,
    lists: pd.DataFrame,
    patients: pd.Series,
    *,
    placed: dict[str, tuple[str, str]],
) -> tuple[dict, pd.DataFrame]:
    """The SL split's payload (without ``inputs``) and table. ``final`` is the final
    split's payload, ``table`` its table, ``patients`` maps every line it lists
    (single-cell and bulk) to its PatientID, and ``placed`` the held-out lines to
    their side and reason."""
    lines = [*split.train, *split.val, *split.test]
    unknown = sorted(set(placed) - set(lines))
    if unknown:
        raise ValueError(f"placed lines not in the final split: {unknown}")
    scored = set(scored_lines(lists, set(table["model_id"])))
    excluded = dict(final["excluded"])
    for model_id, (side, _) in placed.items():
        for other in patients.index[patients == patients[model_id]]:
            if other == model_id or other in excluded:
                continue
            if other in scored:
                if placed.get(other, ("train",))[0] != side:
                    raise ValueError(
                        f"{other} shares PatientID {patients[model_id]} with "
                        f"{model_id} ({side}) and is not placed on {side}"
                    )
            elif other not in placed:
                excluded[other] = (
                    f"shares PatientID {patients[model_id]} with held-out SL line "
                    f"{model_id}"
                )
    held = {s: [m for m in lines if placed.get(m, ("train",))[0] == s] for s in SIDES}
    gone = set(placed) | set(excluded)
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "policy": POLICY.format(val=len(held["val"]), test=len(held["test"])),
        "train": [m for m in lines if m not in gone],
        "val": held["val"],
        "test": held["test"],
        "unlabeled_train": [m for m in split.unlabeled_train if m not in gone],
        "bulk_labelled": [m for m in final["bulk_labelled"] if m not in gone],
        "bulk_unlabelled": [m for m in final["bulk_unlabelled"] if m not in gone],
        "excluded": dict(sorted(excluded.items())),
        "placed": {m: r for m, (_, r) in sorted(placed.items())},
    }
    order = [*payload["train"], *payload["val"], *payload["test"]]
    rows = table.set_index("model_id").loc[order].reset_index()[table.columns]
    rows["split"] = (
        ["train"] * len(payload["train"])
        + ["val"] * len(payload["val"])
        + ["test"] * len(payload["test"])
    )
    without = set(split.unlabeled_train)
    rows["group"] = [GROUPS[m in without] for m in rows["model_id"]]
    return payload, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload, table = build_sl_split(
        json.loads(args.final.read_text()),
        load_geneeffect_split(args.final),
        pd.read_csv(args.final.with_suffix(".csv"), dtype=str),
        pd.read_csv(args.lists),
        read_models(args.model)["patient_id"],
        placed=read_placements(args.placements),
    )
    payload["inputs"] = {
        "final_split": {"path": str(args.final)},
        "lists": {"path": str(args.lists)},
        "placements": {"path": str(args.placements)},
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
