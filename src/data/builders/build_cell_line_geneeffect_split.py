"""Build the final cell-line split (docs/data/cell-line-geneeffect-split.md), its JSON
and ``.csv`` table, from the 226 split and the two extra-line files, which stay its
provenance and enter no fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.extra_lines import (
    ExtraLines,
    ExtraSingleCellLines,
    load_extra_lines,
    load_extra_single_cell_lines,
)
from src.data.splits import SPLIT_SCHEMA_VERSION, FixedSplit, load_geneeffect_split

POLICY = (
    "The final split. Validation and test are the 226's 27 and 28 lines. The "
    "single-cell training lines are the 226's 171 training lines (PC9 and HeLa "
    "without 26Q1 GeneEffect) and the built single-cell extras; the bulk-only lines "
    "join the training side through bulk RNA. No line sharing a PatientID with a "
    "validation or test line is read."
)
TABLE_COLUMNS = ["split", "model_id", "patient_id", "lineage", "expression_source"]


def build_split(
    split: FixedSplit,
    table: pd.DataFrame,
    single_cell: ExtraSingleCellLines,
    bulk: ExtraLines,
) -> tuple[dict, pd.DataFrame]:
    """The final split's payload (without ``inputs``) and table."""
    if single_cell.pending:
        raise ValueError(
            f"single-cell extras not built yet: {sorted(single_cell.pending)}"
        )
    cells = sorted(single_cell.labelled)
    for name, group in (("unlabelled", bulk.unlabelled), ("excluded", bulk.excluded)):
        shared = sorted(set(cells) & set(group))
        if shared:
            raise ValueError(
                f"built single-cell extras the bulk file calls {name}: {shared}"
            )
    if set(table["model_id"]) != set(split.all_model_ids):
        raise ValueError("the 226 table and the 226 split list different lines")
    patients = {
        **table.set_index("model_id")["patient_id"].to_dict(),
        **{m: single_cell.labelled[m]["patient_id"] for m in cells},
    }
    held = {patients[m] for m in (*split.val, *split.test)}
    leaking = sorted(m for m in (*split.train, *cells) if patients[m] in held)
    if leaking:
        raise ValueError(
            f"training lines sharing a patient with a held-out line: {leaking}"
        )
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "policy": POLICY,
        "train": [*split.train, *cells],
        "val": list(split.val),
        "test": list(split.test),
        "unlabeled_train": list(split.unlabeled_train),
        "bulk_labelled": sorted(set(bulk.labelled) - set(cells)),
        "bulk_unlabelled": sorted(bulk.unlabelled),
        "excluded": dict(sorted({**bulk.excluded, **single_cell.excluded}.items())),
    }
    extras = pd.DataFrame(
        {
            "split": "train",
            "model_id": cells,
            "patient_id": [single_cell.labelled[m]["patient_id"] for m in cells],
            "lineage": [single_cell.labelled[m]["lineage"] for m in cells],
            "expression_source": [single_cell.labelled[m]["source"] for m in cells],
        }
    )
    rows = pd.concat([table[TABLE_COLUMNS], extras], ignore_index=True)
    order = {m: i for i, m in enumerate([*payload["train"], *split.val, *split.test])}
    rows = rows.sort_values("model_id", key=lambda ids: ids.map(order))
    return payload, rows.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-226", type=Path, required=True)
    parser.add_argument("--single-cell", type=Path, required=True)
    parser.add_argument("--bulk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    split = load_geneeffect_split(args.split_226)
    payload, table = build_split(
        split,
        pd.read_csv(args.split_226.with_suffix(".csv"), dtype=str),
        load_extra_single_cell_lines(args.single_cell, split),
        load_extra_lines(args.bulk, split),
    )
    inputs = {
        "split_226": args.split_226,
        "split_226_table": args.split_226.with_suffix(".csv"),
        "single_cell": args.single_cell,
        "bulk": args.bulk,
    }
    payload["inputs"] = {name: {"path": str(path)} for name, path in inputs.items()}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    print(
        f"{args.out}: train {len(payload['train'])} "
        f"({len(payload['unlabeled_train'])} unlabelled), val {len(payload['val'])}, "
        f"test {len(payload['test'])}; bulk-only {len(payload['bulk_labelled'])} "
        f"labelled, {len(payload['bulk_unlabelled'])} unlabelled; "
        f"{len(payload['excluded'])} excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
