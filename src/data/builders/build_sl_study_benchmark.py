"""Build the multi-study SL benchmark's labels (docs/data/sl-study-inventory.md): run
the reader (the ``reader`` column) of every ready row of
configs/benchmarks/sl_label_rules.csv, map each list's line to its reviewed ModelID and
each gene to its 26Q1 GeneEffect symbol, and write one row per (list, gene pair).
Constructs of one pair in one list are merged: SL when any is, negative only when every
one is called negative. An anchor that is not a gene (``drug:<name>``) keeps its name.
Sides come later, from the SL split."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.data.builders.build_sl_screen_benchmark import geneeffect_namer, symbol_mapper
from src.data.builders.sl_study_readers import COLUMNS, READERS
from src.data.io import write_json, write_parquet
from src.data.labels import parse_gene_symbol

LINE = re.compile(r"^(?P<line>.*?)\s*\[(?P<model_id>ACH-\d{6})\]$")


def line_map(rules: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(study_id, line as the file spells it) -> ModelID, from the rules' ``lines``."""
    out = {}
    for study, field in zip(rules["study_id"], rules["lines"]):
        for entry in filter(None, (part.strip() for part in str(field).split(";"))):
            match = LINE.match(entry)
            if match:
                out[(study, match["line"])] = match["model_id"]
    return out


def merge_constructs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (list, gene_a, gene_b)."""
    keys = ["list_id", "gene_a", "gene_b"]
    grouped = rows.groupby(keys, sort=False)
    label = grouped["label"].agg(
        lambda x: 1 if (x == 1).any() else (0 if x.notna().all() else pd.NA)
    )
    first = grouped.first()
    merged = first.assign(
        screen_gene_a=grouped["screen_gene_a"].agg(lambda x: "|".join(sorted(set(x)))),
        screen_gene_b=grouped["screen_gene_b"].agg(lambda x: "|".join(sorted(set(x)))),
        score=grouped["score"].mean(),
        fdr=grouped["fdr"].min(),
        label=label.astype("Int8"),
        unlabelled_reason=grouped["unlabelled_reason"].agg(
            lambda x: "; ".join(sorted(set(filter(None, x))))
        ),
        n_constructs=grouped.size(),
    )
    merged.loc[merged["label"].notna(), "unlabelled_reason"] = ""
    return merged.reset_index()


def build(
    rules_path: Path, raw: Path, gene_effect: Path, hgnc_path: Path, out: Path
) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rules = pd.read_csv(rules_path, dtype=str, keep_default_na=False)
    ready = rules[rules["status"] == "ready"]
    missing = sorted(set(ready["reader"]) - set(READERS))
    if missing:
        raise KeyError(f"ready rules without a reader: {missing}")
    geneeffect = {
        parse_gene_symbol(column)
        for column in pd.read_csv(gene_effect, nrows=0).columns[1:]
    }
    hgnc = pd.read_csv(
        hgnc_path,
        sep="\t",
        usecols=["symbol", "prev_symbol", "alias_symbol"],
        dtype=str,
    )
    name = geneeffect_namer(symbol_mapper(hgnc), geneeffect)
    lines = line_map(ready)
    frames, studies = [], {}
    for reader, design in zip(ready["reader"], ready["design"]):
        frame = READERS[reader](raw)
        if tuple(frame.columns) != COLUMNS:
            raise ValueError(f"{reader} returned columns {list(frame.columns)}")
        frames.append(frame.assign(design=design))
    rows = pd.concat(frames, ignore_index=True)
    rows["model_id"] = [
        lines.get(key, "") for key in zip(rows["study_id"], rows["line"])
    ]
    rows["screen_gene_a"], rows["screen_gene_b"] = rows["gene_a"], rows["gene_b"]
    rows["gene_a"] = rows["gene_a"].map(name)
    rows["gene_b"] = rows["gene_b"].map(name)
    pair = rows["design"] == "combinatorial"
    swap = pair & (rows["gene_a"] > rows["gene_b"])
    rows.loc[swap, ["gene_a", "gene_b"]] = rows.loc[
        swap, ["gene_b", "gene_a"]
    ].to_numpy()
    rows.loc[swap, ["screen_gene_a", "screen_gene_b"]] = rows.loc[
        swap, ["screen_gene_b", "screen_gene_a"]
    ].to_numpy()
    rows = rows[rows["gene_a"] != rows["gene_b"]]
    rows = merge_constructs(rows)
    lists = (
        rows.groupby(["study_id", "design", "list_id", "line", "model_id"])["label"]
        .agg(
            n_positive=lambda x: int((x == 1).sum()),
            n_negative=lambda x: int((x == 0).sum()),
            n_unlabelled=lambda x: int(x.isna().sum()),
        )
        .reset_index()
    )
    for study, group in lists.groupby("study_id"):
        studies[study] = {
            "lists": len(group),
            "lists_without_model_id": int((group["model_id"] == "").sum()),
            "positive": int(group["n_positive"].sum()),
            "negative": int(group["n_negative"].sum()),
            "unlabelled": int(group["n_unlabelled"].sum()),
        }
    out.mkdir(parents=True)
    write_parquet(out / "labels.parquet", rows)
    lists.to_csv(out / "lists.csv", index=False)
    manifest = {
        "schema_version": "sl-study-benchmark-v1",
        "inputs": {"rules": str(rules_path), "gene_effect": str(gene_effect),
                   "hgnc": str(hgnc_path)},
        "rows": len(rows),
        "renamed_symbols": int(
            (rows["gene_a"] != rows["screen_gene_a"]).sum()
            + (rows["gene_b"] != rows["screen_gene_b"]).sum()
        ),
        "studies": studies,
        "rules_not_ready": rules[rules["status"] != "ready"]
        .groupby("study_id")["status"]
        .agg(lambda x: "; ".join(sorted(set(x))))
        .to_dict(),
    }  # fmt: skip
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--hgnc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.rules, args.raw, args.gene_effect, args.hgnc, args.out)
    print(f"{args.out}: {manifest['rows']} rows, {len(manifest['studies'])} studies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
