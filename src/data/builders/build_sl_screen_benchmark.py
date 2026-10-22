"""Build the SL screen benchmark (docs/data/sl-screen-benchmark.md, blueprint §5): the
screens of Kebabci et al. 2026 (Ito 2021, Klingbeil 2024, Harle 2025), one row per
(screen, line, unordered gene pair) with the published scores, an SL label and the
line's split side, plus MiT4SL's A549 labels as a secondary external test."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Collection
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.extra_lines import (
    ExtraSingleCellLines,
    load_extra_lines,
    load_extra_single_cell_lines,
)
from src.data.labels import parse_gene_symbol
from src.data.splits import FixedSplit, load_geneeffect_split

#: Published tables under ``--raw``, as fetched (docs/data/sl-screen-benchmark.md).
RAW_FILES = {
    "ito": "ito_2021/41588_2021_967_MOESM2_ESM.xlsx",
    "klingbeil_lfc": "klingbeil_2024/cd-23-1529_supplementary_table_s2_suppst2.xlsx",
    "klingbeil_gemini": "klingbeil_2024/cd-23-1529_supplementary_table_s4_suppst4.xlsx",
    "klingbeil_fdr": "klingbeil_2024/cd-23-1529_supplementary_table_s6_suppst6.xlsx",
    "harle": "harle_2025/13059_2025_3737_MOESM1_ESM.xlsx",
    "mit4sl_a549": "mit4sl/a549_bench.csv",
    "mit4sl_nodes": "mit4sl/Primenode.csv",
}
SCREENS = ("ito_2021", "klingbeil_2024", "harle_2025")
#: Kebabci et al. 2026, ``05_annotate_gemini.ipynb``: a (line, pair) is SL when its
#: GEMINI score is at or above the screen's 90th percentile and its LFC is below the
#: screen's mid-point between one-essential and other pairs.
GEMINI_QUANTILE = 0.9
LFC_THRESHOLD = {"ito_2021": -0.51, "klingbeil_2024": -1.03}
#: Kebabci et al. report 958 SL calls in 41,244 (line, pair) points of Ito's screen,
#: MeWo left out; the build reports its own count on the same lines beside it.
KEBABCI_ITO_REPORTED = {"sl": 958, "points": 41244, "left_out": "MEWO_SKIN"}
CONTROLS = frozenset({"AAVS1", "CUTTING_CONTROL", "NONCUTTING_CONTROL"})
ROW_COLUMNS = (
    "screen",
    "screen_line",
    "model_id",
    "gene_a",
    "gene_b",
    "screen_gene_a",
    "screen_gene_b",
    "both_in_geneeffect",
    "gemini",
    "lfc",
    "gi",
    "fdr",
    "n_constructs",
    "label",
    "unlabelled_reason",
    "split",
    "stratum",
    "line_reason",
)


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def kebabci_labels(
    gemini: pd.Series, lfc: pd.Series, lfc_threshold: float
) -> tuple[pd.Series, float]:
    """Kebabci's rule over one screen's scored rows: label 1 when GEMINI is at or above
    the screen's 90th percentile and LFC is below ``lfc_threshold``. A row without an
    LFC cannot be called and stays unlabelled (NA)."""
    if gemini.isna().any():
        raise ValueError("kebabci_labels takes scored rows only")
    cut = float(np.quantile(gemini.to_numpy(), GEMINI_QUANTILE))
    called = (gemini >= cut) & (lfc < lfc_threshold)
    label = called.astype("Int8").where(lfc.notna(), pd.NA)
    return label, cut


def read_ito(path: Path) -> pd.DataFrame:
    """Ito et al. 2021: GEMINI sensitive-lethality score (Supplementary Table 8), its
    FDR (Table 10) and raw LFC (Table 4), long over (line, pair)."""

    def wide(sheet: str) -> pd.DataFrame:
        frame = pd.read_excel(path, sheet_name=sheet, header=3, index_col=0)
        frame = frame.loc[frame.index.notna()]
        frame.index = frame.index.astype(str)
        return frame.apply(pd.to_numeric, errors="coerce")

    gemini, fdr = wide("Supplementary table 8"), wide("Supplementary table 10")
    if list(gemini.columns) != list(fdr.columns) or not gemini.index.equals(fdr.index):
        raise ValueError("Ito Tables 8 and 10 disagree on lines or pairs")
    lfc = pd.read_excel(path, sheet_name="Supplementary table 4", header=3)
    genes = lfc[["gene 1", "gene 2"]].astype(str)
    lfc = lfc.loc[~genes.isin(CONTROLS).any(axis=1)]
    lfc.index = [
        ";".join(_sorted_pair(a, b)) for a, b in zip(lfc["gene 1"], lfc["gene 2"])
    ]
    lfc = lfc[list(gemini.columns)].apply(pd.to_numeric, errors="coerce")
    if lfc.index.duplicated().any():
        raise ValueError("Ito Table 4 repeats a gene pair")
    keys = [";".join(_sorted_pair(*pair.split(";"))) for pair in gemini.index]
    rows = []
    for line in gemini.columns:
        frame = pd.DataFrame(
            {
                "screen_line": line,
                "pair": gemini.index,
                "gemini": gemini[line].to_numpy(),
                "fdr": fdr[line].to_numpy(),
                "lfc": lfc[line].reindex(keys).to_numpy(),
            }
        )
        rows.append(frame.loc[frame["gemini"].notna()])
    long = pd.concat(rows, ignore_index=True)
    long[["screen_gene_a", "screen_gene_b"]] = long["pair"].str.split(";", expand=True)
    long["n_constructs"] = 1
    long["label"], cut = kebabci_labels(
        long["gemini"], long["lfc"], LFC_THRESHOLD["ito_2021"]
    )
    long.attrs["gemini_cut"] = cut
    return long.drop(columns="pair")


def _construct_key(gene_a: str, domain_a: str, gene_b: str, domain_b: str) -> str:
    left, right = sorted([(gene_a, domain_a), (gene_b, domain_b)])
    return ";".join(f"{gene}:{domain}" for gene, domain in (left, right))


def read_klingbeil(gemini_path: Path, lfc_path: Path, fdr_path: Path) -> pd.DataFrame:
    """Klingbeil et al. 2024: GEMINI score (Table S4), FDR (S6) and LFC (S2) per domain
    construct; Kebabci's rule is applied per construct, then a gene pair is SL in a line
    when any of its constructs is."""

    def wide(path: Path) -> pd.DataFrame:
        frame = pd.read_excel(path, index_col=0)
        frame = frame.loc[frame.index.notna()]
        frame = frame.loc[~frame.index.duplicated()]
        frame.index = [_construct_key(*_split_construct(c)) for c in frame.index]
        return frame.apply(pd.to_numeric, errors="coerce")

    gemini = wide(gemini_path)
    fdr = wide(fdr_path).reindex(index=gemini.index, columns=gemini.columns)
    lfc = pd.read_excel(lfc_path)
    lfc = lfc.loc[~lfc[["GENE 1", "GENE 2"]].isin(CONTROLS).any(axis=1)]
    lfc.index = [
        _construct_key(str(a), str(da), str(b), str(db))
        for a, da, b, db in zip(
            lfc["GENE 1"], lfc["domain_1"], lfc["GENE 2"], lfc["domain_2"]
        )
    ]
    lfc = lfc.loc[~lfc.index.duplicated()]
    lfc = lfc.reindex(index=gemini.index, columns=gemini.columns)
    lfc = lfc.apply(pd.to_numeric, errors="coerce")
    rows = []
    for line in gemini.columns:
        frame = pd.DataFrame(
            {
                "screen_line": line,
                "construct": gemini.index,
                "gemini": gemini[line].to_numpy(),
                "fdr": fdr[line].to_numpy(),
                "lfc": lfc[line].to_numpy(),
            }
        )
        rows.append(frame.loc[frame["gemini"].notna()])
    long = pd.concat(rows, ignore_index=True)
    long["label"], cut = kebabci_labels(
        long["gemini"], long["lfc"], LFC_THRESHOLD["klingbeil_2024"]
    )
    genes = long["construct"].str.split(";", expand=True)
    long["screen_gene_a"] = genes[0].str.split(":").str[0]
    long["screen_gene_b"] = genes[1].str.split(":").str[0]
    long["n_constructs"] = 1
    long.attrs["gemini_cut"] = cut
    long.attrs["constructs_without_lfc"] = int(long["lfc"].isna().sum())
    return long.drop(columns="construct")


def _split_construct(construct: object) -> tuple[str, str, str, str]:
    left, right = str(construct).split(";")
    gene_a, domain_a = left.split(":", 1)
    gene_b, domain_b = right.split(":", 1)
    return gene_a, domain_a, gene_b, domain_b


def aggregate_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (line, gene_a, gene_b) after symbol mapping. Klingbeil screens
    several domain constructs per pair, and some genes under two symbols; a pair is SL
    in a line when any of its constructs is, negative only when every construct is
    called negative, and keeps the strongest GEMINI, LFC, GI and FDR."""
    grouped = rows.groupby(["screen_line", "gene_a", "gene_b"], sort=False)
    label = grouped["label"].agg(
        lambda x: 1 if (x == 1).any() else (0 if x.notna().all() else pd.NA)
    )
    pairs = pd.DataFrame(
        {
            "screen_gene_a": grouped["screen_gene_a"].agg(
                lambda x: "|".join(sorted(set(x)))
            ),
            "screen_gene_b": grouped["screen_gene_b"].agg(
                lambda x: "|".join(sorted(set(x)))
            ),
            "gemini": grouped["gemini"].max(),
            "lfc": grouped["lfc"].min(),
            "gi": grouped["gi"].min(),
            "fdr": grouped["fdr"].min(),
            "n_constructs": grouped["n_constructs"].sum(),
            "single_depleted": grouped["single_depleted"].any(),
            "label": label.astype("Int8"),
        }
    )
    return pairs.reset_index()


def read_harle(path: Path) -> pd.DataFrame:
    """Harle et al. 2025: the authors' per-line call (Table S5) with the mean
    normalised GI, FDR and single-depletion flags (Table S4). A non-hit with either gene
    depleted by both MAGeCK and BAGEL2 in that line is unlabelled: the call cannot make
    it a hit."""
    s4 = pd.read_excel(path, sheet_name="Table S4", header=4)
    calls = pd.read_excel(path, sheet_name="Table S5", header=3)
    calls = calls.set_index("sorted_gene_pair")
    lines = sorted(set(s4["cell_line_label"]))
    if sorted(calls.columns[: len(lines)]) != lines:
        raise ValueError("Harle Tables S4 and S5 disagree on lines")
    stacked = calls[lines].stack()
    stacked.index.names = ["sorted_gene_pair", "cell_line_label"]
    s4 = s4.join(stacked.rename("call"), on=["sorted_gene_pair", "cell_line_label"])
    if s4["call"].isna().any():
        raise ValueError("a Harle Table S4 row has no Table S5 call")
    depleted = (
        (s4["targetA__is_depleted_mageck"] == 1)
        & (s4["targetA__is_depleted_bagel"] == 1)
    ) | (
        (s4["targetB__is_depleted_mageck"] == 1)
        & (s4["targetB__is_depleted_bagel"] == 1)
    )
    label = s4["call"].astype(int).astype("Int8")
    genes = s4["sorted_gene_pair"].str.split("|", expand=True)
    return pd.DataFrame(
        {
            "screen_line": s4["cell_line_label"],
            "screen_gene_a": genes[0],
            "screen_gene_b": genes[1],
            "gi": s4["mean_norm_gi"],
            "fdr": s4["fdr"],
            "n_constructs": 1,
            "label": label.where(~(depleted & (label == 0)), pd.NA),
            "single_depleted": depleted.to_numpy(),
        }
    )


def read_mit4sl(bench_path: Path, nodes_path: Path) -> pd.DataFrame:
    """MiT4SL's A549 benchmark file: PrimeKG node indices and a binary label."""
    nodes = pd.read_csv(nodes_path, usecols=["node_index", "node_name"])
    names = dict(zip(nodes["node_index"].astype(int), nodes["node_name"].astype(str)))
    bench = pd.read_csv(bench_path)
    bench.columns = ["node_a", "node_b", "cell", "label"]
    return pd.DataFrame(
        {
            "screen_gene_a": bench["node_a"].astype(int).map(names),
            "screen_gene_b": bench["node_b"].astype(int).map(names),
            "label": bench["label"].astype(float).astype(int),
        }
    )


def symbol_mapper(hgnc: pd.DataFrame) -> Callable[[str], str]:
    """Map a published symbol to its current HGNC symbol: itself when approved, else the
    one approved symbol it is a previous symbol of, else of an alias; else unchanged."""
    approved = set(hgnc["symbol"].astype(str))
    previous: dict[str, set[str]] = {}
    alias: dict[str, set[str]] = {}
    for symbol, prev, ali in zip(
        hgnc["symbol"].astype(str),
        hgnc["prev_symbol"].fillna(""),
        hgnc["alias_symbol"].fillna(""),
    ):
        for old in filter(None, str(prev).split("|")):
            previous.setdefault(old, set()).add(symbol)
        for other in filter(None, str(ali).split("|")):
            alias.setdefault(other, set()).add(symbol)

    def current(symbol: str) -> str:
        if symbol in approved:
            return symbol
        for table in (previous, alias):
            hits = table.get(symbol, set())
            if len(hits) == 1:
                return next(iter(hits))
        return symbol

    return current


def geneeffect_namer(
    current: Callable[[str], str], geneeffect: Collection[str]
) -> Callable[[str], str]:
    """Name a published symbol as stage one does: the 26Q1 GeneEffect symbol of the same
    current HGNC gene when exactly one exists (DepMap keeps some symbols HGNC has since
    renamed, such as FAM83A), else the current HGNC symbol."""
    stage_one = set(geneeffect)
    by_current: dict[str, set[str]] = {}
    for symbol in stage_one:
        by_current.setdefault(current(symbol), set()).add(symbol)

    def name(symbol: str) -> str:
        if symbol in stage_one:
            return symbol
        hits = by_current.get(current(symbol), set())
        return next(iter(hits)) if len(hits) == 1 else current(symbol)

    return name


def line_sides(
    lines: pd.DataFrame,
    split: FixedSplit,
    extra_bulk_labelled: Collection[str],
    extra_single_cell: ExtraSingleCellLines,
) -> pd.DataFrame:
    """Each screen line's split side and input stratum.

    Validation and test lines are the 226's (read from basal single cells). Training
    lines are the 226's labelled training lines and the labelled single-cell extras
    (stratum ``single_cell``) or the labelled bulk extras (``bulk``); anything else is
    ``unused`` with a reason.
    """
    sc_labelled = set(extra_single_cell.labelled)
    sc_unlabelled = set(extra_single_cell.unlabelled)
    held = {**{m: "val" for m in split.val}, **{m: "test" for m in split.test}}
    rows = []
    for model_id in lines["model_id"].fillna(""):
        if not model_id:
            side = ("none", "unused", "no DepMap ModelID")
        elif model_id in held:
            side = (held[model_id], "single_cell", "")
        elif model_id in split.unlabeled_train:
            side = ("none", "unused", "226 member without 26Q1 GeneEffect")
        elif model_id in split.train:
            side = ("train", "single_cell", "")
        elif model_id in sc_labelled:
            side = ("train", "single_cell", "")
        elif model_id in extra_bulk_labelled:
            side = ("train", "bulk", "")
        elif model_id in sc_unlabelled:
            side = ("none", "unused", "no 26Q1 GeneEffect")
        else:
            side = ("none", "unused", "no basal single cells or labelled bulk line")
        rows.append(side)
    sides = pd.DataFrame(rows, columns=["split", "stratum", "line_reason"])
    return pd.concat([lines.reset_index(drop=True), sides], axis=1)


def finish_rows(
    frame: pd.DataFrame,
    screen: str,
    sides: pd.DataFrame,
    name: Callable[[str], str],
    geneeffect: Collection[str],
) -> pd.DataFrame:
    """Name genes as stage one does, drop pairs of one gene under two symbols, aggregate
    to one row per (line, pair) and attach line sides. The dropped row count is in
    ``attrs["same_gene_rows"]``."""
    mapped_a = frame["screen_gene_a"].astype(str).map(name)
    mapped_b = frame["screen_gene_b"].astype(str).map(name)
    swap = mapped_a > mapped_b
    rows = frame.copy()
    rows["gene_a"] = mapped_a.where(~swap, mapped_b)
    rows["gene_b"] = mapped_b.where(~swap, mapped_a)
    published_a, published_b = frame["screen_gene_a"], frame["screen_gene_b"]
    rows["screen_gene_a"] = published_a.where(~swap, published_b)
    rows["screen_gene_b"] = published_b.where(~swap, published_a)
    same_gene = rows["gene_a"] == rows["gene_b"]
    rows = rows.loc[~same_gene]
    for column in ("gemini", "lfc", "gi", "fdr"):
        if column not in rows:
            rows[column] = np.nan
    if "single_depleted" not in rows:
        rows["single_depleted"] = False
    out = aggregate_pairs(rows)
    out["screen"] = screen
    ge = set(geneeffect)
    out["both_in_geneeffect"] = out["gene_a"].isin(ge) & out["gene_b"].isin(ge)
    unlabelled = out["label"].isna()
    out["unlabelled_reason"] = ""
    out.loc[unlabelled & out["single_depleted"], "unlabelled_reason"] = (
        "single-depleted gene"
    )
    out.loc[unlabelled & ~out["single_depleted"], "unlabelled_reason"] = "no LFC"
    side = sides.loc[sides["screen"] == screen].set_index("screen_line")
    missing = sorted(set(out["screen_line"]) - set(side.index))
    if missing:
        raise ValueError(f"{screen} lines absent from the line map: {missing}")
    for column in ("model_id", "split", "stratum", "line_reason"):
        out[column] = out["screen_line"].map(side[column])
    out = out.loc[:, list(ROW_COLUMNS)]
    out.attrs["same_gene_rows"] = int(same_gene.sum())
    return out


def _summary(rows: pd.DataFrame) -> dict[str, object]:
    labelled = rows["label"].notna()
    by_side = {}
    for (split, stratum), group in rows.groupby(["split", "stratum"]):
        by_side[f"{split}/{stratum}"] = {
            "lines": sorted(group["model_id"].replace("", "-").unique()),
            "rows": int(len(group)),
            "labelled": int(group["label"].notna().sum()),
            "positives": int((group["label"] == 1).sum()),
            "both_in_geneeffect": int(group["both_in_geneeffect"].sum()),
        }
    return {
        "rows": int(len(rows)),
        "pairs": int(rows[["gene_a", "gene_b"]].drop_duplicates().shape[0]),
        "labelled": int(labelled.sum()),
        "positives": int((rows["label"] == 1).sum()),
        "unlabelled_reasons": rows.loc[~labelled, "unlabelled_reason"]
        .value_counts()
        .to_dict(),
        "by_side": by_side,
    }


def build(
    raw: Path,
    lines_path: Path,
    split_path: Path,
    extra_bulk_path: Path,
    extra_single_cell_path: Path,
    gene_effect_path: Path,
    hgnc_path: Path,
    out: Path,
) -> dict[str, object]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    split = load_geneeffect_split(split_path)
    extra_bulk = load_extra_lines(extra_bulk_path, split)
    extra_single_cell = load_extra_single_cell_lines(extra_single_cell_path, split)
    lines = pd.read_csv(lines_path, dtype=str).fillna("")
    sides = line_sides(lines, split, extra_bulk.labelled, extra_single_cell)
    geneeffect = {
        parse_gene_symbol(column)
        for column in pd.read_csv(gene_effect_path, nrows=0).columns[1:]
    }
    hgnc = pd.read_csv(
        hgnc_path,
        sep="\t",
        usecols=["symbol", "prev_symbol", "alias_symbol"],
        dtype=str,
    )
    name = geneeffect_namer(symbol_mapper(hgnc), geneeffect)
    paths = {name: Path(raw) / relative for name, relative in RAW_FILES.items()}

    ito = read_ito(paths["ito"])
    klingbeil = read_klingbeil(
        paths["klingbeil_gemini"], paths["klingbeil_lfc"], paths["klingbeil_fdr"]
    )
    harle = read_harle(paths["harle"])
    parsed = {"ito_2021": ito, "klingbeil_2024": klingbeil, "harle_2025": harle}
    finished = {
        screen: finish_rows(frame, screen, sides, name, geneeffect)
        for screen, frame in parsed.items()
    }
    rows = pd.concat(finished.values(), ignore_index=True)
    mit4sl = read_mit4sl(paths["mit4sl_a549"], paths["mit4sl_nodes"])
    mapped = [
        _sorted_pair(name(a), name(b))
        for a, b in zip(mit4sl["screen_gene_a"], mit4sl["screen_gene_b"])
    ]
    mit4sl["gene_a"] = [a for a, _ in mapped]
    mit4sl["gene_b"] = [b for _, b in mapped]
    mit4sl["both_in_geneeffect"] = mit4sl["gene_a"].isin(geneeffect) & mit4sl[
        "gene_b"
    ].isin(geneeffect)
    mit4sl["model_id"] = "ACH-000681"

    ito_kebabci = rows.loc[
        (rows["screen"] == "ito_2021")
        & (rows["screen_line"] != KEBABCI_ITO_REPORTED["left_out"])
    ]
    symbols = pd.concat(
        [parsed[s][["screen_gene_a", "screen_gene_b"]].stack() for s in SCREENS]
    ).astype(str)
    symbols = sorted(set(symbols))
    renamed = {s: name(s) for s in symbols if name(s) != s}
    manifest = {
        "schema_version": "sl-screen-benchmark-v1",
        "rule": {
            "kebabci_gemini_quantile": GEMINI_QUANTILE,
            "kebabci_lfc_threshold": LFC_THRESHOLD,
            "gemini_cut": {
                "ito_2021": ito.attrs["gemini_cut"],
                "klingbeil_2024": klingbeil.attrs["gemini_cut"],
            },
            "harle_2025": "authors' Table S5 call; non-hits with a gene "
            "single-depleted by MAGeCK and BAGEL2 are unlabelled",
            "klingbeil_constructs": {
                "constructs": int(len(klingbeil)),
                "constructs_without_lfc": klingbeil.attrs["constructs_without_lfc"],
            },
            "same_gene_rows_dropped": {
                screen: frame.attrs["same_gene_rows"]
                for screen, frame in finished.items()
            },
        },
        "kebabci_ito_reproduction": {
            "reported": KEBABCI_ITO_REPORTED,
            "built_sl": int((ito_kebabci["label"] == 1).sum()),
            "built_points": int(len(ito_kebabci)),
        },
        "genes": {
            "screen_symbols": len(symbols),
            "renamed": len(renamed),
            "not_in_geneeffect": len({name(s) for s in symbols} - set(geneeffect)),
        },
        "screens": {
            screen: _summary(rows.loc[rows["screen"] == screen]) for screen in SCREENS
        },
        "mit4sl_a549": {
            "rows": int(len(mit4sl)),
            "positives": int((mit4sl["label"] == 1).sum()),
            "both_in_geneeffect": int(mit4sl["both_in_geneeffect"].sum()),
        },
        "lines": sides.to_dict(orient="records"),
        "inputs": {
            **{name: {"path": str(path)} for name, path in paths.items()},
            **{
                name: {"path": str(path)}
                for name, path in (
                    ("lines", lines_path),
                    ("split", split_path),
                    ("extra_bulk", extra_bulk_path),
                    ("extra_single_cell", extra_single_cell_path),
                    ("gene_effect", gene_effect_path),
                    ("hgnc", hgnc_path),
                )
            },
        },
    }
    staging = out.with_name(f".{out.name}.building")
    if staging.exists():
        raise FileExistsError(f"refusing to replace incomplete staging {staging}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        rows.to_parquet(staging / "pairs.parquet", index=False)
        mit4sl.loc[
            :,
            [
                "model_id",
                "gene_a",
                "gene_b",
                "screen_gene_a",
                "screen_gene_b",
                "both_in_geneeffect",
                "label",
            ],
        ].to_parquet(staging / "mit4sl_a549.parquet", index=False)
        pd.Series(renamed, name="symbol").rename_axis("screen_symbol").to_csv(
            staging / "renamed_symbols.csv"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        )
        staging.rename(out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "raw",
        "lines",
        "split",
        "extra-bulk",
        "extra-single-cell",
        "gene-effect",
        "hgnc",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(
        args.raw,
        args.lines,
        args.split,
        args.extra_bulk,
        args.extra_single_cell,
        args.gene_effect,
        args.hgnc,
        args.out,
    )
    print(json.dumps({k: manifest[k] for k in ("kebabci_ito_reproduction", "genes")}))
    for screen, summary in manifest["screens"].items():
        print(
            screen,
            json.dumps({k: summary[k] for k in ("rows", "labelled", "positives")}),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
