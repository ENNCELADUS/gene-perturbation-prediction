"""Readers of the combinatorial and isogenic screens Fong2024 to Zhang2026, one per
ready row of configs/benchmarks/sl_label_rules.csv. Each takes the data root and returns
one row per labelled unit with the columns in COLUMNS, applying exactly its row's rule;
it raises ValueError when a file does not have the expected shape. Gene symbols are kept
as the file gives them.
"""

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "study_id",
    "list_id",
    "line",
    "anchor_gene",
    "anchor_genotype",
    "gene_a",
    "gene_b",
    "score",
    "fdr",
    "label",
    "unlabelled_reason",
]


def _f(raw, study, name):
    p = Path(raw) / "sl_studies" / "raw" / study / name
    if not p.exists():
        raise ValueError(f"missing file {p}")
    return p


def _sheet(path, sheet, header=0):
    try:
        return pd.read_excel(path, sheet_name=sheet, header=header)
    except ValueError as e:  # pandas raises ValueError for a missing sheet
        raise ValueError(f"{path.name}: sheet {sheet!r}: {e}") from e


def _need(df, cols, where):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{where}: missing columns {miss}")


def _out(
    study,
    line,
    gene_a,
    gene_b,
    score,
    fdr,
    pos,
    unl_reason,
    anchor="",
    genotype="",
    list_suffix="",
):
    """Build the standard frame. pos: bool array; unl_reason: '' where labelled."""
    n = len(gene_a)
    unl = (
        pd.Series(unl_reason, index=range(n)).fillna("").astype(str)
        if not isinstance(unl_reason, str)
        else pd.Series([unl_reason] * n)
    )
    label = pd.array(np.where(np.asarray(pos), 1, 0), dtype="Int8")
    label[unl.values != ""] = pd.NA
    line_s = (
        pd.Series(line, index=range(n))
        if not isinstance(line, str)
        else pd.Series([line] * n)
    )
    anchor_s = (
        pd.Series([anchor] * n)
        if isinstance(anchor, str)
        else pd.Series(anchor, index=range(n))
    )
    lid = (
        study
        + "|"
        + line_s.astype(str)
        + ("|" + anchor_s.astype(str)).where(anchor_s != "", "")
        + list_suffix
    )
    return pd.DataFrame(
        {
            "study_id": study,
            "list_id": lid.values,
            "line": line_s.values,
            "anchor_gene": anchor_s.values,
            "anchor_genotype": genotype,
            "gene_a": np.asarray(gene_a),
            "gene_b": np.asarray(gene_b),
            "score": np.asarray(score, dtype=float),
            "fdr": np.asarray(fdr, dtype=float),
            "label": label,
            "unlabelled_reason": unl.values,
        }
    )[COLUMNS]


def _bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# Fong2024
FONG_LINEAGE = {
    "MCF7": "Breast",
    "MDAMB231": "Breast",
    "MCF10A": "Breast",
    "CAL27": "Oropharyngeal",
    "CAL33": "Oropharyngeal",
    "A549": "Lung",
    "A427": "Lung",
}


def read_Fong2024(raw):
    s4x = _f(raw, "Fong2024", "41588_2024_1971_MOESM4_ESM.xlsx")
    s2x = _f(raw, "Fong2024", "41588_2024_1971_MOESM2_ESM.xlsx")
    sc = _sheet(s4x, "cell-line-scores")
    fd = _sheet(s4x, "cell-line-fdrs")
    lines = list(FONG_LINEAGE)
    _need(sc, ["target_a_id", "target_b_id"] + lines, "cell-line-scores")
    _need(fd, ["target_a_id", "target_b_id"] + lines, "cell-line-fdrs")
    if not (
        sc[["target_a_id", "target_b_id"]].values
        == fd[["target_a_id", "target_b_id"]].values
    ).all():
        raise ValueError("Fong2024: score and FDR sheets are not row-aligned")
    ess = _sheet(s2x, "S3", header=1)
    calls = _sheet(s2x, "S4", header=1)
    _need(ess, ["Essential genes", "Context"], "S3")
    _need(calls, ["Gene A", "Gene B", "context"], "S4")

    def key(a, b):
        return ["|".join(sorted(x)) for x in zip(a, b)]

    calls["k"] = key(calls["Gene A"].astype(str), calls["Gene B"].astype(str))
    aavs = (sc.target_a_id == "AAVS") | (sc.target_b_id == "AAVS")
    keep = ~aavs & (sc.target_a_id != sc.target_b_id)
    k_all = np.array(key(sc.target_a_id.astype(str), sc.target_b_id.astype(str)))
    out = []
    for L in lines:
        p25 = np.nanpercentile(sc.loc[aavs, L], 2.5)  # pi_ab < Percentile2.5(pi_AAVS)
        score, fdr = sc.loc[keep, L].values, fd.loc[keep, L].values
        line_calls = set(calls.loc[calls.context == L, "k"])
        pos = ((score < p25) & (fdr < 0.1)) | np.isin(
            k_all[keep.values], list(line_calls)
        )
        ctx = {
            L,
            "Pan-cancer",
            FONG_LINEAGE[L],
        }  # "Several contexts" and "POLQ" name no lines
        ess_l = set(ess.loc[ess.Context.isin(ctx), "Essential genes"].astype(str))
        a, b = (
            sc.loc[keep, "target_a_id"].astype(str).values,
            sc.loc[keep, "target_b_id"].astype(str).values,
        )
        unl = np.where(
            np.isin(a, list(ess_l)) | np.isin(b, list(ess_l)),
            "single_gene_essential_S3",
            "",
        )
        ok = ~np.isnan(score)
        out.append(
            _out("Fong2024", L, a[ok], b[ok], score[ok], fdr[ok], pos[ok], unl[ok])
        )
    return pd.concat(out, ignore_index=True)


# GonatopoulosPournatzis2020
def read_GonatopoulosPournatzis2020(raw):
    s8 = _sheet(
        _f(raw, "GonatopoulosPournatzis2020", "41587_2020_437_MOESM10_ESM.xlsx"),
        "Summary",
        header=1,
    )
    s7 = _sheet(
        _f(raw, "GonatopoulosPournatzis2020", "41587_2020_437_MOESM9_ESM.xlsx"),
        "DualTargeting Hits",
        header=1,
    )
    cols = {"HAP1": ("hap1", "HAP1 T18"), "RPE1": ("rpe1", "RPE1 T24")}
    _need(
        s8,
        ["gene1", "gene2"]
        + [
            f"{p}_{c}"
            for p, _ in cols.values()
            for c in ("gi_type", "late_effect_size")
        ],
        "Table S8 Summary",
    )
    _need(
        s7,
        [
            f"{t} {k}"
            for _, t in cols.values()
            for k in ("Dual-Targeting Only Hits", "Single- & Dual-Targeting Hits")
        ],
        "Table S7",
    )
    out = []
    for L, (p, t) in cols.items():
        ess = set(
            pd.concat(
                [
                    s7[f"{t} Dual-Targeting Only Hits"],
                    s7[f"{t} Single- & Dual-Targeting Hits"],
                ]
            )
            .dropna()
            .astype(str)
        )
        a, b = s8.gene1.astype(str).values, s8.gene2.astype(str).values
        score = s8[f"{p}_late_effect_size"].values
        pos = (s8[f"{p}_gi_type"] != "none").values & (score < 0)
        unl = np.where(
            np.isin(a, list(ess)) | np.isin(b, list(ess)),
            "single_gene_essential_S7",
            "",
        )
        out.append(
            _out(
                "GonatopoulosPournatzis2020",
                L,
                a,
                b,
                score,
                np.full(len(a), np.nan),
                pos,
                unl,
            )
        )
    return pd.concat(out, ignore_index=True)


# Koferle2022
def read_Koferle2022(raw):
    x = _f(raw, "Koferle2022", "1-s2.0-S2211124722003886-mmc2.xlsx")
    d = _sheet(x, "screen.paralogs.fullinfo")
    e = _sheet(x, "gene.essentiality")
    _need(
        d,
        [
            "cell.line.name",
            "symbol",
            "escore",
            "FDR",
            "group.paralog.family",
            "relative.copy.number",
            "total.abs.copy.number",
        ],
        "screen.paralogs.fullinfo",
    )
    _need(e, ["gene.symbol", "essentiality"], "gene.essentiality")
    core = set(e.loc[e.essentiality == "core-essential", "gene.symbol"].astype(str))
    with np.errstate(divide="ignore"):
        d["deleted"] = (d["total.abs.copy.number"] == 0) & (
            np.log2(d["relative.copy.number"]) < -1
        )
    out = []
    for (L, fam), g in d.groupby(["cell.line.name", "group.paralog.family"]):
        for A in g.loc[g.deleted, "symbol"].astype(str):
            B = g[g.symbol.astype(str) != A]
            if B.empty:
                continue
            pos = ((B.escore < -0.4) & (B.FDR < 0.1)).values
            unl = np.where(
                B.deleted.values,
                "gene_b_deleted",
                np.where(
                    B.symbol.astype(str).isin(core).values, "gene_b_core_essential", ""
                ),
            )
            out.append(
                _out(
                    "Koferle2022",
                    L,
                    [A] * len(B),
                    B.symbol.astype(str).values,
                    B.escore.values,
                    B.FDR.values,
                    pos,
                    unl,
                    anchor=A,
                    genotype="homozygous deletion (abs CN 0)",
                )
            )
    out = pd.concat(out, ignore_index=True)
    # a gene in two paralog families gives the same (line, A, B) unit twice: keep one
    # copy when identical
    out = out.drop_duplicates(
        ["list_id", "gene_a", "gene_b", "score", "fdr"]
    ).reset_index(drop=True)
    dup = out.duplicated(["list_id", "gene_a", "gene_b"], keep=False)
    out.loc[dup, "label"] = pd.NA
    out.loc[dup, "unlabelled_reason"] = "duplicate_row_in_file"
    return out


# Ku2020
def _ku_isogenic(raw, sheet, gene_col, score_col, genotype, suffix):
    d = _sheet(_f(raw, "Ku2020", "41467_2020_16078_MOESM10_ESM.xlsx"), sheet)
    _need(d, [gene_col, score_col], f"Ku2020 {sheet}")
    s = d[score_col].astype(float).values
    fdr = _bh(10.0 ** (-np.abs(s)))  # score = signed log10(p); BH as in Methods
    return _out(
        "Ku2020",
        "MCF10A",
        ["KRAS"] * len(d),
        d[gene_col].astype(str).values,
        s,
        fdr,
        (fdr < 0.1) & (s < 0),
        "",
        anchor="KRAS",
        genotype=genotype,
        list_suffix=suffix,
    )


def read_Ku2020_G12D(raw):
    return _ku_isogenic(
        raw,
        "G12D",
        "gene",
        "G12D_GI_score",
        "KRAS G12D overexpression vs eGFP",
        "|G12D",
    )


def read_Ku2020_G12V(raw):
    return _ku_isogenic(
        raw, "G12V", "Gene", "G12V_Score", "KRAS G12V overexpression vs eGFP", "|G12V"
    )


def read_Ku2020_emap(raw):
    d = _sheet(_f(raw, "Ku2020", "41467_2020_16078_MOESM11_ESM.xlsx"), "EMAP")
    _need(d, ["gene"], "Ku2020 EMAP")
    q = [
        c for c in d.columns if c.startswith("q") and c != "qBMN673"
    ]  # qBMN673 is a drug, not a gene
    if len(q) < 10:
        raise ValueError("Ku2020 EMAP: query columns not found")
    long = d.melt(
        id_vars="gene", value_vars=q, var_name="query", value_name="z"
    ).dropna(subset=["z"])
    long["query"] = long["query"].str[1:]
    long = long[long["query"] != long["gene"].astype(str)]
    dup = long.duplicated(["gene", "query"], keep=False).values
    return _out(
        "Ku2020",
        "MCF10A",
        long["query"].values,
        long["gene"].astype(str).values,
        long.z.values,
        np.full(len(long), np.nan),
        (long.z < -2).values,
        np.where(dup, "duplicate_row_in_file", ""),
        genotype="KRAS G12D, minimal media (E-MAP)",
        list_suffix="|EMAP",
    )


# Najm2023
def _najm(raw, sheet, rule, suffix):
    d = _sheet(_f(raw, "Najm2023", "41467_2023_36150_MOESM5_ESM.xlsx"), sheet)
    _need(
        d,
        ["condition", "gene_a", "gene_b", "lfc", "pair_z_score", "fdr_bh"],
        f"Najm2023 {sheet}",
    )
    d["gene_a"], d["gene_b"] = (
        d.gene_a.astype(str).str.strip(),
        d.gene_b.astype(str).str.strip(),
    )
    d = d[
        (d.gene_a != "Non-targeting")
        & (d.gene_b != "Non-targeting")
        & (d.gene_a != d.gene_b)
    ]
    if d.duplicated(["condition", "gene_a", "gene_b"]).any():
        raise ValueError(
            f"Najm2023 {sheet}: duplicate (condition, gene_a, gene_b) rows"
        )
    return _out(
        "Najm2023",
        d.condition.values,
        d.gene_a.values,
        d.gene_b.values,
        d.pair_z_score.values,
        d.fdr_bh.values,
        rule(d).values,
        "",
        list_suffix=suffix,
    )


def read_Najm2023_40k(raw):
    return _najm(raw, "40k", lambda d: (d.fdr_bh < 0.05) & (d.pair_z_score < 0), "|40k")


NAJM_300K_MEAN_LFC = {"REH": -0.36, "THP1": -0.27}  # Fig. 1b,c legend


def read_Najm2023_300k(raw):
    return _najm(
        raw,
        "300k",
        lambda d: (d.pair_z_score < -4) & (d.lfc < d.condition.map(NAJM_300K_MEAN_LFC)),
        "|300k",
    )


def read_Najm2023_validation(raw):
    return _najm(
        raw,
        "Validation",
        lambda d: (d.fdr_bh < 1e-5) & (d.pair_z_score < 0),
        "|validation",
    )


# Nebenfuehr2026
def _neb(raw, name, phen, logp, cut, suffix):
    d = _sheet(_f(raw, "Nebenfuehr2026", name), "Sheet1")
    _need(d, ["gene1", "gene2", phen, logp], f"Nebenfuehr2026 {name}")
    d = d[d.gene1 != d.gene2]
    pos = (d[logp] > 1.32) & (d[phen] < cut)
    return _out(
        "Nebenfuehr2026",
        "eHAP",
        d.gene1.astype(str).values,
        d.gene2.astype(str).values,
        d[phen].values,
        10.0 ** (-d[logp].values),
        pos.values,
        "",
        list_suffix=suffix,
    )


def read_Nebenfuehr2026_basal(raw):
    return _neb(
        raw,
        "NIHMS2142944-supplement-3.xlsx",
        "basal_obs_phen",
        "log10fisher_basal",
        -0.7791,
        "|basal",
    )


def read_Nebenfuehr2026_ir(raw):
    return _neb(
        raw,
        "NIHMS2142944-supplement-4.xlsx",
        "ir_obs_phen",
        "log10fisher_ir",
        -0.6270,
        "|IR",
    )


# Rosenbluh2016
def read_Rosenbluh2016(raw):
    d = _sheet(
        _f(raw, "Rosenbluh2016", "1-s2.0-S2405471216302915-mmc9.xlsx"),
        "Table_S9",
        header=None,
    )
    lines = ["DLD1", "HCT116", "HT29", "RKO"]
    if (
        str(d.iat[0, 10]).strip() != "GI Score (S Score)"
        or [str(x) for x in d.iloc[1, 10:14]] != lines
        or [str(x) for x in d.iloc[1, :2]] != ["Gene_1", "Gene_2"]
    ):
        raise ValueError("Rosenbluh2016 Table S9: unexpected header layout")
    b = d.iloc[2:].reset_index(drop=True)
    b = b[b[0].astype(str) != b[1].astype(str)]
    out = []
    for i, L in enumerate(lines):
        s = pd.to_numeric(b[10 + i], errors="coerce")
        ok = s.notna().values
        out.append(
            _out(
                "Rosenbluh2016",
                L,
                b[0].astype(str).values[ok],
                b[1].astype(str).values[ok],
                s.values[ok],
                np.full(ok.sum(), np.nan),
                (s.values[ok] < -0.5),
                "",
            )
        )
    return pd.concat(out, ignore_index=True)


# Zhang2026
def read_Zhang2026_combinatorial(raw):
    d = _sheet(_f(raw, "Zhang2026", "1-s2.0-S0167488925001752-mmc4.xlsx"), "Sheet1")
    _need(
        d,
        ["gene1", "gene2", "Pvalue", "GPS_score", "cellline", "Significant Pairs"],
        "Zhang2026 mmc4",
    )
    g1, g2 = d.gene1.astype(str), d.gene2.astype(str)
    ctrl = g1.str.match(r"^(?:NGC|NTC)-") | g2.str.contains(r"(?:NGC|NTC)-")
    d = d[~ctrl]
    dup = d.duplicated(["cellline", "gene1", "gene2"], keep=False).values
    pos = ((d["Significant Pairs"] == "Y") & (d.GPS_score < 0)).values
    return _out(
        "Zhang2026",
        d.cellline.values,
        d.gene1.astype(str).values,
        d.gene2.astype(str).values,
        d.GPS_score.values,
        np.full(len(d), np.nan),
        pos,
        np.where(dup, "duplicate_row_in_file", ""),
    )


def read_Zhang2026_focad_isogenic(raw):
    d = _sheet(_f(raw, "Zhang2026", "1-s2.0-S0167488925001752-mmc5.xlsx"), "Sheet1")
    _need(
        d,
        ["gene_symbol", "Gene Dependency Score", "FDR", "Significant Gene Hits"],
        "Zhang2026 mmc5",
    )
    s = d["Gene Dependency Score"].values
    pos = (
        (d["Significant Gene Hits"] == "Y") & (d["Gene Dependency Score"] < 0.5)
    ).values
    return _out(
        "Zhang2026",
        "MIAPACA2",
        ["FOCAD"] * len(d),
        d.gene_symbol.astype(str).values,
        s,
        d.FDR.values,
        pos,
        "",
        anchor="FOCAD",
        genotype="FOCAD-deleted (empty vector) vs FOCAD cDNA",
    )


# Knoll2025
KNOLL_ARMS = {"LU99": "LU99.MRTX9768", "SW1573": "SW1573.MRTX1719"}


def _knoll(raw, value, anchor, genotype):
    """The authors' rule on value(d, line), each construct's LFC: the pair below -0.5
    and below the stronger single by more than 0.2."""
    d = pd.read_csv(_f(raw, "Knoll2025", "NIHMS2100756-supplement-3.csv"))
    _need(
        d,
        ["GuideTargetSymbol"]
        + [c for L, c in KNOLL_ARMS.items()]
        + [f"{L}.DMSO" for L in KNOLL_ARMS],
        "Knoll2025 supplement-3",
    )
    sym = d.GuideTargetSymbol.astype(str)
    is_pair = (sym.str.count("_") == 1).values
    ab = sym[is_pair].str.split("_", expand=True)
    out = []
    for L in KNOLL_ARMS:
        lfc = pd.Series(value(d, L).values, index=sym.values)
        single = lfc[~lfc.index.duplicated(keep=False)]
        dp = lfc.values[is_pair]
        da, db = single.reindex(ab[0]).values, single.reindex(ab[1]).values
        miss = np.isnan(da) | np.isnan(db)
        pos = (dp < -0.5) & (dp - np.fmin(da, db) < -0.2)
        out.append(
            _out(
                "Knoll2025",
                L,
                ab[0].values,
                ab[1].values,
                dp,
                np.full(len(dp), np.nan),
                pos,
                np.where(miss, "single_gene_missing", ""),
                anchor=anchor,
                genotype=genotype(L),
            )
        )
    return pd.concat(out, ignore_index=True)


def read_Knoll2025_mrtx(raw):
    return _knoll(
        raw,
        lambda d, L: d[KNOLL_ARMS[L]] - d[f"{L}.DMSO"],
        "PRMT5",
        lambda L: f"MTAP-deleted; {KNOLL_ARMS[L].split('.')[1]} arm minus DMSO",
    )


def read_Knoll2025_dmso(raw):
    """The paralog SL of each line: the authors' rule on the DMSO arm."""
    return _knoll(raw, lambda d, L: d[f"{L}.DMSO"], "", lambda L: "DMSO vs pDNA")


# Du2017
DU_CONTROLS = re.compile(r"^(LC|DC\d+)$")  # library and dummy controls


def read_Du2017(raw):
    """Positives are the published pairs with S < -1 (the authors' call); negatives
    are the other gene pairs of the library, assumed all scored: the paper publishes
    only the significant pairs."""
    hits = pd.read_csv(
        _f(raw, "Du2017", "41592_2017_BFnmeth4286_MOESM2_ESM.txt"), sep="\t"
    )
    _need(hits, ["gene1", "gene2", "score"], "Du2017 Data Set 1")
    member = "Supplementary_Dataset_3/double_lib_before_pack.txt"
    zip_path = _f(raw, "Du2017", "41592_2017_BFnmeth4286_MOESM4_ESM.zip")
    with zipfile.ZipFile(zip_path) as z:
        library = pd.read_csv(z.open(member), sep="\t")
    _need(library, ["pair"], f"Du2017 {member}")
    # guides are GENE_<n>, or GENE_PC for the essential-gene controls the paper scores
    genes = library.pair.str.split(":", expand=True).apply(
        lambda g: g.str.extract(r"^(.+)_(?:\d+|PC)$")[0]
    )
    if genes.isna().any().any():
        raise ValueError(f"Du2017 {member}: a guide name is not GENE_<n> or GENE_PC")
    genes = genes[~genes.apply(lambda g: g.str.match(DU_CONTROLS)).any(axis=1)]
    genes = genes[genes[0] != genes[1]]
    tested = pd.DataFrame(np.sort(genes.to_numpy(), axis=1), columns=["a", "b"])
    tested = tested.drop_duplicates().reset_index(drop=True)
    hits[["a", "b"]] = np.sort(hits[["gene1", "gene2"]].to_numpy(), axis=1)
    hits = hits[hits.a != hits.b]
    score = tested.merge(hits[["a", "b", "score"]], on=["a", "b"], how="left").score
    if score.notna().sum() != len(hits):
        raise ValueError("Du2017: a published pair is not in the library")
    return _out(
        "Du2017",
        "HEK293",
        tested.a.values,
        tested.b.values,
        score.values,
        np.full(len(tested), np.nan),
        (score < -1).values,
        "",
    )


READERS = {
    "Fong2024": read_Fong2024,
    "GonatopoulosPournatzis2020": read_GonatopoulosPournatzis2020,
    "Koferle2022": read_Koferle2022,
    "Ku2020_G12D": read_Ku2020_G12D,
    "Ku2020_G12V": read_Ku2020_G12V,
    "Ku2020_emap": read_Ku2020_emap,
    "Najm2023_40k": read_Najm2023_40k,
    "Najm2023_300k": read_Najm2023_300k,
    "Najm2023_validation": read_Najm2023_validation,
    "Nebenfuehr2026_basal": read_Nebenfuehr2026_basal,
    "Nebenfuehr2026_ir": read_Nebenfuehr2026_ir,
    "Rosenbluh2016": read_Rosenbluh2016,
    "Zhang2026_combinatorial": read_Zhang2026_combinatorial,
    "Zhang2026_focad_isogenic": read_Zhang2026_focad_isogenic,
    "Knoll2025_mrtx": read_Knoll2025_mrtx,
    "Knoll2025_dmso": read_Knoll2025_dmso,
    "Du2017": read_Du2017,
}
