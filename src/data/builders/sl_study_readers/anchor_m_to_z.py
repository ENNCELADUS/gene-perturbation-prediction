"""Readers of the anchor-screen studies Mengwasser2019 to van2023_2, one per ready row
of configs/benchmarks/sl_label_rules.csv. Each takes the data root and returns one row
per labelled unit with the columns in COLUMNS, applying exactly its row's rule; it
raises ValueError when the file does not have the expected shape.
"""

import io
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
DUP = "gene symbol appears more than once in this list"


def _f(raw: Path, study: str, name: str) -> Path:
    p = Path(raw) / "sl_studies" / "raw" / study / name
    if not p.is_file():
        raise ValueError(f"missing file {p}")
    return p


def _need(df: pd.DataFrame, cols, where: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{where}: missing columns {miss}")


def _sheet(path, sheet, **kw):
    x = pd.ExcelFile(path, engine=kw.pop("engine", None))
    if sheet not in x.sheet_names:
        raise ValueError(f"{path}: missing sheet {sheet!r}")
    return x.parse(sheet, **kw)


def _out(
    study, list_id, line, anchor, genotype, gene_b, score, fdr, label, reason
) -> pd.DataFrame:
    """Build the output frame. label: array of 1/0/NaN; reason: array of str (''
    where labelled)."""
    gene_b = (
        pd.Series(np.asarray(gene_b, dtype=object)).astype(str).str.strip().to_numpy()
    )
    n = len(gene_b)

    def vec(x, dtype):
        return (
            np.full(n, x, dtype=dtype)
            if np.isscalar(x) or x is None
            else np.asarray(x, dtype=dtype)
        )

    def num(x):
        return (
            pd.to_numeric(pd.Series(vec(x, object)), errors="coerce")
            .astype(float)
            .to_numpy()
        )

    d = pd.DataFrame(
        {
            "study_id": study,
            "list_id": list_id,
            "line": line,
            "anchor_gene": anchor,
            "anchor_genotype": genotype,
            "gene_a": anchor,
            "gene_b": gene_b,
            "score": num(score),
            "fdr": num(fdr),
            "label": pd.array(vec(label, float), dtype="Int8"),
            "unlabelled_reason": pd.Series(vec(reason, object)).fillna("").to_numpy(),
        }
    )
    dup = d.gene_b.duplicated(keep=False)
    d.loc[dup, "label"] = pd.NA
    d.loc[dup, "unlabelled_reason"] = DUP
    d.loc[d.label.notna(), "unlabelled_reason"] = ""
    return d[COLUMNS].reset_index(drop=True)


def _bin(pos, neg_mask=None):
    """1 where pos, else 0; NaN where neg_mask is False (not labelled)."""
    lab = np.where(pos, 1.0, 0.0)
    if neg_mask is not None:
        lab = np.where(pos | neg_mask, lab, np.nan)
    return lab


def _orcs(raw, study, screen):
    p = (
        Path(raw)
        / "sl_studies"
        / "raw"
        / study
        / f"BIOGRID-ORCS-SCREEN_{screen}-2.0.18.screen.tab.txt"
    )
    if not p.is_file():
        raise ValueError(f"missing file {p}")
    d = pd.read_csv(p, sep="\t")
    _need(d, ["OFFICIAL_SYMBOL", "SCORE.1", "SCORE.2", "SCORE.3", "HIT"], str(p))
    if set(d.HIT.unique()) - {"YES", "NO"}:
        raise ValueError(f"{p}: unexpected HIT values {set(d.HIT.unique())}")
    for c in ["SCORE.1", "SCORE.2", "SCORE.3"]:
        d[c] = pd.to_numeric(d[c].replace("-", np.nan), errors="coerce")
    return d


# Mengwasser2019 (edgeR, Table 2)
_MENG_BLOCKS = {
    "PRIMARY shRNA Screen": 1,
    "SECONDARY shRNA Screen": 5,
    "CRISPR COLONIC Screen": 9,
    "CRISPR OVARIAN Screen": 13,
}


def _mengwasser(raw, block, line, list_id):
    p = _f(raw, "Mengwasser2019", "NIHMS1059111-supplement-Supplemental_Tables.xlsx")
    t = _sheet(p, "Table 2. edgeR Results", header=None)
    c0 = _MENG_BLOCKS[block]
    if (
        t.iloc[5, c0] != block
        or list(t.iloc[6, c0 + 1 : c0 + 4]) != ["Direction", "PValue", "FDR"]
        or t.iloc[6, 0] != "Gene"
    ):
        raise ValueError(f"{p}: Table 2 layout changed near {block}")
    d = t.iloc[7:, [0, c0 + 1, c0 + 2, c0 + 3]].copy()
    d.columns = ["Gene", "Direction", "PValue", "FDR"]
    d = d[d.Gene.notna() & (d.Direction != "NIL")]
    if not set(d.Direction) <= {"Up", "Down"}:
        raise ValueError(f"{p}: unexpected Direction values {set(d.Direction)}")
    fdr = d.FDR.astype(float)
    pval = d.PValue.astype(float)
    score = np.where(d.Direction == "Down", 1.0, -1.0) * np.log10(
        pval
    )  # negative = dropout in BRCA2-mutant
    pos = ((fdr < 0.1) & (d.Direction == "Down")).to_numpy()
    return _out(
        "Mengwasser2019",
        list_id,
        line,
        "BRCA2",
        "BRCA2 mutant vs BRCA2 wild-type isogenic pair",
        d.Gene,
        score,
        fdr,
        _bin(pos),
        "",
    )


def read_Mengwasser2019_DLD1_crispr(raw: Path) -> pd.DataFrame:
    return _mengwasser(
        raw, "CRISPR COLONIC Screen", "DLD-1", "Mengwasser2019|DLD-1|BRCA2|CRISPR"
    )


def read_Mengwasser2019_PEO1_crispr(raw: Path) -> pd.DataFrame:
    return _mengwasser(
        raw, "CRISPR OVARIAN Screen", "PEO1", "Mengwasser2019|PEO1|BRCA2|CRISPR"
    )


def read_Mengwasser2019_DLD1_shrna_primary(raw: Path) -> pd.DataFrame:
    return _mengwasser(
        raw, "PRIMARY shRNA Screen", "DLD-1", "Mengwasser2019|DLD-1|BRCA2|shRNA-primary"
    )


def read_Mengwasser2019_DLD1_shrna_secondary(raw: Path) -> pd.DataFrame:
    return _mengwasser(
        raw,
        "SECONDARY shRNA Screen",
        "DLD-1",
        "Mengwasser2019|DLD-1|BRCA2|shRNA-secondary",
    )


# Vichas2021 (Supplementary Data 2 and 3)
_VICHAS = {
    "EGFR": (
        "EGFR synthetic lethal (192)",
        "EGFRL858R/T790M",
        "EGFR T790M/L858R (expressed)",
    ),
    "KRAS": ("KRAS synthetic lethal (194)", "KRASG12V", "KRAS G12V (expressed)"),
    "RIT1": ("RIT1 synthetic lethal (134)", "RIT1M90I", "RIT1 M90I (expressed)"),
}


def _vichas(raw, gene):
    col, pre, genotype = _VICHAS[gene]
    cs = pd.read_excel(
        _f(raw, "Vichas2021", "41467_2021_24841_MOESM4_ESM.xlsx"),
        sheet_name="CRISPR Score",
    )
    ctl_cs, ctl_p = (
        "Control/Luciferase_DMSO_CS_avg",
        "neglog10_Control/Luciferase_DMSO_avg_pvalue",
    )
    _need(cs, ["Gene Id", f"{pre}_DMSO_CS_avg", ctl_cs, ctl_p], "Vichas2021 MOESM4")
    sl = pd.read_excel(
        _f(raw, "Vichas2021", "41467_2021_24841_MOESM5_ESM.xlsx"),
        sheet_name="baseline sythetic lethals",
    )
    _need(sl, [col], "Vichas2021 MOESM5")
    calls = set(sl[col].dropna().astype(str).str.strip())
    if not calls <= set(cs["Gene Id"].astype(str)):
        raise ValueError("Vichas2021: called genes missing from the CRISPR Score table")
    pos = cs["Gene Id"].astype(str).isin(calls).to_numpy()
    ctl_ess = ((cs[ctl_cs] < -0.5) & (cs[ctl_p] > -np.log10(0.05))).to_numpy()
    lab = np.where(pos, 1.0, np.where(ctl_ess, np.nan, 0.0))
    why = np.where(
        np.isnan(lab),
        "essential in the vector-control PC9 line (CS < -0.5, p < 0.05)",
        "",
    )
    return _out(
        "Vichas2021",
        f"Vichas2021|PC-9|{gene}",
        "PC-9",
        gene,
        genotype + " vs vector control",
        cs["Gene Id"],
        cs[f"{pre}_DMSO_CS_avg"],
        np.nan,
        lab,
        why,
    )


def read_Vichas2021_EGFR(raw: Path) -> pd.DataFrame:
    return _vichas(raw, "EGFR")


def read_Vichas2021_KRAS(raw: Path) -> pd.DataFrame:
    return _vichas(raw, "KRAS")


def read_Vichas2021_RIT1(raw: Path) -> pd.DataFrame:
    return _vichas(raw, "RIT1")


# Vizeacoumar2013 (Table S4 validation, S2 dGARP)
_VIZ = {
    "BLM": ("BLMvalidation", "BLM_dGARP", "BLM-/-"),
    "MUS81": ("MUS81validation", "MUS81_dGARP", "MUS81-/-"),
    "PTEN": ("PTENvalidation", "PTEN_dGARP", "PTEN-/-"),
    "PTTG1": ("PTTG1validation", "PTTG1_dGARP", "PTTG1-/-"),
    "KRAS": (
        "KRASvalidation",
        "KRASG13D/+_dGARP",
        "KRAS G13D/+ (parental) vs KRAS +/- (mutant allele deleted)",
    ),
}
_VIZ_CALL = "80% Confidence Interval (P<0.2)"


def _vizeacoumar(raw, gene):
    sheet, dcol, genotype = _VIZ[gene]
    v = pd.read_excel(_f(raw, "Vizeacoumar2013", "msb201354-s5.xlsx"), sheet_name=sheet)
    _need(
        v,
        ["Gene.ID", "human gene", "Expression", _VIZ_CALL],
        f"Vizeacoumar2013 {sheet}",
    )
    if not set(v.Expression.dropna()) <= {
        "Expressed",
        "Expression below threshold",
        "no si",
    }:
        raise ValueError(f"Vizeacoumar2013 {sheet}: unexpected Expression values")
    if not set(v[_VIZ_CALL].dropna()) <= {0, 1}:
        raise ValueError(f"Vizeacoumar2013 {sheet}: unexpected call values")
    v = v[v.Expression != "no si"]  # no siRNA: not tested in the validation screen
    g = pd.read_excel(
        _f(raw, "Vizeacoumar2013", "msb201354-s3.xlsx"), sheet_name="dGARP"
    )
    _need(g, ["Gene.ID", dcol], "Vizeacoumar2013 dGARP")
    score = v["Gene.ID"].map(g.drop_duplicates("Gene.ID").set_index("Gene.ID")[dcol])
    expressed = (v.Expression == "Expressed").to_numpy()
    lab = np.where(expressed, (v[_VIZ_CALL] == 1).astype(float), np.nan)
    why = np.where(expressed, "", "target mRNA below the authors' expression threshold")
    return _out(
        "Vizeacoumar2013",
        f"Vizeacoumar2013|HCT116|{gene}",
        "HCT116",
        gene,
        genotype,
        v["human gene"],
        score,
        np.nan,
        lab,
        why,
    )


def read_Vizeacoumar2013_BLM(raw: Path) -> pd.DataFrame:
    return _vizeacoumar(raw, "BLM")


def read_Vizeacoumar2013_MUS81(raw: Path) -> pd.DataFrame:
    return _vizeacoumar(raw, "MUS81")


def read_Vizeacoumar2013_PTEN(raw: Path) -> pd.DataFrame:
    return _vizeacoumar(raw, "PTEN")


def read_Vizeacoumar2013_PTTG1(raw: Path) -> pd.DataFrame:
    return _vizeacoumar(raw, "PTTG1")


def read_Vizeacoumar2013_KRAS(raw: Path) -> pd.DataFrame:
    return _vizeacoumar(raw, "KRAS")


# Schick2019 (Supplementary Table 5, arrayed siRNA)
_SCHICK_GENE = {
    "ACTIN": "ACTB",
    "ARID1A 10": "ARID1A",
    "SMARCA4 6": "SMARCA4",
}  # Table 1 clone names
_SCHICK_CONTROLS = {"PLK1", "ON-TARGETplus Non-targeting Control"}


def _schick(raw, cell_line):
    d = pd.read_excel(
        _f(raw, "Schick2019", "EMS83287-supplement-Supplementary_Table_5.xlsx")
    )
    _need(
        d,
        [
            "CellLine",
            "siRNA targets",
            "Hit",
            "deltaPOC_noOutliers",
            "pValue_noOutliers",
        ],
        "Schick2019 Table 5",
    )
    d = d[
        (d.CellLine == cell_line)
        & d["siRNA targets"].notna()
        & ~d["siRNA targets"].isin(_SCHICK_CONTROLS)
    ]
    if d.empty:
        raise ValueError(f"Schick2019: no rows for cell line {cell_line!r}")
    a = (
        d.groupby("siRNA targets", sort=True)
        .agg(
            hit=("Hit", "max"),
            hit_n=("Hit", "nunique"),
            dpoc=("deltaPOC_noOutliers", "first"),
        )
        .reset_index()
    )
    if (a.hit_n != 1).any():
        raise ValueError("Schick2019: Hit differs between replicates of one siRNA")
    anchor = _SCHICK_GENE.get(cell_line, cell_line)
    pool = a["siRNA targets"].str.contains("_").to_numpy()
    self_ = (a["siRNA targets"] == anchor).to_numpy()
    lab = np.where(pool | self_, np.nan, a.hit.astype(float))
    why = np.where(
        pool,
        "siRNA pool targets several genes",
        np.where(self_, "siRNA targets the knocked-out gene", ""),
    )
    return _out(
        "Schick2019",
        f"Schick2019|HAP1|{anchor}",
        "HAP1",
        anchor,
        f"{anchor} KO clone ({cell_line}) vs HAP1 wild type",
        a["siRNA targets"],
        a.dpoc,
        np.nan,
        lab,
        why,
    )


def read_Schick2019_ACTB(raw):
    return _schick(raw, "ACTIN")


def read_Schick2019_ARID1A(raw):
    return _schick(raw, "ARID1A 10")


def read_Schick2019_ARID1B(raw):
    return _schick(raw, "ARID1B")


def read_Schick2019_ARID2(raw):
    return _schick(raw, "ARID2")


def read_Schick2019_BCL11A(raw):
    return _schick(raw, "BCL11A")


def read_Schick2019_BCL11B(raw):
    return _schick(raw, "BCL11B")


def read_Schick2019_BCL7A(raw):
    return _schick(raw, "BCL7A")


def read_Schick2019_BCL7B(raw):
    return _schick(raw, "BCL7B")


def read_Schick2019_BRD7(raw):
    return _schick(raw, "BRD7")


def read_Schick2019_BRD9(raw):
    return _schick(raw, "BRD9")


def read_Schick2019_DPF1(raw):
    return _schick(raw, "DPF1")


def read_Schick2019_DPF2(raw):
    return _schick(raw, "DPF2")


def read_Schick2019_DPF3(raw):
    return _schick(raw, "DPF3")


def read_Schick2019_PBRM1(raw):
    return _schick(raw, "PBRM1")


def read_Schick2019_PHF10(raw):
    return _schick(raw, "PHF10")


def read_Schick2019_SMARCA2(raw):
    return _schick(raw, "SMARCA2")


def read_Schick2019_SMARCA4(raw):
    return _schick(raw, "SMARCA4 6")


def read_Schick2019_SMARCC1(raw):
    return _schick(raw, "SMARCC1")


def read_Schick2019_SMARCC2(raw):
    return _schick(raw, "SMARCC2")


def read_Schick2019_SMARCD1(raw):
    return _schick(raw, "SMARCD1")


def read_Schick2019_SMARCD2(raw):
    return _schick(raw, "SMARCD2")


def read_Schick2019_SMARCD3(raw):
    return _schick(raw, "SMARCD3")


# SeguraBayona2026 (Supplementary Table 1, MAGeCK RRA)
def _segura(raw, day):
    p = _f(raw, "SeguraBayona2026", "41594_2026_1827_MOESM2_ESM.xlsx")
    d = _sheet(p, f"Supplementary Table 1_{day}", header=2)
    _need(d, ["id", "fdr", "lfc"], f"SeguraBayona2026 {day}")
    d = d[d["id"].notna()]
    pos = (
        d.fdr < 0.05
    ).to_numpy()  # negative-selection FDR: dropout in ATRX-null vs parental
    return _out(
        "SeguraBayona2026",
        f"SeguraBayona2026|eHAP|ATRX|{day}",
        "eHAP iCas9",
        "ATRX",
        f"ATRX-null vs parental eHAP iCas9 ({day})",
        d["id"],
        d.lfc,
        d.fdr,
        _bin(pos),
        "",
    )


def read_SeguraBayona2026_d6(raw: Path) -> pd.DataFrame:
    return _segura(raw, "d6")


def read_SeguraBayona2026_d16(raw: Path) -> pd.DataFrame:
    return _segura(raw, "d16")


# Zhang2023 (Figure 1 source data 1, MAGeCK MLE)
def _zhang(raw, clone):
    d = _sheet(_f(raw, "Zhang2023", "elife-91611-fig1-data1.xlsx"), "All Genes")
    b, f = f"Mut{clone}|beta", f"Mut{clone}|fdr"
    _need(d, ["Gene", b, f], "Zhang2023 All Genes")
    pos = ((d[f] < 0.25) & (d[b] < 0)).to_numpy()
    return _out(
        "Zhang2023",
        f"Zhang2023|OCI-AML2|PPM1D|clone{clone}",
        "OCI-AML2",
        "PPM1D",
        f"PPM1D-mutant clone {clone} vs PPM1D wild-type",
        d.Gene,
        d[b],
        d[f],
        _bin(pos),
        "",
    )


def read_Zhang2023_clone2102(raw: Path) -> pd.DataFrame:
    return _zhang(raw, 2102)


def read_Zhang2023_clone2113(raw: Path) -> pd.DataFrame:
    return _zhang(raw, 2113)


# Zimmermann2018 (Supplementary Table 1, olaparib)
_ZIM = {
    "HeLa": ("HeLa", "HELA", "olaparib 2 uM"),
    "SUM149PT": ("SUM", "SUM", "olaparib 0.2 uM"),
    "RPE1": ("RPE", "RPE", "olaparib 0.5 uM"),
}


def _zimmermann(raw, line):
    pre, col, dose = _ZIM[line]
    p = _f(raw, "Zimmermann2018", "NIHMS78070-supplement-3.xlsx")
    hc = _sheet(p, "High-confidence hits", header=2)
    g = _sheet(p, "Gene summary")
    _need(hc, [col], "Zimmermann2018 High-confidence hits")
    _need(g, ["Gene", f"{pre} normZ", f"{pre} fdr"], "Zimmermann2018 Gene summary")
    calls = set(hc[col].dropna().astype(str).str.strip())
    extra = sorted(
        calls - set(g.Gene.astype(str))
    )  # called, but the Gene summary has no row (no score)
    g = pd.concat([g, pd.DataFrame({"Gene": extra})], ignore_index=True)
    pos = g.Gene.astype(str).isin(calls).to_numpy()
    noscore = g.Gene.astype(str).isin(extra).to_numpy()
    lab = np.where(noscore, np.nan, _bin(pos))
    why = np.where(
        noscore, "called by the authors but absent from the Gene summary (no score)", ""
    )
    return _out(
        "Zimmermann2018",
        f"Zimmermann2018|{line}|olaparib",
        line,
        "drug:olaparib",
        f"PARP inhibitor {dose} vs untreated",
        g.Gene,
        g[f"{pre} normZ"],
        g[f"{pre} fdr"],
        lab,
        why,
    )


def read_Zimmermann2018_HeLa(raw: Path) -> pd.DataFrame:
    return _zimmermann(raw, "HeLa")


def read_Zimmermann2018_SUM149PT(raw: Path) -> pd.DataFrame:
    return _zimmermann(raw, "SUM149PT")


def read_Zimmermann2018_RPE1(raw: Path) -> pd.DataFrame:
    return _zimmermann(raw, "RPE1")


# Wang2017_2 (Table 3, AML CRISPR scores)
# The Results name five genes "required only in the context of oncogenic Ras". The
# Ras-dependent lines are the six whose own KRAS or NRAS score is below -1.9 (every
# other line is above -1.7); Figure 6B, the authors' annotation, was not read.
_WANG_RAS = {"NB4": "KRAS", "OCI-AML3": "NRAS", "P31/FUJ": "NRAS",
             "PL-21": "KRAS", "SKM-1": "KRAS", "THP-1": "NRAS"}  # fmt: skip
_WANG_SL = {"RCE1", "ICMT", "RAF1", "SHOC2", "PREX1"}


def read_Wang2017_2(raw: Path) -> pd.DataFrame:
    d = _sheet(
        _f(raw, "Wang2017_2", "1-s2.0-S0092867417300612-mmc3.xlsx"),
        "3. Human AML screen CS",
        header=1,
    )
    _need(d, ["Gene"], "Wang2017_2 Table 3")
    pos = d.Gene.isin(_WANG_SL).to_numpy()
    if pos.sum() != len(_WANG_SL):
        raise ValueError("Wang2017_2 Table 3: a Ras-SL gene is missing")
    frames = []
    for line, ras in _WANG_RAS.items():
        # NB4 has two replicate columns: their mean
        cols = [c for c in d.columns if str(c).split(" (replicate")[0] == line]
        if not cols:
            raise ValueError(f"Wang2017_2 Table 3: no column for {line}")
        score = d[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        frames.append(
            _out(
                "Wang2017_2",
                f"Wang2017_2|{line}|{ras}",
                line,
                ras,
                f"oncogenic {ras}, Ras-dependent",
                d.Gene,
                score,
                np.nan,
                _bin(pos),
                "",
            )
        )
    return pd.concat(frames, ignore_index=True)


# van2021 (source data, DrugZ)
def _drugz(d, where):
    _need(d, ["GENE", "normZ", "fdr_synth"], where)
    return d[d.GENE.notna()]


def _van2021(raw, fname, sheet, cut, list_id, anchor, genotype):
    d = _drugz(_sheet(_f(raw, "van2021", fname), sheet), f"van2021 {sheet}")
    pos = (d.fdr_synth < cut).to_numpy()
    return _out(
        "van2021",
        list_id,
        "RPE1-iCas9",
        anchor,
        genotype,
        d.GENE,
        d.normZ,
        d.fdr_synth,
        _bin(pos),
        "",
    )


def read_van2021_illudin_WT(raw: Path) -> pd.DataFrame:
    return _van2021(
        raw,
        "41556_2021_688_MOESM5_ESM.xlsx",
        "Figure 1a",
        0.01,
        "van2021|RPE1-iCas9|IlludinS",
        "drug:Illudin S",
        "Illudin S vs untreated, wild-type RPE1-iCas9",
    )


def read_van2021_ELOF1(raw: Path) -> pd.DataFrame:
    return _van2021(
        raw,
        "41556_2021_688_MOESM6_ESM.xlsx",
        "Figure 2a",
        0.1,
        "van2021|RPE1-iCas9|ELOF1",
        "ELOF1",
        "ELOF1-KO vs wild-type RPE1-iCas9",
    )


def read_van2021_illudin_CSBKO(raw: Path) -> pd.DataFrame:
    return _van2021(
        raw,
        "41556_2021_688_MOESM12_ESM.xlsx",
        "Figure 7a",
        0.01,
        "van2021|RPE1-iCas9-CSB-KO|IlludinS",
        "drug:Illudin S",
        "Illudin S vs untreated in CSB (ERCC6)-KO RPE1-iCas9",
    )


def read_van2021_illudin_ELOF1KO(raw: Path) -> pd.DataFrame:
    return _van2021(
        raw,
        "41556_2021_688_MOESM12_ESM.xlsx",
        "Figure 7b",
        0.01,
        "van2021|RPE1-iCas9-ELOF1-KO|IlludinS",
        "drug:Illudin S",
        "Illudin S vs untreated in ELOF1-KO RPE1-iCas9",
    )


# van2023_2 (Table S1 .xlsb inside the zip, DrugZ)
def _van2023(raw, sheet, cut, anchor, genotype):
    z = zipfile.ZipFile(_f(raw, "van2023_2", "gkad756_supplemental_files.zip"))
    names = [n for n in z.namelist() if n.endswith("Table S1 CRISPR Screen data.xlsb")]
    if len(names) != 1:
        raise ValueError("van2023_2: Table S1 .xlsb not found in the zip")
    d = _drugz(
        _sheet(io.BytesIO(z.read(names[0])), sheet, engine="pyxlsb"),
        f"van2023_2 {sheet}",
    )
    pos = (d.fdr_synth < cut).to_numpy()
    return _out(
        "van2023_2",
        f"van2023_2|RPE1|{anchor}",
        "RPE1",
        anchor,
        genotype,
        d.GENE,
        d.normZ,
        d.fdr_synth,
        _bin(pos),
        "",
    )


def read_van2023_2_ESCO2(raw: Path) -> pd.DataFrame:
    return _van2023(
        raw,
        "6 - Analysis WT vs. ESCO2mut",
        0.05,
        "ESCO2",
        "ESCO2 mutant vs wild-type RPE1",
    )


def read_van2023_2_DDX11(raw: Path) -> pd.DataFrame:
    return _van2023(
        raw, "7 - Analysis WT vs. DDX11ko", 0.05, "DDX11", "DDX11-KO vs wild-type RPE1"
    )


def read_van2023_2_PAXIP1(raw: Path) -> pd.DataFrame:
    return _van2023(
        raw,
        "8 - Analysis WT vs. PAXIP1ko",
        0.1,
        "PAXIP1",
        "PAXIP1-KO vs wild-type RPE1",
    )


# Neggers2020 (worksheet 18, modifier screen)
def read_Neggers2020_VPS4A(raw: Path) -> pd.DataFrame:
    p = _f(raw, "Neggers2020", "NIHMS1657052-supplement-2.xlsx")
    d = _sheet(p, "18. CRISPR Screen results")
    _need(
        d,
        ["Gene", "ΔFold Change", "-Log10(q-value STARS)", "Significant?"],
        "Neggers2020 worksheet 18",
    )
    calls = {"Yes-neg", "Yes-pos", "No", "No-novalue"}
    if not set(d["Significant?"].dropna()) <= calls:
        raise ValueError(
            f"Neggers2020: unexpected calls {set(d['Significant?'].dropna())}"
        )
    sig = d["Significant?"]
    has_call = sig.isin(["Yes-neg", "Yes-pos", "No"]).to_numpy()
    lab = np.where(has_call, (sig == "Yes-neg").astype(float), np.nan)
    why = np.where(
        has_call,
        "",
        np.where(
            sig == "No-novalue", "no STARS q-value (call 'No-novalue')", "no call"
        ),
    )
    return _out(
        "Neggers2020",
        "Neggers2020|SNU213|VPS4A",
        "SNU213",
        "VPS4A",
        "dox-induced shVPS4A-2 vs no dox, in VPS4B-copy-loss SNU213",
        d.Gene,
        d["ΔFold Change"],
        10 ** -d["-Log10(q-value STARS)"],
        lab,
        why,
    )


# BioGRID ORCS screens (HIT = the authors' call)
def _orcs_call(raw, study, screen, list_id, line, anchor, genotype, fdr_from):
    d = _orcs(raw, study, screen)
    d = d[d.OFFICIAL_SYMBOL.notna()]
    pos = ((d.HIT == "YES") & (d["SCORE.1"] < 0)).to_numpy()
    fdr = d["SCORE.3"] if fdr_from == "score3" else 10 ** -d["SCORE.3"]
    return _out(
        study,
        list_id,
        line,
        anchor,
        genotype,
        d.OFFICIAL_SYMBOL,
        d["SCORE.1"],
        fdr,
        _bin(pos),
        "",
    )


def read_Park2023_WT_etoposide(raw: Path) -> pd.DataFrame:
    return _orcs_call(
        raw,
        "Park2023",
        2082,
        "Park2023|HEK293-A|etoposide",
        "HEK293-A",
        "drug:etoposide",
        "etoposide 100 nM vs untreated, wild-type HEK293A",
        "score3",
    )


def read_Park2023_ZNF451KO_etoposide(raw: Path) -> pd.DataFrame:
    return _orcs_call(
        raw,
        "Park2023",
        2083,
        "Park2023|HEK293-A-ZNF451-KO|etoposide",
        "HEK293-A",
        "drug:etoposide",
        "etoposide 20 nM vs untreated, ZNF451-KO HEK293A",
        "score3",
    )


_TUANO = {  # ORCS screen: (subline, platform, olaparib dose)
    2292: ("HMLE", "CRISPRn", "5 uM"),
    2293: ("mesHMLE", "CRISPRn", "5 uM"),
    2294: ("B80-T17", "CRISPRn", "5 uM"),
    2295: ("B80-T5", "CRISPRn", "2.5 uM"),
    2296: ("K5+/K19-", "CRISPRn", "5 uM"),
    2297: ("HMLE", "CRISPRi", "5 uM"),
    2298: ("mesHMLE", "CRISPRi", "5 uM"),
    2299: ("B80-T17", "CRISPRi", "5 uM"),
    2300: ("B80-T5", "CRISPRi", "2.5 uM"),
    2301: ("K5+/K19-", "CRISPRi", "5 uM"),
}


def _tuano(raw, screen):
    sub, plat, dose = _TUANO[screen]
    return _orcs_call(
        raw,
        "Tuano2023",
        screen,
        f"Tuano2023|HMEC-{sub}|olaparib|{plat}",
        f"HMEC ({sub})",
        "drug:olaparib",
        f"olaparib {dose} vs untreated, {plat} ({sub})",
        "neglog10",
    )


def read_Tuano2023_2292(raw):
    return _tuano(raw, 2292)


def read_Tuano2023_2293(raw):
    return _tuano(raw, 2293)


def read_Tuano2023_2294(raw):
    return _tuano(raw, 2294)


def read_Tuano2023_2295(raw):
    return _tuano(raw, 2295)


def read_Tuano2023_2296(raw):
    return _tuano(raw, 2296)


def read_Tuano2023_2297(raw):
    return _tuano(raw, 2297)


def read_Tuano2023_2298(raw):
    return _tuano(raw, 2298)


def read_Tuano2023_2299(raw):
    return _tuano(raw, 2299)


def read_Tuano2023_2300(raw):
    return _tuano(raw, 2300)


def read_Tuano2023_2301(raw):
    return _tuano(raw, 2301)


READERS = {
    k: v for k, v in dict(globals()).items() if k.startswith("read_") and callable(v)
}
