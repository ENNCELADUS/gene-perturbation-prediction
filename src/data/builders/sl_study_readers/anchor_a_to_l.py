"""Readers of the anchor-screen studies Adhikari2018 to Liu2019, one per ready row of
configs/benchmarks/sl_label_rules.csv: read_<study_id>[_<suffix>](raw) -> DataFrame,
where raw is the data root. Each applies the rule stored in RULES[<function name>] (the
rules file copies these strings) with DataFrame.query.
"""

import io
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

# Library control elements, never screened genes.
_CONTROL = re.compile(
    r"(?i)^(non[-_ ]?targeting|nontargeting|ntc|ctrl|control|lacz|egfp|gfp|"
    r"luciferase|luc|aavs1|safe[-_ ]?harbou?r|intergenic|empty)([-_ ].*)?$"
)

# Rule strings: pandas query over the named fields of each reader's frame.
RULES = {}


def _rule(name, positive, negative, unlabelled=""):
    RULES[name] = {
        "positive_rule": positive,
        "negative_rule": negative,
        "unlabelled_rule": unlabelled,
    }


def _need(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what}: missing columns {missing}; found {list(df.columns)[:20]}"
        )


def _sheet(path, sheet, header=0):
    xl = pd.ExcelFile(path, engine="openpyxl")
    if sheet not in xl.sheet_names:
        raise ValueError(f"{path}: missing sheet {sheet!r}; found {xl.sheet_names}")
    return xl.parse(sheet, header=header)


def _zip_member(path, member):
    z = zipfile.ZipFile(path)
    if member not in z.namelist():
        raise ValueError(f"{path}: missing member {member!r}")
    return io.BytesIO(z.read(member))


def _drop_controls(df, col):
    """Drop control elements and cells that are not text (Excel turned some symbols
    into dates)."""
    df = df[df[col].map(lambda v: isinstance(v, str))].copy()
    df[col] = df[col].str.strip()
    return df[~df[col].str.match(_CONTROL)]


def _unique(df, col):
    """Drop every row of a symbol listed more than once (ambiguous score)."""
    return df[~df[col].duplicated(keep=False)]


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(
            df[c].mask(df[c].astype(str).str.strip() == "-"), errors="coerce"
        )
    return df


def _label(
    df,
    name,
    study_id,
    list_id,
    line,
    anchor_gene,
    anchor_genotype,
    gene_a,
    gene_b,
    score,
    fdr,
    unlabelled_reason="",
    gene_a_col=None,
):
    """Apply RULES[name] to df. gene_b/score/fdr are column names; gene_a is a constant
    unless gene_a_col is given."""
    r = RULES[name]
    # a symbol listed more than once in one list is ambiguous: drop every copy
    keys = [gene_a_col, gene_b] if gene_a_col else [gene_b]
    df = df[~df.duplicated(keys, keep=False)]
    pos = df.index.isin(df.query(r["positive_rule"]).index)
    neg = df.index.isin(df.query(r["negative_rule"]).index)
    unl = (
        df.index.isin(df.query(r["unlabelled_rule"]).index)
        if r["unlabelled_rule"]
        else np.zeros(len(df), bool)
    )
    if (pos & neg).any() or (pos & unl).any() or (neg & unl).any():
        raise ValueError(f"{name}: rules overlap")
    label = pd.array(np.where(pos, 1, np.where(neg, 0, -1)), dtype="Int8")
    label[~(pos | neg)] = pd.NA
    out = pd.DataFrame(
        {
            "study_id": study_id,
            "list_id": list_id,
            "line": line,
            "anchor_gene": anchor_gene,
            "anchor_genotype": anchor_genotype,
            "gene_a": df[gene_a_col].to_numpy() if gene_a_col else gene_a,
            "gene_b": df[gene_b].to_numpy(),
            "score": pd.to_numeric(df[score], errors="coerce").astype(float).to_numpy(),
            "fdr": pd.to_numeric(df[fdr], errors="coerce").astype(float).to_numpy()
            if fdr
            else np.nan,
            "label": label,
            "unlabelled_reason": np.where(
                unl, unlabelled_reason, np.where(pos | neg, "", "not covered by rule")
            ),
        }
    )
    out = out[out.label.notna() | unl]  # rows outside every rule are not units
    return out[COLUMNS].reset_index(drop=True)


def _orcs(raw, study, screen):
    p = (
        raw
        / "sl_studies/raw"
        / study
        / f"BIOGRID-ORCS-SCREEN_{screen}-2.0.18.screen.tab.txt"
    )
    d = pd.read_csv(p, sep="\t", dtype=str)
    _need(d, ["OFFICIAL_SYMBOL", "SCORE.1", "SCORE.2", "SCORE.3", "HIT"], str(p))
    d = _num(d, ["SCORE.1", "SCORE.2", "SCORE.3", "SCORE.4", "SCORE.5"])
    return _unique(_drop_controls(d, "OFFICIAL_SYMBOL"), "OFFICIAL_SYMBOL")


# Alborzinia2023 (SLC-focused KO screens)
_rule(
    "read_Alborzinia2023_A375",
    "`neg|fdr` < 0.05 and `neg|lfc` < 0",
    "not (`neg|fdr` < 0.05 and `neg|lfc` < 0)",
)
_rule(
    "read_Alborzinia2023_HT1080",
    "`neg|fdr` < 0.05 and `neg|lfc` < 0",
    "not (`neg|fdr` < 0.05 and `neg|lfc` < 0)",
)


def _alborzinia(raw, name, sheet, line):
    d = _sheet(raw / "sl_studies/raw/Alborzinia2023/EMMM-15-e18014-s002.xlsx", sheet)
    _need(d, ["symbol", "neg|fdr", "neg|lfc"], sheet)
    d = _num(_drop_controls(d, "symbol"), ["neg|fdr", "neg|lfc"])
    return _label(
        d,
        name,
        "Alborzinia2023",
        f"Alborzinia2023|{line}|LRP8",
        line,
        "LRP8",
        "LRP8-KO vs LRP8-WT",
        "LRP8",
        "symbol",
        "neg|lfc",
        "neg|fdr",
    )


def read_Alborzinia2023_A375(raw: Path) -> pd.DataFrame:
    return _alborzinia(raw, "read_Alborzinia2023_A375", "A375 LRP8-KO vs. WT", "A375")


def read_Alborzinia2023_HT1080(raw: Path) -> pd.DataFrame:
    return _alborzinia(
        raw, "read_Alborzinia2023_HT1080", "HT1080 LRP8-KO vs. WT", "HT1080"
    )


# Amitzi2024 (HAP1 DDX11-KO)
_rule(
    "read_Amitzi2024",
    "`PIScore` < -0.4 and `FDR` <= 0.2",
    "not (`PIScore` < -0.4 and `FDR` <= 0.2)",
)


def read_Amitzi2024(raw: Path) -> pd.DataFrame:
    f = _zip_member(
        raw / "sl_studies/raw/Amitzi2024/jkae052_supplementary_data.zip",
        "Table_S1_G3-2023-404784.xlsx",
    )
    d = pd.read_excel(f, sheet_name="PI scores", engine="openpyxl")
    _need(d, ["Gene", "PIScore", "FDR"], "Amitzi2024 Table S1")
    d = _num(_drop_controls(d, "Gene"), ["PIScore", "FDR"])
    return _label(
        d,
        "read_Amitzi2024",
        "Amitzi2024",
        "Amitzi2024|HAP1|DDX11",
        "HAP1",
        "DDX11",
        "DDX11-KO vs parental",
        "DDX11",
        "Gene",
        "PIScore",
        "FDR",
    )


# Aregger2020 (HAP1 query-KO qGI screens)
_AREGGER = {  # suffix: (qGI column, FDR column, anchor gene, genotype)
    "FASN_a": ("qGI_FASN_a", "fdr_FASN_a", "FASN", "FASN-KO vs parental, screen a"),
    "FASN_b": ("qGI_FASN_b", "fdr_FASN_b", "FASN", "FASN-KO vs parental, screen b"),
    "FASN_c": ("qGI_FASN_c", "fdr_FASN_c", "FASN", "FASN-KO vs parental, screen c"),
    "ACACA": ("ACACA", "fdr_ACACA", "ACACA", "ACACA-KO vs parental"),
    "LDLR": ("LDLR", "fdr_LDLR", "LDLR", "LDLR-KO vs parental"),
    "SREBF1": ("SREBF1", "fdr_SREBF1", "SREBF1", "SREBF1-KO vs parental"),
    "SREBF2": ("SREBF2", "fdr_SREBF2", "SREBF2", "SREBF2-KO vs parental"),
    "C12orf49": ("C12orf49", "fdr_C12orf49", "C12orf49", "C12orf49-KO vs parental"),
}
for _s, (_q, _f, _a, _g) in _AREGGER.items():
    _rule(
        f"read_Aregger2020_{_s}",
        f"`{_q}` < -0.5 and `{_f}` < 0.5",
        f"not (`{_q}` < -0.5 and `{_f}` < 0.5)",
    )


def _aregger(raw, suffix):
    q, f, anchor, genotype = _AREGGER[suffix]
    p = (
        raw
        / "sl_studies/raw/Aregger2020/NIHMS1587165-supplement-1587165_Sup_Tab_2.xlsx"
    )
    s = _sheet(p, "qGI scores", header=1)
    fd = _sheet(p, "FDR", header=0)
    _need(s, ["gene", q], "Aregger2020 qGI scores")
    _need(fd, ["gene", f], "Aregger2020 FDR")
    d = s[["gene", q]].merge(
        fd[["gene", f]], on="gene", how="inner", validate="one_to_one"
    )
    d = _num(_drop_controls(d, "gene"), [q, f])
    return _label(
        d,
        f"read_Aregger2020_{suffix}",
        "Aregger2020",
        f"Aregger2020|HAP1|{suffix}",
        "HAP1",
        anchor,
        genotype,
        anchor,
        "gene",
        q,
        f,
    )


def read_Aregger2020_FASN_a(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "FASN_a")


def read_Aregger2020_FASN_b(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "FASN_b")


def read_Aregger2020_FASN_c(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "FASN_c")


def read_Aregger2020_ACACA(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "ACACA")


def read_Aregger2020_LDLR(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "LDLR")


def read_Aregger2020_SREBF1(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "SREBF1")


def read_Aregger2020_SREBF2(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "SREBF2")


def read_Aregger2020_C12orf49(raw: Path) -> pd.DataFrame:
    return _aregger(raw, "C12orf49")


# Baxter2023 (MCF10A FBXW7-17 and FBXW7-24 clones)
_rule(
    "read_Baxter2023",
    "`HIT_2086` == 'YES' and `HIT_2087` == 'YES'",
    "not (`HIT_2086` == 'YES' and `HIT_2087` == 'YES')",
)


def read_Baxter2023(raw: Path) -> pd.DataFrame:
    a = _orcs(raw, "Baxter2023", 2086)[["OFFICIAL_SYMBOL", "SCORE.1", "HIT"]]
    b = _orcs(raw, "Baxter2023", 2087)[["OFFICIAL_SYMBOL", "SCORE.1", "HIT"]]
    d = a.merge(
        b,
        on="OFFICIAL_SYMBOL",
        how="inner",
        suffixes=("_2086", "_2087"),
        validate="one_to_one",
    )
    d["mean_Z"] = d[["SCORE.1_2086", "SCORE.1_2087"]].mean(axis=1)
    return _label(
        d,
        "read_Baxter2023",
        "Baxter2023",
        "Baxter2023|MCF-10A|FBXW7",
        "MCF-10A",
        "FBXW7",
        "FBXW7-mutant clones FBXW7-17 and FBXW7-24 vs FBXW7-WT (MCF10A TP53-mutant)",
        "FBXW7",
        "OFFICIAL_SYMBOL",
        "mean_Z",
        None,
    )


# Benslimane2021 (NALM-6)
_rule(
    "read_Benslimane2021_TAPR1",
    "`Rank` <= 180 and `GI score` < 0",
    "not (`Rank` <= 180 and `GI score` < 0)",
)
_rule(
    "read_Benslimane2021_BIBR1532",
    "`RANKS_score` < -2 and `FDR` < 0.1",
    "not (`RANKS_score` < -2 and `FDR` < 0.1)",
)


def read_Benslimane2021_TAPR1(raw: Path) -> pd.DataFrame:
    d = _sheet(raw / "sl_studies/raw/Benslimane2021/ACEL-20-e13331-s005.xlsx", "Sheet1")
    _need(d, ["Gene", "GI score", "Rank"], "Benslimane2021 s005")
    d = _num(_drop_controls(d, "Gene"), ["GI score", "Rank"])
    if (
        len(d) // 100 != 180
    ):  # the rule's 180 is the bottom 1% of this table's 18,055 genes
        raise ValueError(f"Benslimane2021 s005: {len(d)} genes, rule expects 18,0xx")
    return _label(
        d,
        "read_Benslimane2021_TAPR1",
        "Benslimane2021",
        "Benslimane2021|NALM-6|HAPSTR1",
        "NALM-6",
        "HAPSTR1",
        "TAPR1 (HAPSTR1)-KO clones #1 and #4 vs WT",
        "HAPSTR1",
        "Gene",
        "GI score",
        None,
    )


def read_Benslimane2021_BIBR1532(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Benslimane2021/ACEL-20-e13331-s003.xlsx",
        "BIBR1532_20µM_RANKS",
    )
    _need(d, ["Gene", "RANKS_score", "FDR"], "Benslimane2021 s003")
    d = _num(_drop_controls(d, "Gene"), ["RANKS_score", "FDR"])
    return _label(
        d,
        "read_Benslimane2021_BIBR1532",
        "Benslimane2021",
        "Benslimane2021|NALM-6|TERT_BIBR1532",
        "NALM-6",
        "TERT",
        "BIBR1532 20 uM vs DMSO (telomerase inhibitor; drug vs vehicle)",
        "TERT",
        "Gene",
        "RANKS_score",
        "FDR",
    )


# Chai2020 (U937 C9ORF72-KO)
_rule(
    "read_Chai2020",
    "`10% FDR` == 1 and `Combo casTLE Effect` < 0",
    "not (`10% FDR` == 1 and `Combo casTLE Effect` < 0)",
)


def read_Chai2020(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Chai2020/1-s2.0-S0006899319306559-mmc2.xlsx",
        "Genome-wide con vs C9KO screen",
    )
    _need(d, ["Symbol", "Combo casTLE Effect", "10% FDR"], "Chai2020 mmc2")
    d = _num(_drop_controls(d, "Symbol"), ["Combo casTLE Effect", "10% FDR"])
    return _label(
        d,
        "read_Chai2020",
        "Chai2020",
        "Chai2020|U937|C9orf72",
        "U937",
        "C9orf72",
        "C9ORF72-KO vs control, PMA-differentiated",
        "C9orf72",
        "Symbol",
        "Combo casTLE Effect",
        None,
    )


# Clements2020 (HeLa, olaparib vs DMSO)
_CLE = "`FDR negative selection` < 0.05 and `Log2 fold change negative selection` < 0"
_rule("read_Clements2020_WT", _CLE, f"not ({_CLE})")
_rule("read_Clements2020_BRCA2KO", _CLE, f"not ({_CLE})")


def _clements(raw, name, fname, line_genotype, list_id):
    d = _sheet(raw / "sl_studies/raw/Clements2020" / fname, "Gene rank")
    cols = [
        "Gene name",
        "FDR negative selection",
        "Log2 fold change negative selection",
    ]
    _need(d, cols, fname)
    d = _num(_drop_controls(d, "Gene name"), cols[1:])
    return _label(
        d,
        name,
        "Clements2020",
        list_id,
        "HeLa",
        "PARP1",
        line_genotype,
        "PARP1",
        "Gene name",
        "Log2 fold change negative selection",
        "FDR negative selection",
    )


def read_Clements2020_WT(raw: Path) -> pd.DataFrame:
    return _clements(
        raw,
        "read_Clements2020_WT",
        "41467_2020_19961_MOESM4_ESM.xlsx",
        "olaparib 5 uM vs DMSO in parental HeLa (PARP inhibitor; drug vs vehicle)",
        "Clements2020|HeLa|PARP1_olaparib",
    )


def read_Clements2020_BRCA2KO(raw: Path) -> pd.DataFrame:
    return _clements(
        raw,
        "read_Clements2020_BRCA2KO",
        "41467_2020_19961_MOESM5_ESM.xlsx",
        "olaparib 4 uM vs DMSO in HeLa BRCA2-KO (PARP inhibitor; drug vs vehicle)",
        "Clements2020|HeLa_BRCA2KO|PARP1_olaparib",
    )


# Drainas2020 (hTERT-RPE1 TP53-KO vs WT, 2D)
_rule(
    "read_Drainas2020",
    "`cell_specificity_fdr` < 0.05 and `TP53ko_vsWT_fc` < 1",
    "not (`cell_specificity_fdr` < 0.05 and `TP53ko_vsWT_fc` < 1)",
)


def read_Drainas2020(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Drainas2020/mmc3.xlsx", "ST2_181212_RPEgrowth_TP53ko_vsW"
    )
    _need(d, ["gene_id", "cell_specificity_fdr", "TP53ko_vsWT_fc"], "Drainas2020 mmc3")
    d = _num(_drop_controls(d, "gene_id"), ["cell_specificity_fdr", "TP53ko_vsWT_fc"])
    return _label(
        d,
        "read_Drainas2020",
        "Drainas2020",
        "Drainas2020|hTERT-RPE1|TP53",
        "hTERT-RPE1",
        "TP53",
        "TP53-KO vs WT, 2D growth",
        "TP53",
        "gene_id",
        "TP53ko_vsWT_fc",
        "cell_specificity_fdr",
    )


# Feng2022 (HEK293A tumour-suppressor KO panel)
# suffix: (Table S5 column, Table S4 sheet, Table S2 column, anchor gene, genotype)
_FENG22 = {
    "ARID1A": (
        "ARID1A KD",
        "ARID1A KD",
        "293A_ARID1A KD",
        "ARID1A",
        "ARID1A knockdown vs WT",
    ),
    "BAP1": ("BAP1 KO", "BAP1 KO", "293A_BAP1 KO", "BAP1", "BAP1-KO vs WT"),
    "CDH1": ("CDH1 KO", "CDH1 KO", "293A_CDH1 KO", "CDH1", "CDH1-KO vs WT"),
    "KEAP1": ("KEAP1 KO", "KEAP1 KO", "293A_KEAP1 KO", "KEAP1", "KEAP1-KO vs WT"),
    "STK11": ("STK11 KO", "STK11 KO", "293A_LKB1 KO", "STK11", "STK11 (LKB1)-KO vs WT"),
    "NF1": ("NF1 KO", "NF1 KO", "293A_NF1 KO", "NF1", "NF1-KO vs WT"),
    "NF2": ("NF2 KO", "NF2 KO", "293A_NF2 KO", "NF2", "NF2-KO vs WT"),
    "PBRM1": ("PBRM1 KO", "PBRM1 KO", "293A_PBRM1 KO", "PBRM1", "PBRM1-KO vs WT"),
    "PTEN": ("PTEN KO", "PTEN KO", "293A_PTEN KO", "PTEN", "PTEN-KO vs WT"),
    "RB1": ("RB1 KO", "RB1 KO", "293A_RB1 KO", "RB1", "RB1-KO vs WT"),
    "TP53": ("TP53 KO", "TP53 KO", "293A_TP53 KO", "TP53", "TP53-KO vs WT"),
    "VHL": ("VHL KO", "VHL KO", "293A_VHL KO", "VHL", "VHL-KO vs WT"),
    "TP53BP1": (
        "TP53BP1 KO",
        "TP53BP1 KO",
        "293A_TP53BP1 KO",
        "TP53BP1",
        "TP53BP1-KO vs WT",
    ),
}
_FENG22_WT = ["293A_WT_XF804", "293A_WT_XF646", "293A_WT_XF821", "293A_WT_XF498"]
for _s in _FENG22:
    _rule(
        f"read_Feng2022_{_s}",
        "`in_table_S5` == 1",
        "`in_table_S5` == 0 and `wt_essential_calls` < 3",
        "`in_table_S5` == 0 and `wt_essential_calls` >= 3",
    )


def _feng2022(raw, suffix):
    s5col, s4sheet, s2col, anchor, genotype = _FENG22[suffix]
    zp = raw / "sl_studies/raw/Feng2022/sciadv.abm6638_tables_s1_to_s7.zip"
    s2 = pd.read_excel(
        _zip_member(zp, "sciadv.abm6638_table_s2.xlsx"), engine="openpyxl"
    )
    s2.columns = [str(c).strip() for c in s2.columns]
    _need(s2, ["GENE", s2col] + _FENG22_WT, "Feng2022 Table S2")
    s4 = pd.read_excel(
        _zip_member(zp, "sciadv.abm6638_table_s4.xlsx"),
        sheet_name=None,
        engine="openpyxl",
    )
    if s4sheet not in s4:
        raise ValueError(
            f"Feng2022 Table S4: missing sheet {s4sheet!r}; found {list(s4)}"
        )
    s4 = s4[s4sheet]
    _need(s4, ["GENE", "normZ", "fdr_synth"], f"Feng2022 Table S4 {s4sheet}")
    s5 = pd.read_excel(
        _zip_member(zp, "sciadv.abm6638_table_s5.xlsx"), engine="openpyxl"
    )
    s5.columns = [str(c).strip() for c in s5.columns]
    _need(s5, [s5col], "Feng2022 Table S5")
    hits = set(s5[s5col].dropna().astype(str).str.strip())
    d = _drop_controls(s2[["GENE"] + _FENG22_WT], "GENE")
    d["wt_essential_calls"] = d[_FENG22_WT].sum(axis=1)
    d["in_table_S5"] = d["GENE"].isin(hits).astype(int)
    if d["in_table_S5"].sum() != len(hits):
        raise ValueError(
            f"Feng2022 {suffix}: {len(hits - set(d.GENE))} Table S5 genes "
            "not in Table S2"
        )
    s4 = _unique(_drop_controls(s4[["GENE", "normZ", "fdr_synth"]], "GENE"), "GENE")
    d = d.merge(s4, on="GENE", how="left", validate="one_to_one")
    return _label(
        d,
        f"read_Feng2022_{suffix}",
        "Feng2022",
        f"Feng2022|HEK293A|{anchor}",
        "HEK293-A",
        anchor,
        genotype,
        anchor,
        "GENE",
        "normZ",
        "fdr_synth",
        unlabelled_reason="essential in WT HEK293A (>=3 of 4 WT screens)",
    )


def read_Feng2022_ARID1A(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "ARID1A")


def read_Feng2022_BAP1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "BAP1")


def read_Feng2022_CDH1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "CDH1")


def read_Feng2022_KEAP1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "KEAP1")


def read_Feng2022_STK11(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "STK11")


def read_Feng2022_NF1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "NF1")


def read_Feng2022_NF2(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "NF2")


def read_Feng2022_PBRM1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "PBRM1")


def read_Feng2022_PTEN(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "PTEN")


def read_Feng2022_RB1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "RB1")


def read_Feng2022_TP53(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "TP53")


def read_Feng2022_VHL(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "VHL")


def read_Feng2022_TP53BP1(raw: Path) -> pd.DataFrame:
    return _feng2022(raw, "TP53BP1")


# Feng2024_2 (RPE1-hTERT TP53-/- FANCM-KO clone 3)
_rule(
    "read_Feng2024_2_FANCM_clone3",
    "`CCA (clone3)` > 1 and `RPE1_dFANCM_clone3_BAGEL_T18` > 10",
    "not (`CCA (clone3)` > 1 and `RPE1_dFANCM_clone3_BAGEL_T18` > 10)",
)


def read_Feng2024_2_FANCM_clone3(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Feng2024_2/biorxiv_DC2_TableS2.xlsx",
        "FANCM KO SL screen",
        header=1,
    )
    cols = ["Gene", "CCA (clone3)", "RPE1_dFANCM_clone3_BAGEL_T18"]
    _need(d, cols, "Feng2024_2 Table S2")
    d = _num(_drop_controls(d, "Gene"), cols[1:])
    d["CCA (clone3)"] = d["CCA (clone3)"].where(
        d["CCA (clone3)"] > -9999
    )  # -9999.88 is a missing-value sentinel
    return _label(
        d,
        "read_Feng2024_2_FANCM_clone3",
        "Feng2024_2",
        "Feng2024_2|RPE1|FANCM_clone3",
        "RPE1",
        "FANCM",
        "FANCM-/- clone 3 vs parental RPE1-hTERT TP53-/- PACKO Cas9",
        "FANCM",
        "Gene",
        "CCA (clone3)",
        None,
    )


# Fielden2025 (dual-guide CRISPRi, combinatorial)
_rule("read_Fielden2025_RPE1", "`sens.score` <= -1", "`sens.score` > -1")


def _pairs(d, col):
    g = d[col].astype(str).str.split(";", expand=True)
    if g.shape[1] != 2 or g.isna().any().any():
        raise ValueError(f"Fielden2025: {col} is not 'geneA;geneB'")
    d = d.assign(gene_a=g[0].str.strip(), gene_b=g[1].str.strip())
    return d[~d.gene_a.str.match(_CONTROL) & ~d.gene_b.str.match(_CONTROL)]


def read_Fielden2025_RPE1(raw: Path) -> pd.DataFrame:
    d = pd.read_csv(raw / "sl_studies/raw/Fielden2025/41586_2025_8815_MOESM5_ESM.csv")
    _need(d, ["gene_combination", "sens.score"], "Fielden2025 MOESM5")
    d = _num(_pairs(d, "gene_combination"), ["sens.score"])
    return _label(
        d,
        "read_Fielden2025_RPE1",
        "Fielden2025",
        "Fielden2025|RPE-1",
        "RPE-1",
        "",
        "",
        "",
        "gene_b",
        "sens.score",
        None,
        gene_a_col="gene_a",
    )


# Hill2019 (NALM-6 UBE2R1/UBE2R2 KO, EKO library)
_HILL = {  # suffix: (ORCS screen, anchor gene, genotype)
    "1576_UBE2R1": (1576, "CDC34", "UBE2R1 (CDC34)-KO vs control populations"),
    "1577_UBE2R2": (1577, "UBE2R2", "UBE2R2-KO vs control populations, screen 1577"),
    "1578_UBE2R2": (1578, "UBE2R2", "UBE2R2-KO vs control populations, screen 1578"),
    "1579_UBE2R1_UBE2R2": (
        1579,
        "CDC34;UBE2R2",
        "UBE2R1/UBE2R2 double KO vs control populations, replicate 1579",
    ),
    "1580_UBE2R1_UBE2R2": (
        1580,
        "CDC34;UBE2R2",
        "UBE2R1/UBE2R2 double KO vs control populations, replicate 1580",
    ),
}
for _s in _HILL:
    _rule(
        f"read_Hill2019_{_s}",
        "`SCORE.3` < 0.05 and `SCORE.1` < 0",
        "not (`SCORE.3` < 0.05 and `SCORE.1` < 0)",
    )


def _hill(raw, suffix):
    screen, anchor, genotype = _HILL[suffix]
    d = _orcs(raw, "Hill2019", screen)
    return _label(
        d,
        f"read_Hill2019_{suffix}",
        "Hill2019",
        f"Hill2019|NALM-6|{suffix}",
        "NALM-6",
        anchor,
        genotype,
        anchor,
        "OFFICIAL_SYMBOL",
        "SCORE.1",
        "SCORE.3",
    )


def read_Hill2019_1576_UBE2R1(raw: Path) -> pd.DataFrame:
    return _hill(raw, "1576_UBE2R1")


def read_Hill2019_1577_UBE2R2(raw: Path) -> pd.DataFrame:
    return _hill(raw, "1577_UBE2R2")


def read_Hill2019_1578_UBE2R2(raw: Path) -> pd.DataFrame:
    return _hill(raw, "1578_UBE2R2")


def read_Hill2019_1579_UBE2R1_UBE2R2(raw: Path) -> pd.DataFrame:
    return _hill(raw, "1579_UBE2R1_UBE2R2")


def read_Hill2019_1580_UBE2R1_UBE2R2(raw: Path) -> pd.DataFrame:
    return _hill(raw, "1580_UBE2R1_UBE2R2")


# Jung2022 (Jurkat FANCD2-/- vs WT, metabolism library)
_rule(
    "read_Jung2022",
    "`adjusted p value` < 0.05 and `Direction` == 'Down'",
    "not (`adjusted p value` < 0.05 and `Direction` == 'Down')",
)


def read_Jung2022(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Jung2022/biorxiv_DC1_TableS1.xlsx", "Sheet1", header=2
    )
    _need(d, ["Gene", "Direction", "p value", "adjusted p value"], "Jung2022 Table S1")
    d = _num(_drop_controls(d, "Gene"), ["p value", "adjusted p value"])
    if not d.Direction.isin(["Up", "Down"]).all():
        raise ValueError("Jung2022 Table S1: Direction outside Up/Down")
    # signed score: log10 p, negative for depletion in FANCD2-/- (Down), positive for Up
    d["signed_log10p"] = np.log10(d["p value"].clip(lower=1e-300)) * np.where(
        d.Direction == "Down", 1, -1
    )
    return _label(
        d,
        "read_Jung2022",
        "Jung2022",
        "Jung2022|Jurkat|FANCD2",
        "Jurkat",
        "FANCD2",
        "FANCD2-/- vs WT",
        "FANCD2",
        "Gene",
        "signed_log10p",
        "adjusted p value",
    )


# Liang2020 (PLC/PRF/5 ATRX-KO vs WT)
_rule("read_Liang2020", "`label` == 'synthetic lethal'", "`label` == 'other gene'")


def read_Liang2020(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw
        / "sl_studies/raw/Liang2020"
        / "00085472can183374-sup-211068_3_supp_5741479_pwz5ml.xlsx",
        "Sheet1",
        header=12,
    )
    _need(d, ["Gene", "KO|beta", "KO|fdr", "label"], "Liang2020 Table S1")
    d = _num(_drop_controls(d, "Gene"), ["KO|beta", "KO|fdr"])
    if not d.label.isin(["synthetic lethal", "other gene"]).all():
        raise ValueError(
            "Liang2020: unexpected labels "
            f"{set(d.label) - {'synthetic lethal', 'other gene'}}"
        )
    return _label(
        d,
        "read_Liang2020",
        "Liang2020",
        "Liang2020|PLC/PRF/5|ATRX",
        "PLC/PRF/5",
        "ATRX",
        "ATRX-KO vs WT",
        "ATRX",
        "Gene",
        "KO|beta",
        "KO|fdr",
    )


# Liu2019 (WA09 hESC TP53-KO, cisplatin vs DMF)
_rule(
    "read_Liu2019",
    "`negative selection p-value` < 0.01",
    "`negative selection p-value` >= 0.01",
)


def read_Liu2019(raw: Path) -> pd.DataFrame:
    d = _sheet(
        raw / "sl_studies/raw/Liu2019/NIHMS1524158-supplement-5.xlsx",
        "Cisplatin_DMF_gene_summary.txt",
    )
    _need(
        d,
        ["id", "negative selection score", "negative selection p-value"],
        "Liu2019 supplement 5",
    )
    d = _num(
        _drop_controls(d, "id"),
        ["negative selection score", "negative selection p-value"],
    )
    return _label(
        d,
        "read_Liu2019",
        "Liu2019",
        "Liu2019|WA09|TP53_cisplatin",
        "WA09",
        "TP53",
        "TP53-KO; cisplatin 200 nM vs DMF vehicle (drug vs vehicle)",
        "TP53",
        "id",
        "negative selection score",
        None,
    )


READERS = [n for n in RULES]
