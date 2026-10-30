"""Readers of the SLKB-deposited studies and of the combinatorial screens Chan2025 to
Flister2025, one per ready row of configs/benchmarks/sl_label_rules.csv. Each takes the
data root and returns one row per labelled unit with the columns in COLS, applying its
row's rule.
"""

import functools
import io
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

COLS = [
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
S = Path("sl_studies/raw")


def _need(df, cols, what):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{what}: missing columns {miss}")


def _excel(path, sheet, header):
    if not Path(path).exists():
        raise ValueError(f"missing file {path}")
    try:
        return pd.read_excel(path, sheet_name=sheet, header=header)
    except ValueError as e:  # missing sheet
        raise ValueError(f"{path}: {e}") from e


def _zip_member(path, member):
    with zipfile.ZipFile(path) as z:
        if member not in z.namelist():
            raise ValueError(f"{path}: no member {member}")
        return z.read(member)


def _mac_tsv(data):
    # Najm 2018 tables use old Mac line ends (CR only).
    return pd.read_csv(
        io.BytesIO(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")), sep="\t"
    )


def _frame(
    study,
    line,
    gene_a,
    gene_b,
    score,
    fdr,
    label,
    reason="",
    anchor_gene="",
    anchor_genotype="",
):
    n = len(gene_a)
    df = pd.DataFrame(
        {
            "study_id": study,
            "list_id": f"{study}|{line}"
            + (f"|{anchor_gene}:{anchor_genotype}" if anchor_gene else ""),
            "line": str(line),
            "anchor_gene": anchor_gene,
            "anchor_genotype": anchor_genotype,
            "gene_a": list(gene_a),
            "gene_b": list(gene_b),
            "score": pd.to_numeric(pd.Series(list(score)), errors="coerce").astype(
                float
            ),
            "fdr": pd.to_numeric(pd.Series(list(fdr)), errors="coerce").astype(float)
            if fdr is not None
            else float("nan"),
            "label": pd.array(list(label), dtype="Int8"),
            "unlabelled_reason": list(reason)
            if not isinstance(reason, str)
            else [reason] * n,
        }
    )
    df.loc[df.label.notna(), "unlabelled_reason"] = ""
    return df[COLS]


def _concat(frames):
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["list_id", "gene_a", "gene_b"]).any():
        raise ValueError("duplicate units within a list")
    return out


@functools.lru_cache(maxsize=1)
def _slkb_original(zip_path):
    """SLKB cdko_original_sl_results: the study-reported scores and calls as SLKB
    transcribed them."""
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE cdko_original_sl_results (id INTEGER, gene_pair_id INTEGER,"
        " gene_pair TEXT, study_origin TEXT, cell_line_origin TEXT, gene_1 TEXT,"
        " gene_2 TEXT, SL_or_not TEXT, SL_score REAL, statistical_score REAL,"
        " SL_score_cutoff REAL, statistical_score_cutoff REAL)"
    )
    prefix = "INSERT INTO cdko_original_sl_results VALUES("
    batch = []
    with zipfile.ZipFile(zip_path) as z, z.open("SQL_Dumps/SLKB-sqlite3_dump.sql") as f:
        for raw_line in io.TextIOWrapper(f, encoding="utf-8"):
            if raw_line.startswith(prefix):
                batch.append(raw_line)
    if not batch:
        raise ValueError(f"{zip_path}: no cdko_original_sl_results rows")
    con.executescript("".join(batch))
    return pd.read_sql("SELECT * FROM cdko_original_sl_results", con)


def _slkb(raw, study, pmid, lines):
    t = _slkb_original(str(raw / S / "SLKB_deposited/SQL_Dumps.zip"))
    frames = []
    for line in lines:
        d = t[(t.study_origin == pmid) & (t.cell_line_origin == line)]
        if d.empty:
            raise ValueError(f"SLKB has no rows for {pmid} {line}")
        if not set(d.SL_or_not) <= {"SL", "Not SL"}:
            raise ValueError(f"unexpected SL_or_not values {set(d.SL_or_not)}")
        frames.append(
            _frame(
                study,
                line,
                d.gene_1,
                d.gene_2,
                d.SL_score,
                d.statistical_score,
                (d.SL_or_not == "SL").astype(int),
            )
        )
    return _concat(frames)


# SLKB studies


def read_Han2017(raw: Path) -> pd.DataFrame:
    d = _excel(raw / S / "Han2017/41587_2017_BFnbt3834_MOESM267_ESM.xlsx", 0, 0)
    _need(d, ["Gene_Pair", "GeneAB GIT score"], "Han2017 Table 4")
    qcol = [c for c in d.columns if str(c).startswith("q-values based on GIM")]
    if len(qcol) != 1:
        raise ValueError("Han2017: q-value column not found")
    top = (
        _excel(raw / S / "Han2017/41587_2017_BFnbt3834_MOESM277_ESM.xlsx", 0, 0)
        .iloc[:, 0]
        .dropna()
    )
    if len(top) != 30 or not top.astype(str).str.contains("__").all():
        raise ValueError("Han2017 Table 14: expected 30 gene pairs")
    g = d.Gene_Pair.astype(str).str.split("__", expand=True)
    d = d.assign(a=g[0], b=g[1])[g[0] != g[1]]  # drop same-gene pairs
    if not top.isin(d.Gene_Pair).all():
        raise ValueError("Han2017: a top-30 pair is not in Table 4")
    return _frame(
        "Han2017",
        "K562",
        d.a,
        d.b,
        d["GeneAB GIT score"],
        d[qcol[0]],
        d.Gene_Pair.isin(top).astype(int),
    )


def read_Horlbeck2018(raw: Path) -> pd.DataFrame:
    p = raw / S / "Horlbeck2018/1-s2.0-S0092867418307359-mmc5.xlsx"
    d = _excel(p, "gene GI scores and correlations", None)
    hdr = d.iloc[:3, :].astype(str).agg("|".join).tolist()
    want = {
        6: "K562|Replicate Average|GI score",
        12: "Jurkat|Replicate Average|GI score",
    }
    for i, h in want.items():
        if hdr[i] != h:
            raise ValueError(f"Horlbeck2018: column {i} is {hdr[i]}, expected {h}")
    d = d.iloc[4:]
    d = d[(d[0] != d[1]) & (d[0] != "negative") & (d[1] != "negative")]
    frames = []
    for col, line in [(6, "K562"), (12, "Jurkat")]:
        x = d[pd.to_numeric(d[col], errors="coerce").notna()]
        s = x[col].astype(float)
        frames.append(
            _frame("Horlbeck2018", line, x[0], x[1], s, None, (s < -3).astype(int))
        )
    return _concat(frames)


def read_Najm2018(raw: Path) -> pd.DataFrame:
    t = _mac_tsv(
        _zip_member(
            raw / S / "Najm2018/NIHMS925040-supplement-supp_tables.zip",
            "Supplementary Tables/Supplementary Table 4 SynLet FDRs.txt",
        )
    )
    _need(
        t,
        ["Gene 1", "Gene 2", "Group 1", "Group 2", "Source", "SynLet q-value"],
        "Najm2018 Table 4",
    )
    frames = []
    for line in ["786O", "A375", "A549", "HT29", "Meljuso", "OVCAR8"]:
        d = t[
            (t.Source == line)
            & (t["Gene 1"] != t["Gene 2"])
            & (t["Group 1"] != "Ctrl")
            & (t["Group 2"] != "Ctrl")
        ]
        d = d.assign(
            k=d[["Gene 1", "Gene 2"]].apply(lambda r: "|".join(sorted(r)), axis=1)
        ).drop_duplicates("k")
        if len(d) != 300:
            raise ValueError(f"Najm2018 {line}: {len(d)} pairs, expected 300")
        q = d["SynLet q-value"]
        frames.append(
            _frame(
                "Najm2018", line, d["Gene 1"], d["Gene 2"], q, q, (q < 0.05).astype(int)
            )
        )
    return _concat(frames)


def read_Najm2018_apoptosis(raw: Path) -> pd.DataFrame:
    t = _mac_tsv(
        _zip_member(
            raw / S / "Najm2018/NIHMS925040-supplement-supp_tables.zip",
            "Supplementary Tables/Supplementary Table 7 Apoptosis FDRs.txt",
        )
    )
    _need(
        t,
        [
            "Gene 1",
            "Gene 2",
            "Group 1",
            "Group 2",
            "Treatment",
            "Cell Line",
            "SynLet q-value",
        ],
        "Najm2018 Table 7",
    )
    frames = []
    for line in ["Meljuso", "OVCAR8"]:
        d = t[
            (t.Treatment == "no drug")
            & (t["Cell Line"] == line)
            & (t["Gene 1"] != t["Gene 2"])
            & (t["Group 1"] != "control")
            & (t["Group 2"] != "control")
        ]
        d = d.assign(
            k=d[["Gene 1", "Gene 2"]].apply(lambda r: "|".join(sorted(r)), axis=1)
        )
        if (d.groupby("k")["SynLet q-value"].nunique() > 1).any():
            raise ValueError("Najm2018 apoptosis: orientations disagree")
        d = d.drop_duplicates("k")
        q = d["SynLet q-value"]
        frames.append(
            _frame(
                "Najm2018_apoptosis",
                line,
                d["Gene 1"],
                d["Gene 2"],
                q,
                q,
                (q < 0.05).astype(int),
            )
        )
    # its own lists (list_id Najm2018_apoptosis|line) within the study Najm2018
    return _concat(frames).assign(study_id="Najm2018")


def read_Parrish2021(raw: Path) -> pd.DataFrame:
    d = _excel(raw / S / "Parrish2021/NIHMS1737332-supplement-5.xlsx", 0, 1)
    frames = []
    for line in ["PC9", "HeLa"]:
        _need(
            d,
            [
                "target1",
                "target2",
                f"{line}_GI_flag",
                f"{line}_GI_score",
                f"{line}_GI_fdr",
            ],
            "Parrish2021 S4",
        )
        frames.append(
            _frame(
                "Parrish2021",
                line,
                d.target1,
                d.target2,
                d[f"{line}_GI_score"],
                d[f"{line}_GI_fdr"],
                (d[f"{line}_GI_flag"] == f"SL_in_{line}").astype(int),
            )
        )
    return _concat(frames)


def read_Shen2017(raw: Path) -> pd.DataFrame:
    d = _excel(
        raw / S / "Shen2017/41592_2017_BFnmeth4225_MOESM4_ESM.xlsx",
        "Table_S2B_CRISPR-KO-assay",
        0,
    )
    h = _excel(
        raw / S / "Shen2017/41592_2017_BFnmeth4225_MOESM5_ESM.xlsx",
        "TableS3B_CRISPR-KO-hits",
        1,
    )
    _need(h, ["Interaction", "Hit_Cell_Line", "Interaction_type"], "Shen2017 S3B")
    d = d[d.gene_gene.notna()]
    h = h[h.Interaction_type == "Synthetic Lethal"]
    if not h.Interaction.isin(d.gene_gene).all():
        raise ValueError("Shen2017: a hit is not in Table S2B")
    frames = []
    for line in ["HeLa", "A549", "293T"]:
        _need(d, ["geneA", "geneB", f"{line}_Z", f"FDR_left_{line}"], "Shen2017 S2B")
        hit = h[
            h.Hit_Cell_Line.str.lower()
            .str.split(", ")
            .apply(lambda xs: line.lower() in xs)
        ].Interaction
        frames.append(
            _frame(
                "Shen2017",
                line,
                d.geneA,
                d.geneB,
                d[f"{line}_Z"],
                d[f"FDR_left_{line}"],
                d.gene_gene.isin(hit).astype(int),
            )
        )
    return _concat(frames)


def read_Tang2022(raw: Path) -> pd.DataFrame:
    return _slkb(raw, "Tang2022", "36060092", ["22RV1"])


def read_Thompson2021(raw: Path) -> pd.DataFrame:
    d = _excel(raw / S / "Thompson2021/41467_2021_21478_MOESM4_ESM.xlsx", "1", 2)
    hits = _excel(raw / S / "Thompson2021/41467_2021_21478_MOESM8_ESM.xlsx", "5", 3)
    d = d[d["Gene pair class"] != "Control pair (non-targeting controls paired)"]
    g = d["GENE PAIR"].astype(str).str.split("_", expand=True)
    if g.shape[1] != 2:
        raise ValueError("Thompson2021: GENE PAIR is not A_B")
    frames = []
    for line, passes in [
        ("A375", "Passes A375 filter?"),
        ("Mewo", "Passes the Mewo filter?"),
        ("RPE", "Passes RPE filter?"),
    ]:
        rra, t = f"{line}_D28_rra_fdr_low", f"{line}_D28_t_fdr_low"
        _need(d, [rra, t, passes], "Thompson2021 SD1")
        _need(hits, [line], "Thompson2021 SD5")
        pos = d["GENE PAIR"].isin(hits[line].dropna())
        if pos.sum() != hits[line].notna().sum():
            raise ValueError(f"Thompson2021 {line}: a hit is not in SD1")
        fdr = pd.concat(
            [
                pd.to_numeric(d[rra], errors="coerce"),
                pd.to_numeric(d[t], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1)
        unl = (d[passes] == "No") & ~pos
        lab = pos.astype(int).astype("Int8").mask(unl)
        k = fdr.notna()  # 5 pairs have no statistics in any line: not analysed
        why = unl.map(
            {
                True: (
                    "a gene is lethal alone (fails the authors' essential-gene filter)"
                ),
                False: "",
            }
        )
        frames.append(
            _frame(
                "Thompson2021", line, g[0][k], g[1][k], fdr[k], fdr[k], lab[k], why[k]
            )
        )
    return _concat(frames)


def read_Wong2016(raw: Path) -> pd.DataFrame:
    return _slkb(raw, "Wong2016", "26864203", ["OVCAR8"])


def read_Zhao2018(raw: Path) -> pd.DataFrame:
    frames = []
    for line, sheet in [("A549", "A549 pi scores"), ("HeLa", "HeLa pi socres")]:
        d = _excel(raw / S / "Zhao2018/1-s2.0-S1097276518300467-mmc4.xlsx", sheet, 0)
        top = _excel(
            raw / S / "Zhao2018/1-s2.0-S1097276518300467-mmc5.xlsx",
            f"{line} Top hits",
            0,
        )
        _need(d, ["gene_gene", "geneA", "geneB", "z", "FDR left"], f"Zhao2018 {sheet}")
        _need(top, ["gene_gene"], f"Zhao2018 {line} Top hits")
        pos = d.gene_gene.isin(top.gene_gene)
        if pos.sum() != len(top):
            raise ValueError(f"Zhao2018 {line}: a top hit is not in the pi table")
        unl = (d.z < -3) & ~pos
        lab = pos.astype(int).astype("Int8").mask(unl)
        frames.append(
            _frame(
                "Zhao2018",
                line,
                d.geneA,
                d.geneB,
                d.z,
                d["FDR left"],
                lab,
                unl.map(
                    {
                        True: (
                            "z < -3 but removed by the authors' RPKM < 0.15 "
                            "expression filter"
                        ),
                        False: "",
                    }
                ),
            )
        )
    return _concat(frames)


# combinatorial screens


def read_Chan2025(raw: Path) -> pd.DataFrame:
    p = raw / S / "Chan2025/41588_2025_2222_MOESM4_ESM.xlsx"
    d = _excel(p, "Supplementary Table 6", 2)
    hits = _excel(p, "Supplementary Table 7", 3)
    cols = [
        "gene_pair_id",
        "targetA",
        "targetB",
        "cell_line_label",
        "mean_norm_gi",
        "fdr",
        "targetA__is_single_depleted",
        "targetB__is_single_depleted",
    ]
    _need(d, cols, "Chan2025 ST6")
    _need(hits, ["gene_pair_id", "cell_line_label"], "Chan2025 ST7")
    d = d[d.cell_line_label.notna()].copy()
    d["line"] = d.cell_line_label.astype(str)
    key = set(zip(hits.gene_pair_id, hits.cell_line_label.astype(str)))
    pos = pd.Series([k in key for k in zip(d.gene_pair_id, d.line)], index=d.index)
    if pos.sum() != len(key):
        raise ValueError("Chan2025: an ST7 hit is not in ST6")
    single = (d.targetA__is_single_depleted == 1) | (d.targetB__is_single_depleted == 1)
    unl = single & ~pos
    d["lab"] = pos.astype(int).astype("Int8").mask(unl)
    d["why"] = unl.map(
        {True: "a gene is depleted alone (targetA/B__is_single_depleted)", False: ""}
    )
    return _concat(
        [
            _frame(
                "Chan2025",
                line,
                x.targetA,
                x.targetB,
                x.mean_norm_gi,
                x.fdr,
                x.lab,
                x.why,
            )
            for line, x in d.groupby("line", sort=False)
        ]
    )


def read_Dede2020(raw: Path) -> pd.DataFrame:
    d = pd.read_csv(
        raw / S / "Dede2020/13059_2020_2173_MOESM3_ESM.txt", sep="\t", index_col=0
    )
    _need(d, ["A549", "HT29", "OVCAR8"], "Dede2020 Table S2")
    g = d.index.to_series().str.split("_", expand=True)
    if g.shape[1] != 2:
        raise ValueError("Dede2020: pair name is not A_B")
    frames = []
    for line in ["A549", "HT29", "OVCAR8"]:
        s = d[line]
        k = s.notna()
        frames.append(
            _frame(
                "Dede2020", line, g[0][k], g[1][k], s[k], None, (s[k] < -3).astype(int)
            )
        )
    return _concat(frames)


EBRIGHT_LINES = [
    "BXPC3",
    "COLO205",
    "COLO678",
    "HCT15",
    "HPAFII",
    "IPC298",
    "KP363T",
    "PACADD165",
    "PANC1",
    "PANC0203",
    "SNU410",
    "SNU478",
]


def read_Ebright2026(raw: Path) -> pd.DataFrame:
    d = pd.read_csv(raw / S / "Ebright2026/biorxiv_DC5_SupplementaryTable4.csv")
    _need(
        d,
        ["GuideTargetSymbol"] + [f"{x}_CombinatoryScore" for x in EBRIGHT_LINES],
        "Ebright2026 Table S4",
    )
    g = d.GuideTargetSymbol.str.split("_", expand=True)
    if g.shape[1] != 2:
        raise ValueError("Ebright2026: GuideTargetSymbol is not A_B")
    frames = []
    for line in EBRIGHT_LINES:
        s = d[f"{line}_CombinatoryScore"]
        k = s.notna()
        frames.append(
            _frame(
                "Ebright2026",
                line,
                g[0][k],
                g[1][k],
                s[k],
                None,
                (s[k] <= -0.5).astype(int),
            )
        )
    return _concat(frames)


def read_EsmaeiliAnvar2024(raw: Path) -> pd.DataFrame:
    d = _excel(raw / S / "EsmaeiliAnvar2024/41467_2024_47795_MOESM10_ESM.xlsx", 0, 0)
    lines = ["k562_T8", "A549_T21", "MeljusoCP1905_T21", "A375CP1906_T21"]
    _need(d, ["Gene"] + lines, "EsmaeiliAnvar2024 SD7")
    d = d.set_index(d.Gene.astype(str))
    n = d.index.str.count("_")
    pairs = d[n == 1]  # 3- and 4-gene families (n > 1) are not pairs: left out
    single = d[n == 0]
    g = pairs.index.to_series().str.split("_", expand=True)
    if g.shape[1] != 2:
        raise ValueError("EsmaeiliAnvar2024: pair name is not A_B")
    frames = []
    for line in lines:
        # dLFC = observed pair LFC - (LFC_A + LFC_B): the multiplicative null in log
        # space (Methods)
        exp = (
            single[line].reindex(g[0]).to_numpy()
            + single[line].reindex(g[1]).to_numpy()
        )
        dlfc = pairs[line].to_numpy() - exp
        k = pd.notna(dlfc)
        frames.append(
            _frame(
                "EsmaeiliAnvar2024",
                line,
                g[0][k],
                g[1][k],
                dlfc[k],
                None,
                (dlfc[k] < -1).astype(int),
            )
        )
    return _concat(frames)


def read_Flister2025_genomewide(raw: Path) -> pd.DataFrame:
    lab = _excel(
        raw / S / "Flister2025_genomewide/1-s2.0-S2211124725012835-mmc4.xlsx", 0, 1
    )
    sc = _excel(raw / S / "Flister2025_hybrid/1-s2.0-S2211124725012835-mmc3.xlsx", 0, 1)
    _need(lab, ["label", "NCIH1299", "MDAMB231"], "Flister2025 Table S3")
    _need(sc, ["label", "H1299_diff", "diff"], "Flister2025 Table S2")
    sc = sc.drop_duplicates("label").set_index("label")
    lab = lab[~lab.label.astype(str).str.contains("NTC|ctrl|control", case=False)]
    g = lab.label.astype(str).str.split("_", expand=True)
    if g.shape[1] != 2:
        raise ValueError("Flister2025 genome-wide: label is not A_B")
    frames = []
    for line, col in [("NCIH1299", "H1299_diff"), ("MDAMB231", "diff")]:
        s = sc[col].reindex(lab.label)
        if s.isna().any():
            raise ValueError(
                f"Flister2025 {line}: a Table S3 pair has no Table S2 score"
            )
        frames.append(
            _frame(
                "Flister2025_genomewide",
                line,
                g[0],
                g[1],
                s,
                None,
                (lab[line] == "Lethal").astype(int),
            )
        )
    return _concat(frames)


def read_Flister2025_hybrid(raw: Path) -> pd.DataFrame:
    p = raw / S / "Flister2025_hybrid/1-s2.0-S2211124725012835-mmc6.xlsx"
    d = _excel(p, 0, 1)
    if d.shape[1] != 183 or list(d.columns[:3]) != ["Gene A", "Gene B", "Pair"]:
        raise ValueError("Flister2025 hybrid Table S5: unexpected layout")
    frames = []
    for i in range(3, 183, 9):
        name = str(d.columns[i])
        if not name.startswith("Gene_A_B") or not str(d.columns[i + 7]).startswith(
            "diff_z"
        ):
            raise ValueError(f"Flister2025 hybrid: block at column {i} is {name}")
        line = name.replace("Gene_A_B", "").strip(" _()").strip()
        blk = d.iloc[:, i : i + 9]
        k = blk.iloc[:, 7].notna()
        frames.append(
            _frame(
                "Flister2025_hybrid",
                line,
                d["Gene A"][k],
                d["Gene B"][k],
                blk.iloc[:, 7][k],
                None,
                (blk.iloc[:, 8][k] == "Lethal").astype(int),
            )
        )
    return _concat(frames)
