"""PromptCASL (Lu et al. 2025; official code at third_party/promptcasl, commit afcf7e9)
on the multi-study SL benchmark: the inputs, built with the official functions.

Decisions (read from the official code on 2026-11-26):

- Prompt version v2 (``src/data_prepare.py``). v1, the paper default, writes no prompt
for
  a gene whose copy number or mutation is missing ("No description for <gene>"); v2 and
  v3
  write one (a missing copy number reads "no copy-number alteration", a missing mutation
  "wild-type"). v3 adds NCBI gene summaries, a file that is not released. So v2, the
  closest released mode that handles the 21 labelled lines without copy number.
- Omics mode ``pair_pathway_evidence`` with the official defaults of ``main_loco.py``
  (the README command needs ``--omics_encoder 0`` too: its default 1 is refused with an
  omics mode). The three line tables are written in the official file format under
  ``<out>/ccle`` (GE.csv, CNGene.csv, MU.csv; no name2ens.json: the official prompt step
  needs Ensembl IDs only to look up mutations, which come from DepMap here by symbol).
  GE.csv is DepMap's log2(TPM + 1), or log2(CPM + 1) of the summed raw UMI of the line's
  cells for the lines the table lacks; CNGene.csv is the relative copy number
  (2^x - 1 of DepMap's log2 value; a missing line keeps NaN); MU.csv lists the genes of
  the damaging and hotspot mutation matrices (the matrices of hpc/mvgcn_isl).
- The official pathway step z-scores each gene over every line of GE.csv and averages
the
  pathway activity over the lines of one lineage. Here both statistics come from the fit
  lines only (``fit_only_zscore``, ``fit_lineage_means``).
- The mean-input control is a line ``MEAN_INPUT``: the fit lines' mean expression and
  copy number, no mutation, a description without line name and disease.
- PrimeKG: the MiT4SL copy (data/mit4sl/official, no Harvard Dataverse on the H20), its
  triples rebuilt from the HeteroData, every edge between the two genes of a benchmark
  pair removed. The official ConvE script runs on it (its ``PrimeKG`` dataset call is
  replaced by this graph; its 16 x 16 reshape of the 768-dim vectors, which pykeen
  1.10.2 refuses, becomes 16 x 48). Neighbour tables are not written: train.py selects
  the 100 neighbours of each pair as ``scripts/build_kg_neighbors.py`` and
  ``src/preprocess_datasets.py`` do, when it builds a row.

    .venv-promptcasl/bin/python hpc/promptcasl/prepare.py \
        --out data/promptcasl/benchmark
"""

import argparse
import importlib
import importlib.util
import json
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENDORED = ROOT / "third_party" / "promptcasl"
sys.path.insert(0, str(ROOT / "hpc"))

import sl_common  # noqa: E402

MEAN_INPUT = "MEAN_INPUT"
DEPMAP = ROOT / "data/sl_dependency_v0/raw/depmap"
OFFICIAL = ROOT / "data/promptcasl/official"
KG_DATA = ROOT / "data/mit4sl/official/MultiOmics_feature/kg_data"
UMI = ROOT / "data/sl_dependency_v0/processed/basal_raw_umi"
PROMPT_VERSION = "v2"
NEIGHBOR_COUNT = 100  # min_kg_neighbors of src/preprocess_datasets.py


def vendored(name: str):
    """A module of the vendored package (a top-level ``src``, like the repository's own;
    the harness scripts never put the repository root on the path)."""
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    return importlib.import_module(name)


def verbalizer():
    return vendored("src.data_prepare")


# --- omics tables ----------------------------------------------------------------


def pseudo_bulk(counts, genes) -> pd.Series:
    """log2(CPM + 1) of the summed raw UMI of a line's cells (cells x genes), on the
    scale of DepMap's log2(TPM + 1)."""
    total = np.asarray(counts.sum(axis=0)).ravel().astype(float)
    return pd.Series(np.log2(total / total.sum() * 1e6 + 1), index=list(genes))


def omics_tables(
    tables: dict[str, pd.DataFrame],
    lines: list[str],
    fit_lines: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """The official GE.csv, CNGene.csv and MU.csv for ``lines``: ``tables`` holds the
    DepMap ``expression`` (log2 TPM + 1), ``copy_number`` (log2 relative + 1) and
    ``mutation`` (lines x genes, > 0 when mutated) and the ``pseudo_bulk`` of the lines
    without expression. A value that a table lacks stays NaN. With ``fit_lines``, the
    line ``MEAN_INPUT`` is their mean expression and copy number (no mutation)."""
    ge = tables["expression"].reindex(lines)
    for line in ge.index[ge.isna().all(axis=1)]:
        if line in tables["pseudo_bulk"].index:
            ge.loc[line] = tables["pseudo_bulk"].loc[line].reindex(ge.columns)
    cn = 2.0 ** tables["copy_number"].reindex(index=lines) - 1
    if fit_lines is not None:
        ge.loc[MEAN_INPUT] = ge.loc[ge.index.intersection(fit_lines)].mean()
        cn.loc[MEAN_INPUT] = cn.loc[cn.index.intersection(fit_lines)].mean()
    mutation = tables["mutation"].reindex(index=lines)
    row, column = np.nonzero(mutation.to_numpy() > 0)  # NaN (no call) is not > 0
    mutated = pd.DataFrame(
        {"ModelID": mutation.index[row], "HugoSymbol": mutation.columns[column]}
    )
    return {"GE": ge, "CNGene": cn, "MU": mutated}


def read_copy_number(depmap, path: Path) -> pd.DataFrame:
    """DepMap's copy number, lines x upper-case symbols (a column named by an Ensembl ID
    only, and a repeated symbol, are dropped)."""
    frame = pd.read_csv(path, index_col=0)
    frame = frame.loc[:, ~frame.columns.str.startswith("ENSG")]
    frame.index = frame.index.astype(str)
    frame.columns = [depmap.depmap_symbol(c) for c in frame.columns]
    return frame.loc[:, ~frame.columns.duplicated()]


def read_pseudo_bulk(line: str, genes) -> pd.Series | None:
    import anndata

    path = UMI / f"{line}.h5ad"
    if not path.exists():
        return None
    adata = anndata.read_h5ad(path)
    symbols = adata.var["gene_symbol"].astype(str).str.upper()
    keep = ~symbols.duplicated().to_numpy()
    series = pseudo_bulk(adata.X[:, keep], symbols[keep])
    return series.reindex(genes)


# --- prompts -----------------------------------------------------------------------


def line_description(name: str | None, disease: str | None, lineage: str | None) -> str:
    """The line description of the prompt, in the form of the official curated ones
    ('A549 cell line is derived from human lung carcinoma, ...')."""
    if not name:
        return "Cell line derived from human cancer."
    return (
        f"{name} cell line is derived from human {disease or 'cancer'}"
        f"{f', a {lineage} cancer' if lineage else ''}."
    )


def prompts(line, name, ge, cn, mu, genes) -> dict[str, str]:
    """The official v2 prompt of each gene (``format_bio_state_description``): the
    line's
    expression, copy number and mutation of the gene. A gene without expression has no
    prompt (the official step skips it, and the row reads "No description for
    <gene>")."""
    dp = verbalizer()
    ge_row, cn_row = ge.loc[line], cn.loc[line]
    # the official copy-number cut points need the line's own values; a line without any
    # reads a placeholder, and no gene uses it (a missing value is "no alteration")
    cn_frame = cn if cn_row.notna().any() else pd.DataFrame({"x": [0.0]}, index=[line])
    thresholds = dp._compute_cell_line_thresholds(line, ge, cn_frame)
    mutated = set(mu.loc[mu["ModelID"] == line, "HugoSymbol"])
    out = {}
    for gene in genes:
        expression = ge_row.get(gene, np.nan)
        if pd.isna(expression):
            continue
        copy = cn_row.get(gene, np.nan)
        state = [
            float(expression),
            None if pd.isna(copy) else float(copy),
            int(gene in mutated),
        ]
        out[gene] = dp.format_bio_state_description(name, gene, state, thresholds)
    return out


# --- labels ----------------------------------------------------------------------


def label_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The official fold-table columns (data/folds_assignment/*_full.csv), one row for
    each (line, pair): SL when half or more of its labels are SL. One fold."""
    pairs = rows.groupby(["model_id", "gene_1", "gene_2"], as_index=False)[
        "label"
    ].mean()
    return pd.DataFrame(
        {
            "gene_pair": pairs["gene_1"] + "|" + pairs["gene_2"],
            "cell_line_origin": pairs["model_id"],
            "gene_1": pairs["gene_1"],
            "gene_2": pairs["gene_2"],
            "SL_or_not": np.where(pairs["label"] >= 0.5, "SL", "Not SL"),
            "fold": 1,
        }
    )


def fit_samples(rows: pd.DataFrame, sides, rng: np.random.Generator) -> pd.DataFrame:
    """For each line of ``sides``, its genetic SL pairs and as many of its non-SL pairs
    (the official balanced setting), drawn once."""
    chosen = rows[rows["side"].isin(sides) & (rows["condition"] == "genetic")]
    pairs = chosen.groupby(["model_id", "gene_1", "gene_2"], as_index=False)[
        "label"
    ].mean()
    pairs["label"] = (pairs["label"] >= 0.5).astype(int)
    parts = [
        sl_common.balanced(part.reset_index(drop=True), rng)
        for _, part in pairs.groupby("model_id")
    ]
    return pd.concat(parts, ignore_index=True)


# --- statistics of the fit lines ---------------------------------------------------


def fit_only_zscore(ge: pd.DataFrame, fit_lines):
    """``pathway_utils.zscore_ge`` with the gene mean and standard deviation of the fit
    lines only (the official one reads every line of GE.csv, the held-out ones too)."""
    fit = ge.loc[ge.index.intersection(fit_lines)]
    mean, std = fit.mean(axis=0), fit.std(axis=0).replace(0, np.nan)
    return lambda frame: ((frame - mean) / std).fillna(0.0)


def fit_lineage_means(activity: dict, lineage_of: dict, fit_lines) -> dict:
    """The lineage mean of the pathway expression (column 0 of the activity), over the
    fit lines of the lineage; a lineage without a fit line takes the mean of all of
    them."""
    fit = [line for line in fit_lines if line in activity]
    everything = np.mean([activity[c][:, 0] for c in fit], axis=0).astype(np.float32)
    out = {}
    for lineage in set(lineage_of.values()):
        members = [c for c in fit if lineage_of.get(c) == lineage]
        out[lineage] = (
            np.mean([activity[c][:, 0] for c in members], axis=0).astype(np.float32)
            if members and lineage != "all"
            else everything
        )
    return out


# --- PrimeKG ---------------------------------------------------------------------


def kg_triples(edges: dict, names: dict) -> pd.DataFrame:
    """(head, relation, tail) names of PrimeKG from MiT4SL's HeteroData: ``edges`` maps
    (source type, relation, target type) to a 2 x n array of node positions in the type,
    ``names`` maps a type to the array of its node names. The ``rev_`` copies are left
    out. A gene or protein keeps its upper-case symbol, any other node is prefixed by
    its
    type (a drug or a disease can share a gene's name)."""

    def label(kind, index):
        raw = np.asarray(names[kind])[index]
        return (
            np.char.upper(raw.astype(str))
            if kind == "gene/protein"
            else (np.char.add(kind + ":", raw.astype(str)))
        )

    frames = [
        pd.DataFrame(
            {
                "head": label(src, index[0]),
                "relation": relation,
                "tail": label(dst, index[1]),
            }
        )
        for (src, relation, dst), index in edges.items()
        if not relation.startswith("rev_")
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def neighbor_ids(adj: dict, a: int, b: int, min_len: int = NEIGHBOR_COUNT):
    """The ``min_len`` neighbour ids of each gene of a pair, as
    ``src/preprocess_datasets.
    get_kg_neighbor_emb_for_pair`` selects them from the table of
    ``scripts/build_kg_neighbors.py``: all 1-hop neighbours (without the partner),
    padded
    with 2-hop ones; half of the list the genes share, the rest their own. Seeded by the
    pair, so a pair always reads the same. None when a gene has fewer than ``min_len``
    (the official step skips such a row)."""
    kg = vendored("scripts.build_kg_neighbors")
    rng = random.Random(f"{a}|{b}")
    na = kg.pad_neighbors(adj, a, {b}, rng, min_len)
    nb = kg.pad_neighbors(adj, b, {a}, rng, min_len)
    shared = list(set(na) & set(nb))
    pick = np.random.default_rng(42)
    half = min_len // 2
    chosen = pick.choice(shared, half, replace=False) if len(shared) >= half else shared
    remain = min_len - len(chosen)
    own = []
    for neighbors in (na, nb):
        unique = [n for n in neighbors if n not in chosen]
        own.append(
            pick.choice(unique, remain, replace=False)
            if len(unique) >= remain
            else unique
        )
    ids = [list(chosen) + list(unique) for unique in own]
    if any(len(i) != min_len for i in ids):
        return None
    return np.array(ids, dtype=np.int32)


# --- steps -------------------------------------------------------------------------


def upper_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """The pair columns in upper case, sorted: DepMap, PrimeKG and the Reactome sets
    are matched by upper-case symbol (852 benchmark genes are not, as ``C1orf112``).
    ``gene_a`` and ``gene_b`` keep the stored spelling and orientation."""
    a, b = rows["gene_a"].str.upper(), rows["gene_b"].str.upper()
    return rows.assign(gene_1=np.minimum(a, b), gene_2=np.maximum(a, b))


def smoke_lines(rows: pd.DataFrame, with_copy_number: set, n: int) -> list[str]:
    """``n`` lines for a short run: about two thirds fit lines, the rest held out, the
    smallest ones with at least 20 SL pairs, and at least one without copy number."""
    genetic = rows[rows["condition"] == "genetic"].drop_duplicates(
        ["model_id", "gene_1", "gene_2"]
    )
    stats = genetic.groupby("model_id").agg(
        size=("label", "size"), sl=("label", "sum"), side=("side", "first")
    )
    stats = stats[stats["sl"] >= 20].sort_values("size")
    n_held = max(1, n // 3)
    held = list(stats.index[stats["side"] == "val"][:n_held])
    fit = list(stats.index[stats["side"] == "train"][: n - n_held])
    if all(line in with_copy_number for line in held + fit):
        bare = [
            m
            for m in stats.index[stats["side"] == "train"]
            if m not in with_copy_number
        ]
        fit = fit[:-1] + bare[:1]
    return fit + held


def build_omics(out: Path, rows: pd.DataFrame, lines: list[str]) -> dict:
    spec = importlib.util.spec_from_file_location(
        "depmap_tables", ROOT / "src/data/depmap.py"
    )
    depmap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(depmap)
    expression = depmap.read_depmap_matrix(
        DEPMAP / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    copy_number = read_copy_number(depmap, DEPMAP / "PortalOmicsCNGeneLog2.csv")
    mutation = pd.concat(
        [
            depmap.read_depmap_matrix(DEPMAP / f"OmicsSomaticMutationsMatrix{kind}.csv")
            for kind in ("Damaging", "Hotspot")
        ]
    )
    mutation = mutation.groupby(level=0).max()
    absent = [m for m in lines if m not in expression.index]
    pseudo = {m: read_pseudo_bulk(m, expression.columns) for m in absent}
    pseudo = pd.DataFrame({m: s for m, s in pseudo.items() if s is not None}).T
    pseudo = pseudo.reindex(columns=expression.columns)
    fit = sorted(rows.loc[rows["side"].isin(sl_common.FIT_SIDES), "model_id"].unique())
    tables = {
        "expression": expression,
        "copy_number": copy_number,
        "mutation": mutation,
        "pseudo_bulk": pseudo,
    }
    built = omics_tables(tables, lines, fit_lines=fit)
    (out / "ccle").mkdir(parents=True, exist_ok=True)
    built["GE"].to_csv(out / "ccle" / "GE.csv")
    built["CNGene"].to_csv(out / "ccle" / "CNGene.csv")
    built["MU"].to_csv(out / "ccle" / "MU.csv", index=False)
    model = pd.read_csv(DEPMAP / "Model.csv").set_index("ModelID")
    info = {}
    for line in lines:
        name = model["StrippedCellLineName"].get(line)
        info[line] = {
            "name": name if isinstance(name, str) else line,
            "lineage": model["OncotreeLineage"].get(line),
            "description": line_description(
                name if isinstance(name, str) else None,
                model["OncotreePrimaryDisease"].get(line),
                model["OncotreeLineage"].get(line),
            ),
            "expression": "depmap"
            if line in expression.index
            else ("pseudo_bulk" if line in pseudo.index else None),
            "copy_number": bool(line in copy_number.index),
            "side": rows.loc[rows["model_id"] == line, "side"].iloc[0],
        }
    info[MEAN_INPUT] = {
        "name": "average",
        "lineage": "all",
        "description": line_description(None, None, None),
        "expression": "fit_mean",
        "copy_number": True,
        "side": "mean",
    }
    (out / "lines.json").write_text(json.dumps(info, indent=1) + "\n")
    # prompts: the genes of each line's rows (the mean line: every held-out gene)
    template = out / "text_template" / PROMPT_VERSION
    template.mkdir(parents=True, exist_ok=True)
    held_genes = set(rows.loc[rows["side"].isin(sl_common.HELD_SIDES), "gene_1"]) | set(
        rows.loc[rows["side"].isin(sl_common.HELD_SIDES), "gene_2"]
    )
    usable = [m for m in lines if info[m]["expression"]]
    for line in usable + [MEAN_INPUT]:
        part = rows[rows["model_id"] == line]
        genes = (
            held_genes
            if line == MEAN_INPUT
            else set(part["gene_1"]) | set(part["gene_2"])
        )
        text = prompts(
            line, info[line]["name"], built["GE"], built["CNGene"], built["MU"], genes
        )
        (template / f"{line}_gene_omics_descriptions_{PROMPT_VERSION}.json").write_text(
            json.dumps(text)
        )
    return {
        "lines": len(lines),
        "lines_without_expression": [m for m in lines if not info[m]["expression"]],
        "lines_with_pseudo_bulk": sorted(pseudo.index),
        "lines_without_copy_number": [m for m in lines if not info[m]["copy_number"]],
    }


def build_reactome(
    out: Path, lines: list[str], fit_lines: list[str], device: int
) -> dict:
    """The official ``scripts/build_reactome_pair_evidence.main`` on this benchmark's
    lines: the paths and the line list are set, the z-score takes the fit lines."""
    info = json.loads((out / "lines.json").read_text())
    names = [m for m in lines if info[m]["expression"]] + [MEAN_INPUT]
    gmt = out / "multi_omics" / "c2.cp.v7.5.1.symbols.upper.gmt"
    gmt.parent.mkdir(parents=True, exist_ok=True)
    with open(OFFICIAL / "c2.cp.v7.5.1.symbols.gmt") as f:
        sets = [line.rstrip("\n").split("\t") for line in f]
    gmt.write_text(  # the DepMap columns are upper case
        "".join("\t".join(x[:2] + [g.upper() for g in x[2:]]) + "\n" for x in sets)
    )
    pu = vendored("src.pathway_utils")
    script = vendored("scripts.build_reactome_pair_evidence")
    ge = pd.read_csv(out / "ccle" / "GE.csv", index_col=0)
    pu._CCLE_GE = str(out / "ccle" / "GE.csv")
    pu._CCLE_CNV = str(out / "ccle" / "CNGene.csv")
    pu._CCLE_MU = str(out / "ccle" / "MU.csv")
    pu._GMT_C2CP = str(gmt)
    pu.zscore_ge = fit_only_zscore(ge, fit_lines)
    script.CELL_LINE_ACH = {m: m for m in names}
    script.MULTI_OMICS_DIR = str(out / "multi_omics")
    (out / "multi_omics").mkdir(parents=True, exist_ok=True)
    sys.argv = ["build_reactome_pair_evidence", "--device", str(device)]
    script.main()
    return json.loads((out / "multi_omics" / "reactome_manifest.json").read_text())


def build_kg(out: Path, rows: pd.DataFrame, device: int, epochs: int) -> dict:
    """PrimeKG without the benchmark pairs' edges, and the official ConvE run on it."""
    from pykeen.triples import TriplesFactory

    nodes = pd.read_csv(KG_DATA / "Primenode.csv")
    with open(KG_DATA / "node_index_dic.json") as f:
        position = json.load(f)  # type -> {node_index: position in the type}
    names = {}
    for kind, mapping in position.items():
        table = nodes[nodes["node_type"] == kind].set_index("node_index")["node_name"]
        order = np.empty(len(mapping), dtype=object)
        for node, local in mapping.items():
            order[local] = table[int(node)]
        names[kind] = order
    with open(KG_DATA / "kgdata.pkl", "rb") as f:
        data = pickle.load(f)
    edges = {e: data[e].edge_index.numpy() for e in data.edge_types}
    triples = kg_triples(edges, names)
    before = len(triples)
    pairs = triples.rename(columns={"head": "gene1", "tail": "gene2"})
    kept = sl_common.drop_pair_edges(pairs, rows)
    triples = triples.loc[kept.index]
    (out / "kg").mkdir(parents=True, exist_ok=True)
    factory = TriplesFactory.from_labeled_triples(
        triples[["head", "relation", "tail"]].to_numpy(dtype=str),
        create_inverse_triples=True,
    )
    train, valid = factory.split([0.99, 0.01], random_state=42)
    np.save(out / "kg" / "train_triples.npy", train.mapped_triples.numpy())
    with open(out / "kg" / "entity_to_id.json", "w") as f:
        json.dump(factory.entity_to_id, f)
    conve = vendored("scripts.train_conve_primekg")
    # the official script scores its validation split at the end: 5,000 triples are
    # enough
    small = SimpleNamespace(
        training=train,
        validation=SimpleNamespace(mapped_triples=valid.mapped_triples[:5000]),
    )
    conve.PrimeKG = lambda create_inverse_triples: small
    # pykeen 1.10.2 refuses the official 16 x 16 reshape of a 768-dim entity vector
    # (ConvE "could not resolve (1, 16, 16) to a valid factorization of 768"); the model
    # needs 768 dims, so the width is 48 (16 x 48 = 768) and nothing else changes
    official_conve = conve.ConvE
    conve.ConvE = lambda **kwargs: official_conve(**{**kwargs, "embedding_width": 48})
    sys.argv = [
        "train_conve_primekg", "--dim", "768", "--epochs", str(epochs),
        "--device", str(device), "--out_dir", str(out / "kg"),
    ]  # fmt: skip
    conve.main()
    return {
        "kg_triples": before,
        "kg_triples_after_pair_edges_removed": len(triples),
        "kg_entities": len(factory.entity_to_id),
        "conve_epochs": epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data/sl_studies/benchmark_v2/labels.parquet",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "configs/benchmarks/cell_line_sl_split.json",
    )
    parser.add_argument("--lines", type=int, help="a short run on this many lines")
    parser.add_argument(
        "--device", type=int, default=3, help="GPU of the pathway and ConvE steps"
    )
    parser.add_argument(
        "--conve-epochs", type=int, default=10, help="the official default"
    )
    cli = parser.parse_args()
    # the official config reads the encoder path when it is imported
    os.environ.setdefault("PROMPTCASL_BIOMEDBERT", str(OFFICIAL / "biomedbert"))
    out = cli.out
    out.mkdir(parents=True, exist_ok=True)
    rows = upper_pairs(sl_common.benchmark_rows(cli.labels, cli.split))
    if cli.lines:
        copy_number = set(
            pd.read_csv(DEPMAP / "PortalOmicsCNGeneLog2.csv", usecols=[0]).iloc[:, 0]
        )
        rows = rows[rows["model_id"].isin(smoke_lines(rows, copy_number, cli.lines))]
    lines = sorted(rows["model_id"].unique())
    fit_lines = sorted(
        rows.loc[rows["side"].isin(sl_common.FIT_SIDES), "model_id"].unique()
    )
    summary = {"lines_total": len(lines), "fit_lines": len(fit_lines)}
    summary |= build_omics(out, rows, lines)
    summary["reactome"] = build_reactome(out, lines, fit_lines, cli.device)
    summary |= build_kg(out, rows, cli.device, cli.conve_epochs)
    rng = np.random.default_rng(0)
    fit = fit_samples(rows, sl_common.FIT_SIDES, rng)
    val = fit_samples(rows, ("val",), rng)
    (out / "labels").mkdir(exist_ok=True)
    label_table(fit).to_csv(out / "labels" / "fit_full.csv", index=False)
    label_table(val).to_csv(out / "labels" / "val_full.csv", index=False)
    rows.to_parquet(out / "rows.parquet", index=False)
    summary |= {
        "fit_rows": len(fit),
        "fit_sl_rows": int(fit["label"].sum()),
        "val_rows": len(val),
        "held_rows": int(rows["side"].isin(sl_common.HELD_SIDES).sum()),
        "prompt_version": PROMPT_VERSION,
    }
    (out / "prepare.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
