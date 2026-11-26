"""Generate the verbalized cell-line prompt templates (v1 / v2 / v3).

Run: python -m src.data_prepare
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CCLE_DIR, CELL_LINES, NCBI_SUMMARY, SLKB_RAW, TEXT_TEMPLATE_DIR

# Manually curated cell-line descriptions (tissue lineage + disease background).
cell_lines_info = {
    "A549": {
        "cell_line_id": "ACH-000681",
        "description": "A549 cell line is derived from human lung carcinoma, used in research of lung cancer and chemotherapeutic responses. It is responsible for the study of cancerous transformations and treatment efficacy in pulmonary cells. These cells are characterized by an epithelial-like structure, making them useful for studying ion transport across alveolar surfaces and drug metabolism.",
    },
    "PK1": {
        "cell_line_id": "ACH-000307",
        "description": "PK1 (PK-1) cell line is derived from human pancreatic ductal adenocarcinoma, used in research of pancreatic cancer and KRAS-driven oncogenic signaling. It is responsible for modeling tumor proliferation, invasion, and therapeutic resistance in pancreatic epithelial cells. These cells are characterized by activating KRAS (G12D) and TP53 mutations, making them a valuable model for studying pancreatic cancer progression and targeted therapies.",
    },
    "IPC298": {
        "cell_line_id": "ACH-000915",
        "description": "IPC-298 cell line is derived from human malignant melanoma, used in research of skin cancer mechanisms and drug efficacy. It is responsible for exploring cellular responses to oncogenic stress. These cells provide a model for understanding melanoma biology and testing novel therapeutic strategies.",
    },
    "MEWO": {
        "cell_line_id": "ACH-000987",
        "description": "MEWO cell line is derived from human skin (melanoma), used in research of melanoma progression and treatment. It is responsible for modeling melanoma spread and response to therapies. These cells are characterized by their ability to form tumors and metastasize, which are key features for oncological studies.",
    },
    "GI1": {
        "cell_line_id": "ACH-000756",
        "description": "GI1 cell line is derived from human glioblastoma multiforme (GBM), a highly malignant tumor of the central nervous system. It is used in research on brain cancer, tumor biology, and therapeutic resistance. These cells exhibit high invasiveness and resistance to radiation, making them an important model for studying the mechanisms of glioma progression and developing targeted therapies for GBM.",
    },
    "HS936T": {
        "cell_line_id": "ACH-000801",
        "description": "HS936T cell line is derived from human melanoma, used in research of skin cancer and metastatic mechanisms. It is employed to study the transformation of melanocytes, tumor invasion, and resistance to treatments. These cells exhibit aggressive proliferation and metastatic potential, making them an essential model for understanding melanoma progression and testing targeted therapies.",
    },
}


def load_omics_data():
    """Load CCLE name-to-Ensembl mapping, expression, mutation and copy-number tables."""
    with open(os.path.join(CCLE_DIR, "name2ens.json")) as file:
        name_id_mapping = json.load(file)

    ge_data = pd.read_csv(os.path.join(CCLE_DIR, "GE.csv"))
    mu_data = pd.read_csv(os.path.join(CCLE_DIR, "MU.csv"), low_memory=False)
    cn_gene_data = pd.read_csv(os.path.join(CCLE_DIR, "CNGene.csv"))

    ge_data.set_index("Unnamed: 0", inplace=True)
    cn_gene_data.set_index("Unnamed: 0", inplace=True)
    return name_id_mapping, ge_data, mu_data, cn_gene_data


def if_mutation(mu_data, cell_line_id, gene_id):
    filtered_rows = mu_data[mu_data["ModelID"] == cell_line_id]
    mu_event = filtered_rows[filtered_rows["EnsemblGeneID"].astype(str) == gene_id]
    return 1 if not mu_event.empty else 0


def get_omics_data(ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id):
    """Return [expression, copy number, mutation] for a gene in a cell line."""
    ge_level = None
    cnv_gene = None
    try:
        gene_columns = [col for col in ge_data.columns if gene_name in col]
        ge_level = ge_data.loc[cell_line_id, gene_columns[0]]
    except (IndexError, KeyError):
        ge_level = None
    try:
        gene_columns = [col for col in cn_gene_data.columns if gene_name in col]
        cnv_gene = cn_gene_data.loc[cell_line_id, gene_columns[0]]
        if pd.isna(cnv_gene):
            cnv_gene = None
    except (IndexError, KeyError):
        cnv_gene = None
    mu_gene = if_mutation(mu_data, cell_line_id, gene_id)
    return [ge_level, cnv_gene, mu_gene]


# Per-cell-line quantile thresholds, computed from the full CCLE transcriptome
# (~17K genes) rather than from SL-pair genes, so they are independent of fold
# assignment and introduce no test leakage.
def _compute_cell_line_thresholds(cell_line_id, ge_data, cn_gene_data):
    ge_row = ge_data.loc[cell_line_id].dropna().values.astype(float)
    ge_nonzero = ge_row[ge_row > 0]
    ge_p20 = float(np.percentile(ge_nonzero, 25))
    ge_p80 = float(np.percentile(ge_nonzero, 75))

    cnv_row = cn_gene_data.loc[cell_line_id].dropna().values.astype(float)
    cnv_p10 = float(np.percentile(cnv_row, 10))
    cnv_p40 = float(np.percentile(cnv_row, 40))
    cnv_p60 = float(np.percentile(cnv_row, 60))
    cnv_p90 = float(np.percentile(cnv_row, 90))

    return {
        "ge_p20": ge_p20,
        "ge_p80": ge_p80,
        "cnv_p10": cnv_p10,
        "cnv_p40": cnv_p40,
        "cnv_p60": cnv_p60,
        "cnv_p90": cnv_p90,
    }


def _verbalize_ge(ge, thresholds):
    if ge is None or ge == 0.0:
        return "not expressed"
    elif ge <= thresholds["ge_p20"]:
        return "lowly expressed"
    elif ge <= thresholds["ge_p80"]:
        return "normally expressed"
    else:
        return "highly expressed"


def _verbalize_cnv(cnv, thresholds):
    if cnv is None:
        return "no copy-number alteration"
    elif cnv <= thresholds["cnv_p10"]:
        return "copy-number deleted"
    elif cnv <= thresholds["cnv_p40"]:
        return "copy-number loss"
    elif cnv <= thresholds["cnv_p60"]:
        return "no copy-number alteration"
    elif cnv <= thresholds["cnv_p90"]:
        return "copy-number gain"
    else:
        return "copy-number amplified"


def _verbalize_mu(mu):
    return "mutated" if mu == 1 else "wild-type"


def format_bio_state_description(cell_line, gene_name, omics_data, thresholds):
    ge_level, cnv_gene, mu_gene = omics_data
    return (
        f"In the {cell_line} cell line, {gene_name} is "
        f"{_verbalize_ge(ge_level, thresholds)}, "
        f"{_verbalize_cnv(cnv_gene, thresholds)}, "
        f"and {_verbalize_mu(mu_gene)}."
    )


def _extract_ncbi_summary(gene_name, ncbi_summaries, max_chars=200):
    raw = ncbi_summaries.get(gene_name, "")
    if not raw:
        return ""
    idx = raw.find("Summary ")
    text = raw[idx + 8:] if idx != -1 else raw
    period_idx = text.find(". ")
    sentence = text[:period_idx] if 0 < period_idx <= max_chars else text[:max_chars]
    return sentence.strip().rstrip(".")


def format_evidence_description(
    cell_line, gene_name, omics_data, ncbi_summaries, thresholds
):
    bio_state = format_bio_state_description(cell_line, gene_name, omics_data, thresholds)
    func = _extract_ncbi_summary(gene_name, ncbi_summaries)
    if func:
        return f"{gene_name}: {func}. {bio_state}"
    return bio_state


def format_cell_line_description(cell_line, gene_name, omics_data):
    """v1 numeric template (paper default)."""
    ge_level, cnv_gene, mu_gene = omics_data
    mutation_status = "mutated" if mu_gene == 1 else "not mutated"
    return (
        f"In the {cell_line} cell line, {gene_name} has a "
        f"gene expression level of {ge_level}, "
        f"a copy number variation of {cnv_gene}, "
        f"and is {mutation_status}."
    )


def _iter_genes(target_df):
    return pd.unique(target_df[["gene_1", "gene_2"]].values.ravel())


def process_cell_line_data_v1(
    cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path
):
    gene_descriptions = {}
    for gene_name in tqdm(_iter_genes(target_df), desc=f"v1 {cell_line}"):
        gene_id = name_id_mapping.get(gene_name)
        if gene_id is None:
            continue
        omics_data = get_omics_data(
            ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id
        )
        if any(data is None for data in omics_data):
            continue
        gene_descriptions[gene_name] = format_cell_line_description(
            cell_line, gene_name, omics_data
        )
    save_file = os.path.join(save_path, f"{cell_line}_gene_omics_descriptions.json")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(gene_descriptions, f, indent=4, ensure_ascii=False)
    print(f"Saved v1 descriptions for {cell_line} to {save_file}")


def process_cell_line_data_v2(
    cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_path
):
    thresholds = _compute_cell_line_thresholds(cell_line_id, ge_data, cn_gene_data)
    gene_descriptions = {}
    for gene_name in tqdm(_iter_genes(target_df), desc=f"v2 {cell_line}"):
        gene_id = name_id_mapping.get(gene_name)
        if gene_id is None:
            continue
        omics_data = get_omics_data(
            ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id
        )
        if any(data is None for data in omics_data):
            continue
        gene_descriptions[gene_name] = format_bio_state_description(
            cell_line, gene_name, omics_data, thresholds
        )
    save_file = os.path.join(save_path, f"{cell_line}_gene_omics_descriptions_v2.json")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(gene_descriptions, f, indent=4, ensure_ascii=False)
    print(f"Saved v2 descriptions for {cell_line} to {save_file}")


def process_cell_line_data_v3(
    cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data,
    ncbi_summaries, save_path,
):
    thresholds = _compute_cell_line_thresholds(cell_line_id, ge_data, cn_gene_data)
    gene_descriptions = {}
    for gene_name in tqdm(_iter_genes(target_df), desc=f"v3 {cell_line}"):
        gene_id = name_id_mapping.get(gene_name)
        if gene_id is None:
            continue
        omics_data = get_omics_data(
            ge_data, cn_gene_data, mu_data, cell_line_id, gene_name, gene_id
        )
        if any(data is None for data in omics_data):
            continue
        gene_descriptions[gene_name] = format_evidence_description(
            cell_line, gene_name, omics_data, ncbi_summaries, thresholds
        )
    save_file = os.path.join(save_path, f"{cell_line}_gene_omics_descriptions_v3.json")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(gene_descriptions, f, indent=4, ensure_ascii=False)
    print(f"Saved v3 descriptions for {cell_line} to {save_file}")


def main():
    name_id_mapping, ge_data, mu_data, cn_gene_data = load_omics_data()

    labels_df = pd.read_csv(SLKB_RAW)

    save_v1 = os.path.join(TEXT_TEMPLATE_DIR, "v1")
    save_v2 = os.path.join(TEXT_TEMPLATE_DIR, "v2")
    save_v3 = os.path.join(TEXT_TEMPLATE_DIR, "v3")
    for path in (save_v1, save_v2, save_v3):
        os.makedirs(path, exist_ok=True)

    with open(NCBI_SUMMARY) as f:
        ncbi_summaries = json.load(f)

    for cell_line in CELL_LINES:
        target_df = labels_df[labels_df["cell_line_origin"] == cell_line].copy()
        cell_line_id = cell_lines_info[cell_line]["cell_line_id"]
        process_cell_line_data_v1(
            cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_v1
        )
        process_cell_line_data_v2(
            cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data, save_v2
        )
        process_cell_line_data_v3(
            cell_line, cell_line_id, target_df, name_id_mapping, ge_data, mu_data, cn_gene_data,
            ncbi_summaries, save_v3,
        )


if __name__ == "__main__":
    main()
