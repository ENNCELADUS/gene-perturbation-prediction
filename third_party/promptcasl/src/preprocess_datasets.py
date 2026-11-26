"""Build the per-fold GenePairDataset ``.pt`` files.

Run: python -m src.preprocess_datasets
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pykeen.datasets import PrimeKG

from src.config import (
    BIOMEDBERT,
    CELL_LINES,
    DATASETS_DIR,
    FOLDS_DIR,
    GENEPT_PKL,
    KG_NEIGHBOR_DIR,
    PRIMEKG_EMB,
    TEXT_TEMPLATE_DIR,
)
from src.data_loader import GenePairDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def get_token_embeddings(model, tokenizer, text, device, max_length=512):
    """Return the embedding-layer token representations of ``text``.

    Returns:
        input_ids (torch.Tensor): [1, seq_len]
        attention_mask (torch.Tensor): [seq_len]
        token_embeddings (torch.Tensor): [seq_len, hidden_dim]
        tokens (list): token strings.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        if hasattr(model, "embeddings"):
            # BERT-family encoders (BiomedBERT, BioBERT, ...).
            token_embeddings = model.embeddings(input_ids=input_ids).squeeze(0)
        elif hasattr(model, "embed_tokens"):
            # Decoder-style LLMs (Qwen, Llama, ...), kept for flexibility.
            token_embeddings = model.embed_tokens(input_ids).squeeze(0)
        else:
            raise ValueError(
                "Model does not have a recognized embedding layer: "
                "'embeddings' or 'embed_tokens'."
            )
    return input_ids, attention_mask.squeeze(0), token_embeddings, tokens


def get_biobert_token_emb_for_pair(
    cell_line, geneA, geneB, cell_lines_info, gene_omics_desc, biobert, tokenizer, device
):
    """Concatenate cell-line description and both gene states, then encode."""
    cell_line_desc = cell_lines_info[cell_line]["description"]
    descA = gene_omics_desc.get(geneA, f"No description for {geneA}")
    descB = gene_omics_desc.get(geneB, f"No description for {geneB}")
    full_text = f"{cell_line_desc} {descA} {descB}"
    input_ids, attention_mask, token_embeddings, _ = get_token_embeddings(
        biobert, tokenizer, full_text, device
    )
    return input_ids, attention_mask, token_embeddings


def get_kg_neighbor_emb_for_pair(
    geneA,
    geneB,
    cell_line,
    entity_to_id,
    kg_entity_embeddings,
    neighbor_dir=KG_NEIGHBOR_DIR,
    min_len=100,
    rd_seed=42,
):
    """Return KG neighbor embeddings for the two genes from a precomputed .tsv.

    The neighbor list for each gene is composed of half the genes shared with the
    partner plus its own unique neighbors, capped at ``min_len``.

    Returns:
        geneA_neighbor_embs, geneB_neighbor_embs: [min_len, emb_dim] numpy arrays.
    """
    geneA_id = entity_to_id.get(geneA)
    geneB_id = entity_to_id.get(geneB)
    if geneA_id is None or geneB_id is None:
        raise ValueError(f"GeneA or GeneB not found in entity_to_id: {geneA}, {geneB}")

    neighbor_file = os.path.join(neighbor_dir, f"{cell_line}_neighbors.tsv")
    df = pd.read_csv(neighbor_file, sep="\t")

    row = df[(df["geneA"] == geneA_id) & (df["geneB"] == geneB_id)]
    if row.empty:
        raise ValueError(f"Gene pair ({geneA}, {geneB}) not found in {neighbor_file}")
    row = row.iloc[0]

    geneA_neighbors = [int(x) for x in str(row["geneA_neighbors"]).split(",") if x]
    geneB_neighbors = [int(x) for x in str(row["geneB_neighbors"]).split(",") if x]
    shared_neighbors = [int(x) for x in str(row["shared_neighbors"]).split(",") if x]

    rng = np.random.default_rng(rd_seed)
    half_min_len = min_len // 2

    if len(shared_neighbors) >= half_min_len:
        selected_shared = rng.choice(shared_neighbors, half_min_len, replace=False)
    else:
        selected_shared = shared_neighbors
    num_shared = len(selected_shared)
    remain = min_len - num_shared

    unique_geneA_neighbors = [n for n in geneA_neighbors if n not in selected_shared]
    if len(unique_geneA_neighbors) >= remain:
        selected_geneA = rng.choice(unique_geneA_neighbors, remain, replace=False)
    else:
        selected_geneA = unique_geneA_neighbors

    unique_geneB_neighbors = [n for n in geneB_neighbors if n not in selected_shared]
    if len(unique_geneB_neighbors) >= remain:
        selected_geneB = rng.choice(unique_geneB_neighbors, remain, replace=False)
    else:
        selected_geneB = unique_geneB_neighbors

    geneA_selected_ids = list(selected_shared) + list(selected_geneA)
    geneB_selected_ids = list(selected_shared) + list(selected_geneB)

    geneA_neighbor_embs = kg_entity_embeddings[np.array(geneA_selected_ids)]
    geneB_neighbor_embs = kg_entity_embeddings[np.array(geneB_selected_ids)]
    return geneA_neighbor_embs, geneB_neighbor_embs


def build_folds_GenePairDataset(
    fold_assign_file,
    cell_line,
    cell_lines_info,
    gene_omics_desc,
    genept_embeddings,
    kg_entity_to_id,
    kg_entity_embeddings,
    fold_output_dir,
    biobert,
    tokenizer,
    device,
    neighbor_dir=KG_NEIGHBOR_DIR,
    min_kg_neighbors=100,
):
    """Build one GenePairDataset per fold and save it as ``{cell_line}_F{fold}.pt``."""
    df = pd.read_csv(fold_assign_file)
    fold2rows = defaultdict(list)
    for _, row in df.iterrows():
        fold2rows[int(row["fold"])].append(row)

    for fold, rows in fold2rows.items():
        embeddings, labels, gene_names = [], [], []
        for row in tqdm(rows, desc=f"Processing fold {fold}"):
            gene_1, gene_2 = row["gene_1"], row["gene_2"]

            # 1. Text token sequence (BiomedBERT).
            _, attention_mask, token_emb = get_biobert_token_emb_for_pair(
                cell_line,
                gene_1,
                gene_2,
                cell_lines_info,
                gene_omics_desc,
                biobert,
                tokenizer,
                device,
            )
            # 2. GenePT embeddings.
            geneA_genept = genept_embeddings.get(gene_1)
            geneB_genept = genept_embeddings.get(gene_2)
            if geneA_genept is None or geneB_genept is None:
                print("missing genept")
                continue
            # 3. PrimeKG entity embeddings for both genes.
            geneA_kgid = kg_entity_to_id.get(gene_1)
            geneB_kgid = kg_entity_to_id.get(gene_2)
            if geneA_kgid is None or geneB_kgid is None:
                print("missing kg id")
                continue
            geneA_kg_emb = kg_entity_embeddings[geneA_kgid]
            geneB_kg_emb = kg_entity_embeddings[geneB_kgid]
            # 4. KG neighbor embeddings (min_kg_neighbors for each gene).
            try:
                geneA_neighbors_emb, geneB_neighbors_emb = get_kg_neighbor_emb_for_pair(
                    gene_1,
                    gene_2,
                    cell_line,
                    kg_entity_to_id,
                    kg_entity_embeddings,
                    neighbor_dir=neighbor_dir,
                    min_len=min_kg_neighbors,
                )
            except Exception as e:
                print(
                    f"Skipping {gene_1}, {gene_2} in {cell_line} due to neighbor error: {e}"
                )
                continue
            # 5. Label.
            label = 1 if str(row["SL_or_not"]).strip().upper() == "SL" else 0

            embedding = (
                token_emb.cpu().numpy(),
                attention_mask.cpu().numpy(),
                geneA_genept,
                geneA_kg_emb,
                geneB_genept,
                geneB_kg_emb,
                geneA_neighbors_emb,
                geneB_neighbors_emb,
            )
            embeddings.append(embedding)
            labels.append(torch.tensor(label).long())
            gene_names.append((row["gene_pair"], cell_line))

        fold_dataset = GenePairDataset(embeddings, labels, gene_names)
        fold_pt_path = os.path.join(fold_output_dir, f"{cell_line}_F{fold}.pt")
        torch.save(fold_dataset, fold_pt_path)
        print(f"saved in {fold_pt_path}")


if __name__ == "__main__":
    # Prompt template version: "v1" (numeric, paper default), "v2", or "v3".
    TEMPLATE_TYPE = os.environ.get("PROMPTCASL_PROMPT", "v1")
    DATASET_NAME = {
        "v1": "cell_line_biomedbert_kg50shared",
        "v2": "cell_line_biomedbert_v2_kg50shared",
        "v3": "cell_line_biomedbert_v3_kg50shared",
    }[TEMPLATE_TYPE]
    fold_output_dir = os.path.join(DATASETS_DIR, DATASET_NAME)
    os.makedirs(fold_output_dir, exist_ok=True)

    # GenePT gene embeddings.
    with open(GENEPT_PKL, "rb") as file:
        genept_gene_embeddings = pickle.load(file)
    for gene, emb in genept_gene_embeddings.items():
        genept_gene_embeddings[gene] = np.array(emb, dtype=np.float32)

    # PrimeKG (ConvE) entity embeddings.
    kg_dataset = PrimeKG(create_inverse_triples=True)
    kg_entity_embeddings = np.load(PRIMEKG_EMB)
    kg_entity_to_id = kg_dataset.training.entity_to_id

    tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT)
    biobert = AutoModel.from_pretrained(BIOMEDBERT).to(device)

    for cell_line_case in CELL_LINES:
        print("Processing ", cell_line_case)
        if TEMPLATE_TYPE == "v1":
            template_file = os.path.join(
                TEXT_TEMPLATE_DIR, "v1", f"{cell_line_case}_gene_omics_descriptions.json"
            )
        else:
            template_file = os.path.join(
                TEXT_TEMPLATE_DIR,
                TEMPLATE_TYPE,
                f"{cell_line_case}_gene_omics_descriptions_{TEMPLATE_TYPE}.json",
            )
        with open(template_file) as f:
            gene_omics_desc = json.load(f)
        build_folds_GenePairDataset(
            os.path.join(FOLDS_DIR, f"{cell_line_case}_filtered_NP1_F.csv"),
            cell_line_case,
            cell_lines_info,
            gene_omics_desc,
            genept_gene_embeddings,
            kg_entity_to_id,
            kg_entity_embeddings,
            fold_output_dir,
            biobert,
            tokenizer,
            device,
        )
