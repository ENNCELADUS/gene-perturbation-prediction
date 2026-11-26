"""Path configuration for PromptCASL.

External data (CCLE, SLKB, GenePT, PrimeKG, NCBI summaries, BiomedBERT) is resolved
from environment variables; project paths are relative to the repository root.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

# ---------------------------------------------------------------------------
# Third-party resources (override with environment variables)
# ---------------------------------------------------------------------------
# CCLE omics tables: GE.csv, CNGene.csv, MU.csv, name2ens.json
CCLE_DIR = os.environ.get("PROMPTCASL_CCLE_DIR", os.path.join(EXTERNAL_DIR, "CCLE"))
# SLKB raw label table (rawSL.csv)
SLKB_RAW = os.environ.get("PROMPTCASL_SLKB_RAW", os.path.join(EXTERNAL_DIR, "rawSL.csv"))
# GenePT gene embeddings (3072-dim, pickled dict {gene: embedding})
GENEPT_PKL = os.environ.get(
    "PROMPTCASL_GENEPT_PKL",
    os.path.join(EXTERNAL_DIR, "GenePT_gene_protein_embedding_model_3_text.pickle"),
)
# PrimeKG entity embeddings produced by ConvE
PRIMEKG_EMB = os.environ.get(
    "PROMPTCASL_PRIMEKG_EMB",
    os.path.join(EXTERNAL_DIR, "conv_primekg_entity_embeddings.npy"),
)
# PrimeKG 128-dim ConvE embeddings (used by the KG-SLomics baseline)
PRIMEKG_EMB128 = os.environ.get(
    "PROMPTCASL_PRIMEKG_EMB128",
    os.path.join(EXTERNAL_DIR, "conv_primekg_entity_emb128.npy"),
)
# PrimeKG name -> entity-id dictionary and the name-level triples table
PRIMEKG_ENTITY_JSON = os.environ.get(
    "PROMPTCASL_PRIMEKG_ENTITY_JSON",
    os.path.join(EXTERNAL_DIR, "primekg_entity_to_id.json"),
)
PRIMEKG_TRIPLES_TSV = os.environ.get(
    "PROMPTCASL_PRIMEKG_TRIPLES_TSV",
    os.path.join(EXTERNAL_DIR, "primeKG_triples_names.tsv"),
)
# SLBench 9845-gene universe (used to align the baselines' evaluation pairs)
SLBENCH_META = os.environ.get(
    "PROMPTCASL_SLBENCH_META",
    os.path.join(EXTERNAL_DIR, "meta_table_9845.csv"),
)
# Sanger ProCan proteomics (4th omics channel of the KG-SLomics baseline)
SANGER_PROCAN = os.environ.get(
    "PROMPTCASL_SANGER_PROCAN",
    os.path.join(EXTERNAL_DIR, "proteomics.csv"),
)
# NCBI gene summary text (used by the v3 prompt)
NCBI_SUMMARY = os.environ.get(
    "PROMPTCASL_NCBI_SUMMARY",
    os.path.join(EXTERNAL_DIR, "NCBI_clipped_summary_of_genes.json"),
)
# MSigDB C2:CP gene sets (Reactome pathways), used to build the pathway evidence
GMT_C2CP = os.environ.get(
    "PROMPTCASL_GMT_C2CP",
    os.path.join(EXTERNAL_DIR, "c2.cp.v7.5.1.symbols.gmt"),
)
# BiomedBERT encoder (a local path or a HuggingFace model id)
BIOMEDBERT = os.environ.get(
    "PROMPTCASL_BIOMEDBERT",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
)

# ---------------------------------------------------------------------------
# Project data (relative to the repository root)
# ---------------------------------------------------------------------------
TEXT_TEMPLATE_DIR = os.path.join(DATA_DIR, "text_template")
FOLDS_DIR = os.path.join(DATA_DIR, "folds_assignment")
MULTI_OMICS_DIR = os.path.join(DATA_DIR, "multi_omics")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
KG_NEIGHBOR_DIR = os.environ.get(
    "PROMPTCASL_KG_NEIGHBORS", os.path.join(DATA_DIR, "KG_neighbors")
)
SAVE_DIR = os.path.join(PROJECT_ROOT, "save")

# Cell lines used in this study.
CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]
