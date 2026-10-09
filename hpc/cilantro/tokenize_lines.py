"""Released notebook 1_tokenizer: build the SL-gene + 500-HVG expression matrix and
tokenize it for Geneformer.

Transcribed cell by cell; only paths change, and the tokenizer is given the run's
dictionary explicitly (gc95M is the fork's default, so the as-released run is
unchanged). The notebook's unused full-matrix h5ad is not written. The released
perturber reads its gene list from Human_SL_genes.pkl, which the Drive folder does not
hold; it is rebuilt here as the tokenized gene universe (the paper: "for all HVGs and
relevant SL and non-SL genes").
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DICTIONARIES,
    OFFICIAL_DATA,
    SYMBOL_MAP,
    configure,
    mark_done,
    paths,
    run_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run_dir)
    p = paths(args.run_dir)
    dictionary = DICTIONARIES[run_config(args.run_dir)["dictionary"]]

    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from geneformer import TranscriptomeTokenizer
    from tqdm import tqdm

    pd.options.mode.chained_assignment = None

    omics_expr = pd.read_csv(p["expression"])
    omics_expr.set_index("Unnamed: 0", inplace=True)
    metadata = pd.read_csv(p["metadata"])
    metadata.set_index("ModelID", inplace=True)

    adata = ad.AnnData(omics_expr)
    adata.obs_names = [str(i).split(" ")[0] for i in omics_expr.index]
    adata.var_names = [str(i).split(" ")[0] for i in omics_expr.columns]
    adata.obs["patient_id"] = omics_expr.index
    adata.obs["cell_line"] = metadata["OncotreeLineage"]
    adata.obs["disease"] = metadata["OncotreePrimaryDisease"]
    for patient in adata.obs_names:
        adata.obs.loc[patient, "cell_line"] = metadata.loc[
            patient, "StrippedCellLineName"
        ]
        if metadata.loc[patient, "OncotreePrimaryDisease"] != "Non-Cancerous":
            adata.obs.loc[patient, "disease_state"] = "Cancerous"
        else:
            adata.obs.loc[patient, "disease_state"] = "Non-Cancerous"
        adata.obs.loc[patient, "disease"] = metadata.loc[
            patient, "OncotreePrimaryDisease"
        ]

    with open(SYMBOL_MAP, "rb") as f:
        id_gene_dict = pickle.load(f)

    def query_id(g):
        if g in id_gene_dict:
            return id_gene_dict[g]
        return " "

    with open(OFFICIAL_DATA / "Human_SL_pairs.pkl", "rb") as f:
        SL_genes = pickle.load(f)
    SL_ensembl = [query_id(g) for g in SL_genes.keys()]
    SL_ensembl = list(filter(lambda x: x != " ", SL_ensembl))

    lst, genes = [], []
    for gene in adata.var_names:
        gene2 = gene.split(" ")[0]
        if gene2 in id_gene_dict:
            lst.append(id_gene_dict[gene2])
            genes.append(gene2)
        else:
            lst.append(None)
    filtered_results = [val for val in lst if val is not None]

    # 26Q1 adds an ATXN8 column, which the gc95M map sends to TWNK's Ensembl ID; the
    # notebook's HVG loop cannot handle a repeated ID, so keep the first column per ID.
    first = ~pd.Index(filtered_results).duplicated()
    dropped = [g for g, keep in zip(genes, first) if not keep]
    genes = [g for g, keep in zip(genes, first) if keep]
    filtered_results = [e for e, keep in zip(filtered_results, first) if keep]
    print(f"columns dropped for a repeated Ensembl ID: {dropped}")

    adata2 = adata[:, genes]
    adata2.var_names = filtered_results
    adata2.var["ensembl_id"] = filtered_results
    adata2.obs["n_counts"] = adata2.X.sum(axis=1)
    adata2.obs["cell_type"] = adata.obs["cell_line"]
    adata2.obs["disease"] = adata.obs["disease"]
    adata2.obs["disease_state"] = adata.obs["disease_state"]
    adata2.obs["batch"] = 0
    adata2.obs["patient_id"] = adata.obs["patient_id"]
    adata2.var["gene_name"] = genes

    sc.pp.highly_variable_genes(adata2, n_top_genes=500, inplace=True)
    hvgs = adata2.var["highly_variable"]
    total_overlap = total_missing = 0
    for gene in tqdm(SL_ensembl):
        if gene in hvgs:
            if hvgs[gene]:
                total_overlap += 1
            hvgs[gene] = True
        else:
            total_missing += 1
    print(
        f"SL genes already HVG: {total_overlap}; "
        f"SL genes not in the expression matrix: {total_missing}"
    )

    adata_hvg = adata2[:, hvgs]
    p["tokenize_input"].mkdir(parents=True, exist_ok=True)
    adata_hvg.write_h5ad(p["sl_hvg_h5ad"])

    tk = TranscriptomeTokenizer(
        {
            "cell_type": "cell_type",
            "disease": "disease",
            "disease_state": "disease_state",
            "patient_id": "patient_id",
        },
        nproc=16,
        special_token=False,
        model_input_size=2048,
        gene_median_file=dictionary["median"],
        token_dictionary_file=dictionary["token"],
        gene_mapping_file=dictionary["mapping"],
    )
    p["tokenized"].parent.mkdir(parents=True, exist_ok=True)
    tk.tokenize_data(
        p["tokenize_input"],
        p["tokenized"].parent,
        "hvg_500_w_SL_tokenized_2048",
        file_format="h5ad",
    )

    genes_to_perturb = list(dict.fromkeys(adata_hvg.var_names))
    with open(p["genes_to_perturb"], "wb") as f:
        pickle.dump(genes_to_perturb, f)
    mark_done(
        p["tokenized"].parent,
        lines=int(adata_hvg.n_obs),
        genes=int(adata_hvg.n_vars),
        genes_to_perturb=len(genes_to_perturb),
        sl_genes_mapped=len(SL_ensembl),
        sl_genes_already_hvg=total_overlap,
        sl_genes_missing=total_missing,
        columns_dropped_for_repeated_ensembl_id=dropped,
    )


if __name__ == "__main__":
    main()
