"""Released notebook 4_setup_input: append each gene's Gene2vec embedding to its delta-
embedding rows.

Transcribed cell by cell. The notebook leaves `sub_embs.set_index("Unnamed: 0")`
commented out, but its loop parses the index as a (line, gene) tuple string, which only
exists after that call; it is applied here.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_DATA, mark_done, paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    p = paths(args.run_dir)

    import pandas as pd
    import torch
    from tqdm import tqdm

    gene2vec = torch.load(OFFICIAL_DATA / "gene2vec_embs.pt")
    sub_embs = pd.read_csv(p["generated"] / "sub_embs_only.csv")
    sub_embs.rename(columns={"512": "viability score"}, inplace=True)
    res_file = p["generated"] / "gene2vec_emb_mat.h5"
    sub_embs.set_index("Unnamed: 0", inplace=True)

    gene_dict = {}
    df_no_via = sub_embs.drop(columns="viability score")
    via = sub_embs["viability score"]
    print("Concatenating gene embeddings...")
    for entry in tqdm(sub_embs.index):
        e = ast.literal_eval(entry)
        gene_id = e[1][5:]
        if gene_id in gene2vec:
            gene_repr = gene2vec[gene_id].tolist()
            gene_dict[entry] = list(df_no_via.loc[entry]) + gene_repr + [via.loc[entry]]

    df_results = pd.DataFrame.from_dict(gene_dict, orient="index")
    orig_cols = [str(c) for c in df_no_via.columns]
    gene_cols = [f"gene_dim_{i}" for i in range(len(gene_repr))]
    df_results.columns = orig_cols + gene_cols + ["viability score"]
    print(f"Saving combined matrix to {res_file}")
    df_results.to_hdf(res_file, key="table")
    mark_done(
        p["generated"] / "gene2vec",
        rows=len(df_results),
        rows_without_gene2vec=len(sub_embs) - len(df_results),
    )


if __name__ == "__main__":
    main()
