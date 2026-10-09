"""Released notebook 3_sampler: SynLethDB SL/non-SL pairs plus expression-correlation
negatives up to 5x positives.

Transcribed cell by cell. Dropped, because nothing downstream reads them: the HDF copy
of sub_embs_only.csv, the CSV round trip of the gene-gene correlation matrix, and the
pickle of the sorted correlation list. Only the first column of sub_embs_only.csv is
read; the notebook reads the whole file to get the same index.
"""

from __future__ import annotations

import argparse
import pickle as pkl
import sys
from ast import literal_eval
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_DATA, SYMBOL_MAP, mark_done, paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    p = paths(args.run_dir)

    import pandas as pd
    from tqdm.auto import tqdm

    sub_embs = pd.read_csv(p["generated"] / "sub_embs_only.csv", usecols=[0])
    sub_embs.set_index("Unnamed: 0", inplace=True)
    sub_embs["gene"] = list(map(lambda x: literal_eval(x)[1], list(sub_embs.index)))
    sub_embs["gene"] = sub_embs["gene"].apply(lambda x: x[5:])
    valid_ensembl = set(sub_embs["gene"])

    SL_df = pd.read_csv(OFFICIAL_DATA / "Human_SL.csv")
    nonSL_df = pd.read_csv(OFFICIAL_DATA / "Human_nonSL.csv")

    # Pull all pos / neg pairs. If a pair is present in both, delete it
    res_pairs = {}
    for i in range(len(SL_df)):
        row = SL_df.iloc[i]
        res_pairs[(row["n1.name"], row["n2.name"])] = True
    num_overlap = 0
    for i in range(len(nonSL_df)):
        row = nonSL_df.iloc[i]
        g1, g2 = row["n1.name"], row["n2.name"]
        if (g1, g2) in res_pairs:
            num_overlap += 1
            del res_pairs[(g1, g2)]
        else:
            res_pairs[(g1, g2)] = False
    p["pairs"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["pairs"].parent / "all_pairs_dict.pkl", "wb") as f:
        pkl.dump(res_pairs, f)
    pair_list = res_pairs

    df = pd.read_csv(p["expression"])
    df.set_index("Unnamed: 0", inplace=True)
    corr_mat = df.corr()
    corr_mat.index = list(map(lambda x: x.split(" ")[0], corr_mat.index))
    corr_mat.columns = list(map(lambda x: x.split(" ")[0], corr_mat.columns))
    corr_dict = corr_mat.to_dict()
    del corr_mat

    res = set()
    for k, v in tqdm(corr_dict.items()):
        for k2, v2 in v.items():
            if k < k2:
                res.add((k, k2, v2))
            else:
                res.add((k2, k, v2))
    del corr_dict
    sorted_items = sorted(res, key=lambda item: item[2])
    del res

    with open(SYMBOL_MAP, "rb") as f:
        id_gene_dict = pkl.load(f)

    def query_id(g):
        if g in id_gene_dict:
            if id_gene_dict[g] in valid_ensembl:
                return id_gene_dict[g]
        return " "

    # subset pairlist to only valid genes
    pair_list = {
        (g1, g2): v
        for ((g1, g2), v) in pair_list.items()
        if query_id(g1) != " " and query_id(g2) != " "
    }
    pos_pairs = {key: value for key, value in pair_list.items() if value}
    sorted_items_f = list(
        filter(lambda x: query_id(x[0]) != " " and query_id(x[1]) != " ", sorted_items)
    )
    del sorted_items

    num_pos = len(pos_pairs)
    num_neg = len(pair_list) - num_pos

    includes_all = []
    num_added = 0
    sample_param = 5
    for i, (g1, g2, corr) in tqdm(enumerate(sorted_items_f)):
        g1_ens, g2_ens = query_id(g1), query_id(g2)
        if g1_ens == " " or g2_ens == " ":
            continue
        if (g1, g2, corr) in includes_all or (g1, g2) in pair_list:
            continue
        if len(includes_all) + num_neg >= sample_param * num_pos:
            break
        includes_all.append((g1, g2, corr))
        num_added += 1
        if num_added % 10000 == 0:
            print(f"added {num_added} so far")

    for g1, g2, _ in includes_all:
        pair_list[(g1, g2)] = False

    with open(p["pairs"], "wb") as f:
        pkl.dump(pair_list, f)
    n_pos = sum(1 for v in pair_list.values() if v)
    mark_done(
        p["pairs"].parent,
        valid_genes=len(valid_ensembl),
        sl_nonsl_overlap_removed=num_overlap,
        pairs=len(pair_list),
        positives=n_pos,
        negatives=len(pair_list) - n_pos,
        sampled_negatives=len(includes_all),
        sampled_correlation_range=[includes_all[0][2], includes_all[-1][2]]
        if includes_all
        else None,
    )


if __name__ == "__main__":
    main()
