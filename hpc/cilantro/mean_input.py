"""Cilantro-SL's mean-input control (the "recall: own - mean input" column): the
stage-one regressor and classifier that hpc/cilantro/benchmark.py saved score each
held-out (line, gene a, gene b) row again, with each gene's deletion embedding replaced
by its mean over the split's fitted-side lines (single-cell training and bulk labelled
lines), so the score no longer reads the held-out line. A row is scored when it is
covered in benchmark.py (both genes among the line's deleted genes) and both genes have
a fitted-side mean; every other row gets 0.5, as there. The own-input scores are
computed again from the same weights; their largest difference from
benchmark/scores.parquet goes to benchmark/mean_input.json (0 when the saved weights
are the ones that scored).

    .venv-cilantro/bin/python hpc/cilantro/mean_input.py \
        --run-dir outputs/cilantro_reproduction/benchmark_gc95M
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import NAMES, read_gene  # noqa: E402
from common import (  # noqa: E402
    OFFICIAL_DATA,
    SL_FIT_SIDES,
    SL_HELD_SIDES,
    SL_LABELS,
    SYMBOL_MAP,
    configure,
    paths,
    sl_split_sides,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run_dir)
    p = paths(args.run_dir)
    out = args.run_dir / "benchmark"

    import pickle

    import numpy as np
    import pandas as pd
    import pair_classifier as pc_module
    import torch
    from fast_classifier import fast_pair_classifier, fast_via_film
    from net_layers import FiLMNet, SLNet

    sides = sl_split_sides()
    files = sorted((p["perturb"] / "subtracted_embs").glob("*.csv"))
    with ProcessPoolExecutor(32) as pool:
        rows = pd.concat(pool.map(read_gene, files, chunksize=64), ignore_index=True)
    rows = rows[rows["patient_id"].isin(sides)]
    gene2vec = torch.load(OFFICIAL_DATA / "gene2vec_embs.pt")
    rows = rows[rows["gene"].isin(gene2vec)].reset_index(drop=True)
    side = rows["patient_id"].map(sides)
    columns = [str(i) for i in range(512)]
    fit_mean = rows.loc[side.isin(SL_FIT_SIDES)].groupby("gene")[columns].mean()
    held = rows[side.isin(SL_HELD_SIDES)].reset_index(drop=True)

    def embedding(delta: np.ndarray, frame: pd.DataFrame, ve) -> pd.DataFrame:
        """V(line, gene) of benchmark.py's stage one for the rows of ``frame``."""
        genes = np.stack([gene2vec[g].numpy() for g in frame["gene"]])
        features = pd.DataFrame(np.hstack([delta, genes.astype(np.float64)]))
        _, v = ve.extract_nn_second_last(features)
        return pd.DataFrame(
            v.to_numpy(),
            index=pd.MultiIndex.from_arrays([frame["patient_id"], frame["gene"]]),
        )

    ve = fast_via_film(None, input_length=640)
    ve.net = FiLMNet(protein_size=128).to(ve.device)
    ve.net.load_state_dict(
        torch.load(p["models"] / "benchmark_stage_one.pth", map_location=ve.device)
    )
    own = embedding(held[columns].to_numpy(np.float64), held, ve)
    with_mean = held[held["gene"].isin(fit_mean.index)]
    mean = embedding(
        fit_mean.loc[with_mean["gene"]].to_numpy(np.float64), with_mean, ve
    )

    pc = fast_pair_classifier(
        pd.DataFrame({"SL": []}), model_type=pc_module.ModelType.COMBINED
    )
    pc.net = SLNet().to(pc.device)
    pc.net.load_state_dict(
        torch.load(p["models"] / "benchmark_classifier.pth", map_location=pc.device)
    )

    with open(SYMBOL_MAP, "rb") as f:
        to_ensembl = pickle.load(f)
    columns_read = ["model_id", "gene_a", "gene_b", "label"]
    labels = pd.read_parquet(SL_LABELS, columns=columns_read)
    labels = labels[
        labels["label"].notna() & labels["model_id"].map(sides).isin(SL_HELD_SIDES)
    ]
    every = labels[["model_id", "gene_a", "gene_b"]].drop_duplicates()
    every = every.assign(
        ens_a=every["gene_a"].map(to_ensembl), ens_b=every["gene_b"].map(to_ensembl)
    )

    def score(v: pd.DataFrame) -> pd.DataFrame:
        """benchmark.py's pair frame and classifier where ``v`` covers; else 0.5."""
        have = set(v.index)
        covered = [
            (m, a) in have and (m, b) in have
            for m, a, b in zip(every["model_id"], every["ens_a"], every["ens_b"])
        ]
        pairs = every[covered]
        va = v.reindex(pd.MultiIndex.from_arrays([pairs["model_id"], pairs["ens_a"]]))
        vb = v.reindex(pd.MultiIndex.from_arrays([pairs["model_id"], pairs["ens_b"]]))
        frame = pd.DataFrame(np.hstack([va.to_numpy(), vb.to_numpy()]))
        frame["SL"] = 0
        keys = ["model_id", "gene_a", "gene_b", "ens_a", "ens_b"]
        for i, (name, key) in enumerate(zip(NAMES, keys)):
            frame.insert(i, name, pairs[key].to_numpy())
        _, _, _, probs = pc._predict(*pc._tensors(frame))
        scored = pairs[["model_id", "gene_a", "gene_b"]].assign(score=probs[:, 1])
        result = every[["model_id", "gene_a", "gene_b"]].merge(
            scored, how="left", on=["model_id", "gene_a", "gene_b"]
        )
        return result.assign(
            covered=result["score"].notna(), score=result["score"].fillna(0.5)
        )

    own_scores = score(own)
    mean_scores = score(mean)
    saved = pd.read_parquet(out / "scores.parquet")
    check = own_scores.merge(
        saved, on=["model_id", "gene_a", "gene_b"], suffixes=("", "_saved")
    )
    if (
        len(check) != len(own_scores)
        or not (check["covered"] == check["covered_saved"]).all()
    ):
        raise ValueError("the recomputed own-input rows differ from scores.parquet")
    mean_scores.to_parquet(out / "scores_mean_input.parquet", index=False)
    report = {
        "held_rows": len(mean_scores),
        "own_input_covered": int(own_scores["covered"].sum()),
        "mean_input_covered": int(mean_scores["covered"].sum()),
        "held_gene_rows_without_a_fitted_side_mean": int(
            (~held["gene"].isin(fit_mean.index)).sum()
        ),
        "own_input_max_abs_difference_from_saved": float(
            (check["score"] - check["score_saved"]).abs().max()
        ),
    }
    (out / "mean_input.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
