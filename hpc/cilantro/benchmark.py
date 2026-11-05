"""Cilantro-SL on the multi-study SL benchmark: the released stage one and classifier,
refitted on the SL split's lines (run.json "benchmark": true; after the deletion step).

- Rows: each (line, gene) deletion embedding of the run (perturb/subtracted_embs, which
  also holds the lines without GeneEffect), with the gene's Gene2vec embedding appended
  as notebook 4 does; a gene without Gene2vec has no row.
- Stage one: the released FiLM regressor and settings (``train_and_test(0.0)``, batch
  512, lr 0.001) on the rows of the split's fitted sides (single-cell training and bulk
  labelled lines) that have GeneEffect, never on validation or test lines; its 32-d
  penultimate layer is the viability embedding V(line, gene) of every row.
- Classifier: the released SLNet and settings (class-weighted cross entropy, batch 128,
  lr 0.0001) on [V(line, a); V(line, b)] of the training lines' genetic rows, one per
  (line, gene a, gene b), SL when half or more of its rows are SL. The released code
  copies one global SynLethDB label to every line; here each line has its own.
- Output: the SL probability of each held-out (line, gene a, gene b); a pair whose genes
  are not both among the line's deleted genes (its top 2,048 tokens) gets 0.5 and ties.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    OFFICIAL_DATA,
    SL_FIT_SIDES,
    SL_HELD_SIDES,
    SL_LABELS,
    SYMBOL_MAP,
    configure,
    mark_done,
    pandarallel_over_pipes,
    paths,
    seed_everything,
    sl_split_sides,
)

NAMES = ["patient", "gene 1", "gene 2", "gene 1 ensembl", "gene 2 ensembl"]


def read_gene(path: Path):
    import pandas as pd

    frame = pd.read_csv(path)
    # one row per line, as the released process_sub keeps the first viability
    frame = frame.drop_duplicates("patient_id").assign(gene=path.stem)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    configure(args.run_dir)
    seed_everything(0)
    p = paths(args.run_dir)
    out = args.run_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    p["models"].mkdir(parents=True, exist_ok=True)

    import pickle

    import numpy as np
    import pandas as pd
    import pair_classifier as pc_module
    import torch
    import via_film  # noqa: F401  # initialises pandarallel, as the notebooks do
    from fast_classifier import fast_pair_classifier, fast_via_film
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    pandarallel_over_pipes()
    sides = sl_split_sides()

    files = sorted((p["perturb"] / "subtracted_embs").glob("*.csv"))
    with ProcessPoolExecutor(32) as pool:
        rows = pd.concat(pool.map(read_gene, files, chunksize=64), ignore_index=True)
    rows = rows[rows["patient_id"].isin(sides)]
    gene2vec = torch.load(OFFICIAL_DATA / "gene2vec_embs.pt")
    rows = rows[rows["gene"].isin(gene2vec)].reset_index(drop=True)
    delta = rows[[str(i) for i in range(512)]].to_numpy(np.float64)
    genes = np.stack([gene2vec[g].numpy() for g in rows["gene"]]).astype(np.float64)
    features = pd.DataFrame(np.hstack([delta, genes]))
    side = rows["patient_id"].map(sides)

    # stage one
    fit = side.isin(SL_FIT_SIDES) & rows["viability"].notna()
    ve = fast_via_film(
        features[fit].assign(viability=rows.loc[fit, "viability"]), input_length=640
    )
    ve.init_regression(batch_size=512, lr=0.001, num_epochs=args.epochs)
    ve.train_and_test(0.0, str(p["models"] / "benchmark_stage_one.pth"))
    predicted, embedding = ve.extract_nn_second_last(features)
    held_ge = side.isin(SL_HELD_SIDES) & rows["viability"].notna()
    stage_one = {
        name: {
            "rows": int(mask.sum()),
            "spearman": float(
                spearmanr(predicted[mask.to_numpy()], rows.loc[mask, "viability"])[0]
            ),
        }
        for name, mask in (("fitted rows", fit), ("held-out rows", held_ge))
    }
    v = pd.DataFrame(
        embedding.to_numpy(),
        index=pd.MultiIndex.from_arrays([rows["patient_id"], rows["gene"]]),
    )

    # classifier
    with open(SYMBOL_MAP, "rb") as f:
        to_ensembl = pickle.load(f)
    labels = pd.read_parquet(
        SL_LABELS, columns=["model_id", "gene_a", "gene_b", "label", "condition"]
    )
    labels = labels[labels["label"].notna() & labels["model_id"].isin(sides)]
    labels = labels.assign(
        side=labels["model_id"].map(sides),
        ens_a=labels["gene_a"].map(to_ensembl),
        ens_b=labels["gene_b"].map(to_ensembl),
    )
    have = set(v.index)
    labels["covered"] = [
        (m, a) in have and (m, b) in have
        for m, a, b in zip(labels["model_id"], labels["ens_a"], labels["ens_b"])
    ]

    def pair_frame(chosen: pd.DataFrame) -> pd.DataFrame:
        keys = ["model_id", "gene_a", "gene_b", "ens_a", "ens_b"]
        rate = chosen.groupby(keys)["label"].mean().reset_index()
        va = v.reindex(pd.MultiIndex.from_arrays([rate["model_id"], rate["ens_a"]]))
        vb = v.reindex(pd.MultiIndex.from_arrays([rate["model_id"], rate["ens_b"]]))
        frame = pd.DataFrame(np.hstack([va.to_numpy(), vb.to_numpy()]))
        frame["SL"] = (rate["label"] >= 0.5).to_numpy().astype(int)
        for i, (name, key) in enumerate(zip(NAMES, keys)):
            frame.insert(i, name, rate[key].to_numpy())
        return frame

    genetic = labels[labels["covered"] & (labels["condition"] == "genetic")]
    train = pair_frame(genetic[genetic["side"].isin(SL_FIT_SIDES)])
    val = pair_frame(genetic[genetic["side"] == "val"])
    held = labels[labels["side"].isin(SL_HELD_SIDES) & labels["covered"]]
    held = pair_frame(held)
    pc = fast_pair_classifier(train.copy(), model_type=pc_module.ModelType.COMBINED)
    pc.init_classification(batch_size=128, lr=0.0001, num_epochs=args.epochs)
    pc.setup_dataloaders_from_sets(val.copy(), train.copy())
    pc.train_nn(str(p["models"] / "benchmark_classifier.pth"))
    _, _, _, probs = pc._predict(*pc._tensors(held.copy()))
    held_scores = held[["patient", "gene 1", "gene 2"]].assign(score=probs[:, 1])
    held_scores.columns = ["model_id", "gene_a", "gene_b", "score"]

    every = labels[labels["side"].isin(SL_HELD_SIDES)]
    every = every[["model_id", "gene_a", "gene_b"]].drop_duplicates()
    scores = every.merge(held_scores, how="left", on=["model_id", "gene_a", "gene_b"])
    scores = scores.assign(
        covered=scores["score"].notna(), score=scores["score"].fillna(0.5)
    )
    scores.to_parquet(out / "scores.parquet", index=False)
    val_scores = held_scores.merge(
        val[["patient", "gene 1", "gene 2", "SL"]].set_axis(
            ["model_id", "gene_a", "gene_b", "SL"], axis=1
        )
    )
    held_rows = labels[labels["side"].isin(SL_HELD_SIDES)]
    report = {
        "deletion_rows": len(rows),
        "lines": int(rows["patient_id"].nunique()),
        "genes": int(rows["gene"].nunique()),
        "stage_one": stage_one,
        "stage_one_loss": ve.loss_arr,
        "classifier_training_pairs": len(train),
        "classifier_training_sl": int(train["SL"].sum()),
        "classifier_loss": pc.loss_arr,
        "validation_pairs_auroc": float(
            roc_auc_score(val_scores["SL"].astype(int), val_scores["score"])
        ),
        "held_rows": len(held_rows),
        "held_rows_covered_share": float(held_rows["covered"].mean()),
        "held_genetic_rows_covered_share": float(
            held_rows.loc[held_rows["condition"] == "genetic", "covered"].mean()
        ),
        "epochs": args.epochs,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=1) + "\n")
    mark_done(out, **{k: v for k, v in report.items() if "loss" not in k})


if __name__ == "__main__":
    main()
