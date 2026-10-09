"""Released notebook 5_pretraining: FiLM viability regression on every row, then 32-d
viability embeddings per pair.

Transcribed; the released framework trains on all rows and then scores a random 99.9% of
the same rows, so the stage-one fit recorded here (RMSE, Spearman, R2) is in-sample, the
quantity the released code can produce. --epochs overrides the notebook's 100 for smoke
runs only.

By default the regressor is ``fast_classifier.fast_via_film``, the released class with
its data paths replaced by resident tensors (same batches, same per-row forward
passes; see its docstring and the design's §4.4). --released uses the released class
as is.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import configure, mark_done, paths, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--released", action="store_true")
    args = parser.parse_args()
    configure(args.run_dir)
    seed_everything(0)
    p = paths(args.run_dir)
    p["models"].mkdir(parents=True, exist_ok=True)
    p["metrics"].mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd
    import training_framework
    import via_film
    from fast_classifier import fast_via_film
    from scipy.stats import spearmanr

    regressor = via_film.via_film if args.released else fast_via_film
    name = "gene2vec_5x"
    # The released df_to_SL_embs skips training when its embedding file exists; an
    # interrupted run (no done marker) would then resume without a model. Start over.
    for stale in (f"{name}_all.h5", f"{name}_all_double.h5"):
        (p["generated"] / stale).unlink(missing_ok=True)
    df = pd.read_hdf(p["generated"] / "gene2vec_emb_mat.h5", "table")
    print(f"df shape is {df.shape}")
    ve = regressor(df, input_length=640)
    framework = training_framework.Framework(
        ve, None, metrics_path=str(p["metrics"] / "pretraining.pkl")
    )
    # Notebook 5 passes the pickle's path, but create_SL_embs iterates its argument as
    # the pair dictionary (it would loop over the path's characters); the notebook's
    # commented-out alternative passes the loaded dictionary, as done here.
    with open(p["pairs"], "rb") as f:
        pair_list = pickle.load(f)
    framework.df_to_SL_embs(
        pair_list,
        str(p["generated"] / f"{name}_all.h5"),
        str(p["generated"] / f"{name}_all_double.h5"),
        batch_size=512,
        lr=0.001,
        num_epochs=args.epochs,
        name=name,
    )

    predicted, observed = (np.asarray(x, dtype=float) for x in ve.test_nn())
    residual = predicted - observed
    fit = {
        "scope": "in-sample: random 99.9% of the rows stage one was trained on",
        "rows": int(len(observed)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(predicted, observed).correlation),
        "r2": float(
            1 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)
        ),
        "epochs": args.epochs,
        "path": "released" if args.released else "fast",
    }
    (p["metrics"] / "stage_one_fit.json").write_text(json.dumps(fit, indent=1) + "\n")
    mark_done(p["metrics"] / "pretrain", **fit)


if __name__ == "__main__":
    main()
