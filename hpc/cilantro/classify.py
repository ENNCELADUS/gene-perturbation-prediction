"""Released notebook 6_classifier: five-fold SL classification and conformal prediction
for one split regime, one fold per process.

  classify.py --run-dir R --cv 1|3 --fold K [--epochs N] [--released]   # fold K
  classify.py --run-dir R --cv 1|3 --merge [--epochs N] [--released]    # then this

--cv 1 is pair-holdout and --cv 3 gene-holdout (the released split functions).
Transcribed. The notebook's ``run_cv`` trains the five folds in sequence, each fold's
network initialised from the global torch RNG where the previous fold left it. Here
each fold runs in its own process: it rebuilds the same folds from seed 0, replays the
RNG draws of folds 0..K-1 without training (``fast_pair_classifier.skip_fold``), then
runs ``run_cv`` on fold K alone, so fold K's network, batches, weights and outputs are
the sequential run's (design §4.4). --merge assembles the metrics pickle ``run_cv``
writes after five folds (per-fold probabilities, the fold means accumulated in fold
order as its sums are, the per-fold metrics) from the folds' pickles, then, as in the
notebook, runs conformal prediction on the first fold only (``framework.folds = 1``).
The notebook pickles the full train/test DataFrames of every fold; here each fold's
test rows are written with both class probabilities instead, which is what the report
reads. --epochs overrides the notebook's 100 for smoke runs only.

By default the classifier is ``fast_classifier.fast_pair_classifier``, the released
class with its data paths replaced by resident tensors (same batches, same per-row
forward passes; see its docstring and the design's §4.4). --released uses the released
class as is.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    configure,
    mark_done,
    pandarallel_over_pipes,
    paths,
    seed_everything,
)

FOLDS = 5
# run_cv's metric names, in the order it fills its result dictionaries.
METRICS = ("accuracy", "precision", "recall", "f1", "auc", "aupr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cv", type=int, choices=(1, 3), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fold", type=int, choices=range(FOLDS))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--released", action="store_true")
    args = parser.parse_args()
    configure(args.run_dir)
    seed_everything(0)
    p = paths(args.run_dir)
    out = p["metrics"] / f"cv{args.cv}"
    model_dir = p["models"] / f"cv{args.cv}"
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import pair_classifier as pc_module
    import training_framework
    import via_film  # noqa: F401  # notebook 6 imports it; it initialises pandarallel
    from fast_classifier import as_matrix, fast_pair_classifier

    pandarallel_over_pipes()

    classifier = pc_module.pair_classifier if args.released else fast_pair_classifier
    d = pd.read_hdf(p["generated"] / "gene2vec_5x_all_double.h5", "table")
    pc = classifier(d, model_type=pc_module.ModelType.COMBINED)
    nn_save_paths = [str(model_dir / f"nn_path_{i}.pth") for i in range(FOLDS)]
    test, train = pc.setup_cv(FOLDS, cv=args.cv)
    path = "released" if args.released else "fast"

    if args.fold is not None:
        k = args.fold
        for _ in range(k):
            fast_pair_classifier.skip_fold(args.epochs)
        framework = training_framework.Framework(
            None, pc, metrics_path=str(out / f"fold{k}.pkl")
        )
        framework.run_cv(
            [test[k]],
            [train[k]],
            folds=1,
            cv=args.cv,
            nn_save_paths=[nn_save_paths[k]],
            num_epochs=args.epochs,
        )
        scores = test[k][["patient", "gene 1", "gene 2", "SL"]].copy()
        probs = as_matrix(framework.metrics["probs"][0])
        scores["p_non_sl"] = [float(prob) for prob in probs[:, 0]]
        scores["p_sl"] = [float(prob) for prob in probs[:, 1]]
        scores.to_csv(out / f"fold{k}_test_scores.csv.gz", index=False)
        mark_done(
            out / f"fold{k}",
            test_rows=len(test[k]),
            train_rows=len(train[k]),
            results=framework.metrics["general_results"],
            epochs=args.epochs,
            path=path,
        )
        return

    folds = []
    for i in range(FOLDS):
        with open(out / f"fold{i}.pkl", "rb") as f:
            folds.append(pickle.load(f))
    individual = {
        name: [fold["individual_results"][name][0] for fold in folds]
        for name in METRICS
    }
    general = {}
    for name in METRICS:
        total = 0
        for value in individual[name]:
            total += value / FOLDS
        general[name] = total
    framework = training_framework.Framework(
        None, pc, metrics_path=str(out / "gene2vec_5x.pkl")
    )
    framework.metrics = {
        "probs": [fold["probs"][0] for fold in folds],
        "general_results": general,
        "individual_results": individual,
    }
    with open(framework.metrics_path, "wb") as f:
        pickle.dump(framework.metrics, f)

    framework.all_test = test
    framework.all_train = train
    framework.folds = 1

    def extract_name(x):
        return x["patient"]

    framework.uncertainty_quantification(
        training_framework.UQ.MONDRIAN_CONFORMAL,
        net=nn_save_paths,
        mondrian_class_dict={},
        extract_name=extract_name,
    )
    mark_done(
        out,
        folds=FOLDS,
        test_rows=[len(t) for t in test],
        train_rows=[len(t) for t in train],
        general_results=general,
        epochs=args.epochs,
        path=path,
    )


if __name__ == "__main__":
    main()
