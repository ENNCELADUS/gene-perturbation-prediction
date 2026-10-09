"""Released notebook 6_classifier: five-fold SL classification and conformal prediction
for one split regime.

--cv 1 is pair-holdout and --cv 3 gene-holdout (the released split functions).
Transcribed; as in the notebook, conformal prediction runs on the first fold only
(`framework.folds = 1`). The notebook pickles the full train/test DataFrames of every
fold; here each fold's test rows are written with both class probabilities instead,
which is what the report reads. --epochs overrides the notebook's 100 for smoke runs
only.

By default the classifier is ``fast_classifier.fast_pair_classifier``, the released
class with its data paths replaced by resident tensors (same batches, same per-row
forward passes; see its docstring and the design's §4.4). --released uses the released
class as is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import configure, mark_done, paths, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cv", type=int, choices=(1, 3), required=True)
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

    classifier = pc_module.pair_classifier if args.released else fast_pair_classifier
    d = pd.read_hdf(p["generated"] / "gene2vec_5x_all_double.h5", "table")
    pc = classifier(d, model_type=pc_module.ModelType.COMBINED)
    framework = training_framework.Framework(
        None, pc, metrics_path=str(out / "gene2vec_5x.pkl")
    )
    nn_save_paths = [str(model_dir / f"nn_path_{i}.pth") for i in range(5)]

    test, train = pc.setup_cv(5, cv=args.cv)
    framework.run_cv(
        test, train, cv=args.cv, nn_save_paths=nn_save_paths, num_epochs=args.epochs
    )
    for i, fold in enumerate(test):
        scores = fold[["patient", "gene 1", "gene 2", "SL"]].copy()
        probs = as_matrix(framework.metrics["probs"][i])
        scores["p_non_sl"] = [float(prob) for prob in probs[:, 0]]
        scores["p_sl"] = [float(prob) for prob in probs[:, 1]]
        scores.to_csv(out / f"fold{i}_test_scores.csv.gz", index=False)

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
        folds=5,
        test_rows=[len(t) for t in test],
        train_rows=[len(t) for t in train],
        general_results=framework.metrics["general_results"],
        epochs=args.epochs,
        path="released" if args.released else "fast",
    )


if __name__ == "__main__":
    main()
