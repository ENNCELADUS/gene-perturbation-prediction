"""Equivalence check of the harness's fast paths against the released code, on the
inputs and outputs of a finished (smoke) run.

  .venv-cilantro/bin/python hpc/cilantro/check_equivalence.py --reference RUN_DIR
      --out DIR [--genes 5] [--epochs 2] [--released-deletion] [--benchmark-rows N]
      [--skip-deletion] [--skip-classifier] [--skip-stage-one]

Deletion: ``fast_perturb.Deleter`` recomputes the first --genes genes (sorted) of the
reference run from its tokenized dataset into DIR/fast and the two CSVs per gene are
compared byte for byte with the reference's; on a difference the maximum absolute
difference of the parsed values is reported. --released-deletion also runs the released
``via_perturber`` on the same genes into DIR/released (fresh evidence of determinism,
and a timing of the released path), compared the same way.

Classifier: on the reference run's pair table, folds 0 and 1 of the pair-holdout split
in sequence for --epochs epochs, trained and scored once by the released
``pair_classifier`` and once by ``fast_pair_classifier`` from the same RNG state;
compares per fold the trained weights, predictions, probabilities, released metrics
and the RNG state after scoring (which seeds the next fold), and the conformal outputs
(calibration scores, p-values) of fold 0's saved model.

Stage one: the same for ``via_film`` against ``fast_via_film``: weights, the in-sample
test predictions and targets, and the penultimate-layer table written to HDF5.

--benchmark-rows N times one training epoch of both classifiers on a synthetic
table of N rows (the per-epoch cost at full scale; no equivalence claim).

Writes DIR/equivalence.json and prints a summary. GPU required. Reference outputs are
read only; DIR/run holds symlinks to the reference's inputs for ``configure``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import configure, paths, seed_everything  # noqa: E402


def rng_snapshot():
    import numpy as np
    import torch

    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )


def rng_restore(state):
    import numpy as np
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def rng_equal(a, b) -> bool:
    import numpy as np
    import torch

    return (
        a[0] == b[0]
        and all(np.array_equal(x, y) for x, y in zip(a[1], b[1]))
        and torch.equal(a[2], b[2])
        and torch.equal(a[3], b[3])
    )


def state_dict_equal(a: dict, b: dict) -> bool:
    import torch

    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def csv_compare(a: Path, b: Path) -> dict:
    import numpy as np
    import pandas as pd

    if a.read_bytes() == b.read_bytes():
        return {"bytes_equal": True, "max_abs_diff": 0.0}
    x = pd.read_csv(a, index_col=0)
    y = pd.read_csv(b, index_col=0)
    same_index = list(x.index) == list(y.index) and list(x.columns) == list(y.columns)
    diff = (
        float(np.nanmax(np.abs(x.to_numpy(dtype=float) - y.to_numpy(dtype=float))))
        if same_index
        else float("inf")
    )
    return {"bytes_equal": False, "same_index": same_index, "max_abs_diff": diff}


def scratch_run_dir(reference: Path, out: Path) -> Path:
    run_dir = out / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(reference / "run.json", run_dir / "run.json")
    for name in ("tokenized", "inputs", "tokenize_input"):
        link = run_dir / name
        if not link.exists():
            link.symlink_to((reference / name).resolve())
    return run_dir


def check_deletion(
    reference: Path, out: Path, n_genes: int, run_released: bool
) -> dict:
    from fast_perturb import Deleter

    run_dir = scratch_run_dir(reference, out)
    p = paths(run_dir)
    from common import DICTIONARIES, GENEFORMER_MODEL, SYMBOL_MAP, run_config

    genes = sorted(
        f.stem for f in (reference / "perturb" / "subtracted_embs").glob("*.csv")
    )
    genes = genes[:n_genes]
    result = {"genes": genes, "fast": {}, "released": {}}

    t0 = time.time()
    deleter = Deleter(
        dataset_path=p["tokenized"],
        model_dir=GENEFORMER_MODEL,
        token_dictionary=DICTIONARIES[run_config(run_dir)["dictionary"]]["token"],
        ensembl_mapping=SYMBOL_MAP,
        gene_effect_csv=p["gene_effect"],
        out_dir=out / "fast",
    )
    result["fast"]["startup_seconds"] = round(time.time() - t0, 1)
    for gene in genes:
        for sub in ("subtracted_embs", "perturbed_embs"):
            (out / "fast" / sub / f"{gene}.csv").unlink(missing_ok=True)
        t0 = time.time()
        deleter.perturb_gene(gene)
        seconds = time.time() - t0
        entry = {"seconds": round(seconds, 1)}
        for sub in ("subtracted_embs", "perturbed_embs"):
            entry[sub] = csv_compare(
                out / "fast" / sub / f"{gene}.csv",
                reference / "perturb" / sub / f"{gene}.csv",
            )
        entry["lines"] = (
            sum(1 for _ in open(out / "fast" / "subtracted_embs" / f"{gene}.csv")) - 1
        )
        result["fast"][gene] = entry
    del deleter
    import torch

    torch.cuda.empty_cache()

    if run_released:
        from perturb import released_perturber

        shard_dir = out / "released_shard"
        gene_list = shard_dir / "genes.pkl"
        shard_dir.mkdir(parents=True, exist_ok=True)
        with open(gene_list, "wb") as f:
            pickle.dump(genes, f)
        released_dir = out / "released"
        for sub in ("subtracted_embs", "perturbed_embs"):
            shutil.rmtree(released_dir / sub, ignore_errors=True)
        os.environ["CILANTRO_RES_DIR"] = str(shard_dir / "res") + "/"
        vp = released_perturber(run_dir, shard_dir, gene_list)
        vp.embs_prefix = str(released_dir) + "/"
        for sub in ("subtracted_embs", "perturbed_embs"):
            (released_dir / sub).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        vp.perturb_all_genes()
        result["released"]["seconds_total"] = round(time.time() - t0, 1)
        for gene in genes:
            entry = {}
            for sub in ("subtracted_embs", "perturbed_embs"):
                entry[sub + "_vs_reference"] = csv_compare(
                    released_dir / sub / f"{gene}.csv",
                    reference / "perturb" / sub / f"{gene}.csv",
                )
                entry[sub + "_vs_fast"] = csv_compare(
                    released_dir / sub / f"{gene}.csv",
                    out / "fast" / sub / f"{gene}.csv",
                )
            result["released"][gene] = entry
    return result


def check_classifier(reference: Path, out: Path, epochs: int) -> dict:
    import pandas as pd
    import pair_classifier as pc_module
    import via_film  # noqa: F401  # initialises pandarallel, as the harness imports it
    from fast_classifier import as_matrix, fast_pair_classifier

    seed_everything(0)
    d = pd.read_hdf(reference / "generated_df" / "gene2vec_5x_all_double.h5", "table")
    released = pc_module.pair_classifier(
        d.copy(), model_type=pc_module.ModelType.COMBINED
    )
    fast = fast_pair_classifier(d.copy(), model_type=pc_module.ModelType.COMBINED)
    test, train = released.setup_cv(5, cv=1)
    start = rng_snapshot()

    def run(pc, tag):
        """Folds 0 and 1 in sequence, as run_cv does, so the RNG stream fold 1 starts
        from is the one fold 0 leaves; conformal prediction on fold 0, as notebook 6."""
        rng_restore(start)
        folds = []
        for i in range(2):
            path = str(out / f"classifier_{tag}_fold{i}.pth")
            pc.init_classification(batch_size=128, lr=0.0001, num_epochs=epochs)
            pc.setup_dataloaders_from_sets(test[i], train[i])
            t0 = time.time()
            pc.train_nn(path)
            t_train = time.time() - t0
            t0 = time.time()
            vals, corr, _, probs = pc.test_nn()
            t_test = time.time() - t0
            metrics = pc.compute_acc(vals, corr, probs)
            folds.append(
                {
                    "state": {
                        k: v.cpu().clone() for k, v in pc.net.state_dict().items()
                    },
                    "vals": vals,
                    "corr": corr,
                    "probs": as_matrix(probs),
                    "metrics": [float(m) for m in metrics],
                    "rng_after": rng_snapshot(),
                    "timing": {
                        "train_seconds_per_epoch": round(t_train / epochs, 2),
                        "test_seconds": round(t_test, 2),
                        "train_rows": len(train[i]),
                        "test_rows": len(test[i]),
                    },
                }
            )
        pc.load_net(str(out / f"classifier_{tag}_fold0.pth"))
        calib_dl, test_dl = pc.get_calib_test(
            test[0], 0.1, mondrian_class_dict={}, extract_name=lambda x: x["patient"]
        )
        t0 = time.time()
        c_vals, c_corr, conf_scores, _, calib_scores, c_probs = pc.conformal_pred(
            calib_dl,
            test_dl,
            mondrian_class_dict={},
            extract_name=lambda x: x["patient"],
        )
        return {
            "folds": folds,
            "c_vals": c_vals,
            "c_corr": c_corr,
            "conf_scores": conf_scores,
            "calib_scores": calib_scores,
            "c_probs": as_matrix(c_probs),
            "conformal_seconds": round(time.time() - t0, 2),
        }

    r = run(released, "released")
    f = run(fast, "fast")
    import numpy as np

    folds = []
    for rf, ff in zip(r["folds"], f["folds"]):
        folds.append(
            {
                "weights_equal": state_dict_equal(rf["state"], ff["state"]),
                "predictions_equal": rf["vals"] == ff["vals"]
                and rf["corr"] == ff["corr"],
                "probs_equal": np.array_equal(rf["probs"], ff["probs"]),
                "probs_max_abs_diff": float(np.abs(rf["probs"] - ff["probs"]).max()),
                "metrics_equal": rf["metrics"] == ff["metrics"],
                "metrics": {"released": rf["metrics"], "fast": ff["metrics"]},
                "rng_after_equal": rng_equal(rf["rng_after"], ff["rng_after"]),
                "timing": {"released": rf["timing"], "fast": ff["timing"]},
            }
        )
    return {
        "epochs": epochs,
        "folds": folds,
        "conformal_predictions_equal": r["c_vals"] == f["c_vals"]
        and r["c_corr"] == f["c_corr"],
        "conformal_p_values_equal": r["conf_scores"] == f["conf_scores"],
        "calibration_scores_equal": r["calib_scores"] == f["calib_scores"],
        "conformal_probs_equal": np.array_equal(r["c_probs"], f["c_probs"]),
        "conformal_seconds": {
            "released": r["conformal_seconds"],
            "fast": f["conformal_seconds"],
        },
    }


def check_stage_one(reference: Path, out: Path, epochs: int) -> dict:
    import numpy as np
    import pandas as pd
    import via_film
    from fast_classifier import fast_via_film

    seed_everything(0)
    df = pd.read_hdf(reference / "generated_df" / "gene2vec_emb_mat.h5", "table")
    start = rng_snapshot()

    def run(cls, tag):
        rng_restore(start)
        ve = cls(df, input_length=640)
        path = str(out / f"stage_one_{tag}.pth")
        layers = str(out / f"stage_one_{tag}_layers.h5")
        ve.init_regression(batch_size=512, lr=0.001, num_epochs=epochs)
        t0 = time.time()
        ve.train_and_test(0.0, path)
        t_train = time.time() - t0
        ve.setup_dataloaders(0.999)
        t0 = time.time()
        vals, y_true = ve.test_nn()
        t_test = time.time() - t0
        after = rng_snapshot()
        t0 = time.time()
        ve.extract_nn_second_last(ve.df.drop(columns="viability score"), layers)
        t_extract = time.time() - t0
        return {
            "state": {k: v.cpu().clone() for k, v in ve.net.state_dict().items()},
            "vals": vals,
            "y_true": y_true,
            "rng_after": after,
            "layers": pd.read_hdf(layers, "table"),
            "timing": {
                "train_seconds_per_epoch": round(t_train / epochs, 2),
                "test_seconds": round(t_test, 2),
                "extract_seconds": round(t_extract, 2),
                "rows": len(df),
            },
        }

    r = run(via_film.via_film, "released")
    f = run(fast_via_film, "fast")
    return {
        "epochs": epochs,
        "weights_equal": state_dict_equal(r["state"], f["state"]),
        "test_predictions_equal": r["vals"] == f["vals"] and r["y_true"] == f["y_true"],
        "test_predictions_max_abs_diff": float(
            np.abs(np.asarray(r["vals"]) - np.asarray(f["vals"])).max()
        ),
        "rng_after_equal": rng_equal(r["rng_after"], f["rng_after"]),
        "layers_index_equal": list(r["layers"].index) == list(f["layers"].index)
        and list(r["layers"].columns) == list(f["layers"].columns),
        "layers_dtype": [
            str(r["layers"].dtypes.iloc[0]),
            str(f["layers"].dtypes.iloc[0]),
        ],
        "layers_values_equal": np.array_equal(
            r["layers"].to_numpy(), f["layers"].to_numpy()
        ),
        "layers_max_abs_diff": float(
            np.abs(
                r["layers"].to_numpy(dtype=float) - f["layers"].to_numpy(dtype=float)
            ).max()
        ),
        "timing": {"released": r["timing"], "fast": f["timing"]},
    }


def benchmark(rows: int, out: Path) -> dict:
    """One training epoch of both classifiers on a synthetic table (timing only)."""
    import numpy as np
    import pandas as pd
    import pair_classifier as pc_module
    from fast_classifier import fast_pair_classifier

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.standard_normal((rows, 64), dtype=np.float32), columns=range(64)
    )
    for name in ("patient", "gene 1", "gene 2", "gene 1 ensembl", "gene 2 ensembl"):
        frame.insert(0, name, "x")
    frame["SL"] = rng.integers(0, 2, rows)
    test = frame.iloc[:1000]
    result = {"rows": rows}
    for tag, cls in (
        ("released", pc_module.pair_classifier),
        ("fast", fast_pair_classifier),
    ):
        seed_everything(0)
        pc = cls(frame.copy(), model_type=pc_module.ModelType.COMBINED)
        pc.init_classification(batch_size=128, lr=0.0001, num_epochs=1)
        pc.setup_dataloaders_from_sets(test, frame)
        t0 = time.time()
        pc.train_nn(None)
        result[tag + "_seconds_per_epoch"] = round(time.time() - t0, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--genes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--released-deletion", action="store_true")
    parser.add_argument("--benchmark-rows", type=int, default=0)
    parser.add_argument("--skip-deletion", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--skip-stage-one", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = scratch_run_dir(args.reference, args.out)
    configure(run_dir, res_dir=args.out / "res")
    os.environ["CILANTRO_MODEL_DIR"] = str(args.out)

    report = {"reference": str(args.reference)}
    if not args.skip_deletion:
        report["deletion"] = check_deletion(
            args.reference, args.out, args.genes, args.released_deletion
        )
    if not args.skip_classifier:
        report["classifier"] = check_classifier(args.reference, args.out, args.epochs)
    if not args.skip_stage_one:
        report["stage_one"] = check_stage_one(args.reference, args.out, args.epochs)
    if args.benchmark_rows:
        report["benchmark"] = benchmark(args.benchmark_rows, args.out)
    (args.out / "equivalence.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
