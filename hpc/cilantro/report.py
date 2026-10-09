"""Summarise one Cilantro-SL harness run: classifier, conformal, stage one.

For each split regime (pair-holdout cv1, gene-holdout cv3) and fold, from the test rows
classify.py writes: the released metrics (two-class AP, which is what
`average_precision_score` on one-hot labels returns, and positive-class F1 at the
argmax), then positive-class AUPR, its prior and AUPR minus prior, and AUROC, at row
level (one row per line and pair, as the released code scores) and at pair level (the
line scores averaged, as the paper describes). Conformal coverage and mean set size over
an alpha grid come from the released conformal outputs (first fold only, as notebook 6
runs it). Writes report.json and report.md in the run directory.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import configure, paths  # noqa: E402

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
REGIMES = {"cv1": "pair-holdout", "cv3": "gene-holdout"}


def classifier_metrics(labels, p_non_sl, p_sl) -> dict:
    """The released compute_acc() metrics from both softmax columns (two-class AP; F1
    at the argmax; AUROC and two-class AP are 0 on a single-class fold, as released),
    then positive-class AUPR, prior, AUPR minus prior and AUROC (undefined there)."""
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probs = np.column_stack([p_non_sl, p_sl]).astype(float)
    both = len(np.unique(labels)) == 2
    released_ap = average_precision_score(np.eye(2)[labels], probs) if both else 0.0
    aupr = average_precision_score(labels, probs[:, 1]) if both else float("nan")
    prior = float(labels.mean())
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "two_class_ap": float(released_ap),
        "f1_argmax": float(
            f1_score(labels, probs.argmax(axis=1), average="binary", zero_division=0)
        ),
        "aupr": float(aupr),
        "prior": prior,
        "aupr_minus_prior": float(aupr - prior),
        "auroc": float(roc_auc_score(labels, probs[:, 1])) if both else float("nan"),
    }


def conformal(metrics: dict) -> dict:
    import numpy as np

    p_best, p_other = (np.asarray(x) for x in zip(*metrics["mondrian_conf_scores"][0]))
    predicted = np.asarray(metrics["mondrian_pred_vals"][0])
    truth = np.asarray(metrics["mondrian_corr"][0])
    p_true = np.where(predicted == truth, p_best, p_other)
    rows = []
    for alpha in ALPHAS:
        size = (p_best > alpha).astype(int) + (p_other > alpha).astype(int)
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "coverage": float(np.mean(p_true > alpha)),
                "mean_set_size": float(np.mean(size)),
                "singleton_share": float(np.mean(size == 1)),
            }
        )
    return {
        "fold": 0,
        "test_rows": int(len(truth)),
        "calibration_rows": int(
            sum(len(v) for v in metrics["calibration_scores"][0].values())
        ),
        "grid": rows,
    }


def summarise(per_fold: list[dict]) -> dict:
    import numpy as np

    keys = [k for k in per_fold[0] if k not in ("n", "positives")]
    return {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "sd": float(np.std([f[k] for f in per_fold], ddof=1)),
        }
        for k in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run_dir)
    p = paths(args.run_dir)

    import pandas as pd

    report = {"run": args.run_dir.name, "regimes": {}}
    fit = p["metrics"] / "stage_one_fit.json"
    if fit.exists():
        report["stage_one_fit"] = json.loads(fit.read_text())
    for cv, regime in REGIMES.items():
        directory = p["metrics"] / cv
        if not (directory / "done.json").exists():
            continue
        rows, pairs = [], []
        for i in range(5):
            scores = pd.read_csv(directory / f"fold{i}_test_scores.csv.gz")
            rows.append(
                classifier_metrics(scores["SL"], scores["p_non_sl"], scores["p_sl"])
            )
            pair = scores.groupby(["gene 1", "gene 2"], sort=False).agg(
                SL=("SL", "first"),
                p_non_sl=("p_non_sl", "mean"),
                p_sl=("p_sl", "mean"),
            )
            pairs.append(classifier_metrics(pair["SL"], pair["p_non_sl"], pair["p_sl"]))
        with open(directory / "gene2vec_5x.pkl", "rb") as f:
            released = pickle.load(f)
        report["regimes"][regime] = {
            "row_level": {"folds": rows, "summary": summarise(rows)},
            "pair_level": {"folds": pairs, "summary": summarise(pairs)},
            "released_general_results": released["general_results"],
            "conformal": conformal(released),
        }
    (args.run_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    lines = [f"# Cilantro-SL harness run `{args.run_dir.name}`", ""]
    if "stage_one_fit" in report:
        s = report["stage_one_fit"]
        lines += [
            f"Stage-one fit ({s['scope']}): RMSE {s['rmse']:.3f}, Spearman "
            f"{s['spearman']:.3f}, R2 {s['r2']:.3f} over {s['rows']:,} rows.",
            "",
        ]
    for regime, r in report["regimes"].items():
        lines += [
            f"## {regime}",
            "",
            "| Level | Metric | Mean | SD |",
            "| --- | --- | ---: | ---: |",
        ]
        for level in ("row_level", "pair_level"):
            for k, v in r[level]["summary"].items():
                lines.append(f"| {level} | {k} | {v['mean']:.4f} | {v['sd']:.4f} |")
        lines += [
            "",
            "| alpha | target | coverage | mean set size | singletons |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for g in r["conformal"]["grid"]:
            lines.append(
                f"| {g['alpha']} | {g['target_coverage']:.2f} | {g['coverage']:.4f} | "
                f"{g['mean_set_size']:.3f} | {g['singleton_share']:.3f} |"
            )
        lines.append("")
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
