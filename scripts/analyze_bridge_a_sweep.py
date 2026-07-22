"""Analyze the Bridge A full-sweep pair scores against Horlbeck gi_score.

Hypothesis under test (T1 kill-test): the symmetrized Bridge A co-dependency
score ``s_a_seed_mean`` should rank synthetic-lethal pairs high. Horlbeck
``gi_score`` is negative for synergistic/synthetic-sick-lethal pairs, so:

* Spearman(s_A, gi_score) should be NEGATIVE (large s_A <-> low gi_score);
* equivalently Spearman(s_A, -gi_score) POSITIVE;
* AUROC(s_A -> is_strong_sl) should be > 0.5.

This is a development kill-test diagnostic, NOT a formal MECHANISTIC verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_BOOL_COLUMNS = (
    "high_effective_n",
    "is_strong_sl",
    "bootstrapped_a_to_b",
    "bootstrapped_b_to_a",
    "seed_variance_material",
)


def _coerce_bools(frame: pd.DataFrame) -> pd.DataFrame:
    for col in _BOOL_COLUMNS:
        if col in frame.columns and frame[col].dtype == object:
            frame[col] = frame[col].astype(str).str.strip().str.lower() == "true"
    return frame


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def _auroc_strong_sl(frame: pd.DataFrame) -> tuple[float, int, int]:
    labels = frame["is_strong_sl"].to_numpy(dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    # larger s_A should predict strong-SL (True). AUROC from average ranks:
    # (sum_ranks(pos) - n_pos*(n_pos+1)/2) / (n_pos*n_neg), ties averaged.
    scores = frame["s_a_seed_mean"].to_numpy()
    ranks = stats.rankdata(scores)
    rank_sum_pos = float(ranks[labels].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc), n_pos, n_neg


def _report_slice(name: str, frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.dropna(subset=["s_a_seed_mean", "gi_score"])
    rho, rho_p = _spearman(frame["s_a_seed_mean"], frame["gi_score"])
    rho_neg, _ = _spearman(frame["s_a_seed_mean"], -frame["gi_score"])
    pearson = (
        float(np.corrcoef(frame["s_a_seed_mean"], frame["gi_score"])[0, 1])
        if len(frame) >= 3
        else float("nan")
    )
    auroc, n_pos, n_neg = _auroc_strong_sl(frame)
    row = {
        "slice": name,
        "n_pairs": int(len(frame)),
        "n_strong_sl": n_pos,
        "spearman_s_a_vs_gi": round(rho, 4),
        "spearman_p": f"{rho_p:.2e}",
        "spearman_s_a_vs_neg_gi": round(rho_neg, 4),
        "pearson_s_a_vs_gi": round(pearson, 4),
        "auroc_s_a_to_strong_sl": round(auroc, 4),
    }
    return row


def analyze(pairs_csv: Path, convention: str) -> list[dict[str, object]]:
    frame = _coerce_bools(pd.read_csv(pairs_csv))
    non_boot = ~(frame["bootstrapped_a_to_b"] | frame["bootstrapped_b_to_a"])
    slices = {
        "full": frame,
        "high_effective_n (primary, both>=64)": frame[frame["high_effective_n"]],
        "non_bootstrapped (both dirs)": frame[non_boot],
        "high_eff_n & not_seed_material": frame[
            frame["high_effective_n"] & ~frame["seed_variance_material"]
        ],
    }
    rows = []
    for name, sub in slices.items():
        row = _report_slice(name, sub)
        row = {"convention": convention, **row}
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/experiments/05_aivc_a_to_b_to_c/bridge_a/sweep"),
    )
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    for convention in ("self", "control"):
        csv = args.sweep_dir / f"pilot_pairs_{convention}.csv"
        if not csv.exists():
            print(f"MISSING: {csv}")
            continue
        all_rows.extend(analyze(csv, convention))

    out = pd.DataFrame(all_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(args.sweep_dir / "horlbeck_correlation_summary.csv", index=False)
    print(f"\nwrote {args.sweep_dir / 'horlbeck_correlation_summary.csv'}")
    print(
        "\nInterpretation: SL signal => spearman_s_a_vs_gi NEGATIVE "
        "(and auroc_s_a_to_strong_sl > 0.5). Positive/zero => no recovered signal."
    )


if __name__ == "__main__":
    main()
