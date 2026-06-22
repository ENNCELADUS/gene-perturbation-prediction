"""Generate the data figure(s) for the transcriptome-SL paper.

Only the difficulty-ladder plot is data-driven; the teaser and pipeline
schematics are drawn separately (see fig_teaser.md / fig_pipeline.md). All
numbers flow through make_tables.read_metric so the figure can never drift from
the artifacts.
"""
from __future__ import annotations

import importlib.util
import logging
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

_HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("make_tables", _HERE / "make_tables.py")
mt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mt)

FIGURES = _HERE.parent / "figures"


def build_difficulty_ladder() -> None:
    """NDCG@10 across the CV1/CV2/CV3 difficulty ladder for three signal sets.

    CV1 is marked as the degree-gameable diagnostic; the honest message is the
    CV2 transcriptome lift and the CV3 collapse for every signal.
    """
    splits = ["CV1", "CV2", "CV3"]
    series = {
        "Dependency-only floor (exp06)": [
            mt.read_metric(mt.EXP06, s, "B", "ndcg@10") for s in splits
        ],
        "+ Observed transcriptome (exp07)": [
            mt.read_metric(mt.EXP07, s, "B_transcript", "ndcg@10", slice="full_universe")
            for s in splits
        ],
        "+ Cross-line selectivity (exp09)": [
            mt.read_metric(mt.EXP09, s, "B_xcl", "ndcg@10", slice="full_universe")
            for s in splits
        ],
    }

    import pandas as pd  # noqa: E402
    _pub = pd.read_csv(_HERE.parent / "tables" / "benchmark_published.csv")
    _grsmf = _pub.loc[_pub["model"] == "GRSMF"].iloc[0]
    grsmf_ndcg = [float(_grsmf["cv1_ndcg10"]), float(_grsmf["cv2_ndcg10"]),
                  float(_grsmf["cv3_ndcg10"])]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = range(len(splits))
    markers = ["o", "s", "^"]
    for (label, ys), marker in zip(series.items(), markers):
        ax.plot(list(x), ys, marker=marker, linewidth=1.8, markersize=6, label=label)
    ax.plot(list(x), grsmf_ndcg, linestyle="--", color="0.4", linewidth=1.5,
            marker="x", markersize=6, label="GRSMF (best published ranker)")

    # shade CV1 as the degree-gameable diagnostic region
    ax.axvspan(-0.4, 0.4, color="0.92", zorder=0)
    ax.annotate(
        "degree-gameable\n(diagnostic)",
        xy=(0, ax.get_ylim()[1]),
        xytext=(0, 0.92),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["CV1\n(pair-level)", "CV2\n(one held out)", "CV3\n(both cold)"], fontsize=9
    )
    ax.set_ylabel("NDCG@10 (per-anchor)")
    ax.set_title("Ranking quality down the cold-start difficulty ladder", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig_difficulty_ladder.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_difficulty_ladder()


if __name__ == "__main__":
    main()
