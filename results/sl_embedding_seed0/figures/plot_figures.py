"""Draw the figures of this run record from the copied host report.

Run from the repository root:
    .venv/bin/python results/sl_embedding_seed0/figures/plot_figures.py
Every number comes from host/benchmark_summary.csv or host/benchmark_results.md.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.experiments.sl.benchmark_baselines import NAMES, STRATA, SUBSETS  # noqa: E402

OUT = Path(__file__).resolve().parent
HOST = OUT.parent / "host"
# validation and test: the first two slots of the categorical palette, with a shape each
VAL, TEST, TRAIN = "#eb6834", "#2a78d6", "#52514e"
INK, MUTED, GRID, LINK = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": LINK,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.size": 0,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)
#: the models whose two values are written beside their marks
LABELLED = ("embedding", "embedding_mean_token", "kebabci_paralog_predicted")
MARK = {"markeredgecolor": "white", "markeredgewidth": 1.2, "linestyle": "none"}
SHORT = {
    "embedding": "Gene-knowledge, cell-set model",
    "embedding_mean_token": "Gene-knowledge model without cell tokens",
    "embedding_cellnet": "Gene-knowledge, cell-set model with CellNet",
    "buffer_head": "Logistic head on predicted GeneEffect",
    "kebabci_paralog_predicted": "Kebabci 2026 paralog classifier",
    "kg4sl": "KG4SL",
    "dekegel_paralog": "De Kegel 2021 paralog classifier",
    "paris": "PARIS",
    "mit4sl": "MiT4SL",
    "cilantro_sl": "Cilantro-SL",
    "slidr": "SLIdR",
    "pan_essentiality": "Gene mean GeneEffect (reads no line)",
    "random": "Random score",
}
summary = pd.read_csv(HOST / "benchmark_summary.csv").set_index(
    ["split", "stratum", "scorer"]
)["recall_top10"]


def recall(split: str, rows: str, scorer: str) -> float | None:
    key = (split, rows, scorer)
    return float(summary[key]) if key in summary.index else None


def split_legend(fig, training: bool = False) -> None:
    handles = [
        Line2D([], [], marker="D", color=VAL, markersize=6.5, **MARK),
        Line2D([], [], marker="o", color=TEST, markersize=7.5, **MARK),
    ]
    labels = ["Validation lines (select)", "Test lines (reported)"]
    if training:
        handles.insert(0, Line2D([], [], marker="s", color=TRAIN, markersize=6, **MARK))
        labels.insert(0, "Training lines (in sample)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0),
               ncol=len(handles), handletextpad=0.2, columnspacing=1.6)  # fmt: skip


def dumbbell(ax, y: float, val: float | None, test: float | None) -> None:
    if val is None or test is None:
        return
    ax.plot([val, test], [y, y], color=LINK, linewidth=1.5, zorder=2)
    ax.plot(val, y, marker="D", color=VAL, markersize=6.5, zorder=3, **MARK)
    ax.plot(test, y, marker="o", color=TEST, markersize=7.5, zorder=4, **MARK)


def all_pairs() -> None:
    """Recall in the top 10% of every model on all pairs."""
    groups = [
        ("This study", ["embedding", "embedding_mean_token", "embedding_cellnet"]),
        ("Earlier model of this project", ["buffer_head"]),
        ("Published baselines", ["kebabci_paralog_predicted", "kg4sl",
                                 "dekegel_paralog", "paris", "mit4sl", "cilantro_sl",
                                 "slidr"]),
        ("Controls", ["pan_essentiality", "random"]),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y, ticks, labels = 0.0, [], []
    for title, members in groups:
        ax.text(-0.005, y - 0.05, title, transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=MUTED, fontsize=8,
                style="italic")  # fmt: skip
        y += 0.85
        for scorer in members:
            val, test = recall("val", "all", scorer), recall("test", "all", scorer)
            dumbbell(ax, y, val, test)
            if scorer in LABELLED:
                ax.text(max(val, test) + 0.012, y, f"{val:.3f} / {test:.3f}",
                        va="center", color=MUTED, fontsize=8)  # fmt: skip
            ticks.append(y)
            labels.append(SHORT[scorer])
            y += 1
        y += 0.35
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.6, -0.5)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel(
        "Recall in the top 10%, all pairs, mean over held-out lines (higher is better)"
    )
    split_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "all_pairs.svg", facecolor="white")
    plt.close(fig)


def row_sets() -> None:
    """The same recall on the training lines and on three sets of held-out pairs."""
    models = ["embedding", "embedding_mean_token", "buffer_head",
              "kebabci_paralog_predicted", "kg4sl", "pan_essentiality",
              "random"]  # fmt: skip
    panels = [
        ("Seen lines, seen pairs\n(the training lines)", None),
        ("Unseen lines,\nseen pairs", "seen"),
        ("Unseen lines,\nnew pairs", "new_pair"),
        ("Unseen lines, seen pairs:\nno SL call in training", "seen_never"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9, 3.5), sharey=True)
    for ax, (title, rows) in zip(axes, panels):
        for y, scorer in enumerate(models):
            if rows is None:
                train = recall("train", "all", scorer)
                if train is not None:
                    ax.plot(train, y, marker="s", color=TRAIN, markersize=6, **MARK)
            else:
                val, test = recall("val", rows, scorer), recall("test", rows, scorer)
                dumbbell(ax, y, val, test)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    axes[0].set_yticks(range(len(models)), [SHORT[m] for m in models])
    axes[0].set_ylim(len(models) - 0.5, -0.5)
    fig.supxlabel(
        "Recall in the top 10%, each set ranked inside it in each screen "
        "(higher is better)",
        fontsize=9, color=MUTED, x=0.62,
    )  # fmt: skip
    split_legend(fig, training=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "row_sets.svg", facecolor="white")
    plt.close(fig)


def paired_differences() -> dict[tuple[str, str, str], dict[str, tuple[float, ...]]]:
    """Table 5 of the harness report: (model, control, row set) -> split -> (mean,
    low, high)."""
    by_name = {v: k for k, v in NAMES.items()}
    sets = {v: k for k, v in ({"all": "All pairs"} | STRATA | SUBSETS).items()}
    text = (HOST / "benchmark_results.md").read_text()
    table = text[text.index("**Table 5.**") :]
    out = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0].split(" − ")[0] not in by_name:
            continue
        model, control = cells[0].split(" − ")
        numbers = [tuple(map(float, re.findall(r"[+-]\d\.\d+", c))) for c in cells[2:]]
        out[(by_name[model], by_name[control], sets[cells[1]])] = dict(
            zip(("val", "test"), numbers)
        )
    return out


def line_input() -> None:
    """Each model minus the same weights with the line's input replaced."""
    differences = paired_differences()
    sets = [("all", "All pairs"), ("seen", "Seen pairs"),
            ("seen_never", "Seen pairs, no training\nline called them SL"),
            ("seen_mixed", "Seen pairs, the\ntraining lines differ"),
            ("new_pair", "New pairs")]  # fmt: skip
    models = [("embedding", "Model with cell tokens"),
              ("embedding_mean_token", "Model without cell tokens")]  # fmt: skip
    controls = [("mean_input", "Own input minus no line input\n(context-ablated)"),
                ("other_line", "Own input minus\nanother line's input")]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True, sharey=True)
    for i, (model, model_name) in enumerate(models):
        for j, (control, control_name) in enumerate(controls):
            ax = axes[i, j]
            ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
            for y, (rows, _) in enumerate(sets):
                entry = differences[(model, f"{model}_{control}", rows)]
                for split, colour, marker, size, shift in (
                    ("val", VAL, "D", 6.5, -0.17), ("test", TEST, "o", 7.5, 0.17)
                ):  # fmt: skip
                    mean, low, high = entry[split]
                    ax.plot([low, high], [y + shift] * 2, color=colour, linewidth=1.8,
                            solid_capstyle="round", zorder=2)  # fmt: skip
                    ax.plot(mean, y + shift, marker=marker, color=colour,
                            markersize=size, zorder=3, **MARK)  # fmt: skip
            if i == 0:
                ax.set_title(control_name, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel(model_name, color=INK, fontweight="bold", labelpad=8)
    axes[0, 0].set_yticks(range(len(sets)), [name for _, name in sets])
    axes[0, 0].set_ylim(len(sets) - 0.5, -0.5)
    axes[0, 0].set_xlim(-0.36, 0.47)
    fig.supxlabel(
        "Paired difference in recall in the top 10%, with its 95% interval over "
        "lines (above zero: the line's own input helps)",
        fontsize=9, color=MUTED,
    )  # fmt: skip
    split_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "line_input.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    all_pairs()
    row_sets()
    line_input()
