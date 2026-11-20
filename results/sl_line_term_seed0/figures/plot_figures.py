"""Draw the figures of this run record from the copied host report.

Run from the repository root:
    uv run python results/sl_line_term_seed0/figures/plot_figures.py
Every number comes from host/benchmark_results.md (Tables 1 and 5).
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.experiments.sl.benchmark_baselines import NAMES, STRATA, SUBSETS  # noqa: E402

OUT = Path(__file__).resolve().parent
HOST = OUT.parent / "host"
# validation and test: the first two slots of the categorical palette, with a shape each
VAL, TEST = "#eb6834", "#2a78d6"
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
MARK = {"markeredgecolor": "white", "markeredgewidth": 1.2, "linestyle": "none"}
#: the short row names of the run record (its "Rows" section defines them)
SHORT = {
    "direct": "Direct target, Tx1-3B cell tokens",
    "direct_mean_token": "Direct target, no cell tokens",
    "direct_line_term": "Direct target, Tx1-3B cell tokens + line term (own expression)",  # noqa: E501
    "direct_mean_token_line_term": "Direct target, no cell tokens + line term (own expression)",  # noqa: E501
    "direct_pie_change_line_term": "Direct target, Tx1-3B cell tokens + line term (own expression, PIE change)",  # noqa: E501
    "direct_tx1_gene_line_term": "Direct target, Tx1-3B cell tokens + line term (own expression, Tx1-3B gene embeddings)",  # noqa: E501
    "embedding": "Residual target, Tx1-3B cell tokens",
    "embedding_mean_token": "Residual target, no cell tokens",
    "embedding_cellnet": "Residual target, CellNet cell tokens",
    "buffer_head": "Backbone head on the buffer-aware reader's terms",
    "kebabci_paralog_predicted": "Kebabci 2026 paralog classifier",
    "kg4sl": "KG4SL",
    "dekegel_paralog": "De Kegel 2021 paralog classifier",
    "paris": "PARIS",
    "mit4sl": "MiT4SL",
    "cilantro_sl": "Cilantro-SL",
    "slidr": "SLIdR",
    "pan_essentiality": "Pan-essentiality (reads no line)",
    "random": "Random score",
}  # fmt: skip
#: the figure's shorter names of the line-term rows
LINE_TERM = {
    "direct_line_term": "Tx1-3B cell tokens,\nown expression",
    "direct_mean_token_line_term": "No cell tokens,\nown expression",
    "direct_pie_change_line_term": "Tx1-3B cell tokens,\nown expression + PIE change",
    "direct_tx1_gene_line_term": "Tx1-3B cell tokens, own expression\n+ Tx1-3B gene embeddings",  # noqa: E501
}


def _rows(text: str, start: str) -> list[list[str]]:
    """The cells of each body row of the Markdown table after ``start``."""
    block = text[text.index(start) :].split("\n\n", 2)[1]
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in block.splitlines()[2:]
    ]


def _number(cell: str) -> float | None:
    found = re.search(r"\d\.\d+", cell)
    return None if found is None else float(found.group())


def harness() -> dict:
    """Tables 1, 3 and 5 of the harness report, keyed by scorer and row set.

    ``recall[scorer]`` = {"train", "val", "test"}; ``row_sets[(scorer, set)]`` =
    (val, test); ``differences[(model, control, set)]`` = {split: (mean, low, high,
    lines)}."""
    by_name = {v: k for k, v in NAMES.items()}
    sets = {v: k for k, v in ({"all": "All pairs"} | STRATA | SUBSETS).items()}
    text = (HOST / "benchmark_results.md").read_text()
    recall = {
        by_name[c[0]]: dict(zip(("train", "val", "test"), map(_number, c[1:])))
        for c in _rows(text, "**Table 1.**")
    }
    header = text[text.index("**Table 3.**") :].split("\n\n", 2)[1].splitlines()[0]
    columns = [sets[c.strip()] for c in header.strip().strip("|").split("|")[1:]]
    row_sets = {}
    for c in _rows(text, "**Table 3.**"):
        for column, cell in zip(columns, c[1:]):
            row_sets[(by_name[c[0]], column)] = tuple(map(float, cell.split(" / ")))
    differences = {}
    for c in _rows(text, "**Table 5.**"):
        model, control = c[0].split(" − ")
        entry = {}
        for split, cell in zip(("val", "test"), c[2:]):
            mean, low, high = map(float, re.findall(r"[+-]\d\.\d+", cell))
            entry[split] = (mean, low, high, int(re.search(r"\((\d+)\)", cell)[1]))
        differences[(by_name[model], by_name[control], sets[c[1]])] = entry
    return {"recall": recall, "row_sets": row_sets, "differences": differences}


def split_legend(fig) -> None:
    handles = [
        Line2D([], [], marker="D", color=VAL, markersize=6.5, **MARK),
        Line2D([], [], marker="o", color=TEST, markersize=7.5, **MARK),
    ]
    labels = ["Validation lines (select)", "Test lines (reported)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0),
               ncol=2, handletextpad=0.2, columnspacing=1.6)  # fmt: skip


def all_pairs(recall: dict) -> None:
    """Recall in the top 10% of the main rows on all pairs."""
    groups = [
        ("Direct GeneEffect target (this study)",
         ["direct", "direct_mean_token", "direct_line_term",
          "direct_mean_token_line_term", "direct_pie_change_line_term",
          "direct_tx1_gene_line_term"]),
        ("Residual GeneEffect target (reference, 2026-11-19)",
         ["embedding", "embedding_mean_token", "embedding_cellnet"]),
        ("Earlier model of this project", ["buffer_head"]),
        ("Published baselines", ["kebabci_paralog_predicted", "kg4sl",
                                 "dekegel_paralog", "paris", "mit4sl", "cilantro_sl",
                                 "slidr"]),
        ("Controls", ["pan_essentiality", "random"]),
    ]  # fmt: skip
    labelled = ("direct", "direct_mean_token", "embedding", "kebabci_paralog_predicted")
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    y, ticks, labels = 0.0, [], []
    for title, members in groups:
        ax.text(-0.005, y - 0.05, title, transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=MUTED, fontsize=8,
                style="italic")  # fmt: skip
        y += 0.85
        for scorer in members:
            val, test = recall[scorer]["val"], recall[scorer]["test"]
            ax.plot([val, test], [y, y], color=LINK, linewidth=1.5, zorder=2)
            ax.plot(val, y, marker="D", color=VAL, markersize=6.5, zorder=3, **MARK)
            ax.plot(test, y, marker="o", color=TEST, markersize=7.5, zorder=4, **MARK)
            if scorer in labelled:
                ax.text(max(val, test) + 0.012, y, f"{val:.3f} / {test:.3f}",
                        va="center", color=MUTED, fontsize=8)  # fmt: skip
            ticks.append(y)
            label = SHORT[scorer].removeprefix("Direct target, ")
            label = label.removeprefix("Residual target, ")
            labels.append(label[0].upper() + label[1:])
            y += 1
        y += 0.35
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.6, -0.5)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel(
        "Recall in the top 10%, all pairs, mean over held-out lines (higher is better)"
    )
    split_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "all_pairs.svg", facecolor="white")
    plt.close(fig)


def line_term(differences: dict) -> None:
    """Each line-term row minus its line-input controls and minus its own model."""
    panels = [
        ("mean_input", "Minus the context-ablated model\n(the fit lines' mean input)"),
        ("line_term_other_line", "Minus the model and the line term\non another line's input"),  # noqa: E501
        ("", "Minus the model without\nthe line term"),
    ]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True, sharex=True)
    for ax, (control, title) in zip(axes, panels):
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
        for y, row in enumerate(LINE_TERM):
            base = row.removesuffix("_line_term")
            entry = differences[(row, f"{base}_{control}" if control else base, "all")]
            for split, colour, marker, size, shift in (
                ("val", VAL, "D", 6.5, -0.15), ("test", TEST, "o", 7.5, 0.15)
            ):  # fmt: skip
                mean, low, high, _ = entry[split]
                ax.plot([low, high], [y + shift] * 2, color=colour, linewidth=1.8,
                        solid_capstyle="round", zorder=2)  # fmt: skip
                ax.plot(mean, y + shift, marker=marker, color=colour, markersize=size,
                        zorder=3, **MARK)  # fmt: skip
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_yticks(range(len(LINE_TERM)), list(LINE_TERM.values()))
    axes[0].set_ylim(len(LINE_TERM) - 0.5, -0.5)
    axes[0].set_xlim(-0.09, 0.24)
    fig.supxlabel(
        "Paired difference in recall in the top 10%, all pairs, with its 95% interval "
        "over lines (above zero: the row with the line term is higher)",
        fontsize=9, color=MUTED, x=0.6,
    )  # fmt: skip
    split_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "line_term.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    tables = harness()
    all_pairs(tables["recall"])
    line_term(tables["differences"])
