"""Draw the result figures of docs/03-geneeffect-protocol.md from tracked evidence.

Run from the repository root:  uv run python docs/figures/plot_geneeffect_protocol.py
Every number comes from a file under results/; nothing is typed in by hand.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
P1 = ROOT / "results" / "p1_response_pathway_diagnostics" / "evidence"
MM = 1 / 25.4

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
    }
)

# One signal colour for the explicit context slope, one for the linear context
# baseline it is measured against, neutrals for everything else.
SLOPE = "#0F4D92"
RIDGE = "#42949E"
MLP = "#767676"
NEUTRAL = "#272727"
REF = "#9A9A9A"


def panel_label(ax, letter):
    ax.text(
        -0.02,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def save(fig, name):
    for suffix, kwargs in ((".svg", {}), (".pdf", {})):
        fig.savefig(
            OUT / f"{name}{suffix}", bbox_inches="tight", facecolor="white", **kwargs
        )
    plt.close(fig)


def read_jsonl(path):
    return pd.DataFrame(
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    )


# --------------------------------------------------------------------------- Figure 2
def learning_curves():
    slope = read_jsonl(P1 / "p1a" / "heads" / "A2" / "metrics.jsonl")
    mlp = read_jsonl(P1 / "p1a" / "heads" / "A0" / "metrics.jsonl")
    audit = json.loads((P1 / "p1a" / "analysis" / "audit.json").read_text())[
        "baselines"
    ]
    selected = int(slope.loc[slope["val_geneeffect_loss"].idxmin(), "epoch"])
    heads = ((slope, SLOPE, "Explicit context slope"), (mlp, MLP, "Shared MLP"))

    fig, (ax_h, ax_p) = plt.subplots(
        1, 2, figsize=(150 * MM, 55 * MM), constrained_layout=True
    )
    for frame, colour, label in heads:
        ax_h.plot(
            frame["epoch"],
            frame["val_geneeffect_loss"],
            color=colour,
            marker="o",
            markersize=2.8,
            label=label,
        )
        ax_p.plot(
            frame["epoch"],
            frame["val_residual_pearson_macro_per_gene"],
            color=colour,
            marker="o",
            markersize=2.8,
        )
        ax_p.plot(
            frame["epoch"],
            frame["train_residual_pearson_macro_per_gene"],
            color=colour,
            ls="--",
            lw=0.9,
            alpha=0.6,
        )
    for ax, column in (
        (ax_h, "val_geneeffect_loss"),
        (ax_p, "val_residual_pearson_macro_per_gene"),
    ):
        ax.plot(
            selected,
            slope.loc[slope["epoch"] == selected, column].item(),
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=SLOPE,
            lw=0,
        )
    for ax, ref, ref_label, colour in (
        (ax_h, audit["gene_mean"]["geneeffect_loss"], "gene mean", REF),
        (
            ax_p,
            audit["PCA8-ridge"]["residual_pearson_macro_per_gene"],
            "context-PCA ridge (validation)",
            RIDGE,
        ),
    ):
        ax.axhline(ref, color=colour, lw=0.8, ls=":", zorder=0)
        ax.annotate(
            ref_label,
            (0.7, ref),
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=5.5,
            va="bottom",
            color=colour,
        )
    ax_h.annotate(
        "selected",
        xy=(
            selected,
            slope.loc[slope["epoch"] == selected, "val_geneeffect_loss"].item(),
        ),
        xytext=(0, -11),
        textcoords="offset points",
        ha="center",
        fontsize=5.5,
        color=SLOPE,
    )
    ax_h.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.5f"))
    ax_h.set_title("Validation Huber loss (selection criterion)", loc="left")
    ax_h.set_ylim(0.01615, 0.01656)
    ax_h.legend(loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.2)
    ax_p.plot([], [], color=MLP, label="validation")
    ax_p.plot([], [], color=MLP, ls="--", lw=0.9, alpha=0.6, label="training")
    ax_p.legend(loc="upper center", ncol=2, handlelength=1.8, columnspacing=1.2)
    ax_p.set_ylim(0, 0.41)
    ax_p.set_title("Residual Pearson", loc="left")
    for ax, letter in ((ax_h, "a"), (ax_p, "b")):
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.6, 8.4)
        panel_label(ax, letter)
    save(fig, "geneeffect_readout_learning_curves")


# --------------------------------------------------------------------------- Figure 3
VALIDATION_METHODS = [
    # label, source, colour
    ("Explicit context slope", ("head", "A2"), SLOPE),
    ("Context-PCA ridge (Tx1, 8 PCs)", ("baseline", "PCA8-ridge"), RIDGE),
    ("Shared MLP", ("head", "A0"), MLP),
    ("Joint backbone readout", ("baseline", "P0"), NEUTRAL),
]


def validation_rows():
    audit = json.loads((P1 / "p1a" / "analysis" / "audit.json").read_text())[
        "baselines"
    ]
    heads = (
        pd.read_csv(P1 / "p1a" / "comparison" / "selected.csv")
        .query("split == 'val'")
        .set_index("arm")
    )
    rows = []
    for label, (kind, key), colour in VALIDATION_METHODS:
        if kind == "head":
            pearson = heads.loc[key, "val_residual_pearson_macro_per_gene"]
        else:
            pearson = audit[key]["residual_pearson_macro_per_gene"]
        rows.append({"label": label, "colour": colour, "pearson": pearson})
    return pd.DataFrame(rows)


def head_seed_pearsons():
    """Validation residual Pearson of A2 (slope) and A0 (MLP) at head seeds 1 and 2."""
    out = {"A0": [], "A2": []}
    for seed in (1, 2):
        frame = pd.read_csv(
            P1 / "p1c" / "heads" / f"seed{seed}" / "comparison" / "selected.csv"
        )
        frame = frame.query("split == 'val'").set_index("arm")
        for arm in out:
            out[arm].append(frame.loc[arm, "val_residual_pearson_macro_per_gene"])
    return out


def validation_comparison():
    rows = validation_rows()
    seeds = head_seed_pearsons()
    paired = json.loads((P1 / "p1a" / "comparison" / "paired.json").read_text())
    audit = json.loads((P1 / "p1a" / "analysis" / "audit.json").read_text())

    fig, (ax_p, ax_d) = plt.subplots(
        1,
        2,
        figsize=(170 * MM, 50 * MM),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.1]},
    )
    y = np.arange(len(rows))[::-1]
    for yi, (_, row) in zip(y, rows.iterrows()):
        ax_p.hlines(yi, 0, row["pearson"], color=row["colour"], lw=0.8, alpha=0.5)
        ax_p.plot(row["pearson"], yi, "o", color=row["colour"], markersize=4)
        ax_p.text(
            row["pearson"],
            yi + 0.3,
            f"{row['pearson']:.3f}",
            fontsize=5.5,
            ha="center",
            va="bottom",
            color=row["colour"],
        )
    for arm, label in (("A2", "Explicit context slope"), ("A0", "Shared MLP")):
        match = rows["label"] == label
        ax_p.plot(
            seeds[arm],
            [y[rows.index[match][0]]] * 2,
            "o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=rows.loc[match, "colour"].item(),
            markeredgewidth=0.7,
        )
    ax_p.plot(
        [],
        [],
        "o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=MLP,
        markeredgewidth=0.7,
        label="head seeds 1, 2",
    )
    ax_p.legend(loc="lower right", handletextpad=0.3)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(rows["label"])
    ax_p.set_ylim(-0.6, len(rows) - 0.3)
    ax_p.set_xlim(0, 0.16)
    ax_p.set_xlabel("Residual Pearson (macro over 4,447 genes)")
    ax_p.set_title("Context signal on 27 validation lines", loc="left")
    panel_label(ax_p, "a")

    contrasts = [
        ("Slope − shared MLP", paired["A2-minus-A0"], SLOPE),
        ("Slope − joint backbone readout", audit["A2_vs_P0"], SLOPE),
        ("Slope − context-PCA ridge", audit["A2_vs_PCA"], RIDGE),
    ]
    yd = np.arange(len(contrasts))[::-1]
    for yi, (label, record, colour) in zip(yd, contrasts):
        point = record["point"]["pearson_delta"]
        low, high = record["intervals"]["pearson"]["interval_95"]
        ax_d.hlines(yi, low, high, color=colour, lw=1.4)
        ax_d.plot(point, yi, "o", color=colour, markersize=4.5)
        ax_d.text(
            high + 0.004,
            yi,
            f"{point:+.3f} [{low:+.3f}, {high:+.3f}]",
            va="center",
            fontsize=5.5,
            color=colour,
        )
    ax_d.axvline(0, color=REF, lw=0.6, zorder=0)
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([c[0] for c in contrasts])
    ax_d.set_xlim(-0.03, 0.17)
    ax_d.set_ylim(-0.6, len(contrasts) - 0.4)
    ax_d.set_xlabel("Difference in residual Pearson (95% paired bootstrap)")
    ax_d.set_title("Paired differences", loc="left")
    panel_label(ax_d, "b")
    for ax in (ax_p, ax_d):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, "geneeffect_validation_comparison")


# --------------------------------------------------------------------------- Figure 4
INTERFACES = [
    ("N-native", "Released checkpoint, untrained"),
    ("V1", "Expression-residual interface"),
    ("V2-null", "Native basal path, ESM-2 tokens"),
    ("V2", "Native path + trainable Tx1 term"),
    ("V0", "Adapted Tx1 basal encoder"),
]
LINES = [
    ("jurkat", "Jurkat", "o", "#0F4D92"),
    ("k562", "K562", "s", "#42949E"),
    ("hepg2", "HepG2", "^", "#9A4D8E"),
    ("hct116", "HCT116", "D", "#B64342"),
]


def interface_transfer():
    comparison = P1 / "p1c" / "round2" / "comparison"
    folds = pd.read_csv(comparison / "summary.csv")
    pooled = pd.read_csv(comparison / "variants.csv").set_index("label")

    fig, ax = plt.subplots(figsize=(120 * MM, 58 * MM), constrained_layout=True)
    y = np.arange(len(INTERFACES))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(LINES))
    for yi, (label, name) in zip(y, INTERFACES):
        rows = folds[folds["label"] == label].set_index("fold")
        for offset, (fold, line_name, marker, colour) in zip(offsets, LINES):
            ax.plot(
                rows.loc[fold, "external_ratio"],
                yi + offset,
                marker,
                color=colour,
                markersize=3.6,
                label=line_name if yi == y[0] else None,
            )
        if label in pooled.index:
            record = pooled.loc[label]
            ax.hlines(
                yi,
                record["pooled_ratio_ci_low"],
                record["pooled_ratio_ci_high"],
                color=NEUTRAL,
                lw=3.0,
                alpha=0.35,
            )
            ax.plot(
                record["pooled_ratio"],
                yi,
                "|",
                color=NEUTRAL,
                markersize=9,
                markeredgewidth=1.4,
            )
            ax.text(
                45,
                yi,
                f"{record['pooled_ratio']:.2f}"
                if record["pooled_ratio"] < 10
                else f"{record['pooled_ratio']:.1f}",
                fontsize=5.5,
                va="center",
                ha="left",
                color=NEUTRAL,
            )
    ax.plot(
        [],
        [],
        "|",
        color=NEUTRAL,
        markersize=7,
        markeredgewidth=1.4,
        label="pooled, 95% interval",
    )
    ax.axvline(1, color=REF, lw=0.8, ls="--", zorder=0)
    ax.text(
        1,
        len(INTERFACES) - 0.45,
        "no-change",
        fontsize=5.5,
        ha="center",
        va="bottom",
        color=REF,
    )
    ax.set_xscale("log")
    ax.set_xlim(0.7, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in INTERFACES])
    ax.set_ylim(-0.6, len(INTERFACES) - 0.2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Response loss on the held-out line / no-change loss (log scale)")
    ax.text(
        45,
        len(INTERFACES) - 0.45,
        "pooled",
        fontsize=5.5,
        ha="left",
        va="bottom",
        color=NEUTRAL,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        handletextpad=0.2,
        columnspacing=0.9,
    )
    save(fig, "geneeffect_interface_transfer")


if __name__ == "__main__":
    learning_curves()
    validation_comparison()
    interface_transfer()
