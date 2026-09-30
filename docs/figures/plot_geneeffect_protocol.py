"""Draw the result figures of docs/03-geneeffect-protocol.md from tracked evidence.

Run from the repository root:  uv run python docs/figures/plot_geneeffect_protocol.py
Every number comes from a file under docs/results/; nothing is typed in by hand.
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
P1 = ROOT / "docs" / "results" / "p1_response_pathway_diagnostics" / "evidence"
JOINT = ROOT / "docs" / "results" / "joint_geneeffect_seed0"
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
SLOPE_LIGHT = "#7FA6D6"
RIDGE = "#42949E"
MLP = "#767676"
MLP_LIGHT = "#B5B5B5"
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


def huber_change(loss, gene_mean_loss):
    return 100.0 * (loss / gene_mean_loss - 1.0)


# --------------------------------------------------------------------------- Figure 2
def learning_curves():
    slope = read_jsonl(P1 / "p1a" / "heads" / "A2" / "metrics.jsonl")
    mlp = read_jsonl(P1 / "p1a" / "heads" / "A0" / "metrics.jsonl")
    audit = json.loads((P1 / "p1a" / "analysis" / "audit.json").read_text())[
        "baselines"
    ]
    selected = int(slope.loc[slope["val_geneeffect_loss"].idxmin(), "epoch"])

    panels = [
        (
            "val_geneeffect_loss",
            "Validation Huber loss",
            audit["gene_mean"]["geneeffect_loss"],
            "gene mean",
        ),
        (
            "val_residual_pearson_macro_per_gene",
            "Validation residual Pearson",
            audit["PCA8-ridge"]["residual_pearson_macro_per_gene"],
            "context-PCA ridge",
        ),
        (
            "train_residual_pearson_macro_per_gene",
            "Training residual Pearson",
            None,
            None,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM), constrained_layout=True)
    for ax, letter, (column, title, ref, ref_label) in zip(axes, "abc", panels):
        for frame, colour, label in (
            (slope, SLOPE, "Explicit context slope"),
            (mlp, MLP, "Shared MLP"),
        ):
            ax.plot(
                frame["epoch"],
                frame[column],
                color=colour,
                marker="o",
                markersize=2.8,
                label=label,
            )
        pick = slope.loc[slope["epoch"] == selected, column].item()
        ax.plot(
            selected,
            pick,
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=SLOPE,
            lw=0,
        )
        if ref is not None:
            colour = RIDGE if "ridge" in ref_label else REF
            ax.axhline(ref, color=colour, lw=0.8, ls="--", zorder=0)
            ax.annotate(
                ref_label,
                (0.7, ref),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=5.5,
                va="bottom",
                color=colour,
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.6, 8.4)
        panel_label(ax, letter)
    axes[0].annotate(
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
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.5f"))
    axes[1].set_ylim(0, 0.16)
    axes[2].set_ylim(0, 0.36)
    axes[2].legend(loc="lower right", handlelength=1.6)
    save(fig, "geneeffect_readout_learning_curves")


# --------------------------------------------------------------------------- Figure 3
VALIDATION_METHODS = [
    # label, source, colour
    ("Explicit context slope", ("head", "A2"), SLOPE),
    ("Explicit slope + response block", ("head", "A3"), SLOPE_LIGHT),
    ("Shared MLP", ("head", "A0"), MLP),
    ("Shared MLP + response block", ("head", "A1"), MLP_LIGHT),
    ("Joint backbone readout", ("baseline", "P0"), NEUTRAL),
    ("Context-PCA ridge (Tx1, 8 PCs)", ("baseline", "PCA8-ridge"), RIDGE),
    ("Gene mean", ("baseline", "gene_mean"), REF),
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
    gene_mean_loss = audit["gene_mean"]["geneeffect_loss"]
    rows = []
    for label, (kind, key), colour in VALIDATION_METHODS:
        if kind == "head":
            loss = heads.loc[key, "val_geneeffect_loss"]
            pearson = heads.loc[key, "val_residual_pearson_macro_per_gene"]
            sd_ratio = heads.loc[key, "val_residual_sd_ratio_macro_per_gene"]
        else:
            loss = audit[key]["geneeffect_loss"]
            pearson = audit[key]["residual_pearson_macro_per_gene"]
            sd_ratio = audit[key]["residual_sd_ratio_macro_per_gene"]
        rows.append(
            {
                "label": label,
                "colour": colour,
                "huber_change": huber_change(loss, gene_mean_loss),
                "pearson": np.nan if pearson is None else pearson,
                "sd_ratio": sd_ratio,
            }
        )
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


def dot_column(
    ax, rows, column, xlabel, undefined_at=None, fmt="{:.3f}", zero_line=False
):
    y = np.arange(len(rows))[::-1]
    for yi, (_, row) in zip(y, rows.iterrows()):
        value = row[column]
        if np.isnan(value):
            ax.text(
                undefined_at,
                yi,
                "undefined",
                fontsize=5.5,
                va="center",
                ha="left",
                color=REF,
                style="italic",
            )
            continue
        ax.hlines(
            yi,
            0 if not zero_line else min(0, value),
            value if not zero_line else max(0, value),
            color=row["colour"],
            lw=0.8,
            alpha=0.5,
        )
        ax.plot(value, yi, "o", color=row["colour"], markersize=4)
        ax.text(
            value,
            yi + 0.32,
            fmt.format(value),
            fontsize=5.5,
            ha="center",
            va="bottom",
            color=row["colour"],
        )
    if zero_line:
        ax.axvline(0, color=REF, lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return y


def validation_comparison():
    rows = validation_rows()
    seeds = head_seed_pearsons()
    paired = json.loads((P1 / "p1a" / "comparison" / "paired.json").read_text())
    audit = json.loads((P1 / "p1a" / "analysis" / "audit.json").read_text())

    fig = plt.figure(figsize=(183 * MM, 100 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1.35, 1, 1])
    ax_p = fig.add_subplot(grid[0, 0])
    ax_h = fig.add_subplot(grid[0, 1], sharey=ax_p)
    ax_s = fig.add_subplot(grid[0, 2], sharey=ax_p)
    ax_d = fig.add_subplot(grid[1, :])

    y = dot_column(
        ax_p,
        rows,
        "pearson",
        "Residual Pearson (macro over 4,447 genes)",
        undefined_at=0.002,
    )
    ax_p.set_yticklabels(rows["label"])
    for arm, label in (("A2", "Explicit context slope"), ("A0", "Shared MLP")):
        yi = y[rows.index[rows["label"] == label][0]]
        ax_p.plot(
            seeds[arm],
            [yi] * 2,
            "o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=rows.loc[rows["label"] == label, "colour"].item(),
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
    ax_p.set_xlim(0, 0.16)
    ax_p.set_title("Context signal recovered", loc="left")
    panel_label(ax_p, "a")

    dot_column(
        ax_h,
        rows,
        "huber_change",
        "Huber loss vs gene mean (%)",
        fmt="{:+.2f}",
        zero_line=True,
    )
    plt.setp(ax_h.get_yticklabels(), visible=False)
    ax_h.set_xlim(-1.1, 0.6)
    ax_h.set_title("Total error", loc="left")
    panel_label(ax_h, "b")

    dot_column(ax_s, rows, "sd_ratio", "SD ratio (prediction / target)", fmt="{:.2f}")
    plt.setp(ax_s.get_yticklabels(), visible=False)
    ax_s.set_xlim(-0.01, 0.32)
    ax_s.set_title("Amplitude", loc="left")
    panel_label(ax_s, "c")

    contrasts = [
        ("Slope − shared MLP", paired["A2-minus-A0"], SLOPE),
        ("Slope − joint backbone readout", audit["A2_vs_P0"], SLOPE),
        ("Slope − context-PCA ridge", audit["A2_vs_PCA"], RIDGE),
        ("Adding response block, slope readout", paired["A3-minus-A2"], MLP),
        ("Adding response block, MLP readout", paired["A1-minus-A0"], MLP),
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
    ax_d.set_xlim(-0.04, 0.16)
    ax_d.set_ylim(-0.6, len(contrasts) - 0.4)
    ax_d.spines["left"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    ax_d.set_xlabel(
        "Difference in residual Pearson (paired bootstrap over 27 lines, 95% interval)"
    )
    ax_d.set_title("Paired differences", loc="left")
    panel_label(ax_d, "d")
    save(fig, "geneeffect_validation_comparison")


# --------------------------------------------------------------------------- Figure 4
TEST_METHODS = [
    ("joint_best_epoch3", "Joint backbone", NEUTRAL),
    ("context_pca_ridge[tx1]", "Context-PCA ridge, Tx1", RIDGE),
    ("context_pca_ridge[hvg]", "Context-PCA ridge, HVG", "#8CC2C8"),
    ("nearest_line[tx1]", "Nearest line, Tx1", MLP),
    ("nearest_line[hvg]", "Nearest line, HVG", MLP_LIGHT),
    ("gene_mean", "Gene mean", REF),
    ("copy_prior", "K562 copy-prior", REF),
]


def test_record():
    frame = pd.read_csv(JOINT / "test_comparison.csv").set_index("method")
    gene_mean_loss = frame.loc["gene_mean", "test_geneeffect_loss"]
    rows = pd.DataFrame(
        {
            "label": label,
            "colour": colour,
            "pearson": frame.loc[key, "test_residual_pearson_macro_per_gene"],
            "huber_change": huber_change(
                frame.loc[key, "test_geneeffect_loss"], gene_mean_loss
            ),
        }
        for key, label, colour in TEST_METHODS
    )
    fig, (ax_p, ax_near, ax_far) = plt.subplots(
        1,
        3,
        figsize=(160 * MM, 52 * MM),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.3, 0.75, 0.45]},
    )
    dot_column(
        ax_p, rows, "pearson", "Residual Pearson (macro over genes)", undefined_at=0.002
    )
    ax_p.set_yticklabels(rows["label"])
    ax_p.set_xlim(0, 0.14)
    ax_p.set_title("Context signal, test split (27 lines)", loc="left")
    panel_label(ax_p, "a")
    # Broken axis: the context-blind and contextual methods sit within 2% of the gene
    # mean, the nearest-line and copy-prior controls 75-110% above it.
    near = rows.where(rows["huber_change"] < 10)
    far = rows.where(rows["huber_change"] >= 10)
    near["colour"], far["colour"] = rows["colour"], rows["colour"]
    for ax, part in ((ax_near, near), (ax_far, far)):
        y = np.arange(len(part))[::-1]
        for yi, (_, row) in zip(y, part.iterrows()):
            if np.isnan(row["huber_change"]):
                continue
            ax.plot(row["huber_change"], yi, "o", color=row["colour"], markersize=4)
            ax.text(
                row["huber_change"],
                yi + 0.32,
                f"{row['huber_change']:+.2f}"
                if ax is ax_near
                else f"{row['huber_change']:+.0f}",
                fontsize=5.5,
                ha="center",
                va="bottom",
                color=row["colour"],
            )
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=False)
    ax_near.axvline(0, color=REF, lw=0.6, zorder=0)
    ax_near.set_xlim(-0.6, 2.0)
    ax_far.set_xlim(65, 120)
    ax_near.set_xlabel("Huber loss vs gene mean (%)")
    ax_near.xaxis.set_label_coords(0.8, -0.16)
    for x, ax in ((1, ax_near), (0, ax_far)):
        ax.plot(
            [x],
            [0],
            marker=[(-0.6, -1), (0.6, 1)],
            markersize=6,
            color="k",
            mew=0.6,
            transform=ax.transAxes,
            clip_on=False,
        )
    ax_near.set_title("Total error", loc="left")
    panel_label(ax_near, "b")
    save(fig, "geneeffect_backbone_test")


# --------------------------------------------------------------------------- Figure 5
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
    test_record()
    interface_transfer()
