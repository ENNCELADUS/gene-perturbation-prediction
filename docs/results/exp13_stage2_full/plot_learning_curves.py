"""Render the formal Exp13 Stage 2 warmup and joint learning curves."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "learning_curves.csv"
OUTPUT_PATH = HERE / "learning_curves.png"

TRAIN_COLOR = "#0072B2"
SECONDARY_COLOR = "#009E73"
TERTIARY_COLOR = "#CC79A7"
VAL_COLOR = "#D55E00"
GRAY = "#666666"


def load_curves() -> dict[str, list[dict[str, float]]]:
    """Load and validate the fixed formal-run telemetry."""
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    curves: dict[str, list[dict[str, float]]] = {"warmup": [], "joint": []}
    numeric = (
        "epoch",
        "train_total",
        "train_huber",
        "train_pearson",
        "train_response",
        "train_dependency",
        "train_lambda_dep",
        "validation_macro_per_gene_spearman",
        "selected",
    )
    for raw in raw_rows:
        phase = raw["phase"]
        if phase not in curves:
            raise ValueError(f"unexpected phase {phase!r}")
        row = {
            key: (float(raw[key]) if raw[key] else math.nan)
            for key in numeric
        }
        if any(
            not math.isfinite(value)
            for value in row.values()
            if not math.isnan(value)
        ):
            raise ValueError(f"non-finite telemetry in {phase}")
        curves[phase].append(row)
    expected = {"warmup": list(range(16)), "joint": list(range(3))}
    for phase, epochs in expected.items():
        observed = [int(row["epoch"]) for row in curves[phase]]
        if observed != epochs:
            raise ValueError(f"{phase} epochs are not the expected {epochs}")
        if sum(row["selected"] == 1.0 for row in curves[phase]) != 1:
            raise ValueError(f"{phase} must have exactly one selected epoch")
    return curves


def configure_style() -> None:
    """Apply a compact, print-safe style matching the KD1 reference figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def decorate(ax: Axes, rows: list[dict[str, float]], title: str, ylabel: str) -> None:
    """Apply shared axes and selected-epoch annotation."""
    epochs = [row["epoch"] for row in rows]
    selected = next(row["epoch"] for row in rows if row["selected"] == 1.0)
    ax.axvline(selected, color=GRAY, linewidth=1.0, linestyle=":", zorder=0)
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        selected + 0.08,
        0.97,
        "selected",
        transform=blend,
        rotation=90,
        ha="left",
        va="top",
        fontsize=7,
        color=GRAY,
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel("Epoch (0-based)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([int(epoch) for epoch in epochs])
    ax.set_xlim(min(epochs) - 0.35, max(epochs) + 0.35)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def line(
    ax: Axes,
    rows: list[dict[str, float]],
    key: str,
    color: str,
    label: str,
) -> None:
    """Draw one telemetry series."""
    ax.plot(
        [row["epoch"] for row in rows],
        [row[key] for row in rows],
        color=color,
        linewidth=1.6,
        marker="o",
        markersize=3.0,
        label=label,
    )


def main() -> None:
    """Render the six-panel formal learning-curve figure at 300 DPI."""
    configure_style()
    curves = load_curves()
    warmup, joint = curves["warmup"], curves["joint"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), constrained_layout=True)

    line(axes[0, 0], warmup, "train_total", TRAIN_COLOR, "total")
    line(axes[0, 0], warmup, "train_pearson", SECONDARY_COLOR, "Pearson penalty")
    decorate(axes[0, 0], warmup, "(a) Warmup train objective", "Loss")
    axes[0, 0].legend(frameon=False)

    line(axes[0, 1], warmup, "train_huber", TRAIN_COLOR, "Huber")
    decorate(axes[0, 1], warmup, "(b) Warmup Huber", "Huber loss")

    line(
        axes[0, 2],
        warmup,
        "validation_macro_per_gene_spearman",
        VAL_COLOR,
        "validation",
    )
    decorate(
        axes[0, 2],
        warmup,
        "(c) Warmup selection metric",
        "Macro per-gene Spearman",
    )

    joint_x = [row["epoch"] for row in joint]
    weighted_dependency = [
        row["train_dependency"] * row["train_lambda_dep"] for row in joint
    ]
    line(axes[1, 0], joint, "train_total", TRAIN_COLOR, "total")
    axes[1, 0].plot(
        joint_x,
        weighted_dependency,
        color=SECONDARY_COLOR,
        linewidth=1.6,
        marker="o",
        markersize=3.0,
        label="lambda_dep x dependency",
    )
    line(axes[1, 0], joint, "train_response", TERTIARY_COLOR, "response")
    decorate(axes[1, 0], joint, "(d) Joint train objective", "Weighted loss")
    axes[1, 0].legend(frameon=False)

    line(axes[1, 1], joint, "train_dependency", TRAIN_COLOR, "dependency")
    decorate(axes[1, 1], joint, "(e) Joint dependency term", "Raw dependency loss")

    line(
        axes[1, 2],
        joint,
        "validation_macro_per_gene_spearman",
        VAL_COLOR,
        "validation",
    )
    decorate(axes[1, 2], joint, "(f) Joint selection metric", "Macro per-gene Spearman")

    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, marker="o", linewidth=1.6, label="Train"),
        Line2D(
            [0], [0], color=VAL_COLOR, marker="o", linewidth=1.6, label="Validation"
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linestyle=":",
            linewidth=1.0,
            label="Selected epoch",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
