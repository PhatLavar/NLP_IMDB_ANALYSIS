"""Two report figures from the tables written by eval/error_analysis.py.

Outputs (results/eval/figures/):
    confusion_matrices.png   5 confusion matrices (counts, colour = % of the true class)
    error_categories.png     hand-labelled error categories in the 100-review DL error sample

Usage (from the repository root, after eval/error_analysis.py):
    python eval/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/eval/figures"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE = "#2a78d6"
BLUES = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE})


def confusion(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.2))
    for ax, r in zip(axes, metrics.itertuples()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1f}%", ha="center", va="center",
                        color="white" if pct[i, j] > 50 else INK)
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_xlabel("Predicted")
        ax.set_title(f"{r.model}\naccuracy {r.accuracy:.4f}", fontsize=9)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True")
    fig.suptitle("Test-set confusion matrices (25,000 reviews; % = share of the true class)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def categories(counts):
    order = ["mixed", "other", "label_noise", "sarcasm", "negation", "long_truncated"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, (kind, title) in zip(axes, [("main", "Main category (first value)"),
                                        ("any", "Any category value (multi-label)")]):
        part = counts[(counts["label_kind"] == kind) & (counts["slice"] == "all")].set_index("category").loc[order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, part["n"], height=0.6, color=BLUE)
        for yi, n in zip(y, part["n"]):
            ax.text(n + 1, yi, str(n), va="center", color=INK2)
        ax.set_yticks(y, order)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=9, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.set_xlabel("reviews (of 100)")
    fig.suptitle("Hand-labelled reasons for DistilBERT errors (sample counts, 100 reviews)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_categories.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    confusion(pd.read_csv(ROOT / "results/eval/metrics_table.csv", encoding="utf-8-sig"))
    categories(pd.read_csv(ROOT / "results/eval/error_analysis/category_counts.csv", encoding="utf-8-sig"))
    print(f"Wrote confusion_matrices.png and error_categories.png to {OUT}")


if __name__ == "__main__":
    main()
