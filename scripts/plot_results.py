"""Plot held-out predictions and residuals from evaluate.py's saved arrays."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/evaluation.predictions.npz")
    parser.add_argument("--output", default="results/evaluation.png")
    args = parser.parse_args()
    with np.load(ROOT / args.predictions) as data:
        target = data["target_mm"] * 1000
        neural, ridge = data["neural"] * 1000, data["ridge_full_window"] * 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    bins = np.histogram_bin_edges(np.concatenate((neural - target, ridge - target)), bins=45)
    for values, label, color in ((neural, "MLP", "#2563eb"), (ridge, "Ridge", "#d97706")):
        axes[0].scatter(target, values, s=8, alpha=0.35, label=label, color=color)
        axes[1].hist(values - target, bins=bins, alpha=0.55, label=label, color=color)
    low, high = min(target.min(), neural.min(), ridge.min()), max(target.max(), neural.max(), ridge.max())
    axes[0].plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Synthetic target (µm)", ylabel="Prediction (µm)", title="Held-out predictions")
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set(xlabel="Prediction − target (µm)", ylabel="Window count", title="Residual distribution")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.15)
    fig.suptitle("Synthetic test runs only · overlapping windows are correlated", fontsize=11)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
