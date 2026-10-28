"""
Stage 3 - Generate figures from results/metrics.csv into results/figures/.

Usage: python experiments/plot.py <experiment_name>

The loading side is already written: it filters by experiment, groups runs by
label, and averages across seeds. Which figures to draw, and how, is left to
the analysis.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.experiment_utils import FIGURES_DIR, METRICS_CSV


def load_curves(experiment):
    """Return {run_label: {epoch: [accuracy for each seed]}}."""
    curves = defaultdict(lambda: defaultdict(list))
    with open(METRICS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] != experiment:
                continue
            curves[row["run_label"]][int(row["epoch"])].append(
                float(row["eval_accuracy"]))
    return curves


def mean_curve(by_epoch):
    """Average across seeds; return (epochs, means, standard deviations)."""
    epochs = sorted(by_epoch)
    means = [statistics.mean(by_epoch[e]) for e in epochs]
    stdevs = [statistics.stdev(by_epoch[e]) if len(by_epoch[e]) > 1 else 0.0
              for e in epochs]
    return epochs, means, stdevs


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python experiments/plot.py <experiment_name>")
    experiment = sys.argv[1]

    curves = load_curves(experiment)
    if not curves:
        sys.exit(f"No rows for experiment={experiment} in metrics.csv. "
                 "Run the experiment first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: this is the plainest possible training curve. Extend it as needed,
    # for example a +/- 1 standard deviation band with fill_between, a log
    # scale on the x axis, or - for data_efficiency - a plot of final accuracy
    # against training set size, grouped by hidden size.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in sorted(curves):
        epochs, means, _ = mean_curve(curves[label])
        ax.plot(epochs, means, marker="o", markersize=3, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(experiment)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / f"{experiment}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
