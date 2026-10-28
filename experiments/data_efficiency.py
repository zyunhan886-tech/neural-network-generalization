"""
Stage 2c - Training set size x network capacity.

Research questions:
  1. How does validation accuracy grow with the number of training examples,
     and where does it start to saturate?
  2. Do larger networks only pay off once there is enough data? With little
     data, does the larger network actually do worse because it overfits?

This is the only two-dimensional sweep in the project. If the interaction is
real, the curves for different hidden sizes should cross on a plot of final
accuracy against training set size.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "data_efficiency"

# TODO: pick training set sizes on a logarithmic scale, covering "clearly not
# enough" through to the full set, for example [1000, 5000, 20000, 50000].
N_TRAIN_GRID = [
    # 1000,
    # 5000,
    # 20000,
    # 50000,
]

# TODO: at least two capacities are needed to see an interaction effect,
# for example [[10], [100]].
HIDDEN_GRID = [
    # [10],
    # [100],
]

# Controlled variables.
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not N_TRAIN_GRID or not HIDDEN_GRID:
        sys.exit("Fill in N_TRAIN_GRID and HIDDEN_GRID first.")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    # Subsets are taken as a fixed prefix. The MNIST training set is already
    # shuffled, so the first n examples are roughly class-balanced. Stratified
    # sampling would be stricter; this choice is noted in the report's
    # Limitations section.
    for n_train in N_TRAIN_GRID:
        subset = training_data[:n_train]
        for hidden in HIDDEN_GRID:
            for seed in args.seeds:
                rows = run_trial(
                    experiment=EXPERIMENT,
                    training_data=subset,
                    evaluation_data=validation_data,
                    hidden=hidden,
                    eta=ETA,
                    mini_batch_size=MINI_BATCH_SIZE,
                    epochs=args.epochs,
                    seed=seed,
                    run_label=f"n={n_train},hidden={'-'.join(map(str, hidden))}",
                )
                append_rows(rows)


if __name__ == "__main__":
    main()
