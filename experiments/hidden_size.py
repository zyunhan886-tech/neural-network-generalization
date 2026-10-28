"""
Stage 2a - Hidden layer size.

Research question: with every other hyper-parameter fixed, how does the width
of the hidden layer affect convergence speed and final validation accuracy?
Is there a point where adding more neurons stops helping?

Controlled variables: eta, mini_batch_size, epochs, training set size and the
random seeds are all held fixed; only the hidden layer size changes. Each
configuration is run with several seeds so that a real effect can be told
apart from random variation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "hidden_size"

# TODO: choose the widths to compare. They should span roughly an order of
# magnitude for the diminishing-returns point to be visible, for example
# [[10], [30], [100]] to start with, optionally [5], [200] or a second hidden
# layer such as [30, 30].
HIDDEN_GRID = [
    [16],
    [30],
    [64],
    [128],
]

# Controlled variables - kept identical to the baseline.
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not HIDDEN_GRID:
        sys.exit("Fill in HIDDEN_GRID with the configurations to compare first.")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    for hidden in HIDDEN_GRID:
        for seed in args.seeds:
            rows = run_trial(
                experiment=EXPERIMENT,
                training_data=training_data,
                evaluation_data=validation_data,
                hidden=hidden,
                eta=ETA,
                mini_batch_size=MINI_BATCH_SIZE,
                epochs=args.epochs,
                seed=seed,
                run_label=f"hidden={'-'.join(map(str, hidden))}",
            )
            append_rows(rows)


if __name__ == "__main__":
    main()
