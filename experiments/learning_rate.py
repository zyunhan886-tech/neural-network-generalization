"""
Stage 2b - Learning rate.

Research question: how does the learning rate eta change the shape of the
training curve? How slow is training when eta is too small, and at what order
of magnitude does it start to oscillate or diverge? Where does the useful
range lie?

Controlled variables: network architecture, mini_batch_size, epochs and
training set size are fixed; only eta changes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "learning_rate"

# TODO: sweep the learning rate on a logarithmic scale rather than in equal
# steps. Something like [0.01, 0.1, 1.0, 3.0, 10.0] works well: the extremes
# are needed to show both failure modes - too small to make progress, and
# large enough to oscillate - which is more informative than only sampling
# the range that works.
ETA_GRID = [
    0.01,
    0.1,
    1.0,
    3.0,
    10.0,
]

# Controlled variables - kept identical to the baseline.
HIDDEN = [30]
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if not ETA_GRID:
        sys.exit("Fill in ETA_GRID with the learning rates to compare first.")

    training_data, validation_data, _ = load_data_wrapper()
    clear_experiment(EXPERIMENT)

    for eta in ETA_GRID:
        for seed in args.seeds:
            rows = run_trial(
                experiment=EXPERIMENT,
                training_data=training_data,
                evaluation_data=validation_data,
                hidden=HIDDEN,
                eta=eta,
                mini_batch_size=MINI_BATCH_SIZE,
                epochs=args.epochs,
                seed=seed,
                run_label=f"eta={eta}",
            )
            append_rows(rows)


if __name__ == "__main__":
    main()
