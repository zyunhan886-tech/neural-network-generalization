"""
Stage 1 - Reproduce Nielsen's baseline network.

Research question: does a from-scratch 784-30-10 network trained with
mini-batch SGD reproduce the ~95% validation accuracy reported in the book,
and how much does the result vary between random seeds?

All hyper-parameter comparisons in this project are made on the validation
set. The test set is held out and used only once, after the final
configuration has been chosen.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "baseline"

# The standard configuration from Chapter 1 of the book.
HIDDEN = [30]
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="several seeds, to estimate run-to-run variance")
    args = parser.parse_args()

    print("Loading MNIST ...")
    training_data, validation_data, test_data = load_data_wrapper()
    print(f"train={len(training_data)} val={len(validation_data)} test={len(test_data)}")

    clear_experiment(EXPERIMENT)

    finals = []
    for seed in args.seeds:
        rows = run_trial(
            experiment=EXPERIMENT,
            training_data=training_data,
            evaluation_data=validation_data,
            hidden=HIDDEN,
            eta=ETA,
            mini_batch_size=MINI_BATCH_SIZE,
            epochs=args.epochs,
            seed=seed,
            run_label=f"seed={seed}",
        )
        append_rows(rows)
        finals.append(rows[-1]["eval_accuracy"])

    mean = sum(finals) / len(finals)
    spread = max(finals) - min(finals)
    print("\n=== Baseline summary (validation set) ===")
    for seed, acc in zip(args.seeds, finals):
        print(f"  seed={seed}: {acc:.4f}")
    print(f"  mean={mean:.4f}  spread={spread:.4f}")


if __name__ == "__main__":
    main()
