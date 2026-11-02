""" Creating neural networks, initiating training, and recording experimental results. """

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics.csv"
FIGURES_DIR = ROOT / "results" / "figures"

# columns name of CSV file
FIELDNAMES = [
    "experiment",       # which experiment, e.g. "baseline" or "hidden_size"
    "run_label",        # readable label for this configuration, e.g. "hidden=30"
    "seed",
    "hidden",           # hidden layer sizes, e.g. "30" or "30-20"
    "eta",              # learning rate
    "mini_batch_size",
    "n_train",          # number of training examples actually used
    "epoch",
    "eval_accuracy",
    "eval_correct",
    "eval_total",
    "seconds",          # wall-clock time since the start of training
]


def bootstrap_path():
    """Allow scripts in experiments/ to import src.* from any directory."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def run_trial(experiment, training_data, evaluation_data, hidden, eta, mini_batch_size, epochs, seed, run_label=None, verbose=True):
    """Train one configuration and return rows ready to be written to CSV."""

    # Determine the network architecture
    from src.network import Network

    hidden = list(hidden)
    sizes = [784] + hidden + [10]
    label = run_label or "-".join(str(h) for h in hidden)   # if no label is given, use the hidden layer sizes as a label

    if verbose:
        print(f"\n[{experiment}] {label}  sizes={sizes} eta={eta} "
              f"batch={mini_batch_size} n_train={len(training_data)} seed={seed}")

    # Create the network and train it
    net = Network(sizes, seed=seed)
    history = net.SGD(training_data, epochs, mini_batch_size, eta, evaluation_data=evaluation_data, verbose=verbose)

    # Convert the per-epoch history into a list of rows for CSV output
    rows = []
    for rec in history:
        rows.append({
            "experiment": experiment,
            "run_label": label,
            "seed": seed,
            "hidden": "-".join(str(h) for h in hidden),
            "eta": eta,
            "mini_batch_size": mini_batch_size,
            "n_train": len(training_data),
            "epoch": rec["epoch"],
            "eval_accuracy": round(rec.get("eval_accuracy", float("nan")), 6),
            "eval_correct": rec.get("eval_correct"),
            "eval_total": rec.get("eval_total"),
            "seconds": rec["seconds"],
        })
    return rows


# write the result to CSV file
def append_rows(rows, path=METRICS_CSV):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  -> wrote {len(rows)} rows to {path.relative_to(ROOT)}")


def clear_experiment(experiment, path=METRICS_CSV):
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="") as f:
        kept = [r for r in csv.DictReader(f) if r["experiment"] != experiment]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
