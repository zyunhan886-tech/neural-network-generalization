"""
Shared experiment infrastructure: run a single training trial and append its
per-epoch metrics to results/metrics.csv.

All experiments share one CSV in tidy (long) format, where each row is one
epoch of one run. This keeps the baseline, hidden_size, learning_rate and
data_efficiency results in a single table that is easy to filter and plot.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics.csv"
FIGURES_DIR = ROOT / "results" / "figures"

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


def run_trial(experiment, training_data, evaluation_data, hidden, eta,
              mini_batch_size, epochs, seed, run_label=None, verbose=True):
    """Train one configuration and return rows ready to be written to CSV."""
    from src.network import Network

    hidden = list(hidden)
    sizes = [784] + hidden + [10]
    label = run_label or "-".join(str(h) for h in hidden)

    if verbose:
        print(f"\n[{experiment}] {label}  sizes={sizes} eta={eta} "
              f"batch={mini_batch_size} n_train={len(training_data)} seed={seed}")

    net = Network(sizes, seed=seed)
    history = net.SGD(training_data, epochs, mini_batch_size, eta,
                      evaluation_data=evaluation_data, verbose=verbose)

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


def append_rows(rows, path=METRICS_CSV):
    """Append rows to the CSV, writing the header if the file is new."""
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
    """
    Drop the previous records of one experiment before re-running it, so that
    repeated runs do not accumulate duplicate rows.
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="") as f:
        kept = [r for r in csv.DictReader(f) if r["experiment"] != experiment]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
