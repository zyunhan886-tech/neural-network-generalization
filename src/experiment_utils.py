"""
实验公共设施：跑一次训练 trial、把每个 epoch 的指标追加进 results/metrics.csv。

所有实验共用一张 tidy（长格式）CSV，每行 = 一个 run 的一个 epoch。
这样 hidden_size / learning_rate / data_efficiency 的结果可以统一读取和画图。
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics.csv"
FIGURES_DIR = ROOT / "results" / "figures"

FIELDNAMES = [
    "experiment",      # 哪个对照实验，如 "baseline" / "hidden_size"
    "run_label",       # 这一组配置的可读标签，如 "hidden=30"
    "seed",
    "hidden",          # 隐藏层结构，如 "30" 或 "30-20"
    "eta",
    "mini_batch_size",
    "n_train",         # 实际使用的训练样本数
    "epoch",
    "eval_accuracy",
    "eval_correct",
    "eval_total",
    "seconds",         # 从训练开始累计的墙钟时间
]


def bootstrap_path():
    """让 experiments/ 下的脚本能 import src.*，无论从哪个目录运行。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def run_trial(experiment, training_data, evaluation_data, hidden, eta,
              mini_batch_size, epochs, seed, run_label=None, verbose=True):
    """跑一次完整训练，返回可直接写进 CSV 的行列表。"""
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
    """追加写入 CSV，文件不存在时自动写表头。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  -> 已写入 {len(rows)} 行到 {path.relative_to(ROOT)}")


def clear_experiment(experiment, path=METRICS_CSV):
    """重跑某个实验前，先删掉它上一次的记录，避免重复累积。"""
    path = Path(path)
    if not path.exists():
        return
    with open(path, newline="") as f:
        kept = [r for r in csv.DictReader(f) if r["experiment"] != experiment]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
