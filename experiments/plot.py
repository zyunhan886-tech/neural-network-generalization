"""
阶段 3 —— 从 results/metrics.csv 生成图表到 results/figures/。

用法: python experiments/plot.py <experiment_name>

读取部分已经写好（按 experiment 过滤、按 run_label 分组、跨 seed 取均值），
具体要画哪些图、怎么画由你决定。
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
    """返回 {run_label: {epoch: [各 seed 的 accuracy]}}"""
    curves = defaultdict(lambda: defaultdict(list))
    with open(METRICS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] != experiment:
                continue
            curves[row["run_label"]][int(row["epoch"])].append(
                float(row["eval_accuracy"]))
    return curves


def mean_curve(by_epoch):
    """跨 seed 求均值，返回 (epochs, means, stdevs)"""
    epochs = sorted(by_epoch)
    means = [statistics.mean(by_epoch[e]) for e in epochs]
    stdevs = [statistics.stdev(by_epoch[e]) if len(by_epoch[e]) > 1 else 0.0
              for e in epochs]
    return epochs, means, stdevs


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: python experiments/plot.py <experiment_name>")
    experiment = sys.argv[1]

    curves = load_curves(experiment)
    if not curves:
        sys.exit(f"metrics.csv 里没有 experiment={experiment} 的数据，先去跑实验。")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # TODO(你来画): 下面是最朴素的训练曲线图，按需改造 ——
    # 比如加 ±1 std 的误差带 (plt.fill_between)、改成 log x 轴、
    # 或者对 data_efficiency 改画 "final accuracy vs n_train, 按 hidden 分组"。
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
    print(f"图已保存: {out}")


if __name__ == "__main__":
    main()
