"""
阶段 1 —— 复现 Nielsen 的基础网络，确立 baseline。

Research question: 手写实现的 784-30-10 全连接网络 + SGD，在 MNIST 上
能否复现书中报告的 ~95% 准确率？不同随机种子之间的波动有多大？

注意：调参阶段一律在 validation set 上评估，test set 留到最后一次性使用，
避免信息泄漏。
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_utils import append_rows, clear_experiment, run_trial
from src.mnist_loader import load_data_wrapper

EXPERIMENT = "baseline"

# Nielsen 书中第 1 章的标准配置
HIDDEN = [30]
ETA = 3.0
MINI_BATCH_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="多个种子用来估计 run-to-run 方差")
    args = parser.parse_args()

    print("加载 MNIST ...")
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
    print("\n=== Baseline 汇总 (validation) ===")
    for seed, acc in zip(args.seeds, finals):
        print(f"  seed={seed}: {acc:.4f}")
    print(f"  mean={mean:.4f}  spread={spread:.4f}")


if __name__ == "__main__":
    main()
